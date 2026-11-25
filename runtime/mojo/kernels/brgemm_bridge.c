#include <limits.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include "oneapi/dnnl/dnnl_ukernel.h"

#ifndef DNNL_EXPERIMENTAL_UKERNEL
#error "oneDNN ukernel support (DNNL_EXPERIMENTAL_UKERNEL) is required"
#endif

typedef struct IntentBrgemmKey {
  int64_t M, N, K, lda, ldb, ldc;
  int32_t a_dt, b_dt, c_dt, add_c;
} IntentBrgemmKey;

typedef struct IntentBrgemmEntry {
  IntentBrgemmKey key;
  dnnl_brgemm_t brg;
  size_t scratch_bytes;
  struct IntentBrgemmEntry *next;
} IntentBrgemmEntry;

#define INTENT_BRGEMM_BUCKETS 256
static IntentBrgemmEntry *g_buckets[INTENT_BRGEMM_BUCKETS];
static pthread_rwlock_t g_lock = PTHREAD_RWLOCK_INITIALIZER;
static uint64_t g_epoch = 1;

static dnnl_data_type_t map_dtype(int32_t code) {
  switch (code) {
  case 0:
    return dnnl_f32;
  case 1:
    return dnnl_bf16;
  case 2:
    return dnnl_f16;
  case 3:
    return dnnl_s8;
  case 4:
    return dnnl_u8;
  case 5:
    return dnnl_s32;
  default:
    return dnnl_data_type_undef;
  }
}

static uint64_t hash_key(const IntentBrgemmKey *k) {
  uint64_t h = 1469598103934665603ULL;
  const uint64_t *words = (const uint64_t *)k;
  for (size_t i = 0; i < 6; ++i)
    h = (h ^ words[i]) * 1099511628211ULL;
  h = (h ^ (uint64_t)(uint32_t)k->a_dt) * 1099511628211ULL;
  h = (h ^ (uint64_t)(uint32_t)k->b_dt) * 1099511628211ULL;
  h = (h ^ (uint64_t)(uint32_t)k->c_dt) * 1099511628211ULL;
  h = (h ^ (uint64_t)(uint32_t)k->add_c) * 1099511628211ULL;
  return h;
}

static int same_key(const IntentBrgemmKey *a, const IntentBrgemmKey *b) {
  return a->M == b->M && a->N == b->N && a->K == b->K && a->lda == b->lda &&
         a->ldb == b->ldb && a->ldc == b->ldc && a->a_dt == b->a_dt &&
         a->b_dt == b->b_dt && a->c_dt == b->c_dt && a->add_c == b->add_c;
}

const void *intent_brgemm_get_ex(int64_t M, int64_t N, int64_t K, int64_t lda,
                                 int64_t ldb, int64_t ldc, int32_t a_dt,
                                 int32_t b_dt, int32_t c_dt, int32_t add_c) {
  static __thread const IntentBrgemmEntry *tls_cache[8];
  static __thread unsigned tls_slot;
  static __thread uint64_t tls_epoch;
  if (M <= 0 || N <= 0 || K <= 0)
    return NULL;
  add_c = add_c ? 1 : 0;
  IntentBrgemmKey key = {M, N, K, lda, ldb, ldc, a_dt, b_dt, c_dt, add_c};
  uint64_t cur_epoch = __atomic_load_n(&g_epoch, __ATOMIC_ACQUIRE);
  if (tls_epoch != cur_epoch) {
    for (unsigned i = 0; i < 8; ++i)
      tls_cache[i] = NULL;
    tls_epoch = cur_epoch;
  } else {
    for (unsigned i = 0; i < 8; ++i) {
      const IntentBrgemmEntry *ce = tls_cache[i];
      if (ce && same_key(&ce->key, &key))
        return ce;
    }
  }
  size_t idx = (size_t)(hash_key(&key) & (INTENT_BRGEMM_BUCKETS - 1));
  pthread_rwlock_rdlock(&g_lock);
  cur_epoch = g_epoch;
  for (IntentBrgemmEntry *e = g_buckets[idx]; e; e = e->next) {
    if (same_key(&e->key, &key)) {
      pthread_rwlock_unlock(&g_lock);
      tls_epoch = cur_epoch;
      tls_cache[tls_slot++ & 7u] = e;
      return e;
    }
  }
  pthread_rwlock_unlock(&g_lock);

  pthread_rwlock_wrlock(&g_lock);
  cur_epoch = g_epoch;
  for (IntentBrgemmEntry *e = g_buckets[idx]; e; e = e->next) {
    if (same_key(&e->key, &key)) {
      pthread_rwlock_unlock(&g_lock);
      tls_epoch = cur_epoch;
      tls_cache[tls_slot++ & 7u] = e;
      return e;
    }
  }
  dnnl_brgemm_t brg = NULL;
  if (dnnl_brgemm_create(&brg, M, N, K, 1, lda, ldb, ldc, map_dtype(a_dt),
                         map_dtype(b_dt), map_dtype(c_dt)) != dnnl_success ||
      (add_c && dnnl_brgemm_set_add_C(brg, 1) != dnnl_success) ||
      dnnl_brgemm_finalize(brg) != dnnl_success ||
      dnnl_brgemm_generate(brg) != dnnl_success) {
    if (brg)
      dnnl_brgemm_destroy(brg);
    pthread_rwlock_unlock(&g_lock);
    return NULL;
  }
  size_t scratch = 0;
  if (dnnl_brgemm_get_scratchpad_size(brg, &scratch) != dnnl_success ||
      scratch > (size_t)INT64_MAX - 64u) {
    dnnl_brgemm_destroy(brg);
    pthread_rwlock_unlock(&g_lock);
    return NULL;
  }
  scratch = scratch < 64u ? 64u : ((scratch + 63u) & ~(size_t)63u);
  IntentBrgemmEntry *entry =
      (IntentBrgemmEntry *)calloc(1, sizeof(IntentBrgemmEntry));
  if (!entry) {
    dnnl_brgemm_destroy(brg);
    pthread_rwlock_unlock(&g_lock);
    return NULL;
  }
  entry->key = key;
  entry->brg = brg;
  entry->scratch_bytes = scratch;
  entry->next = g_buckets[idx];
  g_buckets[idx] = entry;
  pthread_rwlock_unlock(&g_lock);
  tls_epoch = cur_epoch;
  tls_cache[tls_slot++ & 7u] = entry;
  return entry;
}

const void *intent_brgemm_get(int64_t M, int64_t N, int64_t K, int64_t lda,
                              int64_t ldb, int64_t ldc, int32_t a_dt,
                              int32_t b_dt, int32_t c_dt) {
  return intent_brgemm_get_ex(M, N, K, lda, ldb, ldc, a_dt, b_dt, c_dt, 0);
}

void intent_brgemm_cache_clear(void) {
  pthread_rwlock_wrlock(&g_lock);
  __atomic_add_fetch(&g_epoch, 1, __ATOMIC_RELEASE);
  for (size_t i = 0; i < INTENT_BRGEMM_BUCKETS; ++i) {
    IntentBrgemmEntry *e = g_buckets[i];
    g_buckets[i] = NULL;
    while (e) {
      IntentBrgemmEntry *next = e->next;
      if (e->brg)
        dnnl_brgemm_destroy(e->brg);
      free(e);
      e = next;
    }
  }
  pthread_rwlock_unlock(&g_lock);
}

static void __attribute__((destructor)) intent_brgemm_fini(void) {
  intent_brgemm_cache_clear();
}

int64_t intent_brgemm_scratch_bytes(const void *handle) {
  if (!handle)
    return -1;
  return (int64_t)((const IntentBrgemmEntry *)handle)->scratch_bytes;
}

int32_t intent_brgemm_thread_enter(const void *handle) {
  if (!handle)
    return 3;
  return dnnl_brgemm_set_hw_context(((const IntentBrgemmEntry *)handle)->brg) ==
                 dnnl_success
             ? 0
             : 3;
}

void intent_brgemm_thread_leave(void) { dnnl_brgemm_release_hw_context(); }

int32_t intent_brgemm_tile(const void *handle, const void *A, const void *B,
                           void *C, void *scratch) {
  if (!handle || !A || !B || !C || !scratch)
    return 4;
  const dnnl_dim_t offs[2] = {0, 0};
  return dnnl_brgemm_execute(((const IntentBrgemmEntry *)handle)->brg, A, B,
                             offs, C, scratch) == dnnl_success
             ? 0
             : 4;
}

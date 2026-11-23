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
  int32_t a_dt, b_dt, c_dt;
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
  return h;
}

static int same_key(const IntentBrgemmKey *a, const IntentBrgemmKey *b) {
  return a->M == b->M && a->N == b->N && a->K == b->K && a->lda == b->lda &&
         a->ldb == b->ldb && a->ldc == b->ldc && a->a_dt == b->a_dt &&
         a->b_dt == b->b_dt && a->c_dt == b->c_dt;
}

const void *intent_brgemm_get(int64_t M, int64_t N, int64_t K, int64_t lda,
                              int64_t ldb, int64_t ldc, int32_t a_dt,
                              int32_t b_dt, int32_t c_dt) {
  if (M <= 0 || N <= 0 || K <= 0)
    return NULL;
  IntentBrgemmKey key = {M, N, K, lda, ldb, ldc, a_dt, b_dt, c_dt};
  size_t idx = (size_t)(hash_key(&key) & (INTENT_BRGEMM_BUCKETS - 1));
  pthread_rwlock_rdlock(&g_lock);
  for (IntentBrgemmEntry *e = g_buckets[idx]; e; e = e->next) {
    if (same_key(&e->key, &key)) {
      pthread_rwlock_unlock(&g_lock);
      return e;
    }
  }
  pthread_rwlock_unlock(&g_lock);

  pthread_rwlock_wrlock(&g_lock);
  for (IntentBrgemmEntry *e = g_buckets[idx]; e; e = e->next) {
    if (same_key(&e->key, &key)) {
      pthread_rwlock_unlock(&g_lock);
      return e;
    }
  }
  dnnl_brgemm_t brg = NULL;
  if (dnnl_brgemm_create(&brg, M, N, K, 1, lda, ldb, ldc, map_dtype(a_dt),
                         map_dtype(b_dt), map_dtype(c_dt)) != dnnl_success ||
      dnnl_brgemm_finalize(brg) != dnnl_success ||
      dnnl_brgemm_generate(brg) != dnnl_success) {
    if (brg)
      dnnl_brgemm_destroy(brg);
    pthread_rwlock_unlock(&g_lock);
    return NULL;
  }
  size_t scratch = 0;
  dnnl_brgemm_get_scratchpad_size(brg, &scratch);
  scratch = scratch < 64 ? 64 : ((scratch + 63) & ~(size_t)63);
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
  return entry;
}

int64_t intent_brgemm_scratch_bytes(const void *handle) {
  if (!handle)
    return 0;
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
  if (!handle)
    return 4;
  const dnnl_dim_t offs[2] = {0, 0};
  return dnnl_brgemm_execute(((const IntentBrgemmEntry *)handle)->brg, A, B,
                             offs, C, scratch) == dnnl_success
             ? 0
             : 4;
}

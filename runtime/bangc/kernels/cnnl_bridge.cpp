#include <cnnl.h>
#include <cnrt.h>
#include <limits.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

namespace {
enum PlanKind {
  PLAN_MATMUL = 1,
  PLAN_TRANSPOSE = 2,
  PLAN_SOFTMAX = 3,
  PLAN_LAYERNORM = 4
};

struct PlanKey {
  int kind;
  void *queue;
  long long d0, d1, d2;
  int f0, f1, dt0, dt1;
  bool operator==(const PlanKey &o) const {
    return kind == o.kind && queue == o.queue && d0 == o.d0 && d1 == o.d1 &&
           d2 == o.d2 && f0 == o.f0 && f1 == o.f1 && dt0 == o.dt0 &&
           dt1 == o.dt1;
  }
};

struct CachedPlan {
  PlanKey key;
  cnnlHandle_t handle = nullptr;
  cnnlTensorDescriptor_t t0 = nullptr;
  cnnlTensorDescriptor_t t1 = nullptr;
  cnnlTensorDescriptor_t t2 = nullptr;
  cnnlTensorDescriptor_t t3 = nullptr;
  cnnlMatMulDescriptor_t mmDesc = nullptr;
  cnnlMatMulAlgo_t mmAlgo = nullptr;
  cnnlMatMulHeuristicResult_t mmHeur = nullptr;
  cnnlTransposeDescriptor_t trDesc = nullptr;
  void *workspace = nullptr;
  size_t workspaceBytes = 0;
  void *aux0 = nullptr;
  void *aux1 = nullptr;
  CachedPlan *next = nullptr;

  ~CachedPlan() {
    if (aux1)
      cnrtFree(aux1);
    if (aux0)
      cnrtFree(aux0);
    if (workspace)
      cnrtFree(workspace);
    if (trDesc)
      cnnlDestroyTransposeDescriptor(trDesc);
    if (mmHeur)
      cnnlDestroyMatMulHeuristicResult(mmHeur);
    if (mmAlgo)
      cnnlMatMulAlgoDestroy(mmAlgo);
    if (mmDesc)
      cnnlMatMulDescDestroy(mmDesc);
    if (t3)
      cnnlDestroyTensorDescriptor(t3);
    if (t2)
      cnnlDestroyTensorDescriptor(t2);
    if (t1)
      cnnlDestroyTensorDescriptor(t1);
    if (t0)
      cnnlDestroyTensorDescriptor(t0);
    if (handle)
      cnnlDestroy(handle);
  }
};

struct PlanRegistry {
  pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
  CachedPlan *head = nullptr;

  void clearLocked() {
    CachedPlan *cur = head;
    head = nullptr;
    while (cur) {
      CachedPlan *nxt = cur->next;
      delete cur;
      cur = nxt;
    }
  }

  ~PlanRegistry() {
    pthread_mutex_lock(&mu);
    clearLocked();
    pthread_mutex_unlock(&mu);
  }
};

static PlanRegistry g_registry;
static __thread CachedPlan *tls_last_plan = nullptr;
static uint64_t g_epoch = 1;
static __thread uint64_t tls_epoch = 0;

static size_t dtypeBytes(int code) {
  if (code == CNNL_DTYPE_HALF || code == CNNL_DTYPE_BFLOAT16 ||
      code == CNNL_DTYPE_UINT16 || code == CNNL_DTYPE_INT16)
    return 2;
  if (code == CNNL_DTYPE_FLOAT || code == CNNL_DTYPE_INT32 ||
      code == CNNL_DTYPE_UINT32)
    return 4;
  return 1;
}

static CachedPlan *findLocked(const PlanKey &k) {
  if (tls_epoch == g_epoch && tls_last_plan && tls_last_plan->key == k)
    return tls_last_plan;
  for (CachedPlan *p = g_registry.head; p; p = p->next) {
    if (p->key == k) {
      tls_last_plan = p;
      tls_epoch = g_epoch;
      return p;
    }
  }
  return nullptr;
}
} // namespace

extern "C" void intent_cnnl_cleanup() {
  pthread_mutex_lock(&g_registry.mu);
  ++g_epoch;
  tls_last_plan = nullptr;
  g_registry.clearLocked();
  pthread_mutex_unlock(&g_registry.mu);
}

extern "C" int intent_cnnl_matmul(const void *lhs, const void *rhs, void *dst,
                                  long long B, long long M, long long N,
                                  long long K, int trans_a, int trans_b,
                                  long long lhs_batch_stride,
                                  long long rhs_batch_stride,
                                  long long dst_batch_stride, int in_dtype,
                                  int out_dtype, void *queue) {
  if (B <= 0 || M <= 0 || N <= 0 || K <= 0)
    return 0;
  if (M > INT_MAX || N > INT_MAX || K > INT_MAX)
    return (int)CNNL_STATUS_BAD_PARAM;

  PlanKey key{PLAN_MATMUL, queue, M, N, K, trans_a, trans_b, in_dtype, out_dtype};
  pthread_mutex_lock(&g_registry.mu);
  CachedPlan *plan = findLocked(key);
  if (!plan) {
    CachedPlan *np = new CachedPlan();
    np->key = key;
    cnnlStatus_t st = cnnlCreate(&np->handle);
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlSetQueue(np->handle, (cnrtQueue_t)queue);
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlCreateTensorDescriptor(&np->t0);
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlCreateTensorDescriptor(&np->t1);
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlCreateTensorDescriptor(&np->t2);
    int aShape[2] = {trans_a ? (int)K : (int)M, trans_a ? (int)M : (int)K};
    int bShape[2] = {trans_b ? (int)N : (int)K, trans_b ? (int)K : (int)N};
    int cShape[2] = {(int)M, (int)N};
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlSetTensorDescriptor(np->t0, CNNL_LAYOUT_ARRAY,
                                   (cnnlDataType_t)in_dtype, 2, aShape);
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlSetTensorDescriptor(np->t1, CNNL_LAYOUT_ARRAY,
                                   (cnnlDataType_t)in_dtype, 2, bShape);
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlSetTensorDescriptor(np->t2, CNNL_LAYOUT_ARRAY,
                                   (cnnlDataType_t)out_dtype, 2, cShape);
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlSetTensorDescriptorOnchipDataType(np->t2, CNNL_DTYPE_FLOAT);
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlMatMulDescCreate(&np->mmDesc);
    cnnlDataType_t comp = CNNL_DTYPE_FLOAT;
    int zero = 0;
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlSetMatMulDescAttr(np->mmDesc, CNNL_MATMUL_DESC_COMPUTE_TYPE,
                                 &comp, sizeof(comp));
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlSetMatMulDescAttr(np->mmDesc, CNNL_MATMUL_DESC_TRANSA, &trans_a,
                                 sizeof(trans_a));
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlSetMatMulDescAttr(np->mmDesc, CNNL_MATMUL_DESC_TRANSB, &trans_b,
                                 sizeof(trans_b));
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlSetMatMulDescAttr(np->mmDesc, CNNL_MATMUL_ALLOW_TF32, &zero,
                                 sizeof(zero));
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlSetMatMulDescAttr(np->mmDesc, CNNL_MATMUL_USE_BETA, &zero,
                                 sizeof(zero));
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlMatMulAlgoCreate(&np->mmAlgo);
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlCreateMatMulHeuristicResult(&np->mmHeur);
    int returned = 0;
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlGetMatMulAlgoHeuristic(np->handle, np->mmDesc, np->t0, np->t1,
                                      np->t2, np->t2, nullptr, 1, &np->mmHeur,
                                      &returned);
    if (st == CNNL_STATUS_SUCCESS && returned != 1)
      st = CNNL_STATUS_NOT_SUPPORTED;
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlGetMatMulHeuristicResult(np->mmHeur, np->mmAlgo,
                                        &np->workspaceBytes);
    if (st == CNNL_STATUS_SUCCESS && np->workspaceBytes > 0) {
      if (cnrtMalloc(&np->workspace, np->workspaceBytes) != cnrtSuccess)
        st = CNNL_STATUS_ALLOC_FAILED;
    }
    if (st != CNNL_STATUS_SUCCESS) {
      delete np;
      pthread_mutex_unlock(&g_registry.mu);
      return (int)st;
    }
    np->next = g_registry.head;
    g_registry.head = np;
    plan = np;
  }

  float alpha = 1.0f, beta = 0.0f;
  size_t inB = dtypeBytes(in_dtype);
  size_t outB = dtypeBytes(out_dtype);
  cnnlStatus_t runSt = CNNL_STATUS_SUCCESS;
  for (long long b = 0; b < B; ++b) {
    const void *pa =
        (const char *)lhs + (size_t)(b * lhs_batch_stride) * inB;
    const void *pb =
        (const char *)rhs + (size_t)(b * rhs_batch_stride) * inB;
    void *pc = (char *)dst + (size_t)(b * dst_batch_stride) * outB;
    runSt = cnnlMatMul_v2(plan->handle, plan->mmDesc, plan->mmAlgo, &alpha,
                          plan->t0, pa, plan->t1, pb, &beta, plan->t2, pc,
                          plan->workspace, plan->workspaceBytes, plan->t2, pc);
    if (runSt != CNNL_STATUS_SUCCESS)
      break;
  }
  pthread_mutex_unlock(&g_registry.mu);
  return (int)runSt;
}

extern "C" int intent_cnnl_transpose(const void *x, void *y, long long M,
                                     long long N, int elem_bytes, void *queue) {
  if (M <= 0 || N <= 0)
    return 0;
  if (M > INT_MAX || N > INT_MAX)
    return (int)CNNL_STATUS_BAD_PARAM;
  int dt = elem_bytes == 1   ? (int)CNNL_DTYPE_INT8
           : elem_bytes == 2 ? (int)CNNL_DTYPE_UINT16
           : elem_bytes == 4 ? (int)CNNL_DTYPE_INT32
                             : (int)CNNL_DTYPE_INVALID;
  if (dt == (int)CNNL_DTYPE_INVALID)
    return (int)CNNL_STATUS_BAD_PARAM;

  PlanKey key{PLAN_TRANSPOSE, queue, M, N, 0, 0, 0, dt, dt};
  pthread_mutex_lock(&g_registry.mu);
  CachedPlan *plan = findLocked(key);
  if (!plan) {
    CachedPlan *np = new CachedPlan();
    np->key = key;
    cnnlStatus_t st = cnnlCreate(&np->handle);
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlSetQueue(np->handle, (cnrtQueue_t)queue);
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlCreateTensorDescriptor(&np->t0);
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlCreateTensorDescriptor(&np->t1);
    int xDims[2] = {(int)M, (int)N};
    int yDims[2] = {(int)N, (int)M};
    int perm[2] = {1, 0};
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlSetTensorDescriptor(np->t0, CNNL_LAYOUT_ARRAY,
                                   (cnnlDataType_t)dt, 2, xDims);
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlSetTensorDescriptor(np->t1, CNNL_LAYOUT_ARRAY,
                                   (cnnlDataType_t)dt, 2, yDims);
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlCreateTransposeDescriptor(&np->trDesc);
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlSetTransposeDescriptor(np->trDesc, 2, perm);
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlGetTransposeWorkspaceSize(np->handle, np->t0, np->trDesc,
                                         &np->workspaceBytes);
    if (st == CNNL_STATUS_SUCCESS && np->workspaceBytes > 0) {
      if (cnrtMalloc(&np->workspace, np->workspaceBytes) != cnrtSuccess)
        st = CNNL_STATUS_ALLOC_FAILED;
    }
    if (st != CNNL_STATUS_SUCCESS) {
      delete np;
      pthread_mutex_unlock(&g_registry.mu);
      return (int)st;
    }
    np->next = g_registry.head;
    g_registry.head = np;
    plan = np;
  }

  cnnlStatus_t runSt =
      cnnlTranspose_v2(plan->handle, plan->trDesc, plan->t0, x, plan->t1, y,
                       plan->workspace, plan->workspaceBytes);
  pthread_mutex_unlock(&g_registry.mu);
  return (int)runSt;
}

extern "C" int intent_cnnl_softmax(const void *x, void *y, long long M,
                                   long long N, int dtype_code, void *queue) {
  if (M <= 0 || N <= 0)
    return 0;
  if (M > INT_MAX || N > INT_MAX)
    return (int)CNNL_STATUS_BAD_PARAM;

  PlanKey key{PLAN_SOFTMAX, queue, M, N, 0, 0, 0, dtype_code, dtype_code};
  pthread_mutex_lock(&g_registry.mu);
  CachedPlan *plan = findLocked(key);
  if (!plan) {
    CachedPlan *np = new CachedPlan();
    np->key = key;
    cnnlStatus_t st = cnnlCreate(&np->handle);
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlSetQueue(np->handle, (cnrtQueue_t)queue);
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlCreateTensorDescriptor(&np->t0);
    int dims[3] = {(int)M, 1, (int)N};
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlSetTensorDescriptor(np->t0, CNNL_LAYOUT_ARRAY,
                                   (cnnlDataType_t)dtype_code, 3, dims);
    if (st != CNNL_STATUS_SUCCESS) {
      delete np;
      pthread_mutex_unlock(&g_registry.mu);
      return (int)st;
    }
    np->next = g_registry.head;
    g_registry.head = np;
    plan = np;
  }

  cnnlStatus_t runSt = cnnlSoftmaxForward_v2(
      plan->handle, CNNL_SOFTMAX_ACCURATE, CNNL_SOFTMAX_MODE_LOW_DIMENSION,
      CNNL_COMPUTATION_HIGH_PRECISION, nullptr, plan->t0, x, nullptr, plan->t0,
      y);
  pthread_mutex_unlock(&g_registry.mu);
  return (int)runSt;
}

extern "C" int intent_cnnl_layer_norm(const void *x, const void *gain,
                                      const void *bias, void *y, long long M,
                                      long long N, float eps, int dtype_code,
                                      void *queue) {
  if (M <= 0 || N <= 0)
    return 0;
  if (M > INT_MAX || N > INT_MAX)
    return (int)CNNL_STATUS_BAD_PARAM;

  PlanKey key{PLAN_LAYERNORM, queue, M, N, 0, 0, 0, dtype_code, dtype_code};
  pthread_mutex_lock(&g_registry.mu);
  CachedPlan *plan = findLocked(key);
  if (!plan) {
    CachedPlan *np = new CachedPlan();
    np->key = key;
    cnnlStatus_t st = cnnlCreate(&np->handle);
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlSetQueue(np->handle, (cnrtQueue_t)queue);
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlCreateTensorDescriptor(&np->t0);
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlCreateTensorDescriptor(&np->t1);
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlCreateTensorDescriptor(&np->t2);
    int xDims[2] = {(int)M, (int)N};
    int cDim[1] = {(int)N};
    int mDim[1] = {(int)M};
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlSetTensorDescriptor(np->t0, CNNL_LAYOUT_ARRAY,
                                   (cnnlDataType_t)dtype_code, 2, xDims);
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlSetTensorDescriptor(np->t1, CNNL_LAYOUT_ARRAY,
                                   (cnnlDataType_t)dtype_code, 1, cDim);
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlSetTensorDescriptor(np->t2, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT,
                                   1, mDim);
    if (st == CNNL_STATUS_SUCCESS)
      st = cnnlGetLayerNormOpWorkspaceSize(np->handle, 1, np->t0,
                                           &np->workspaceBytes);
    if (st == CNNL_STATUS_SUCCESS && np->workspaceBytes > 0) {
      if (cnrtMalloc(&np->workspace, np->workspaceBytes) != cnrtSuccess)
        st = CNNL_STATUS_ALLOC_FAILED;
    }
    if (st == CNNL_STATUS_SUCCESS) {
      if (cnrtMalloc(&np->aux0, (size_t)M * sizeof(float)) != cnrtSuccess ||
          cnrtMalloc(&np->aux1, (size_t)M * sizeof(float)) != cnrtSuccess)
        st = CNNL_STATUS_ALLOC_FAILED;
    }
    if (st != CNNL_STATUS_SUCCESS) {
      delete np;
      pthread_mutex_unlock(&g_registry.mu);
      return (int)st;
    }
    np->next = g_registry.head;
    g_registry.head = np;
    plan = np;
  }

  cnnlStatus_t runSt = cnnlLayerNormForward(
      plan->handle, plan->t0, x, 1, plan->t1, gain, bias, eps, plan->workspace,
      plan->workspaceBytes, plan->t0, y, plan->t2, plan->aux0, plan->aux1);
  pthread_mutex_unlock(&g_registry.mu);
  return (int)runSt;
}

#define _GNU_SOURCE
#include <stdint.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>

typedef void (*intent_task_range)(void *, int64_t, int64_t);

// GNU OpenMP's synchronous parallel-region ABI. No OpenMP scheduling or
// process-wide thread-count setting is needed by the generated kernel.
extern void GOMP_parallel(void (*)(void *), void *, unsigned, unsigned);
extern int omp_get_thread_num(void);
extern int omp_get_num_threads(void);

struct task_context {
  intent_task_range task;
  void *captures;
  int64_t count;
  int64_t workers;
  cpu_set_t allowed;
};

static void affinity_error(const char *operation) {
  perror(operation);
  abort();
}

static void invoke_range(void *opaque) {
  const struct task_context *context = opaque;
  int64_t rank = omp_get_thread_num();
  int64_t size = omp_get_num_threads();
  if (size > context->workers)
    size = context->workers;
  cpu_set_t previous, intersection;
  int restore = 0;
  if (rank != 0) {
    if (sched_getaffinity(0, sizeof(previous), &previous) != 0)
      affinity_error("Intent CPU dispatch: reading worker affinity");
    CPU_AND(&intersection, &previous, &context->allowed);
    restore = !CPU_EQUAL(&intersection, &previous);
    if (restore && sched_setaffinity(0, sizeof(context->allowed), &context->allowed) != 0)
      affinity_error("Intent CPU dispatch: binding worker affinity");
  }
  if (rank < size) {
    int64_t chunk = context->count / size;
    int64_t extra = context->count % size;
    int64_t begin = rank * chunk + (rank < extra ? rank : extra);
    int64_t end = begin + chunk + (rank < extra);
    context->task(context->captures, begin, end);
  }
  if (restore && sched_setaffinity(0, sizeof(previous), &previous) != 0)
    affinity_error("Intent CPU dispatch: restoring worker affinity");
}

void intent_cpu_dispatch_team(intent_task_range task, void *captures,
                              int64_t count, int64_t workers, int64_t team_size) {
  if (count <= 0)
    return;
  if (workers == 1 || count == 1) {
    task(captures, 0, count);
    return;
  }
  if (workers > count)
    workers = count;
  struct task_context context = { .task = task, .captures = captures,
                                  .count = count, .workers = workers };
  if (sched_getaffinity(0, sizeof(context.allowed), &context.allowed) != 0)
    affinity_error("Intent CPU dispatch: reading caller affinity");
  // A team owns only this synchronous invocation. Captures remain alive until
  // its implicit join, including when OpenMP serializes a nested parallel call.
  // Keep the target's team budget across configurations and successive task
  // regions. Shrinking a GNU OpenMP team retires the excess worker threads;
  // growing it again would measure thread creation instead of hot execution.
  GOMP_parallel(invoke_range, &context, (unsigned)team_size, 0);
}

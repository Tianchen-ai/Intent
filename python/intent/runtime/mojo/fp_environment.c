#include <stdint.h>
#include <xmmintrin.h>

uint32_t intent_cpu_enter_ieee(void) {
  uint32_t previous = _mm_getcsr();
  _mm_setcsr(previous & ~(UINT32_C(0x8040) | UINT32_C(0x6000)));
  return previous;
}

void intent_cpu_leave_ieee(uint32_t previous) {
  _mm_setcsr(previous);
}

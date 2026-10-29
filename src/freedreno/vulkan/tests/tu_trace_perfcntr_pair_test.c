/*
 * Copyright © 2026 Jetha Chan
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>

#include "../tu_trace_perfcntr_pair.h"

static int failures;

#define MHZ 1000000ull

static const struct tu_pc_pair_params P = {
   .width = 64, .tick_hz = 19200000, .fmin_hz = 100 * MHZ, .fmax_hz = 1200 * MHZ,
   .slack_ticks = 40,
};

/* counter 0 = RBBM_ALWAYS_COUNT witness, 1 = SP busy, 2 = CP_NUM_PREEMPTIONS */
static void
check(const char *name, uint64_t b0, uint64_t e0, uint64_t b1, uint64_t e1,
      uint64_t b2, uint64_t e2, uint64_t t0, uint64_t t1, const uint32_t *sel_b,
      const uint32_t *sel_e, uint32_t width, enum tu_pc_pair_class want_cls,
      uint32_t want_reasons, uint64_t want_d1)
{
   static const uint32_t sel_expect[3] = { 0, 3, 3 };
   uint64_t b[3] = { b0, b1, b2 }, e[3] = { e0, e1, e2 }, d[3];
   struct tu_pc_pair_params p = P;
   p.width = width;
   struct tu_pc_pair_in in = {
      .n = 3, .begin = b, .end = e, .ts_begin = t0, .ts_end = t1, .ts_valid = true,
      .cycles_idx = 0, .preempt_idx = 2,
      .sel_expect = sel_b ? sel_expect : NULL, .sel_begin = sel_b, .sel_end = sel_e,
   };
   struct tu_pc_pair_out o;
   enum tu_pc_pair_class c = tu_pc_classify_pair(&p, &in, d, &o);
   int ok = c == want_cls && o.reasons == want_reasons &&
            (want_cls == TU_PC_PAIR_CLOBBERED || d[1] == want_d1);
   printf("%-28s pair=%-11s reasons=0x%02x d1=%llu wrapped=0x%llx %s\n", name,
          tu_pc_pair_class_name(c), o.reasons, (unsigned long long) d[1],
          (unsigned long long) o.wrapped_mask, ok ? "PASS" : "FAIL");
   failures += !ok;
}

int
main(void)
{
   const uint32_t sel_ok[3] = { 0, 3, 3 }, sel_bad[3] = { 0, 0, 3 };
   /* 1 ms = 19200 ticks; 903 MHz -> 903000 cycles. */
   const uint64_t T0 = 1000000, T1 = T0 + 19200, C = 903000;

   /* Clean pair, witness began far above fmax*dt: zap excluded. */
   check("clean_proven", 50000000, 50000000 + C, 10, 510, 4, 4, T0, T1, sel_ok, sel_ok, 64,
         TU_PC_PAIR_OK, 0, 500);
   /* Clean but witness began low (pair started right after the submit-start zap). */
   check("clean_unproven", 1000, 1000 + C, 10, 510, 0, 0, T0, T1, sel_ok, sel_ok, 64,
         TU_PC_PAIR_OK_UNPROVEN, 0, 500);
   /* P2 tex row shape: zap mid-pair, every counter restarts from 0. */
   check("reset_to_zero_negative", 2000000000, 4400000, 40861040, 0, 0, 0, T0, T0 + 4224000,
         sel_ok, sel_ok, 64, TU_PC_PAIR_CLOBBERED, TU_PC_REASON_RESET, 0);
   /* P2 row with positive witness but 22 MHz implied and SP negative. */
   check("reset_partial_positive", 300000, 4893777 + 300000, 40861040, 0, 0, 0, T0, T0 + 4224000,
         sel_ok, sel_ok, 64, TU_PC_PAIR_CLOBBERED, TU_PC_REASON_RESET | TU_PC_REASON_CLOCK_LO, 0);
   /* Zap early in the pair, all counters still above begin: only the clock bound sees it. */
   check("reset_short_clock_low", 1000, 1500, 10, 20, 0, 0, T0, T1, sel_ok, sel_ok, 64,
         TU_PC_PAIR_CLOBBERED, TU_PC_REASON_CLOCK_LO, 0);
   /* Witness jumped by more than fmax*dt (torn read / foreign load). */
   check("clock_high", 50000000, 50000000 + 100 * C, 10, 510, 0, 0, T0, T1, sel_ok, sel_ok, 64,
         TU_PC_PAIR_CLOBBERED, TU_PC_REASON_CLOCK_HI, 0);
   /* 64-bit wraparound of the SP counter: valid modular delta. */
   check("wrap64", 50000000, 50000000 + C, 0xfffffffffffffff0ull, 0x10, 0, 0, T0, T1, sel_ok,
         sel_ok, 64, TU_PC_PAIR_OK, 0, 0x20);
   /* 48-bit counter wraparound. */
   check("wrap48", 50000000, 50000000 + C, 0xfffffffffff0ull, 0x8, 0, 0, T0, T1, sel_ok, sel_ok,
         48, TU_PC_PAIR_OK, 0, 0x18);
   /* Decrease that is not a wrap (begin in the low half). */
   check("decrease_not_wrap", 50000000, 50000000 + C, 0x7000000000000000ull, 5, 0, 0, T0, T1,
         sel_ok, sel_ok, 64, TU_PC_PAIR_CLOBBERED, TU_PC_REASON_RESET, 0);
   /* Selector clobbered by another process between begin and end. */
   check("select_clobbered_end", 50000000, 50000000 + C, 10, 510, 0, 0, T0, T1, sel_ok, sel_bad,
         64, TU_PC_PAIR_CLOBBERED, TU_PC_REASON_SELECT, 0);
   /* Selector already wrong at begin (IFPC reset to countable 0). */
   check("select_lost_begin", 50000000, 50000000 + C, 10, 510, 0, 0, T0, T1, sel_bad, sel_ok,
         64, TU_PC_PAIR_CLOBBERED, TU_PC_REASON_SELECT, 0);
   /* Preemption witness advanced (zap disabled by sysprof). */
   check("preempted", 50000000, 50000000 + C, 10, 510, 4, 6, T0, T1, sel_ok, sel_ok, 64,
         TU_PC_PAIR_CLOBBERED, TU_PC_REASON_PREEMPT, 0);
   /* Timestamps not increasing. */
   check("ts_backwards", 50000000, 50000000 + C, 10, 510, 0, 0, T1, T0, sel_ok, sel_ok, 64,
         TU_PC_PAIR_CLOBBERED, TU_PC_REASON_TS, 0);
   /* No selector readback: still proven by the witness. */
   check("no_selcheck", 50000000, 50000000 + C, 10, 510, 0, 0, T0, T1, NULL, NULL, 64,
         TU_PC_PAIR_OK, 0, 500);
   /* Very short pair (2 ticks): slack keeps a 231 MHz pair valid. */
   check("short_pair_slack", 50000000, 50000000 + 24, 10, 11, 0, 0, T0, T0 + 2, sel_ok, sel_ok,
         64, TU_PC_PAIR_OK, 0, 1);

   printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
   return failures ? 1 : 0;
}

/*
 * Copyright © 2026 Jetha Chan
 * SPDX-License-Identifier: MIT
 */

/* TU_TRACE_PERFCNTRS begin/end pair validity (pure C, no driver state, so
 * it is unit-tested on the host: tests/tu_trace_perfcntr_pair_test.c).
 *
 * Why pairs go bad (msm KMD, a7xx):
 *  - a6xx_set_pagetable() writes RBBM_PERFCTR_SRAM_INIT_CMD on every switch
 *    to another context's submit (zeroes all perf counters) unless sysprof
 *    or a kernel perfcntr stream is active;
 *  - the a7xx preemption KMD postamble does the same when another ring
 *    preempts this one, i.e. INSIDE a submit, between a begin and an end
 *    snapshot (long dispatches are the most exposed);
 *  - IFPC power collapse between submits loses user-programmed selectors
 *    (only kernel-configured selectors are in the power-up list);
 *  - any other process can reprogram the global selectors.
 *  The PERFCNTR_CONFIG non-stream reservation is accounting only: it does
 *  not stop any of the above.
 *
 * Checks (each sets a reason bit; any reason => CLOBBERED, no deltas used):
 *  RESET     a counter decreased and it is not a 2^width wraparound
 *            (begin in the top half, end in the bottom quarter);
 *  TS        timestamps missing or not increasing;
 *  CLOCK_LO  witness core-cycle counter (RBBM_ALWAYS_COUNT) advanced less
 *            than fmin * dt - slack: counters were zeroed inside the pair;
 *  CLOCK_HI  advanced more than fmax * dt + slack (torn read / foreign load);
 *  SELECT    a selector read back at begin or end is not the configured one;
 *  PREEMPT   CP_NUM_PREEMPTIONS witness advanced: another ring ran inside
 *            the pair (only observable when the zap is off, e.g. sysprof).
 * Zap proof: a zap at tz inside the pair leaves the witness at end <=
 *  fmax * (t1 - tz) < fmax * dt + slack. If the witness value at begin is
 *  above that bound, a zap would have made it decrease (RESET), so a pair
 *  with no reason is OK. Otherwise a zap early in the pair cannot be
 *  excluded from the counter values: OK_UNPROVEN.
 */

#ifndef TU_TRACE_PERFCNTR_PAIR_H
#define TU_TRACE_PERFCNTR_PAIR_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum tu_pc_pair_class {
   TU_PC_PAIR_OK = 0,
   TU_PC_PAIR_OK_UNPROVEN,
   TU_PC_PAIR_CLOBBERED,
};

enum tu_pc_pair_reason {
   TU_PC_REASON_RESET = 1u << 0,
   TU_PC_REASON_TS = 1u << 1,
   TU_PC_REASON_CLOCK_LO = 1u << 2,
   TU_PC_REASON_CLOCK_HI = 1u << 3,
   TU_PC_REASON_SELECT = 1u << 4,
   TU_PC_REASON_PREEMPT = 1u << 5,
};

#define TU_PC_REASON_COUNT 6

static inline const char *
tu_pc_reason_name(unsigned bit)
{
   static const char *names[TU_PC_REASON_COUNT] = {
      "reset", "ts", "clock_low", "clock_high", "select", "preempted",
   };
   return bit < TU_PC_REASON_COUNT ? names[bit] : "?";
}

static inline const char *
tu_pc_pair_class_name(enum tu_pc_pair_class c)
{
   return c == TU_PC_PAIR_OK ? "ok" : c == TU_PC_PAIR_OK_UNPROVEN ? "ok_unproven" : "clobbered";
}

struct tu_pc_pair_params {
   uint32_t width;          /* counter width in bits (64 on a6xx/a7xx) */
   uint64_t tick_hz;        /* timestamp tick rate (19.2 MHz always-on) */
   uint64_t fmin_hz;        /* lowest plausible core clock */
   uint64_t fmax_hz;        /* highest plausible core clock */
   uint64_t slack_ticks;    /* snapshot/timestamp skew allowance */
};

struct tu_pc_pair_in {
   uint32_t n;                    /* counters in the pair */
   const uint64_t *begin;         /* [n] */
   const uint64_t *end;           /* [n] */
   uint64_t ts_begin, ts_end;     /* raw ticks */
   bool ts_valid;
   int32_t cycles_idx;            /* RBBM_ALWAYS_COUNT witness, or -1 */
   int32_t preempt_idx;           /* CP_NUM_PREEMPTIONS witness, or -1 */
   /* Optional selector readback, [n] each (NULL = not checked). */
   const uint32_t *sel_expect;
   const uint32_t *sel_begin;
   const uint32_t *sel_end;
};

struct tu_pc_pair_out {
   enum tu_pc_pair_class cls;
   uint32_t reasons;              /* TU_PC_REASON_* */
   uint64_t wrapped_mask;         /* counters whose delta wrapped (first 64) */
   uint64_t reset_mask;           /* counters that decreased (first 64) */
   uint64_t dt_ticks;
   uint64_t cycles;               /* witness delta (0 if the witness reset) */
   bool zap_excluded;
};

static inline uint64_t
tu_pc_width_mask(uint32_t width)
{
   return width >= 64 ? ~(uint64_t) 0 : (((uint64_t) 1 << width) - 1);
}

/* Returns false if end < begin is a reset rather than a wraparound. */
static inline bool
tu_pc_counter_delta(uint32_t width, uint64_t begin, uint64_t end,
                    uint64_t *delta, bool *wrapped)
{
   const uint64_t mask = tu_pc_width_mask(width);
   begin &= mask;
   end &= mask;
   *wrapped = false;
   if (end >= begin) {
      *delta = end - begin;
      return true;
   }
   const uint64_t half = (mask >> 1) + 1;  /* 2^(width-1) */
   const uint64_t quarter = half >> 1;     /* 2^(width-2) */
   if (begin >= half && end < quarter) {
      *delta = (end - begin) & mask;
      *wrapped = true;
      return true;
   }
   *delta = 0;
   return false;
}

/* a * b / c without overflow for the magnitudes used here (clock * ticks). */
static inline uint64_t
tu_pc_muldiv(uint64_t a, uint64_t b, uint64_t c)
{
#if defined(__SIZEOF_INT128__)
   unsigned __int128 r = (unsigned __int128) a * b / (c ? c : 1);
   return r > ~(uint64_t) 0 ? ~(uint64_t) 0 : (uint64_t) r;
#else
   return (uint64_t) ((long double) a * (long double) b / (long double) (c ? c : 1));
#endif
}

/* deltas[] (n entries) receives the deltas; they are meaningful only if the
 * result class is not CLOBBERED.
 */
static inline enum tu_pc_pair_class
tu_pc_classify_pair(const struct tu_pc_pair_params *p,
                    const struct tu_pc_pair_in *in, uint64_t *deltas,
                    struct tu_pc_pair_out *out)
{
   out->reasons = 0;
   out->wrapped_mask = 0;
   out->reset_mask = 0;
   out->dt_ticks = 0;
   out->cycles = 0;
   out->zap_excluded = false;

   for (uint32_t i = 0; i < in->n; i++) {
      bool wrapped;
      if (!tu_pc_counter_delta(p->width, in->begin[i], in->end[i], &deltas[i], &wrapped)) {
         out->reasons |= TU_PC_REASON_RESET;
         if (i < 64)
            out->reset_mask |= (uint64_t) 1 << i;
      }
      if (wrapped && i < 64)
         out->wrapped_mask |= (uint64_t) 1 << i;
   }

   if (!in->ts_valid || in->ts_end <= in->ts_begin)
      out->reasons |= TU_PC_REASON_TS;
   else
      out->dt_ticks = in->ts_end - in->ts_begin;

   if (in->sel_expect && in->sel_begin && in->sel_end) {
      for (uint32_t i = 0; i < in->n; i++) {
         if (in->sel_begin[i] != in->sel_expect[i] || in->sel_end[i] != in->sel_expect[i])
            out->reasons |= TU_PC_REASON_SELECT;
      }
   }

   if (in->preempt_idx >= 0 && (uint32_t) in->preempt_idx < in->n && in->preempt_idx < 64 &&
       !(out->reset_mask & ((uint64_t) 1 << in->preempt_idx)) &&
       deltas[in->preempt_idx] != 0)
      out->reasons |= TU_PC_REASON_PREEMPT;

   if (in->cycles_idx >= 0 && (uint32_t) in->cycles_idx < in->n && in->cycles_idx < 64 &&
       !(out->reasons & TU_PC_REASON_TS) &&
       !(out->reset_mask & ((uint64_t) 1 << in->cycles_idx))) {
      const uint64_t dt = out->dt_ticks;
      const uint64_t c = deltas[in->cycles_idx];
      const uint64_t lo_t = dt > p->slack_ticks ? dt - p->slack_ticks : 0;
      const uint64_t hi_t = dt + p->slack_ticks;
      const uint64_t lo = tu_pc_muldiv(p->fmin_hz, lo_t, p->tick_hz);
      const uint64_t hi = tu_pc_muldiv(p->fmax_hz, hi_t, p->tick_hz);
      out->cycles = c;
      if (c < lo)
         out->reasons |= TU_PC_REASON_CLOCK_LO;
      if (c > hi)
         out->reasons |= TU_PC_REASON_CLOCK_HI;
      /* Zap proof: post-zap witness <= hi < begin value. */
      out->zap_excluded = (in->begin[in->cycles_idx] & tu_pc_width_mask(p->width)) > hi;
   }

   if (out->reasons)
      out->cls = TU_PC_PAIR_CLOBBERED;
   else if (out->zap_excluded)
      out->cls = TU_PC_PAIR_OK;
   else
      out->cls = TU_PC_PAIR_OK_UNPROVEN;
   return out->cls;
}

#ifdef __cplusplus
}
#endif

#endif /* TU_TRACE_PERFCNTR_PAIR_H */

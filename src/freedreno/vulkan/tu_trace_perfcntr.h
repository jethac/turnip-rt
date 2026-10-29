/*
 * Copyright © 2026 Jetha Chan
 * SPDX-License-Identifier: MIT
 */

#ifndef TU_TRACE_PERFCNTR_H
#define TU_TRACE_PERFCNTR_H

#include "tu_common.h"
#include "tu_cs.h"

#define TU_TRACE_PC_MAX_PHYS 48
#define TU_TRACE_PC_MAX_SETS 8
#define TU_TRACE_PC_MAX_DEPTH 64

struct tu_trace_pc_set {
   char name[24];
   uint32_t count;
   uint8_t phys[TU_TRACE_PC_MAX_PHYS];
   const struct fd_perfcntr_group *group[TU_TRACE_PC_MAX_PHYS];
   const struct fd_perfcntr_countable *countable[TU_TRACE_PC_MAX_PHYS];
   bool optional[TU_TRACE_PC_MAX_PHYS];
   int32_t w_cycles, w_ao, w_preempt;
   struct tu_cs_entry select_cs;
   bool built;
};

struct tu_trace_pc {
   uint32_t phys_count;
   const struct fd_perfcntr_group *phys_group[TU_TRACE_PC_MAX_PHYS];
   const struct fd_perfcntr_counter *phys_counter[TU_TRACE_PC_MAX_PHYS];

   uint32_t set_count;
   struct tu_trace_pc_set sets[TU_TRACE_PC_MAX_SETS];

   uint32_t rotate_frames;
   bool rotate_submits;
   uint64_t traced_submits;
   bool raw;
   bool sel_check;
   uint32_t witness;
   uint64_t fmin_hz, fmax_hz;
   struct tu_cs cs;

   /* JSON printer state, touched only from the u_trace queue thread. */
   uint32_t depth;
   char stack_op[TU_TRACE_PC_MAX_DEPTH][40];
   bool stack_valid[TU_TRACE_PC_MAX_DEPTH];
   uint64_t stack_val[TU_TRACE_PC_MAX_DEPTH][TU_TRACE_PC_MAX_PHYS];
   uint32_t stack_sel[TU_TRACE_PC_MAX_DEPTH][TU_TRACE_PC_MAX_PHYS];
   uint64_t stack_ts[TU_TRACE_PC_MAX_DEPTH];
   uint64_t tally[3];
   uint64_t tally_reason[8];
};

/* Bytes of one u_trace timestamp slot: 8 when the knob is off. */
uint32_t
tu_trace_pc_slot_bytes(const struct tu_device *device);

VkResult
tu_trace_pc_init(struct tu_device *device);

/* Installs the JSON printer hook; call after u_trace_context_init(). */
void
tu_trace_pc_install_printer(struct tu_device *device);

void
tu_trace_pc_finish(struct tu_device *device);

/* Selector stream for a traced submission; called with submit_mutex held.
 * Returns NULL (and *set_index = ~0) if it cannot be built.
 */
struct tu_cs_entry *
tu_trace_pc_select_set(struct tu_device *device, uint32_t *set_index);

void
tu_trace_pc_emit_snapshot(struct tu_device *device, struct tu_cs *cs,
                          uint64_t slot_iova);

#endif /* TU_TRACE_PERFCNTR_H */

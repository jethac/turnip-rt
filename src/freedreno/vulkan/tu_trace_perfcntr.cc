/*
 * Copyright © 2026 Jetha Chan
 * SPDX-License-Identifier: MIT
 */

#include "tu_trace_perfcntr.h"
#include "tu_trace_perfcntr_pair.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "util/os_misc.h"
#include "util/u_debug.h"
#include "util/perf/u_trace.h"

#include "tu_cs.h"
#include "tu_device.h"
#include "tu_suballoc.h"

struct tu_trace_pc_preset {
   const char *name;
   const char *list;
};

static const struct tu_trace_pc_preset presets[] = {
   { "bottleneck",
     "RBBM:RBBM_ALWAYS_COUNT,RBBM:RBBM_STATUS_MASKED,"
     "SP:SP_BUSY_CYCLES,SP:SP_ALU_WORKING_CYCLES,"
     "SP:SP_FS_STAGE_FULL_ALU_INSTRUCTIONS,SP:SP_FS_STAGE_HALF_ALU_INSTRUCTIONS,"
     "SP:SP_STALL_CYCLES_TP,SP:SP_STALL_CYCLES_UCHE,SP:SP_NON_EXECUTION_CYCLES,"
     "SP:SP_WAVE_CONTEXT_CYCLES,"
     "TP:TP_BUSY_CYCLES,TP:TP_L1_CACHELINE_REQUESTS,TP:TP_L1_CACHELINE_MISSES,"
     "UCHE:UCHE_VBIF_READ_BEATS_CH0,UCHE:UCHE_VBIF_READ_BEATS_CH1,"
     "UCHE:UCHE_VBIF_WRITE_BEATS_CH0,UCHE:UCHE_VBIF_WRITE_BEATS_CH1" },
   { "memory",
     "RBBM:RBBM_ALWAYS_COUNT,RBBM:RBBM_STATUS_MASKED,"
     "UCHE:UCHE_BUSY_CYCLES,UCHE:UCHE_STALL_CYCLES_ARBITER,"
     "UCHE:UCHE_VBIF_LATENCY_CYCLES,UCHE:UCHE_VBIF_LATENCY_SAMPLES,"
     "UCHE:UCHE_VBIF_READ_BEATS_TP,UCHE:UCHE_VBIF_READ_BEATS_SP,"
     "UCHE:UCHE_VBIF_READ_BEATS_VFD,UCHE:UCHE_VBIF_READ_BEATS_LRZ,"
     "UCHE:UCHE_GMEM_READ_BEATS,UCHE:UCHE_GMEM_WRITE_BEATS,"
     "CMP:CMPDECMP_VBIF_READ_DATA,CMP:CMPDECMP_VBIF_WRITE_DATA,"
     "CCU:CCU_GMEM_READ,CCU:CCU_GMEM_WRITE" },
   { "shader",
     "RBBM:RBBM_ALWAYS_COUNT,RBBM:RBBM_STATUS_MASKED,"
     "SP:SP_BUSY_CYCLES,SP:SP_WAVE_CONTEXTS,SP:SP_WAVE_CONTEXT_CYCLES,"
     "SP:SP_ANY_EU_WORKING_CS_STAGE,SP:SP_ANY_EU_WORKING_FS_STAGE,"
     "SP:SP_ANY_EU_WORKING_VS_STAGE,SP:SP_EFU_WORKING_CYCLES,"
     "SP:SP_GM_LOAD_INSTRUCTIONS,SP:SP_GM_STORE_INSTRUCTIONS,"
     "SP:SP_LM_BANK_CONFLICTS,SP:SP_ICL1_MISSES,SP:SP_GPR_READ_CONFLICT,"
     "SP:SP_TEXTURE_FETCH_LATENCY_CYCLES,SP:SP_TEXTURE_FETCH_LATENCY_SAMPLES" },
   { "raster",
     "RBBM:RBBM_ALWAYS_COUNT,RBBM:RBBM_STATUS_MASKED,"
     "LRZ:LRZ_BUSY_CYCLES,LRZ:LRZ_TOTAL_PIXEL,LRZ:LRZ_VISIBLE_PIXEL_AFTER_LRZ,"
     "LRZ:LRZ_TILE_KILLED,"
     "RB:RB_BUSY_CYCLES,RB:RB_STALL_CYCLES_CCU,RB:RB_3D_PIXELS,"
     "RB:RB_BLENDER_WORKING_CYCLES,"
     "PC:PC_VS_INVOCATIONS,PC:PC_STALL_CYCLES_VFD,"
     "VFD:VFD_BUSY_CYCLES,VFD:VFD_STALL_CYCLES_UCHE,"
     "SP:SP_PIXELS,SP:SP_ANY_EU_WORKING_VS_STAGE" },
};

enum {
   TU_PC_W_CYCLES = 1u << 0,
   TU_PC_W_AO = 1u << 1,
   TU_PC_W_PREEMPT = 1u << 2,
   TU_PC_W_SEL = 1u << 3,
};

static uint32_t
sel_offset(const struct tu_trace_pc *pc)
{
   return sizeof(uint64_t) * (1 + pc->phys_count);
}

uint32_t
tu_trace_pc_slot_bytes(const struct tu_device *device)
{
   const struct tu_trace_pc *pc = device->trace_pc;
   if (!pc)
      return sizeof(uint64_t);
   uint32_t bytes = sel_offset(pc);
   if (pc->sel_check)
      bytes += align(pc->phys_count * sizeof(uint32_t), sizeof(uint64_t));
   return bytes;
}

static const struct fd_perfcntr_countable *
find_countable(const struct fd_perfcntr_group *group, const char *name)
{
   char buf[96];
   const struct fd_perfcntr_countable *c = fd_perfcntrs_countable(group, name);
   if (c)
      return c;
   snprintf(buf, sizeof(buf), "PERF_%s", name);
   if ((c = fd_perfcntrs_countable(group, buf)))
      return c;
   const char *g = group->name;
   if (!strncmp(g, "BV_", 3))
      g += 3;
   snprintf(buf, sizeof(buf), "PERF_%s_%s", g, name);
   return fd_perfcntrs_countable(group, buf);
}

static bool
parse_set(struct tu_device *device, struct tu_trace_pc_set *set,
          const char *text, uint32_t index)
{
   const struct fd_dev_id *id = &device->physical_device->dev_id;
   const char *list = text;

   snprintf(set->name, sizeof(set->name), "custom%u", index);
   for (unsigned i = 0; i < ARRAY_SIZE(presets); i++) {
      if (!strcmp(text, presets[i].name)) {
         list = presets[i].list;
         snprintf(set->name, sizeof(set->name), "%s", presets[i].name);
         break;
      }
   }

   char *dup = strdup(list);
   if (!dup)
      return false;

   bool ok = true;
   char *save = NULL;
   for (char *item = strtok_r(dup, ",", &save); item && ok;
        item = strtok_r(NULL, ",", &save)) {
      char *colon = strchr(item, ':');
      if (!colon) {
         mesa_loge("TU_TRACE_PERFCNTRS: '%s' is not GROUP:COUNTABLE", item);
         ok = false;
         break;
      }
      *colon = '\0';
      const struct fd_perfcntr_group *group = fd_perfcntrs_group(id, item);
      const struct fd_perfcntr_countable *countable =
         group ? find_countable(group, colon + 1) : NULL;
      if (!countable) {
         mesa_loge("TU_TRACE_PERFCNTRS: unknown %s %s:%s",
                   group ? "countable" : "group", item, colon + 1);
         ok = false;
         break;
      }
      for (uint32_t j = 0; j < set->count; j++) {
         if (set->group[j] == group && set->countable[j] == countable) {
            mesa_loge("TU_TRACE_PERFCNTRS: duplicate %s:%s in set %s",
                      item, colon + 1, set->name);
            ok = false;
         }
      }
      if (ok && set->count == TU_TRACE_PC_MAX_PHYS) {
         mesa_loge("TU_TRACE_PERFCNTRS: set %s has more than %u countables",
                   set->name, TU_TRACE_PC_MAX_PHYS);
         ok = false;
      }
      if (ok) {
         set->group[set->count] = group;
         set->countable[set->count] = countable;
         set->count++;
      }
   }
   free(dup);
   set->w_cycles = set->w_ao = set->w_preempt = -1;
   return ok && set->count;
}

static int32_t
find_in_set(const struct tu_trace_pc_set *set, const struct fd_perfcntr_group *group,
            const struct fd_perfcntr_countable *countable)
{
   for (uint32_t i = 0; i < set->count; i++) {
      if (set->group[i] == group && set->countable[i] == countable)
         return i;
   }
   return -1;
}

static int32_t
add_witness(struct tu_device *device, struct tu_trace_pc_set *set,
            const char *group_name, const char *countable_name)
{
   const struct fd_perfcntr_group *group =
      fd_perfcntrs_group(&device->physical_device->dev_id, group_name);
   const struct fd_perfcntr_countable *countable =
      group ? find_countable(group, countable_name) : NULL;
   if (!countable)
      return -1;
   int32_t i = find_in_set(set, group, countable);
   if (i >= 0 || set->count == TU_TRACE_PC_MAX_PHYS)
      return i;
   set->group[set->count] = group;
   set->countable[set->count] = countable;
   set->optional[set->count] = true;
   return set->count++;
}

static void
drop_from_set(struct tu_trace_pc_set *set, uint32_t i)
{
   for (uint32_t j = i; j + 1 < set->count; j++) {
      set->group[j] = set->group[j + 1];
      set->countable[j] = set->countable[j + 1];
      set->optional[j] = set->optional[j + 1];
      set->phys[j] = set->phys[j + 1];
   }
   set->count--;
   int32_t *w[] = { &set->w_cycles, &set->w_ao, &set->w_preempt };
   for (unsigned k = 0; k < ARRAY_SIZE(w); k++) {
      if (*w[k] == (int32_t) i)
         *w[k] = -1;
      else if (*w[k] > (int32_t) i)
         (*w[k])--;
   }
}

static uint32_t
parse_witness(const char *text)
{
   if (!text || !*text)
      return TU_PC_W_CYCLES | TU_PC_W_AO | TU_PC_W_PREEMPT | TU_PC_W_SEL;
   uint32_t w = 0;
   if (strstr(text, "ac"))
      w |= TU_PC_W_CYCLES;
   if (strstr(text, "ao"))
      w |= TU_PC_W_AO;
   if (strstr(text, "preempt"))
      w |= TU_PC_W_PREEMPT;
   if (strstr(text, "sel"))
      w |= TU_PC_W_SEL;
   return w;
}

static void
release_phys(struct tu_device *device, struct tu_trace_pc *pc)
{
   for (uint32_t i = 0; i < pc->phys_count; i++)
      fd_perfcntr_release(device->perfcntrs, pc->phys_counter[i]);
   pc->phys_count = 0;
}

static VkResult
build_select_cs(struct tu_device *device, struct tu_trace_pc *pc,
                struct tu_trace_pc_set *set)
{
   struct tu_cs sub_cs;
   uint32_t size = 4 + set->count * 2 * (1 + 2);
   VkResult result = tu_cs_begin_sub_stream(&pc->cs, size, &sub_cs);
   if (result != VK_SUCCESS)
      return result;

   tu_cs_emit_wfi(&sub_cs);
   for (uint32_t i = 0; i < set->count; i++) {
      const struct fd_perfcntr_counter *counter = pc->phys_counter[set->phys[i]];
      tu_cs_emit_pkt4(&sub_cs, counter->select_reg, 1);
      tu_cs_emit(&sub_cs, set->countable[i]->selector);
      for (unsigned s = 0; s < ARRAY_SIZE(counter->slice_select_regs); s++) {
         if (!counter->slice_select_regs[s])
            break;
         tu_cs_emit_pkt4(&sub_cs, counter->slice_select_regs[s], 1);
         tu_cs_emit(&sub_cs, set->countable[i]->selector);
      }
   }
   tu_cs_emit_wfi(&sub_cs);
   tu_cs_emit_pkt7(&sub_cs, CP_WAIT_FOR_ME, 0);

   set->select_cs = tu_cs_end_sub_stream(&pc->cs, &sub_cs);
   set->built = true;
   return VK_SUCCESS;
}

static void
json_event_extra(struct u_trace_context *utctx, FILE *out, const char *tp_name,
                 void *timestamps, uint64_t offset_B, void *flush_data,
                 bool first_in_batch);

VkResult
tu_trace_pc_init(struct tu_device *device)
{
   const char *knob = os_get_option("TU_TRACE_PERFCNTRS");
   device->trace_pc = NULL;
   if (!knob || !*knob)
      return VK_SUCCESS;

   unsigned gen = fd_dev_gen(&device->physical_device->dev_id);
   if (gen < 6 || gen > 7 || !device->perfcntrs) {
      mesa_loge("TU_TRACE_PERFCNTRS: unsupported on gen %u, disabled", gen);
      return VK_SUCCESS;
   }

   struct tu_trace_pc *pc = (struct tu_trace_pc *)
      vk_zalloc(&device->vk.alloc, sizeof(*pc), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!pc)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   pc->rotate_frames = 60;
   const char *rot = os_get_option("TU_TRACE_PERFCNTRS_ROTATE");
   if (rot && *rot) {
      char *end = NULL;
      unsigned long v = strtoul(rot, &end, 10);
      pc->rotate_submits = end && *end == 's';
      pc->rotate_frames = (uint32_t) CLAMP(v, 1, 1000000);
   }
   pc->raw = debug_get_bool_option("TU_TRACE_PERFCNTRS_RAW", false);
   pc->witness = parse_witness(os_get_option("TU_TRACE_PERFCNTRS_WITNESS"));
   pc->sel_check = pc->witness & TU_PC_W_SEL;
   pc->fmin_hz = 1000000ull * debug_get_num_option("TU_TRACE_PERFCNTRS_FMIN_MHZ", 200);
   pc->fmax_hz = 1000000ull * debug_get_num_option("TU_TRACE_PERFCNTRS_FMAX_MHZ", 1200);

   char *dup = strdup(knob);
   bool ok = dup != NULL;
   char *save = NULL;
   for (char *text = ok ? strtok_r(dup, "+", &save) : NULL; text && ok;
        text = strtok_r(NULL, "+", &save)) {
      if (pc->set_count == TU_TRACE_PC_MAX_SETS) {
         mesa_loge("TU_TRACE_PERFCNTRS: more than %u sets", TU_TRACE_PC_MAX_SETS);
         ok = false;
         break;
      }
      struct tu_trace_pc_set *set = &pc->sets[pc->set_count];
      ok = parse_set(device, set, text, pc->set_count);
      if (ok && (pc->witness & TU_PC_W_CYCLES))
         set->w_cycles = add_witness(device, set, "RBBM", "RBBM_ALWAYS_COUNT");
      if (ok && (pc->witness & TU_PC_W_AO))
         set->w_ao = add_witness(device, set, "RBBM", "RBBM_ALWAYS_ON");
      if (ok && (pc->witness & TU_PC_W_PREEMPT))
         set->w_preempt = add_witness(device, set, "CP", "CP_NUM_PREEMPTIONS");
      pc->set_count++;
   }
   free(dup);
   ok = ok && pc->set_count;

   /* Physical counters: per group, as many as the largest set needs. A set's
    * k-th countable in a group uses that group's k-th physical counter.
    */
   for (uint32_t s = 0; ok && s < pc->set_count; s++) {
      struct tu_trace_pc_set *set = &pc->sets[s];
      for (uint32_t i = 0; ok && i < set->count; i++) {
         uint32_t rank = 0;
         for (uint32_t j = 0; j < i; j++)
            rank += set->group[j] == set->group[i];
         uint32_t seen = 0, p;
         for (p = 0; p < pc->phys_count; p++) {
            if (pc->phys_group[p] == set->group[i] && seen++ == rank)
               break;
         }
         if (p == pc->phys_count) {
            if (pc->phys_count == TU_TRACE_PC_MAX_PHYS) {
               mesa_loge("TU_TRACE_PERFCNTRS: more than %u physical counters",
                         TU_TRACE_PC_MAX_PHYS);
               ok = false;
               break;
            }
            const struct fd_perfcntr_counter *counter =
               fd_perfcntr_reserve(device->perfcntrs, set->group[i], NULL);
            if (!counter && set->optional[i]) {
               mesa_logw("TU_TRACE_PERFCNTRS: no free %s counter for witness %s; "
                         "dropped from set %s", set->group[i]->name,
                         set->countable[i]->name, set->name);
               set->phys[i] = 0xff;
               continue;
            }
            if (!counter) {
               mesa_loge("TU_TRACE_PERFCNTRS: no free %s counter (set %s needs %u)",
                         set->group[i]->name, set->name, rank + 1);
               ok = false;
               break;
            }
            pc->phys_group[p] = set->group[i];
            pc->phys_counter[p] = counter;
            pc->phys_count++;
         }
         set->phys[i] = p;
      }
      for (uint32_t i = set->count; ok && i-- > 0;) {
         if (set->phys[i] == 0xff)
            drop_from_set(set, i);
      }
   }

   if (!ok) {
      release_phys(device, pc);
      vk_free(&device->vk.alloc, pc);
      mesa_loge("TU_TRACE_PERFCNTRS='%s' rejected; counter capture disabled", knob);
      return VK_SUCCESS;
   }

   /* Selector streams are built at the first traced submission, so a set
    * knob with no tracepoints (e.g. TU_GPU_TRACE_GATE closed) allocates no
    * BO and leaves every emitted command and GPU address unchanged.
    */
   tu_cs_init(&pc->cs, device, TU_CS_MODE_SUB_STREAM, 256, "trace perfcntr cs");
   device->trace_pc = pc;

   mesa_logi("TU_TRACE_PERFCNTRS config sets=%u phys=%u slot_bytes=%u rotate=%u%s "
             "raw=%u kernel_reservation=%u witness=0x%x sel_check=%u "
             "fmin_mhz=%" PRIu64 " fmax_mhz=%" PRIu64,
             pc->set_count, pc->phys_count, tu_trace_pc_slot_bytes(device),
             pc->rotate_frames, pc->rotate_submits ? "_submits" : "_frames", pc->raw,
             fd_perfcntr_has_reservation(device->perfcntrs), pc->witness,
             pc->sel_check, pc->fmin_hz / 1000000, pc->fmax_hz / 1000000);
   for (uint32_t p = 0; p < pc->phys_count; p++) {
      const struct fd_perfcntr_group *g = pc->phys_group[p];
      mesa_logi("TU_TRACE_PERFCNTRS phys=%u group=%s counter=%u select=0x%x lo=0x%x",
                p, g->name, (unsigned) (pc->phys_counter[p] - g->counters),
                pc->phys_counter[p]->select_reg, pc->phys_counter[p]->counter_reg_lo);
   }
   for (uint32_t s = 0; s < pc->set_count; s++) {
      const struct tu_trace_pc_set *set = &pc->sets[s];
      for (uint32_t i = 0; i < set->count; i++)
         mesa_logi("TU_TRACE_PERFCNTRS set=%u name=%s phys=%u %s:%s selector=%u",
                   s, set->name, set->phys[i], set->group[i]->name,
                   set->countable[i]->name, set->countable[i]->selector);
      mesa_logi("TU_TRACE_PERFCNTRS set=%u witness cycles=%d always_on=%d preempt=%d",
                s, set->w_cycles, set->w_ao, set->w_preempt);
   }

   return VK_SUCCESS;
}

void
tu_trace_pc_install_printer(struct tu_device *device)
{
   if (device->trace_pc)
      device->trace_context.json_event_extra = json_event_extra;
}

void
tu_trace_pc_finish(struct tu_device *device)
{
   struct tu_trace_pc *pc = device->trace_pc;
   if (!pc)
      return;
   if (pc->tally[0] + pc->tally[1] + pc->tally[2]) {
      mesa_logi("TU_TRACE_PERFCNTRS pairs ok=%" PRIu64 " ok_unproven=%" PRIu64
                " clobbered=%" PRIu64 " (reset=%" PRIu64 " ts=%" PRIu64
                " clock_low=%" PRIu64 " clock_high=%" PRIu64 " select=%" PRIu64
                " preempted=%" PRIu64 ")",
                pc->tally[0], pc->tally[1], pc->tally[2], pc->tally_reason[0],
                pc->tally_reason[1], pc->tally_reason[2], pc->tally_reason[3],
                pc->tally_reason[4], pc->tally_reason[5]);
   }
   release_phys(device, pc);
   tu_cs_finish(&pc->cs);
   vk_free(&device->vk.alloc, pc);
   device->trace_pc = NULL;
}

struct tu_cs_entry *
tu_trace_pc_select_set(struct tu_device *device, uint32_t *set_index)
{
   struct tu_trace_pc *pc = device->trace_pc;
   uint64_t unit = pc->rotate_submits ? pc->traced_submits++ : device->vk.current_frame;
   uint32_t s = (unit / pc->rotate_frames) % pc->set_count;
   struct tu_trace_pc_set *set = &pc->sets[s];

   if (!set->built && build_select_cs(device, pc, set) != VK_SUCCESS) {
      *set_index = ~0u;
      return NULL;
   }
   *set_index = s;
   return &set->select_cs;
}

void
tu_trace_pc_emit_snapshot(struct tu_device *device, struct tu_cs *cs,
                          uint64_t slot_iova)
{
   struct tu_trace_pc *pc = device->trace_pc;

   tu_cs_emit_wfi(cs);
   for (uint32_t p = 0; p < pc->phys_count; p++) {
      tu_cs_emit_pkt7(cs, CP_REG_TO_MEM, 3);
      tu_cs_emit(cs, CP_REG_TO_MEM_0_REG(pc->phys_counter[p]->counter_reg_lo) |
                     CP_REG_TO_MEM_0_IS_64B);
      tu_cs_emit_qw(cs, slot_iova + sizeof(uint64_t) * (1 + p));
   }
   if (pc->sel_check) {
      for (uint32_t p = 0; p < pc->phys_count; p++) {
         tu_cs_emit_pkt7(cs, CP_REG_TO_MEM, 3);
         tu_cs_emit(cs, CP_REG_TO_MEM_0_REG(pc->phys_counter[p]->select_reg) |
                        CP_REG_TO_MEM_0_CNT(1));
         tu_cs_emit_qw(cs, slot_iova + sel_offset(pc) + sizeof(uint32_t) * p);
      }
   }
}

static const char *
short_name(const struct fd_perfcntr_countable *c)
{
   return strncmp(c->name, "PERF_", 5) ? c->name : c->name + 5;
}

static void
print_values(FILE *out, const struct tu_trace_pc_set *set, const uint64_t *v)
{
   for (uint32_t i = 0; i < set->count; i++) {
      fprintf(out, "%s\"%s:%s\": %" PRIu64, i ? ", " : "", set->group[i]->name,
              short_name(set->countable[i]), v[i]);
   }
}

static void
print_end(FILE *out, struct tu_trace_pc *pc, const struct tu_trace_pc_set *set,
          uint32_t d, const uint64_t *slot, bool valid)
{
   const uint64_t *vals = slot + 1;
   const uint32_t *sels = (const uint32_t *) (vals + pc->phys_count);
   uint64_t b[TU_TRACE_PC_MAX_PHYS], e[TU_TRACE_PC_MAX_PHYS], delta[TU_TRACE_PC_MAX_PHYS];
   uint32_t se[TU_TRACE_PC_MAX_PHYS], sb[TU_TRACE_PC_MAX_PHYS], sn[TU_TRACE_PC_MAX_PHYS];
   for (uint32_t i = 0; i < set->count; i++) {
      b[i] = pc->stack_val[d][set->phys[i]];
      e[i] = vals[set->phys[i]];
      se[i] = set->countable[i]->selector;
      sb[i] = pc->stack_sel[d][set->phys[i]];
      sn[i] = pc->sel_check ? sels[set->phys[i]] : 0;
   }
   const struct tu_pc_pair_params params = {
      .width = 64,
      .tick_hz = 19200000,
      .fmin_hz = pc->fmin_hz,
      .fmax_hz = pc->fmax_hz,
      .slack_ticks = 40, /* ~2 us */
   };
   const struct tu_pc_pair_in in = {
      .n = set->count,
      .begin = b,
      .end = e,
      .ts_begin = pc->stack_ts[d],
      .ts_end = slot[0],
      .ts_valid = valid,
      .cycles_idx = set->w_cycles,
      .preempt_idx = set->w_preempt,
      .sel_expect = pc->sel_check ? se : NULL,
      .sel_begin = pc->sel_check ? sb : NULL,
      .sel_end = pc->sel_check ? sn : NULL,
   };
   struct tu_pc_pair_out r;
   enum tu_pc_pair_class cls = tu_pc_classify_pair(&params, &in, delta, &r);
   pc->tally[cls]++;
   for (unsigned k = 0; k < TU_PC_REASON_COUNT; k++)
      pc->tally_reason[k] += !!(r.reasons & (1u << k));

   fprintf(out, ", \"pair\": \"%s\"", tu_pc_pair_class_name(cls));
   if (r.reasons) {
      fprintf(out, ", \"reasons\": [");
      for (unsigned k = 0, n = 0; k < TU_PC_REASON_COUNT; k++) {
         if (r.reasons & (1u << k))
            fprintf(out, "%s\"%s\"", n++ ? ", " : "", tu_pc_reason_name(k));
      }
      fprintf(out, "]");
   }
   if (r.reasons & TU_PC_REASON_SELECT) {
      fprintf(out, ", \"sel_bad\": {");
      for (uint32_t i = 0, n = 0; i < set->count; i++) {
         if (sb[i] != se[i] || sn[i] != se[i])
            fprintf(out, "%s\"%s:%s\": [%u, %u, %u]", n++ ? ", " : "",
                    set->group[i]->name, short_name(set->countable[i]), se[i], sb[i], sn[i]);
      }
      fprintf(out, "}");
   }
   fprintf(out, ", \"witness\": {\"dt_ticks\": %" PRIu64 ", \"zap_excluded\": %s",
           r.dt_ticks, r.zap_excluded ? "true" : "false");
   if (set->w_cycles >= 0) {
      fprintf(out, ", \"cycles_begin\": %" PRIu64, b[set->w_cycles]);
      if (r.dt_ticks && !(r.reasons & TU_PC_REASON_RESET))
         fprintf(out, ", \"mhz\": %.3f",
                 (double) r.cycles * 19.2 / (double) r.dt_ticks);
   }
   if (set->w_ao >= 0 && !(r.reasons & TU_PC_REASON_RESET))
      fprintf(out, ", \"always_on\": %" PRIu64, delta[set->w_ao]);
   fprintf(out, "}");
   if (r.wrapped_mask) {
      fprintf(out, ", \"wrapped\": [");
      for (uint32_t i = 0, n = 0; i < set->count && i < 64; i++) {
         if (r.wrapped_mask & (1ull << i))
            fprintf(out, "%s\"%s:%s\"", n++ ? ", " : "", set->group[i]->name,
                    short_name(set->countable[i]));
      }
      fprintf(out, "]");
   }
   if (cls != TU_PC_PAIR_CLOBBERED) {
      fprintf(out, ", \"delta\": {");
      print_values(out, set, delta);
      fprintf(out, "}");
   } else if (pc->raw) {
      fprintf(out, ", \"raw_begin\": {");
      print_values(out, set, b);
      fprintf(out, "}, \"raw_end\": {");
      print_values(out, set, e);
      fprintf(out, "}");
   }
}

static void
json_event_extra(struct u_trace_context *utctx, FILE *out, const char *tp_name,
                 void *timestamps, uint64_t offset_B, void *flush_data,
                 bool first_in_batch)
{
   struct tu_device *device = container_of(utctx, struct tu_device, trace_context);
   struct tu_trace_pc *pc = device->trace_pc;
   struct tu_u_trace_submission_data *data =
      (struct tu_u_trace_submission_data *) flush_data;

   if (first_in_batch)
      pc->depth = 0;
   if (!timestamps || !data || data->trace_pc_set >= pc->set_count)
      return;

   const struct tu_trace_pc_set *set = &pc->sets[data->trace_pc_set];
   const uint64_t *slot = (const uint64_t *)
      ((const char *) tu_suballoc_bo_map((struct tu_suballoc_bo *) timestamps) + offset_B);
   bool valid = slot[0] != U_TRACE_NO_TIMESTAMP;
   const uint64_t *vals = slot + 1;

   bool start = !strncmp(tp_name, "start_", 6);
   bool end = !strncmp(tp_name, "end_", 4);
   const char *op = start ? tp_name + 6 : end ? tp_name + 4 : NULL;

   if (start) {
      if (pc->depth < TU_TRACE_PC_MAX_DEPTH) {
         snprintf(pc->stack_op[pc->depth], sizeof(pc->stack_op[0]), "%s", op);
         pc->stack_valid[pc->depth] = valid;
         pc->stack_ts[pc->depth] = slot[0];
         memcpy(pc->stack_val[pc->depth], vals, pc->phys_count * sizeof(uint64_t));
         if (pc->sel_check)
            memcpy(pc->stack_sel[pc->depth], vals + pc->phys_count,
                   pc->phys_count * sizeof(uint32_t));
      }
      pc->depth++;
   }

   fprintf(out, ",\n\"perfcntrs\": {\"set\": \"%s\"", set->name);
   if (pc->raw) {
      uint64_t v[TU_TRACE_PC_MAX_PHYS];
      for (uint32_t i = 0; i < set->count; i++)
         v[i] = vals[set->phys[i]];
      fprintf(out, ", \"raw_valid\": %s, \"raw\": {", valid ? "true" : "false");
      print_values(out, set, v);
      fprintf(out, "}");
   }
   if (end) {
      if (!pc->depth) {
         fprintf(out, ", \"pair\": \"unmatched\"");
      } else {
         pc->depth--;
         if (pc->depth >= TU_TRACE_PC_MAX_DEPTH) {
            fprintf(out, ", \"pair\": \"too_deep\"");
         } else if (strcmp(pc->stack_op[pc->depth], op)) {
            fprintf(out, ", \"pair\": \"mismatch\", \"begin_op\": \"%s\"",
                    pc->stack_op[pc->depth]);
         } else if (!valid || !pc->stack_valid[pc->depth]) {
            fprintf(out, ", \"pair\": \"unwritten\"");
         } else {
            print_end(out, pc, set, pc->depth, slot, valid);
         }
      }
   }
   fprintf(out, "}");
}

/*
 * Copyright © 2016 Rob Clark <robclark@freedesktop.org>
 * SPDX-License-Identifier: MIT
 */

#include "rd_private.h"
#include "rd_trace.h"
#include "common/fd5_hw.h"
#include "common/freedreno_common.h"
#include "drm/freedreno_ringbuffer.h"
#include "drm/freedreno_ringbuffer_sp.h"
#include "vk_sync.h"
#include "util/u_math.h"
#include "util/u_atomic.h"
#include "util/os_time.h"
#include "util/u_debug.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct rd_submit_profile {
   uint64_t chunks;
   int64_t total_ns;
   int64_t max_ns;
};

struct rd_gpu_elapsed {
   struct fd_bo *bo;
   void *map;
   const struct rd_dispatch *dispatch;
   const char *reason;
   const char *calibration;
   unsigned sample;
   uint64_t chunks;
   uint64_t submit_count;
};

static bool
rd_gpu_elapsed_reserve(unsigned *sample)
{
   static unsigned attempts;
   unsigned value = p_atomic_read(&attempts);
   while (value < 16) {
      unsigned observed = p_atomic_cmpxchg(&attempts, value, value + 1);
      if (observed == value) {
         *sample = value;
         return true;
      }
      value = observed;
   }
   return false;
}

static void
rd_gpu_elapsed_release(struct rd_gpu_elapsed *sample)
{
   if (sample && sample->bo) {
      fd_bo_del(sample->bo);
      sample->bo = NULL;
   }
}

static void
rd_gpu_elapsed_prepare(struct rd_device *device, struct rd_gpu_elapsed *sample)
{
   sample->bo = fd_bo_new(device->fd, 4096, 0, "radish-gpu-elapsed");
   if (!sample->bo)
      sample->reason = "bo_alloc";
   else if (fd_bo_cpu_prep(sample->bo, device->pipe,
                          FD_BO_PREP_WRITE | FD_BO_PREP_NOSYNC))
      sample->reason = "write_prep";
   else if (!(sample->map = fd_bo_map(sample->bo)))
      sample->reason = "map";
   if (sample->reason) {
      rd_gpu_elapsed_release(sample);
      return;
   }
   const uint64_t initial[4] = {
      UINT64_MAX, UINT64_C(0x1964a5a55a5acafe),
      UINT64_MAX, UINT64_C(0xe69b5a5aa5a53501)
   };
   memcpy(sample->map, initial, sizeof(initial));
   __sync_synchronize();
}

static void
rd_gpu_elapsed_emit(struct fd_ringbuffer *ring, struct fd_bo *bo,
                    unsigned offset)
{
   OUT_PKT7(ring, CP_EVENT_WRITE, 4);
   OUT_RING(ring, CP_EVENT_WRITE_0_EVENT(RB_DONE_TS) |
                  CP_EVENT_WRITE_0_TIMESTAMP);
   OUT_RELOC(ring, bo, offset, 0, 0);
   OUT_RING(ring, 0);
}

static void
rd_gpu_elapsed_finish(struct rd_device *device, struct rd_gpu_elapsed *sample,
                      VkResult result, int64_t start, int64_t done)
{
   if (!sample)
      return;
   uint64_t values[4] = {UINT64_MAX, 0, UINT64_MAX, 0};
   uint64_t delta = 0, gpu_ns = 0;
   int64_t host_ns = start >= 0 && done > start ? done - start : 0;
   if (result != VK_SUCCESS)
      sample->reason = "submit_failed";
   else if (!sample->reason) {
      if (fd_bo_cpu_prep(sample->bo, device->pipe,
                         FD_BO_PREP_READ | FD_BO_PREP_NOSYNC)) {
         sample->reason = "read_prep";
      } else {
         __sync_synchronize();
         memcpy(values, sample->map, sizeof(values));
         if (values[1] != UINT64_C(0x1964a5a55a5acafe) ||
             values[3] != UINT64_C(0xe69b5a5aa5a53501))
            sample->reason = "guard";
         else if ((uint32_t)values[0] == UINT32_MAX ||
                  (uint32_t)(values[0] >> 32) == UINT32_MAX ||
                  (uint32_t)values[2] == UINT32_MAX ||
                  (uint32_t)(values[2] >> 32) == UINT32_MAX)
            sample->reason = "poison";
         else if (!values[0] || !values[2])
            sample->reason = "zero";
         else if (values[2] <= values[0])
            sample->reason = "order";
         else if (values[2] - values[0] > UINT64_C(96000000))
            sample->reason = "ticks";
         else if (!host_ns)
            sample->reason = "host_interval";
         else if (fd_ticks_to_ns(values[2] - values[0]) >
                  (uint64_t)host_ns + 1000)
            sample->reason = "interval";
         else {
            delta = values[2] - values[0];
            gpu_ns = fd_ticks_to_ns(delta);
         }
      }
   }
   char line[1024];
   int length = snprintf(line, sizeof(line),
      "RADISH_GPU_ELAPSED version=1 sample=%u valid=%u reason=%s pipeline=%p grid=4,2,1 local=16,8,1 instrlen=27 chunks=%" PRIu64 " submits=%" PRIu64 " submit_id=%" PRIu64 " start_ticks=%" PRIu64 " stop_ticks=%" PRIu64 " delta_ticks=%" PRIu64 " gpu_ns=%" PRIu64 " host_start_ns=%" PRId64 " host_done_ns=%" PRId64 " submit_ns=%" PRId64 " result=%d\n",
      sample->sample, (unsigned)(sample->reason == NULL), sample->reason ? sample->reason : "ok",
      (void *)sample->dispatch->pipeline, sample->chunks,
      device->submit_count - sample->submit_count, device->submit_count,
      values[0], values[2], delta, gpu_ns, start, done, host_ns, result);
   if (sample->calibration)
      length = snprintf(line, sizeof(line),
         "RADISH_GPU_ELAPSED version=1 sample=%u valid=%u reason=%s pipeline=%p grid=4,2,1 local=16,8,1 instrlen=%u chunks=%" PRIu64 " submits=%" PRIu64 " submit_id=%" PRIu64 " start_ticks=%" PRIu64 " stop_ticks=%" PRIu64 " delta_ticks=%" PRIu64 " gpu_ns=%" PRIu64 " host_start_ns=%" PRId64 " host_done_ns=%" PRId64 " submit_ns=%" PRId64 " result=%d calibration=%s\n",
         sample->sample, (unsigned)(sample->reason == NULL), sample->reason ? sample->reason : "ok",
         (void *)sample->dispatch->pipeline,
         sample->dispatch->pipeline->variant->instrlen, sample->chunks,
         device->submit_count - sample->submit_count, device->submit_count,
         values[0], values[2], delta, gpu_ns, start, done, host_ns, result,
         sample->calibration);
   if (length >= 0 && (size_t)length < sizeof(line))
      fputs(line, stderr);
   else
      fprintf(stderr, "RADISH_GPU_ELAPSED version=1 sample=%u valid=0 reason=report_overflow\n",
              sample->sample);
}

static void
cs_restore_emit(struct fd_ringbuffer *ring, struct rd_device *device)
{
   OUT_PKT7(ring, CP_SET_RENDER_MODE, 5);
   OUT_RING(ring, CP_SET_RENDER_MODE_0_MODE(BYPASS));
   OUT_RING(ring, 0x00000000); /* ADDR_LO */
   OUT_RING(ring, 0x00000000); /* ADDR_HI */
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);

   OUT_PKT4(ring, REG_A5XX_UCHE_CACHE_INVALIDATE_MIN, 5);
   OUT_RING(ring, 0x00000000); /* UCHE_CACHE_INVALIDATE_MIN_LO */
   OUT_RING(ring, 0x00000000); /* UCHE_CACHE_INVALIDATE_MIN_HI */
   OUT_RING(ring, 0x00000000); /* UCHE_CACHE_INVALIDATE_MAX_LO */
   OUT_RING(ring, 0x00000000); /* UCHE_CACHE_INVALIDATE_MAX_HI */
   OUT_RING(ring, 0x00000012); /* UCHE_CACHE_INVALIDATE */

   OUT_WFI5(ring);

   OUT_PKT4(ring, REG_A5XX_HLSQ_UPDATE_CNTL, 1);
   OUT_RING(ring, 0x000fffff);

   OUT_PKT4(ring, REG_A5XX_RB_MODE_CNTL, 1);
   OUT_RING(ring, 0x00000044);

   OUT_PKT4(ring, REG_A5XX_RB_DBG_ECO_CNTL, 1);
   OUT_RING(ring, 0x00100000);

   OUT_PKT4(ring, REG_A5XX_PC_MODE_CNTL, 1);
   OUT_RING(ring, 0x0000001f);

   OUT_PKT4(ring, REG_A5XX_SP_MODE_CNTL, 1);
   OUT_RING(ring, 0x0000001e);

   if (fd_dev_gpu_id(fd_pipe_dev_id(device->pipe)) == 540) {
      OUT_PKT4(ring, REG_A5XX_SP_DBG_ECO_CNTL, 1);
      OUT_RING(ring, 0x00000800);

      OUT_PKT4(ring, REG_A5XX_HLSQ_DBG_ECO_CNTL, 1);
      OUT_RING(ring, 0x00000000);
   } else {
      OUT_PKT4(ring, REG_A5XX_SP_DBG_ECO_CNTL, 1);
      OUT_RING(ring, 0x40000800);
   }

   OUT_PKT4(ring, REG_A5XX_TPL1_MODE_CNTL, 1);
   OUT_RING(ring, 0x00000544);

   OUT_PKT4(ring, REG_A5XX_HLSQ_TIMEOUT_THRESHOLD_0, 2);
   OUT_RING(ring, 0x00000080); /* HLSQ_TIMEOUT_THRESHOLD_0 */
   OUT_RING(ring, 0x00000000); /* HLSQ_TIMEOUT_THRESHOLD_1 */

   OUT_PKT4(ring, REG_A5XX_VPC_DBG_ECO_CNTL, 1);
   OUT_RING(ring, 0x00000400);

   OUT_PKT4(ring, REG_A5XX_HLSQ_MODE_CNTL, 1);
   OUT_RING(ring, 0x00000001);

   OUT_PKT4(ring, REG_A5XX_VPC_MODE_CNTL, 1);
   OUT_RING(ring, 0x00000000);
}

static void
cs_program_emit(struct fd_ringbuffer *ring, struct rd_pipeline *pipeline)
{
   struct ir3_shader_variant *v = pipeline->variant;
   const struct ir3_info *i = &v->info;
   enum a3xx_threadsize thrsz = i->double_threadsize ? FOUR_QUADS : TWO_QUADS;

   /* if shader is more than 32*16 instructions, don't preload it.  Similar
    * to the combined restriction of 64*16 for VS+FS
    */
   unsigned instrlen = (v->instrlen > 32) ? 0 : v->instrlen;

   OUT_PKT4(ring, REG_A5XX_SP_SP_CNTL, 1);
   OUT_RING(ring, 0x00000000); /* SP_SP_CNTL */

   OUT_PKT4(ring, REG_A5XX_HLSQ_CONTROL_0_REG, 1);
   OUT_RING(ring, A5XX_HLSQ_CONTROL_0_REG_FSTHREADSIZE(TWO_QUADS) |
                     A5XX_HLSQ_CONTROL_0_REG_CSTHREADSIZE(thrsz) |
                     0x00000880 /* XXX */);

   OUT_PKT4(ring, REG_A5XX_SP_CS_CTRL_REG0, 1);
   OUT_RING(ring,
            A5XX_SP_CS_CTRL_REG0_THREADSIZE(thrsz) |
               A5XX_SP_CS_CTRL_REG0_HALFREGFOOTPRINT(i->max_half_reg + 1) |
               A5XX_SP_CS_CTRL_REG0_FULLREGFOOTPRINT(i->max_reg + 1) |
               A5XX_SP_CS_CTRL_REG0_BRANCHSTACK(ir3_shader_branchstack_hw(v)) |
               COND(instrlen != 0, A5XX_SP_CS_CTRL_REG0_BUFFER) |
               0x2 /* XXX */);

   OUT_PKT4(ring, REG_A5XX_HLSQ_CS_CONFIG, 1);
   OUT_RING(ring, A5XX_HLSQ_CS_CONFIG_CONSTOBJECTOFFSET(0) |
                     A5XX_HLSQ_CS_CONFIG_SHADEROBJOFFSET(0) |
                     A5XX_HLSQ_CS_CONFIG_ENABLED);

   OUT_PKT4(ring, REG_A5XX_HLSQ_CS_CNTL, 1);
   OUT_RING(ring, A5XX_HLSQ_CS_CNTL_INSTRLEN(instrlen) |
                     COND(pipeline->buffer_count + pipeline->resource_usage.image_count > 0, A5XX_HLSQ_CS_CNTL_SSBO_ENABLE));

   OUT_PKT4(ring, REG_A5XX_SP_CS_CONFIG, 1);
   OUT_RING(ring, A5XX_SP_CS_CONFIG_CONSTOBJECTOFFSET(0) |
                     A5XX_SP_CS_CONFIG_SHADEROBJOFFSET(0) |
                     A5XX_SP_CS_CONFIG_ENABLED);

   OUT_PKT4(ring, REG_A5XX_HLSQ_CS_CONSTLEN, 2);
   OUT_RING(ring, align(v->constlen, 4) / 4); /* HLSQ_CS_CONSTLEN */
   OUT_RING(ring, instrlen);                  /* HLSQ_CS_INSTRLEN */

   OUT_PKT4(ring, REG_A5XX_TPL1_CS_TEX_COUNT, 1);
   OUT_RING(ring, pipeline->resource_usage.images ? ~0u : v->num_samp);

   uint32_t per_fiber_size = v->pvtmem_size;
   uint32_t per_sp_size = pipeline->private_per_sp;

   OUT_PKT4(ring, REG_A5XX_SP_CS_OBJ_START, 6);
   OUT_RELOC(ring, pipeline->bo, 0, 0, 0); /* SP_CS_OBJ_START */
   OUT_RING(ring, A5XX_SP_CS_PVT_MEM_PARAM_MEMSIZEPERITEM(per_fiber_size) |
                     A5XX_SP_CS_PVT_MEM_PARAM_HWSTACKOFFSET(per_sp_size));
   if (v->pvtmem_size > 0) { /* SP_CS_PVT_MEM_ADDR */
      OUT_RELOC(ring, pipeline->private_bo, 0, 0, 0);
      fd_ringbuffer_attach_bo(ring, pipeline->private_bo);
   } else {
      OUT_RING(ring, 0);
      OUT_RING(ring, 0);
   }
   OUT_RING(ring, A5XX_SP_CS_PVT_MEM_SIZE_TOTALPVTMEMSIZE(per_sp_size));

   OUT_PKT4(ring, REG_A5XX_HLSQ_UPDATE_CNTL, 1);
   OUT_RING(ring, 0x01f00000);

   uint32_t local_invocation_id, work_group_id;
   local_invocation_id =
      ir3_find_sysval_regid(v, SYSTEM_VALUE_LOCAL_INVOCATION_ID);
   work_group_id = ir3_find_sysval_regid(v, SYSTEM_VALUE_WORKGROUP_ID);

   OUT_PKT4(ring, REG_A5XX_HLSQ_CS_CNTL_0, 2);
   OUT_RING(ring, A5XX_HLSQ_CS_CNTL_0_WGIDCONSTID(work_group_id) |
                     A5XX_HLSQ_CS_CNTL_0_UNK0(INVALID_REG) |
                     A5XX_HLSQ_CS_CNTL_0_UNK1(INVALID_REG) |
                     A5XX_HLSQ_CS_CNTL_0_LOCALIDREGID(local_invocation_id));
   OUT_RING(ring, 0x1);

   if (instrlen > 0) {
      /* preload the shader into the instruction cache: */
      OUT_PKT7(ring, CP_LOAD_STATE4, 3);
      OUT_RING(ring, CP_LOAD_STATE4_0_DST_OFF(0) |
                        CP_LOAD_STATE4_0_STATE_SRC(SS4_INDIRECT) |
                        CP_LOAD_STATE4_0_STATE_BLOCK(SB4_CS_SHADER) |
                        CP_LOAD_STATE4_0_NUM_UNIT(instrlen));
      OUT_RELOC(ring, pipeline->bo, 0, CP_LOAD_STATE4_1_STATE_TYPE(ST4_SHADER), 0);
   }
}

static void
emit_const(struct fd_ringbuffer *ring, uint32_t constid, uint32_t sizedwords,
           const uint32_t *dwords)
{
   uint32_t align_sz;

   assert((constid % 4) == 0);

   align_sz = align(sizedwords, 4);

   OUT_PKT7(ring, CP_LOAD_STATE4, 3 + align_sz);
   OUT_RING(ring, CP_LOAD_STATE4_0_DST_OFF(constid / 4) |
                     CP_LOAD_STATE4_0_STATE_SRC(SS4_DIRECT) |
                     CP_LOAD_STATE4_0_STATE_BLOCK(SB4_CS_SHADER) |
                     CP_LOAD_STATE4_0_NUM_UNIT(DIV_ROUND_UP(sizedwords, 4)));
   OUT_RING(ring, CP_LOAD_STATE4_1_EXT_SRC_ADDR(0) |
                     CP_LOAD_STATE4_1_STATE_TYPE(ST4_CONSTANTS));
   OUT_RING(ring, CP_LOAD_STATE4_2_EXT_SRC_ADDR_HI(0));
   for (unsigned i = 0; i < sizedwords; i++) {
      OUT_RING(ring, dwords[i]);
   }

   /* Zero-pad to multiple of 4 dwords */
   for (uint32_t i = sizedwords; i < align_sz; i++) {
      OUT_RING(ring, 0);
   }
}


static void
rd_emit_buffers(struct rd_device *device, struct fd_ringbuffer *ring,
                 const struct rd_dispatch *dispatch)
{
   unsigned count = dispatch->pipeline->buffer_count;
   if (!count)
      return;
   OUT_PKT7(ring, CP_LOAD_STATE4, 3 + 2 * count);
   OUT_RING(ring, CP_LOAD_STATE4_0_STATE_SRC(SS4_DIRECT) |
      CP_LOAD_STATE4_0_STATE_BLOCK(SB4_CS_SSBO) | CP_LOAD_STATE4_0_NUM_UNIT(count));
   OUT_RING(ring, CP_LOAD_STATE4_1_STATE_TYPE(ST4_CONSTANTS));
   OUT_RING(ring, 0);
   for (unsigned i = 0; i < count; i++) {
      bool metadata = dispatch->sampling_addresses && i == dispatch->pipeline->sampling_buffer;
      const struct rd_descriptor *desc = metadata ? NULL :
         &dispatch->descriptors[dispatch->pipeline->buffer_descriptors[i]];
      uint32_t bytes = metadata ? 128 : desc->buffer ? desc->range : 0;
      OUT_RING(ring, A5XX_SSBO_1_0_WIDTH(bytes));
      OUT_RING(ring, A5XX_SSBO_1_1_HEIGHT(bytes >> 16));
   }
   OUT_PKT7(ring, CP_LOAD_STATE4, 3 + 2 * count);
   OUT_RING(ring, CP_LOAD_STATE4_0_STATE_SRC(SS4_DIRECT) |
      CP_LOAD_STATE4_0_STATE_BLOCK(SB4_CS_SSBO) | CP_LOAD_STATE4_0_NUM_UNIT(count));
   OUT_RING(ring, CP_LOAD_STATE4_1_STATE_TYPE(ST4_UBO));
   OUT_RING(ring, 0);
   for (unsigned i = 0; i < count; i++) {
      bool metadata = dispatch->sampling_addresses && i == dispatch->pipeline->sampling_buffer;
      const struct rd_descriptor *desc = metadata ? NULL :
         &dispatch->descriptors[dispatch->pipeline->buffer_descriptors[i]];
      if (metadata) {
         OUT_RELOC(ring, dispatch->sampling_addresses, 0, 0, 0);
         for (unsigned j = 0; j < 16; j++)
            if (dispatch->sampling_bos[j])
               fd_ringbuffer_attach_bo(ring, dispatch->sampling_bos[j]);
      } else if (desc->buffer && desc->buffer->memory)
         OUT_RELOC(ring, desc->buffer->memory->bo,
                    desc->buffer->offset + desc->offset, 0, 0);
      else
         OUT_RELOC(ring, device->control, 0, 0, 0);
   }
}

static void
rd_emit_constants(struct fd_ringbuffer *ring, const struct rd_dispatch *dispatch)
{
   const struct ir3_shader_variant *v = dispatch->pipeline->variant;
   const struct ir3_const_state *state = ir3_const_state(v);
   emit_const(ring, 0, MIN2(32, v->constlen * 4), dispatch->push);
   if (v->constant_data_size) {
      if (ir3_const_can_upload(&state->allocs, IR3_CONST_ALLOC_UBO_PTRS,
                              v->constlen)) {
         unsigned base = state->allocs.consts[IR3_CONST_ALLOC_UBO_PTRS].offset_vec4;
         unsigned count = align(state->num_ubos, 2);
         OUT_PKT7(ring, CP_LOAD_STATE4, 3 + 2 * count);
         OUT_RING(ring, CP_LOAD_STATE4_0_DST_OFF(base) |
            CP_LOAD_STATE4_0_STATE_SRC(SS4_DIRECT) |
            CP_LOAD_STATE4_0_STATE_BLOCK(SB4_CS_SHADER) |
            CP_LOAD_STATE4_0_NUM_UNIT(count / 2));
         OUT_RING(ring, CP_LOAD_STATE4_1_STATE_TYPE(ST4_CONSTANTS));
         OUT_RING(ring, 0);
         for (unsigned i = 0; i < count; i++) {
            if (i == (unsigned)state->consts_ubo.idx) {
               OUT_RELOC(ring, dispatch->pipeline->bo,
                          v->info.constant_data_offset, 0, 0);
            } else {
               OUT_RING(ring, 0);
               OUT_RING(ring, 0);
            }
         }
      }
      for (unsigned i = 0; i < state->ubo_state.num_enabled; i++) {
         const struct ir3_ubo_range *range = &state->ubo_state.range[i];
         if (range->offset >= 16 * v->constlen)
            continue;
         unsigned size = MIN2(range->end - range->start,
                              16 * v->constlen - range->offset);
         if (size)
            emit_const(ring, range->offset / 4, size / 4,
               (const uint32_t *)((const uint8_t *)v->bin +
                  v->info.constant_data_offset + range->start));
      }
   }
   unsigned base = state->allocs.max_const_offset_vec4 * 4;
   if (v->imm_state.count && base < v->constlen * 4)
      emit_const(ring, base, MIN2(align(v->imm_state.count, 4),
                                v->constlen * 4 - base), v->imm_state.values);
}

static void
rd_emit_driver_params(struct fd_ringbuffer *ring, const struct rd_dispatch *dispatch,
                      const uint32_t group_base[3])
{
   const struct ir3_shader_variant *v = dispatch->pipeline->variant;
   const struct ir3_const_state *state = ir3_const_state(v);
   if (v->need_driver_params) {
      struct ir3_driver_params_cs params = {
         .num_work_groups_x = dispatch->grid[0],
         .num_work_groups_y = dispatch->grid[1],
         .num_work_groups_z = dispatch->grid[2],
         .work_dim = 3,
         .base_group_x = dispatch->base[0] + group_base[0],
         .base_group_y = dispatch->base[1] + group_base[1],
         .base_group_z = dispatch->base[2] + group_base[2],
         .subgroup_size = 16,
         .subgroup_id_shift = 4,
         .local_group_size_x = v->local_size[0],
         .local_group_size_y = v->local_size[1],
         .local_group_size_z = v->local_size[2],
      };
      unsigned base = state->allocs.consts[IR3_CONST_ALLOC_DRIVER_PARAMS].offset_vec4 * 4;
      unsigned size = MIN2(state->num_driver_params, v->constlen * 4 - base);
      if (size)
         emit_const(ring, base, size, (const uint32_t *)&params);
   }
}

void
rd_flush(struct rd_device *device, struct fd_ringbuffer *ring)
{
   uint32_t seqno = ++device->seqno;
   OUT_PKT7(ring, CP_EVENT_WRITE, 4);
   OUT_RING(ring, CACHE_FLUSH_TS);
   OUT_RELOC(ring, device->control, 0, 0, 0);
   OUT_RING(ring, seqno);
   OUT_PKT7(ring, CP_WAIT_REG_MEM, 6);
   OUT_RING(ring, CP_WAIT_REG_MEM_0_FUNCTION(WRITE_EQ) |
                  CP_WAIT_REG_MEM_0_POLL(POLL_MEMORY));
   OUT_RELOC(ring, device->control, 0, 0, 0);
   OUT_RING(ring, seqno);
   OUT_RING(ring, ~0u);
   OUT_RING(ring, CP_WAIT_REG_MEM_5_DELAY_LOOP_CYCLES(16));
   rd_trace_submit_end(device, ring);
}

VkResult
rd_finish_submit(struct rd_device *device, struct fd_submit *submit,
                  struct fd_ringbuffer *ring, struct rd_submit_profile *profile,
                  struct rd_gpu_elapsed *sample)
{
   device->submit_count++;
   int64_t start = profile ? os_time_get_nano() : 0;
   int64_t t0 = device->trace ? os_time_get_nano() : 0;
   struct fd_fence *fence = fd_submit_flush(submit, -1, false);
   VkResult result = VK_ERROR_DEVICE_LOST;
   if (fence) {
      fd_fence_flush(fence);
      int64_t t1 = device->trace ? os_time_get_nano() : 0;
      int wait = fd_pipe_wait_timeout(device->pipe, fence, 5000000000ull);
      if (device->trace) {
         int64_t t2 = os_time_get_nano();
         device->trace->flush_ns += t1 - t0;
         device->trace->wait_ns += t2 - t1;
      }
      if (!wait) {
         uint64_t faults = 0;
         if (!fd_pipe_get_param(device->pipe, FD_CTX_FAULTS, &faults) &&
             faults == device->faults)
            result = VK_SUCCESS;
      }
   }
   int64_t done = profile ? os_time_get_nano() : 0;
   if (profile) {
      int64_t elapsed = done - start;
      profile->total_ns += elapsed;
      profile->max_ns = MAX2(profile->max_ns, elapsed);
   }
   rd_gpu_elapsed_finish(device, sample, result, start, done);
   if (fence)
      fd_fence_del(fence);
   fd_ringbuffer_del(ring);
   fd_submit_del(submit);
   return result;
}

struct rd_pending_submit {
   struct fd_submit *submit;
   struct fd_ringbuffer *ring;
   struct fd_fence *fence;
   struct fd_bo *cs_backing;
};

struct rd_submit_window {
   struct rd_pending_submit pending[8];
   unsigned count, chunks, cs_bytes, bo_refs, backing_bytes;
   size_t metadata_bytes;
};

static void
rd_window_report(const char *action, const char *reason,
                 const struct rd_submit_window *window, VkResult result)
{
   static int counter;
   if (!debug_get_bool_option("TRANSCRIBE_RESOURCE_DIAGNOSTICS", false) ||
       p_atomic_inc_return(&counter) > 256)
      return;
   fprintf(stderr, "RADISH_DEFERRED_%s slots=%u chunks=%u cs_bytes=%u bo_refs=%u metadata_bytes=%zu backing_bytes=%u reason=%s result=%d\n",
      action, window->count, window->chunks, window->cs_bytes,
      window->bo_refs, window->metadata_bytes, window->backing_bytes, reason, result);
   fflush(stderr);
}

static VkResult
rd_window_drain(struct rd_device *device, struct rd_submit_window *window,
                const char *reason)
{
   if (!window->count)
      return VK_SUCCESS;
   struct fd_fence *last = window->pending[window->count - 1].fence;
   VkResult result = VK_ERROR_DEVICE_LOST;
   int64_t t0 = device->trace ? os_time_get_nano() : 0;
   int wait = fd_pipe_wait_timeout(device->pipe, last, 5000000000ull);
   if (device->trace)
      device->trace->wait_ns += os_time_get_nano() - t0;
   if (!wait) {
      uint64_t faults = 0;
      if (!fd_pipe_get_param(device->pipe, FD_CTX_FAULTS, &faults) &&
          faults == device->faults)
         result = VK_SUCCESS;
   }
   rd_window_report("DRAIN", reason, window, result);
   for (unsigned i = 0; i < window->count; i++) {
      fd_fence_del(window->pending[i].fence);
      fd_ringbuffer_del(window->pending[i].ring);
      fd_submit_del(window->pending[i].submit);
   }
   memset(window, 0, sizeof(*window));
   return result;
}

static bool
rd_window_pipeline(struct rd_device *device, const struct rd_dispatch *dispatch)
{
   const struct rd_pipeline *p = dispatch->pipeline;
   const struct ir3_shader_variant *v = p->variant;
   const uint64_t threads = (uint64_t)v->local_size[0] *
      v->local_size[1] * v->local_size[2];
   const uint64_t wave_size = (uint64_t)device->info.threadsize_base *
      (v->info.double_threadsize ? 2 : 1);
   if (!wave_size)
      return false;
   const uint64_t resident_threads = v->has_barrier ?
      DIV_ROUND_UP(threads, wave_size) * wave_size : wave_size;
   if (fd_dev_gpu_id(fd_pipe_dev_id(device->pipe)) != 506 ||
       fd_device_version(device->fd) < FD_VERSION_SOFTPIN || dispatch->indirect ||
       p->sampling_nir || p->sampling_variants || !p->batch_local_memory ||
       p->linear_subgroups || p->linear_barriers ||
       v->shader_options.workgroup_resident_threads ||
       !v->shader_options.nir_options.a5xx_ssbo_size_consts ||
       v->pvtmem_size || v->constant_data_size || v->info.preamble_instrs_count ||
       p->private_bo || !threads || threads > 128 ||
       resident_threads * ir3_shader_halfregs(v) > 4096 ||
       v->shared_size > 16384 || p->resource_count < 1 || p->resource_count > 5 ||
       v->num_ssbos != p->resource_count ||
       p->descriptor_count != p->resource_count || p->buffer_count != p->resource_count ||
       p->resource_usage.texture_count || p->resource_usage.sampler_count ||
       p->resource_usage.image_count || p->sampling_mask || v->image_mapping.num_tex)
      return false;
   for (unsigned i = 0; i < p->resource_count; i++) {
      if (p->resources[i].type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
         return false;
   }
   return true;
}

static bool
rd_window_footprint(struct fd_submit *submit, struct fd_ringbuffer *ring,
                    uint32_t *initial_start, struct fd_bo **backing,
                    unsigned *references, size_t *metadata)
{
   const struct fd_submit_sp *sp = to_fd_submit_sp(submit);
   const struct fd_ringbuffer_sp *rp = to_fd_ringbuffer_sp(ring);
   if (ring->start != initial_start || ring->size != 32768 ||
       rp->u.nr_cmds != 1 || rp->u.max_cmds > 5 ||
       sp->nr_bos > 12 || sp->max_bos > 20 ||
       sp->nr_suballoc_bos > 12 || sp->max_suballoc_bos > 20 ||
       sp->bo_table->size > 43 || sp->suballoc_bo_table->size > 43 ||
       sp->suballoc_ring || !sp->ring_pool.parent ||
       (uint64_t)sp->ring_pool.parent->element_size *
          sp->ring_pool.parent->num_elements + sizeof(void *) > 4096)
      return false;
   struct fd_bo *bo = rp->ring_bo;
   *backing = bo->handle ? bo : fd_bo_heap_block(bo);
   if (!*backing || (*backing)->size > 4 * 1024 * 1024)
      return false;
   *references = sp->nr_bos + sp->nr_suballoc_bos;
   *metadata = sizeof(*sp) + sizeof(*rp) + sizeof(struct fd_fence) + 4096 +
      (sp->max_bos + sp->max_suballoc_bos) * sizeof(struct fd_bo *) +
      rp->u.max_cmds * sizeof(struct fd_cmd_sp) +
      2 * sizeof(struct hash_table) +
      (sp->bo_table->size + sp->suballoc_bo_table->size) * sizeof(struct hash_entry);
   return *metadata <= 16384;
}

static VkResult
rd_finish_dispatch_submit(struct rd_device *device, struct fd_submit *submit,
                          struct fd_ringbuffer *ring,
                          struct rd_submit_profile *profile,
                          struct rd_gpu_elapsed *sample,
                          struct rd_submit_window *window,
                          uint32_t *initial_start, unsigned chunks)
{
   if (!window)
      return rd_finish_submit(device, submit, ring, profile, sample);
   device->submit_count++;
   int64_t t0 = device->trace ? os_time_get_nano() : 0;
   struct fd_fence *fence = fd_submit_flush(submit, -1, false);
   if (fence)
      fd_fence_flush(fence);
   if (device->trace)
      device->trace->flush_ns += os_time_get_nano() - t0;
   if (!fence) {
      rd_window_drain(device, window, "error");
      fd_ringbuffer_del(ring);
      fd_submit_del(submit);
      return VK_ERROR_DEVICE_LOST;
   }
   if (fence->submit_error) {
      rd_window_drain(device, window, "error");
      fd_fence_del(fence);
      fd_ringbuffer_del(ring);
      fd_submit_del(submit);
      return VK_ERROR_DEVICE_LOST;
   }
   struct fd_bo *backing = NULL;
   unsigned references = 0;
   size_t metadata = 0;
   bool bounded = rd_window_footprint(submit, ring, initial_start,
                                       &backing, &references, &metadata);
   bool known_backing = false;
   for (unsigned i = 0; i < window->count; i++)
      known_backing |= window->pending[i].cs_backing == backing;
   const uint64_t backing_bytes = window->backing_bytes +
      (bounded && !known_backing ? backing->size : 0);
   if (!bounded || backing_bytes > 4 * 1024 * 1024) {
      VkResult before = rd_window_drain(device, window, "fallback");
      int64_t w0 = device->trace ? os_time_get_nano() : 0;
      VkResult result = fd_pipe_wait_timeout(device->pipe, fence, 5000000000ull) ?
         VK_ERROR_DEVICE_LOST : VK_SUCCESS;
      if (device->trace)
         device->trace->wait_ns += os_time_get_nano() - w0;
      uint64_t faults = 0;
      if (fd_pipe_get_param(device->pipe, FD_CTX_FAULTS, &faults) ||
          faults != device->faults)
         result = VK_ERROR_DEVICE_LOST;
      fd_fence_del(fence);
      fd_ringbuffer_del(ring);
      fd_submit_del(submit);
      return before == VK_SUCCESS ? result : before;
   }
   window->pending[window->count++] = (struct rd_pending_submit){submit, ring, fence, backing};
   window->chunks += chunks;
   window->cs_bytes += 32768;
   window->bo_refs += references;
   window->metadata_bytes += metadata;
   window->backing_bytes = backing_bytes;
   rd_window_report("ENQUEUE", "gpu", window, VK_SUCCESS);
   if (window->count == ARRAY_SIZE(window->pending))
      return rd_window_drain(device, window, "capacity");
   return VK_SUCCESS;
}

VkResult
rd_query_timestamp(struct rd_device *device, struct rd_query_pool *pool,
                    uint64_t *value)
{
   if (!pool->bo || !pool->map)
      return VK_ERROR_UNKNOWN;
   struct fd_submit *submit = fd_submit_new(device->pipe);
   if (!submit)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   struct fd_ringbuffer *ring = fd_submit_new_ringbuffer(submit, 0,
      FD_RINGBUFFER_PRIMARY | FD_RINGBUFFER_GROWABLE);
   if (!ring) {
      fd_submit_del(submit);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   cs_restore_emit(ring, device);
   OUT_PKT7(ring, CP_WAIT_FOR_IDLE, 0);
   rd_gpu_elapsed_emit(ring, pool->bo, 0);
   OUT_PKT7(ring, CP_WAIT_FOR_IDLE, 0);
   rd_flush(device, ring);
   VkResult result = rd_finish_submit(device, submit, ring, NULL, NULL);
   if (result != VK_SUCCESS)
      return result;
   if (fd_bo_cpu_prep(pool->bo, device->pipe, FD_BO_PREP_READ | FD_BO_PREP_NOSYNC))
      return VK_ERROR_DEVICE_LOST;
   __sync_synchronize();
   memcpy(value, pool->map, sizeof(*value));
   return VK_SUCCESS;
}

static void
rd_emit_dispatch_chunk(struct rd_device *device, struct fd_ringbuffer *ring,
                       const struct rd_dispatch *dispatch,
                       const uint32_t grid[3], const uint32_t base[3],
                       bool first_chunk, struct rd_gpu_elapsed *sample)
{
   const struct ir3_shader_variant *v = dispatch->pipeline->variant;
   if (device->trace)
      device->trace->chunks++;
   if (first_chunk) {
      rd_trace_submit_begin(device, ring);
      cs_restore_emit(ring, device);
      if (sample && sample->bo)
         rd_gpu_elapsed_emit(ring, sample->bo, 0);
      cs_program_emit(ring, dispatch->pipeline);
      rd_emit_constants(ring, dispatch);
      rd_emit_buffers(device, ring, dispatch);
      rd_emit_shader_resources(device, ring, dispatch);
      rd_emit_image_constants(ring, dispatch);
   }
   rd_emit_driver_params(ring, dispatch, base);
   OUT_PKT4(ring, REG_A5XX_HLSQ_CS_NDRANGE_0, 7);
   OUT_RING(ring, A5XX_HLSQ_CS_NDRANGE_0_KERNELDIM(3) |
      A5XX_HLSQ_CS_NDRANGE_0_LOCALSIZEX(v->local_size[0] - 1) |
      A5XX_HLSQ_CS_NDRANGE_0_LOCALSIZEY(v->local_size[1] - 1) |
      A5XX_HLSQ_CS_NDRANGE_0_LOCALSIZEZ(v->local_size[2] - 1));
   for (unsigned i = 0; i < 3; i++) {
      OUT_RING(ring, v->local_size[i] * grid[i]);
      OUT_RING(ring, 0);
   }
   OUT_PKT4(ring, REG_A5XX_HLSQ_CS_KERNEL_GROUP_X, 3);
   OUT_RING(ring, 1);
   OUT_RING(ring, 1);
   OUT_RING(ring, 1);
   OUT_PKT7(ring, CP_EXEC_CS, 4);
   OUT_RING(ring, 0);
   OUT_RING(ring, grid[0]);
   OUT_RING(ring, grid[1]);
   OUT_RING(ring, grid[2]);
   OUT_WFI5(ring);
}

static bool
rd_shared_batch_resources(const struct rd_pipeline *pipeline)
{
   const struct ir3_shader_variant *v = pipeline->variant;
   if (pipeline->resource_count != 4 || pipeline->descriptor_count != 4 ||
       pipeline->buffer_count != 4 || pipeline->used_descriptors != UINT64_C(15) ||
       pipeline->resource_usage.texture_count || pipeline->resource_usage.sampler_count ||
       pipeline->resource_usage.image_count || pipeline->sampling_mask ||
       v->image_mapping.num_tex || v->num_ssbos != 4 || v->constant_data_size ||
       pipeline->linear_subgroups || pipeline->linear_barriers || pipeline->packed_1d_barrier ||
       v->shader_options.workgroup_resident_threads ||
       !v->shader_options.nir_options.a5xx_ssbo_size_consts)
      return false;
   for (unsigned i = 0; i < 4; i++) {
      if (pipeline->resources[i].type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
          pipeline->resources[i].buffer_slot != i || pipeline->buffer_descriptors[i] != i)
         return false;
   }
   uint64_t resident_threads = (uint64_t)v->local_size[0] *
      v->local_size[1] * v->local_size[2];
   return resident_threads * ir3_shader_halfregs(v) <= 4096;
}

static bool
rd_short_buffer_batch_resources(struct rd_device *device,
                                 const struct rd_pipeline *pipeline)
{
   const struct ir3_shader_variant *v = pipeline->variant;
   const unsigned short_instruction_slots = 32 * 32 / 2;
   if (fd_dev_gpu_id(fd_pipe_dev_id(device->pipe)) != 506 ||
       v->local_size[0] != 16 || v->local_size[1] != 1 || v->local_size[2] != 4 ||
       v->loops || v->has_barrier || v->shared_size || v->pvtmem_size ||
       v->constant_data_size || v->info.double_threadsize || v->info.preamble_instrs_count ||
       !v->info.instrs_count || v->info.instrs_count > short_instruction_slots * 8 ||
       v->info.instrs_per_cat[6] > short_instruction_slots ||
       pipeline->resource_count != 2 || pipeline->descriptor_count != 2 ||
       pipeline->buffer_count != 2 || pipeline->used_descriptors != UINT64_C(3) ||
       pipeline->resource_usage.texture_count || pipeline->resource_usage.sampler_count ||
       pipeline->resource_usage.image_count || pipeline->sampling_mask ||
       v->image_mapping.num_tex || v->num_ssbos != 2 ||
       pipeline->linear_subgroups || pipeline->linear_barriers || pipeline->packed_1d_barrier ||
       v->shader_options.workgroup_resident_threads ||
       !v->shader_options.nir_options.a5xx_ssbo_size_consts)
      return false;
   for (unsigned i = 0; i < 2; i++) {
      if (pipeline->resources[i].type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
          pipeline->resources[i].buffer_slot != i || pipeline->buffer_descriptors[i] != i)
         return false;
   }
   return (uint64_t)device->info.threadsize_base * ir3_shader_halfregs(v) <= 4096;
}

static bool
rd_long_binary_batch_resources(struct rd_device *device,
                               const struct rd_pipeline *pipeline)
{
   const struct ir3_shader_variant *v = pipeline->variant;
   if (fd_dev_gpu_id(fd_pipe_dev_id(device->pipe)) != 506 ||
       v->local_size[0] != 128 || v->local_size[1] != 1 || v->local_size[2] != 1 ||
       v->loops || v->has_barrier || v->shared_size || v->pvtmem_size ||
       v->constant_data_size || v->info.double_threadsize || v->info.preamble_instrs_count ||
       v->instrlen > 202 || v->info.max_reg + 1 > 16 || v->info.max_half_reg + 1 > 12 ||
       !pipeline->batch_local_memory || pipeline->private_bo ||
       pipeline->resource_count != 3 || pipeline->descriptor_count != 3 ||
       pipeline->buffer_count != 3 || pipeline->used_descriptors != UINT64_C(7) ||
       pipeline->resource_usage.texture_count || pipeline->resource_usage.sampler_count ||
       pipeline->resource_usage.image_count || pipeline->sampling_mask ||
       v->image_mapping.num_tex || v->num_ssbos != 3 ||
       pipeline->linear_subgroups || pipeline->linear_barriers || pipeline->packed_1d_barrier ||
       v->shader_options.workgroup_resident_threads ||
       !v->shader_options.nir_options.a5xx_ssbo_size_consts ||
       pipeline->layout->push_range_count != 1 ||
       pipeline->layout->push_ranges[0].offset != 0 ||
       pipeline->layout->push_ranges[0].size != 116 ||
       pipeline->layout->push_ranges[0].stageFlags != VK_SHADER_STAGE_COMPUTE_BIT)
      return false;
   for (unsigned i = 0; i < 3; i++) {
      if (pipeline->resources[i].type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
          pipeline->resources[i].buffer_slot != i || pipeline->buffer_descriptors[i] != i)
         return false;
   }
   return (uint64_t)device->info.threadsize_base * ir3_shader_halfregs(v) <= 4096;
}

static bool
rd_packed_quant_batch_resources(struct rd_device *device,
                                const struct rd_pipeline *pipeline)
{
   const struct ir3_shader_variant *v = pipeline->variant;
   const bool mm = pipeline->logical_local_size[0] == 64 &&
      v->local_size[0] == 32 && v->shared_size == 4096 && (v->loops == 3 || v->loops == 4);
   const bool mv = pipeline->logical_local_size[0] == 128 &&
      v->local_size[0] == 64 && v->shared_size == 512 && (v->loops == 2 || v->loops == 3);
   if (fd_dev_gpu_id(fd_pipe_dev_id(device->pipe)) != 506 || (!mm && !mv) ||
       pipeline->logical_local_size[1] != 1 || pipeline->logical_local_size[2] != 1 ||
       v->local_size[1] != 2 || v->local_size[2] != 1 || !v->has_barrier ||
       !pipeline->packed_1d_barrier || !pipeline->batch_local_memory ||
       pipeline->linear_subgroups || pipeline->linear_barriers ||
       v->shader_options.workgroup_resident_threads ||
       !v->shader_options.nir_options.a5xx_ssbo_size_consts ||
       v->pvtmem_size || v->constant_data_size || v->info.double_threadsize ||
       v->info.preamble_instrs_count || v->instrlen > (mm && v->loops == 4 ? 54 : 53) ||
       v->info.max_reg + 1 > 12 || v->info.max_half_reg + 1 > 12 ||
       pipeline->resource_count != 3 || pipeline->descriptor_count != 3 ||
       pipeline->buffer_count != 3 || pipeline->used_descriptors != UINT64_C(7) ||
       pipeline->resource_usage.texture_count || pipeline->resource_usage.sampler_count ||
       pipeline->resource_usage.image_count || pipeline->sampling_mask ||
       v->image_mapping.num_tex || v->num_ssbos != 3 ||
       pipeline->layout->push_range_count != 1 ||
       pipeline->layout->push_ranges[0].offset != 0 ||
       pipeline->layout->push_ranges[0].size != 68 ||
       pipeline->layout->push_ranges[0].stageFlags != VK_SHADER_STAGE_COMPUTE_BIT)
      return false;
   for (unsigned i = 0; i < 3; i++) {
      if (pipeline->resources[i].type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
          pipeline->resources[i].buffer_slot != i || pipeline->buffer_descriptors[i] != i)
         return false;
   }
   return (uint64_t)v->local_size[0] * v->local_size[1] * v->local_size[2] *
      ir3_shader_halfregs(v) <= 4096;
}

static bool
rd_packed_rms_batch_resources(struct rd_device *device,
                                const struct rd_pipeline *pipeline)
{
   const struct ir3_shader_variant *v = pipeline->variant;
   if (fd_dev_gpu_id(fd_pipe_dev_id(device->pipe)) != 506 ||
       pipeline->logical_local_size[0] != 128 || v->local_size[0] != 64 ||
       v->shared_size != 512 || v->loops != 2 ||
       pipeline->logical_local_size[1] != 1 || pipeline->logical_local_size[2] != 1 ||
       v->local_size[1] != 2 || v->local_size[2] != 1 || !v->has_barrier ||
       !pipeline->packed_1d_barrier || !pipeline->batch_local_memory ||
       pipeline->linear_subgroups || pipeline->linear_barriers ||
       v->shader_options.workgroup_resident_threads ||
       !v->shader_options.nir_options.a5xx_ssbo_size_consts ||
       v->pvtmem_size || v->constant_data_size || v->info.double_threadsize ||
       v->info.preamble_instrs_count || v->instrlen > 477 ||
       v->info.max_reg + 1 > 10 || v->info.max_half_reg + 1 > 12 ||
       pipeline->resource_count != 3 || pipeline->descriptor_count != 3 ||
       pipeline->buffer_count != 3 ||
       (pipeline->used_descriptors & UINT64_C(5)) != UINT64_C(5) ||
       (pipeline->used_descriptors & ~UINT64_C(7)) ||
       pipeline->resource_usage.texture_count || pipeline->resource_usage.sampler_count ||
       pipeline->resource_usage.image_count || pipeline->sampling_mask ||
       v->image_mapping.num_tex || v->num_ssbos != 3 ||
       pipeline->layout->push_range_count != 1 ||
       pipeline->layout->push_ranges[0].offset != 0 ||
       pipeline->layout->push_ranges[0].size != 116 ||
       pipeline->layout->push_ranges[0].stageFlags != VK_SHADER_STAGE_COMPUTE_BIT)
      return false;
   for (unsigned i = 0; i < 3; i++) {
      if (pipeline->resources[i].type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
          pipeline->resources[i].buffer_slot != i || pipeline->buffer_descriptors[i] != i)
         return false;
   }
   return (uint64_t)v->local_size[0] * v->local_size[1] * v->local_size[2] *
      ir3_shader_halfregs(v) <= 4096;
}

static bool
rd_local_two_buffer_batch_resources(struct rd_device *device,
                                const struct rd_pipeline *pipeline)
{
   const struct ir3_shader_variant *v = pipeline->variant;
   const bool transpose = pipeline->logical_local_size[0] == 32 &&
      pipeline->logical_local_size[1] == 4 && v->local_size[0] == 32 &&
      v->local_size[1] == 4 && !pipeline->packed_1d_barrier &&
      v->shared_size == 4224 && v->loops == 3 && v->instrlen <= 73 &&
      v->info.max_reg + 1 <= 8 && v->info.max_half_reg + 1 <= 16;
   const bool norm = pipeline->logical_local_size[0] == 128 &&
      pipeline->logical_local_size[1] == 1 && v->local_size[0] == 64 &&
      v->local_size[1] == 2 && pipeline->packed_1d_barrier &&
      v->shared_size == 1024 && v->loops == 2 && v->instrlen <= 30 &&
      v->info.max_reg + 1 <= 8 && v->info.max_half_reg + 1 <= 2;
   if (fd_dev_gpu_id(fd_pipe_dev_id(device->pipe)) != 506 || (!transpose && !norm) ||
       pipeline->logical_local_size[2] != 1 || v->local_size[2] != 1 ||
       !v->has_barrier || !pipeline->batch_local_memory ||
       pipeline->linear_subgroups || pipeline->linear_barriers ||
       v->shader_options.workgroup_resident_threads ||
       !v->shader_options.nir_options.a5xx_ssbo_size_consts ||
       v->pvtmem_size || v->constant_data_size || v->info.double_threadsize ||
       v->info.preamble_instrs_count ||
       pipeline->resource_count != 2 || pipeline->descriptor_count != 2 ||
       pipeline->buffer_count != 2 || pipeline->used_descriptors != UINT64_C(3) ||
       pipeline->resource_usage.texture_count || pipeline->resource_usage.sampler_count ||
       pipeline->resource_usage.image_count || pipeline->sampling_mask ||
       v->image_mapping.num_tex || v->num_ssbos != 2 ||
       pipeline->layout->push_range_count != 1 ||
       pipeline->layout->push_ranges[0].offset != 0 ||
       pipeline->layout->push_ranges[0].size != 120 ||
       pipeline->layout->push_ranges[0].stageFlags != VK_SHADER_STAGE_COMPUTE_BIT)
      return false;
   for (unsigned i = 0; i < 2; i++) {
      if (pipeline->resources[i].type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
          pipeline->resources[i].buffer_slot != i || pipeline->buffer_descriptors[i] != i)
         return false;
   }
   return (uint64_t)v->local_size[0] * v->local_size[1] * v->local_size[2] *
      ir3_shader_halfregs(v) <= 4096;
}

static bool
rd_packed_attention_batch_resources(struct rd_device *device,
                                    const struct rd_pipeline *pipeline)
{
   const struct ir3_shader_variant *v = pipeline->variant;
   const bool mm = v->shared_size == 8704 && v->instrlen <= 358 &&
      v->info.max_reg + 1 <= 48 && v->info.max_half_reg + 1 <= 48;
   const bool mv = v->shared_size == 256 && v->instrlen <= 536 &&
      v->info.max_reg + 1 <= 12 && v->info.max_half_reg + 1 <= 48;
   const bool softmax = v->shared_size == 128 && v->instrlen <= 65 &&
      v->loops == 3 && v->info.max_reg + 1 <= 10 && v->info.max_half_reg + 1 <= 9;
   const unsigned buffers = mm ? 3 : mv ? 5 : 4;
   const unsigned push_size = mv ? 52 : 68;
   if (fd_dev_gpu_id(fd_pipe_dev_id(device->pipe)) != 506 || (!mm && !mv && !softmax) ||
       pipeline->logical_local_size[0] != 32 || pipeline->logical_local_size[1] != 1 ||
       pipeline->logical_local_size[2] != 1 || v->local_size[0] != 16 ||
       v->local_size[1] != 2 || v->local_size[2] != 1 || !v->has_barrier ||
       !pipeline->packed_1d_barrier || !pipeline->batch_local_memory ||
       pipeline->linear_subgroups || pipeline->linear_barriers ||
       v->shader_options.workgroup_resident_threads ||
       !v->shader_options.nir_options.a5xx_ssbo_size_consts ||
       v->pvtmem_size || v->constant_data_size || v->info.double_threadsize ||
       v->info.preamble_instrs_count ||
       pipeline->resource_count != buffers || pipeline->descriptor_count != buffers ||
       pipeline->buffer_count != buffers ||
       (pipeline->used_descriptors & UINT64_C(7)) != UINT64_C(7) ||
       (pipeline->used_descriptors & ~((UINT64_C(1) << buffers) - 1)) ||
       pipeline->resource_usage.texture_count || pipeline->resource_usage.sampler_count ||
       pipeline->resource_usage.image_count || pipeline->sampling_mask ||
       v->image_mapping.num_tex || v->num_ssbos != buffers ||
       pipeline->layout->push_range_count != 1 ||
       pipeline->layout->push_ranges[0].offset != 0 ||
       pipeline->layout->push_ranges[0].size != push_size ||
       pipeline->layout->push_ranges[0].stageFlags != VK_SHADER_STAGE_COMPUTE_BIT)
      return false;
   for (unsigned i = 0; i < buffers; i++) {
      if (pipeline->resources[i].type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
          pipeline->resources[i].buffer_slot != i || pipeline->buffer_descriptors[i] != i)
         return false;
   }
   return UINT64_C(32) * ir3_shader_halfregs(v) <= 4096;
}

static VkResult
rd_execute_dispatch(struct rd_device *device, const struct rd_dispatch *dispatch,
                    struct rd_submit_profile *profile, struct rd_submit_window *window)
{
   if (!dispatch->grid[0] || !dispatch->grid[1] || !dispatch->grid[2])
      return VK_SUCCESS;
   for (unsigned i = 0; i < dispatch->pipeline->descriptor_count; i++) {
      const struct rd_descriptor *desc = &dispatch->descriptors[i];
      if ((dispatch->pipeline->used_descriptors & (UINT64_C(1) << i)) &&
          !rd_descriptor_ready(desc))
         return VK_ERROR_UNKNOWN;
   }
   const struct ir3_shader_variant *v = dispatch->pipeline->variant;
   if (window && (profile || !rd_window_pipeline(device, dispatch))) {
      VkResult drained = rd_window_drain(device, window, "unsupported");
      if (drained != VK_SUCCESS)
         return drained;
      window = NULL;
   }
   uint32_t tile[3], budget = v->has_barrier ? 1 : 64;
   int64_t shared_groups = debug_get_num_option("RADISH_SHARED_GROUPS", 1);
   const bool packed_shared = dispatch->pipeline->packed_1d_barrier &&
      (rd_packed_quant_batch_resources(device, dispatch->pipeline) ||
       rd_packed_attention_batch_resources(device, dispatch->pipeline) ||
       rd_packed_rms_batch_resources(device, dispatch->pipeline) ||
       rd_local_two_buffer_batch_resources(device, dispatch->pipeline));
   if (v->has_barrier &&
       ((uint64_t)v->local_size[0] * v->local_size[1] * v->local_size[2] <= 32 || packed_shared) &&
       shared_groups >= 1 && shared_groups <= 8)
      budget = shared_groups;
   bool tiled_x2 = debug_get_num_option("RADISH_TILED_GROUPS_X", 1) == 2 &&
      dispatch->grid[0] >= 2 &&
      fd_dev_gpu_id(fd_pipe_dev_id(device->pipe)) == 506 &&
      v->local_size[0] == 16 && v->local_size[1] == 8 &&
      v->local_size[2] == 1 && v->has_barrier && v->loops == 4 &&
      v->shared_size == 16384 && !v->pvtmem_size &&
      !v->info.double_threadsize && v->info.max_reg + 1 <= 16 &&
      v->info.max_half_reg == -1 &&
      dispatch->pipeline->descriptor_count == 4 && v->instrlen == 31;
   if (tiled_x2)
      budget = 2;
   /* Experiment knobs: workgroups per CP_EXEC_CS chunk. */
   const int64_t barrier_groups = debug_get_num_option("RADISH_BARRIER_GROUPS", 0);
   const int64_t plain_groups = debug_get_num_option("RADISH_GROUPS", 0);
   if (v->has_barrier && barrier_groups >= 1 && barrier_groups <= 65535)
      budget = barrier_groups;
   if (!v->has_barrier && plain_groups >= 1 && plain_groups <= 65535)
      budget = plain_groups;
   for (unsigned i = 0; i < 3; i++) {
      if (dispatch->grid[i] > 65535)
         return VK_ERROR_FEATURE_NOT_PRESENT;
      tile[i] = MIN2(budget, dispatch->grid[i]);
      budget /= tile[i];
   }
   int64_t requested_chunks = debug_get_num_option("RADISH_BATCH_CHUNKS", 1);
   unsigned max_chunks = 1;
   if (!v->loops && !v->has_barrier &&
       (v->instrlen <= 32 || rd_short_buffer_batch_resources(device, dispatch->pipeline) ||
        rd_long_binary_batch_resources(device, dispatch->pipeline)) &&
       ((requested_chunks >= 1 && requested_chunks <= 4) || requested_chunks == 8))
      max_chunks = requested_chunks;
   if ((requested_chunks == 2 || requested_chunks == 4 || requested_chunks == 8) &&
       (rd_packed_quant_batch_resources(device, dispatch->pipeline) ||
        rd_packed_attention_batch_resources(device, dispatch->pipeline) ||
        rd_packed_rms_batch_resources(device, dispatch->pipeline) ||
        rd_local_two_buffer_batch_resources(device, dispatch->pipeline)))
      max_chunks = requested_chunks;
   int64_t tiled_chunks = debug_get_num_option("RADISH_TILED_BATCH_CHUNKS", 1);
   if ((tiled_chunks == 2 || tiled_chunks == 4 || tiled_chunks == 8) &&
       fd_dev_gpu_id(fd_pipe_dev_id(device->pipe)) == 506 &&
       v->local_size[0] == 16 && v->local_size[1] == 8 &&
       v->local_size[2] == 1 && v->has_barrier &&
       ((v->loops == 4 && v->shared_size == 16384) ||
        (v->loops == 3 && v->shared_size == 10240 &&
         dispatch->pipeline->layout->push_range_count == 1 &&
         dispatch->pipeline->layout->push_ranges[0].offset == 0 &&
         dispatch->pipeline->layout->push_ranges[0].size == 12 &&
         dispatch->pipeline->layout->push_ranges[0].stageFlags == VK_SHADER_STAGE_COMPUTE_BIT)) &&
       !v->pvtmem_size && !v->info.double_threadsize && v->info.max_reg + 1 <= 16 &&
       rd_shared_batch_resources(dispatch->pipeline))
      max_chunks = tiled_chunks;
   if (tiled_x2)
      max_chunks = MIN2(max_chunks, 4);
   if (rd_packed_quant_batch_resources(device, dispatch->pipeline) ||
       (rd_packed_attention_batch_resources(device, dispatch->pipeline) &&
        (v->shared_size == 8704 || v->shared_size == 256)))
      max_chunks = 1;
   struct rd_gpu_elapsed storage = {0};
   struct rd_gpu_elapsed *sample = NULL;
   const bool zero_base = !dispatch->base[0] && !dispatch->base[1] && !dispatch->base[2];
   if (zero_base && profile && !dispatch->indirect && !tiled_x2 && max_chunks == 8 &&
       fd_dev_gpu_id(fd_pipe_dev_id(device->pipe)) == 506 &&
       v->local_size[0] == 16 && v->local_size[1] == 8 &&
       v->local_size[2] == 1 && v->has_barrier && v->loops == 4 &&
       v->shared_size == 16384 && !v->pvtmem_size &&
       !v->info.double_threadsize && v->info.max_reg + 1 <= 16 &&
       v->info.max_half_reg == -1 &&
       dispatch->pipeline->descriptor_count == 4 && v->instrlen == 27 &&
       dispatch->grid[0] == 4 && dispatch->grid[1] == 2 &&
       dispatch->grid[2] == 1 && tile[0] == 1 && tile[1] == 1 && tile[2] == 1) {
      const char *enabled = getenv("RADISH_GPU_ELAPSED");
      const char *profiling = getenv("RADISH_PROFILE");
      if (enabled && !strcmp(enabled, "1") &&
          profiling && !strcmp(profiling, "1") &&
          rd_gpu_elapsed_reserve(&storage.sample)) {
         sample = &storage;
         sample->dispatch = dispatch;
         sample->chunks = 0;
         sample->submit_count = device->submit_count;
         rd_gpu_elapsed_prepare(device, sample);
      }
   }
   if (!sample && zero_base && profile && !dispatch->indirect && !tiled_x2 &&
       max_chunks == 1 && fd_dev_gpu_id(fd_pipe_dev_id(device->pipe)) == 506 &&
       v->local_size[0] == 16 && v->local_size[1] == 8 &&
       v->local_size[2] == 1 && !v->has_barrier && v->loops == 1 &&
       !v->shared_size && !v->pvtmem_size && !v->info.double_threadsize &&
       v->info.max_half_reg == -1 && ir3_shader_halfregs(v) <= 32 &&
       dispatch->grid[0] == 4 && dispatch->grid[1] == 2 &&
       dispatch->grid[2] == 1 && tile[0] == 4 && tile[1] == 2 && tile[2] == 1) {
      const char *enabled = getenv("RADISH_GPU_ELAPSED");
      const char *profiling = getenv("RADISH_PROFILE");
      const char *requested = getenv("RADISH_GPU_ELAPSED_CALIBRATION");
      const char *calibration = NULL;
      if (requested && !strcmp(requested, "mad") && v->instrlen == 8 &&
          v->info.max_reg == 15 && dispatch->pipeline->descriptor_count == 3)
         calibration = "mad";
      else if (requested && !strcmp(requested, "mad-unroll4") && v->instrlen == 12 &&
               v->info.max_reg == 15 && dispatch->pipeline->descriptor_count == 3)
         calibration = "mad-unroll4";
      else if (requested && !strcmp(requested, "copy") && v->instrlen == 3 &&
               v->info.max_reg == 3 && dispatch->pipeline->descriptor_count == 2)
         calibration = "copy";
      if (enabled && !strcmp(enabled, "1") &&
          profiling && !strcmp(profiling, "1") && calibration &&
          rd_gpu_elapsed_reserve(&storage.sample)) {
         sample = &storage;
         sample->dispatch = dispatch;
         sample->calibration = calibration;
         sample->chunks = 0;
         sample->submit_count = device->submit_count;
         rd_gpu_elapsed_prepare(device, sample);
      }
   }
   if (!sample && zero_base && profile && !dispatch->indirect && !tiled_x2 &&
       max_chunks == 1 && tiled_chunks == 8 &&
       fd_dev_gpu_id(fd_pipe_dev_id(device->pipe)) == 506 &&
       v->local_size[0] == 16 && v->local_size[1] == 8 &&
       v->local_size[2] == 1 && v->has_barrier && v->loops == 4 &&
       v->shared_size == 16384 && !v->pvtmem_size &&
       !v->info.double_threadsize && v->info.max_reg == 15 &&
       v->info.max_half_reg == -1 &&
       dispatch->pipeline->descriptor_count == 4 && v->instrlen == 25 &&
       dispatch->grid[0] == 4 && dispatch->grid[1] == 2 &&
       dispatch->grid[2] == 1 && tile[0] == 1 && tile[1] == 1 && tile[2] == 1) {
      const char *enabled = getenv("RADISH_GPU_ELAPSED");
      const char *profiling = getenv("RADISH_PROFILE");
      const char *requested = getenv("RADISH_GPU_ELAPSED_CALIBRATION");
      if (enabled && !strcmp(enabled, "1") &&
          profiling && !strcmp(profiling, "1") &&
          requested && !strcmp(requested, "shared-hot")) {
         max_chunks = 8;
         if (rd_gpu_elapsed_reserve(&storage.sample)) {
            sample = &storage;
            sample->dispatch = dispatch;
            sample->calibration = "shared-hot";
            sample->chunks = 0;
            sample->submit_count = device->submit_count;
            rd_gpu_elapsed_prepare(device, sample);
         }
      }
   }
   if (!sample && zero_base && profile && !dispatch->indirect && !tiled_x2 &&
       max_chunks == 8 && fd_dev_gpu_id(fd_pipe_dev_id(device->pipe)) == 506 &&
       v->local_size[0] == 16 && v->local_size[1] == 8 &&
       v->local_size[2] == 1 && v->has_barrier && v->loops == 4 &&
       v->shared_size == 16384 && !v->pvtmem_size &&
       !v->info.double_threadsize && v->info.max_reg == 15 &&
       v->info.max_half_reg == -1 &&
       dispatch->pipeline->descriptor_count == 4 && v->instrlen == 28 &&
       dispatch->grid[0] == 4 && dispatch->grid[1] == 2 &&
       dispatch->grid[2] == 1 && tile[0] == 1 && tile[1] == 1 && tile[2] == 1) {
      const char *enabled = getenv("RADISH_GPU_ELAPSED");
      const char *profiling = getenv("RADISH_PROFILE");
      const char *requested = getenv("RADISH_GPU_ELAPSED_CALIBRATION");
      if (enabled && !strcmp(enabled, "1") &&
          profiling && !strcmp(profiling, "1") &&
          requested && !strcmp(requested, "packed-a") &&
          rd_gpu_elapsed_reserve(&storage.sample)) {
         sample = &storage;
         sample->dispatch = dispatch;
         sample->calibration = "packed-a";
         sample->chunks = 0;
         sample->submit_count = device->submit_count;
         rd_gpu_elapsed_prepare(device, sample);
      }
   }
   struct fd_submit *submit = NULL;
   struct fd_ringbuffer *ring = NULL;
   static unsigned diagnostic_dispatches;
   const bool diagnostics = debug_get_bool_option("RADISH_RESOURCE_DIAGNOSTICS", false) &&
      p_atomic_inc_return(&diagnostic_dispatches) <= 4096;
   const uint64_t first_submit = device->submit_count;
   if (diagnostics) {
      fprintf(stderr, "RADISH_DISPATCH_BEGIN native=%u grid=%u,%u,%u tile=%u,%u,%u batch=%u packed=%u local_memory=%u\n",
         v->instrlen, dispatch->grid[0], dispatch->grid[1], dispatch->grid[2],
         tile[0], tile[1], tile[2], max_chunks, dispatch->pipeline->packed_1d_barrier,
         dispatch->pipeline->batch_local_memory);
      fflush(stderr);
   }
   unsigned chunks = 0;
   uint32_t *initial_ring_start = NULL;
   uint32_t base[3];
   for (base[2] = 0; base[2] < dispatch->grid[2]; base[2] += tile[2]) {
      for (base[1] = 0; base[1] < dispatch->grid[1]; base[1] += tile[1]) {
         for (base[0] = 0; base[0] < dispatch->grid[0]; base[0] += tile[0]) {
            if (!submit) {
               submit = fd_submit_new(device->pipe);
               if (!submit) {
                  rd_gpu_elapsed_finish(device, sample,
                     VK_ERROR_OUT_OF_HOST_MEMORY, 0, 0);
                  rd_gpu_elapsed_release(sample);
                  return VK_ERROR_OUT_OF_HOST_MEMORY;
               }
               ring = fd_submit_new_ringbuffer(submit, 0,
                  FD_RINGBUFFER_PRIMARY | FD_RINGBUFFER_GROWABLE);
               if (!ring) {
                  fd_submit_del(submit);
                  rd_gpu_elapsed_finish(device, sample,
                     VK_ERROR_OUT_OF_HOST_MEMORY, 0, 0);
                  rd_gpu_elapsed_release(sample);
                  return VK_ERROR_OUT_OF_HOST_MEMORY;
               }
            }
            if (!chunks)
               initial_ring_start = ring->start;
            uint32_t grid[3];
            for (unsigned i = 0; i < 3; i++)
               grid[i] = MIN2(tile[i], dispatch->grid[i] - base[i]);
            rd_emit_dispatch_chunk(device, ring, dispatch, grid, base,
                                   chunks == 0, sample);
            if (profile)
               profile->chunks++;
            if (sample)
               sample->chunks++;
            if (++chunks == max_chunks) {
               rd_flush(device, ring);
               if (sample && sample->bo) {
                  rd_gpu_elapsed_emit(ring, sample->bo, 16);
                  OUT_WFI5(ring);
               }
               VkResult result = rd_finish_dispatch_submit(device, submit, ring, profile,
                                                   sample, window, initial_ring_start, chunks);
               rd_gpu_elapsed_release(sample);
               sample = NULL;
               submit = NULL;
               ring = NULL;
               chunks = 0;
               if (result != VK_SUCCESS)
                  return result;
            }
         }
      }
   }
   if (submit) {
      rd_flush(device, ring);
      VkResult result = rd_finish_dispatch_submit(device, submit, ring, profile, sample,
                                                    window, initial_ring_start, chunks);
      rd_gpu_elapsed_release(sample);
      if (diagnostics) {
         fprintf(stderr, "RADISH_DISPATCH_END native=%u result=%d submits=%" PRIu64 "\n",
            v->instrlen, result, device->submit_count - first_submit);
         fflush(stderr);
      }
      return result;
   }
   rd_gpu_elapsed_release(sample);
   if (diagnostics) {
      fprintf(stderr, "RADISH_DISPATCH_END native=%u result=0 submits=%" PRIu64 "\n",
         v->instrlen, device->submit_count - first_submit);
      fflush(stderr);
   }
   return VK_SUCCESS;
}

static VkResult
rd_execute_transfer(struct rd_device *device, enum rd_command_type type,
                      const struct rd_transfer *transfer,
                      struct rd_submit_profile *profile)
{
   if (!transfer->dst || !transfer->dst->memory ||
       (type == RD_CMD_COPY && (!transfer->src || !transfer->src->memory)))
      return VK_ERROR_UNKNOWN;
   if (type == RD_CMD_COPY &&
       ((transfer->src_offset | transfer->dst_offset | transfer->size) & 3)) {
      struct fd_bo *src = transfer->src->memory->bo;
      struct fd_bo *dst = transfer->dst->memory->bo;
      if (src == dst) {
         if (fd_bo_cpu_prep(src, device->pipe, FD_BO_PREP_READ | FD_BO_PREP_WRITE))
            return VK_ERROR_DEVICE_LOST;
      } else if (fd_bo_cpu_prep(src, device->pipe, FD_BO_PREP_READ) ||
                 fd_bo_cpu_prep(dst, device->pipe, FD_BO_PREP_WRITE)) {
         return VK_ERROR_DEVICE_LOST;
      }
      void *src_map = fd_bo_map(src);
      void *dst_map = fd_bo_map(dst);
      if (!src_map || !dst_map)
         return VK_ERROR_MEMORY_MAP_FAILED;
      memmove((uint8_t *)dst_map + transfer->dst->offset + transfer->dst_offset,
              (uint8_t *)src_map + transfer->src->offset + transfer->src_offset,
              transfer->size);
      __sync_synchronize();
      return VK_SUCCESS;
   }
   for (VkDeviceSize done = 0; done < transfer->size;) {
      unsigned bytes = MIN2(transfer->size - done, 4096);
      struct fd_submit *submit = fd_submit_new(device->pipe);
      if (!submit)
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      struct fd_ringbuffer *ring = fd_submit_new_ringbuffer(submit, 0,
         FD_RINGBUFFER_PRIMARY | FD_RINGBUFFER_GROWABLE);
      if (!ring) {
         fd_submit_del(submit);
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }
      rd_trace_submit_begin(device, ring);
      struct rd_buffer *dst = transfer->dst;
      unsigned dst_offset = dst->offset + transfer->dst_offset + done;
      if (type == RD_CMD_COPY) {
         struct rd_buffer *src = transfer->src;
         unsigned src_offset = src->offset + transfer->src_offset + done;
         for (unsigned i = 0; i < bytes; i += 4) {
            OUT_PKT7(ring, CP_MEM_TO_MEM, 5);
            OUT_RING(ring, 0);
            OUT_RELOC(ring, dst->memory->bo, dst_offset + i, 0, 0);
            OUT_RELOC(ring, src->memory->bo, src_offset + i, 0, 0);
         }
      } else {
         OUT_PKT7(ring, CP_MEM_WRITE, 2 + bytes / 4);
         OUT_RELOC(ring, dst->memory->bo, dst_offset, 0, 0);
         for (unsigned i = 0; i < bytes; i += 4)
            OUT_RING(ring, type == RD_CMD_FILL ? transfer->value :
               ((const uint32_t *)transfer->data)[(done + i) / 4]);
      }
      OUT_PKT7(ring, CP_WAIT_MEM_WRITES, 0);
      rd_flush(device, ring);
      VkResult result = rd_finish_submit(device, submit, ring, profile, NULL);
      if (result != VK_SUCCESS)
         return result;
      done += bytes;
   }
   return VK_SUCCESS;
}

static VkResult
rd_resolve_indirect(struct rd_device *device, struct rd_dispatch *dispatch)
{
   struct rd_buffer *buffer = dispatch->indirect;
   if (!buffer || !buffer->memory ||
       dispatch->indirect_offset > buffer->vk.size ||
       sizeof(VkDispatchIndirectCommand) > buffer->vk.size - dispatch->indirect_offset)
      return VK_ERROR_UNKNOWN;
   struct fd_bo *bo = buffer->memory->bo;
   if (fd_bo_cpu_prep(bo, device->pipe, FD_BO_PREP_READ))
      return VK_ERROR_DEVICE_LOST;
   const uint8_t *map = fd_bo_map(bo);
   if (!map)
      return VK_ERROR_MEMORY_MAP_FAILED;
   __sync_synchronize();
   VkDispatchIndirectCommand args;
   memcpy(&args, map + buffer->offset + dispatch->indirect_offset, sizeof(args));
   dispatch->grid[0] = args.x;
   dispatch->grid[1] = args.y;
   dispatch->grid[2] = args.z;
   return VK_SUCCESS;
}

static VkResult
rd_queue_fail(struct vk_queue *queue)
{
   VkResult result = vk_queue_set_lost(queue, "Radish execution or synchronization failed");
   rd_wake_device_waiters(queue->base.device);
   return result;
}

VkResult
rd_queue_submit(struct vk_queue *queue, struct vk_queue_submit *submit)
{
   struct rd_device *device = container_of(queue->base.device, struct rd_device, vk);
   bool profile = debug_get_bool_option("RADISH_PROFILE", false);
   struct rd_submit_window window = {0};
   bool deferred = !profile && debug_get_num_option("RADISH_BATCH_CHUNKS", 1) == 8;
   __sync_synchronize();
   const int64_t queue_start = device->trace ? os_time_get_nano() : 0;
   VkResult result = vk_sync_wait_many(&device->vk, submit->wait_count,
      submit->waits, VK_SYNC_WAIT_COMPLETE, UINT64_MAX);
   if (result != VK_SUCCESS)
      return rd_queue_fail(queue);
   for (uint32_t i = 0; i < submit->command_buffer_count; i++) {
      struct rd_command_buffer *command = container_of(submit->command_buffers[i],
         struct rd_command_buffer, vk);
      util_dynarray_foreach(&command->commands, struct rd_command, recorded) {
         if (rd_trace_needs_resolve(device)) {
            result = rd_window_drain(device, &window, "trace");
            if (result != VK_SUCCESS)
               return rd_queue_fail(queue);
            rd_trace_resolve(device, queue_start);
         }
         rd_trace_command_begin(device);
         int64_t start = profile ? os_time_get_nano() : 0;
         uint32_t seqno = device->seqno;
         uint64_t submit_count = device->submit_count;
         struct rd_submit_profile submit_profile = {0};
         const struct rd_dispatch *dispatch = &recorded->dispatch;
         struct rd_dispatch indirect;
         if (recorded->type != RD_CMD_DISPATCH || recorded->dispatch.indirect ||
             !rd_window_pipeline(device, &recorded->dispatch)) {
            result = rd_window_drain(device, &window, "cpu");
            if (result != VK_SUCCESS)
               return rd_queue_fail(queue);
         }
         if (recorded->type == RD_CMD_DISPATCH) {
            result = VK_SUCCESS;
            indirect = *dispatch;
            dispatch = &indirect;
            for (uint32_t i = 0; i < dispatch->pipeline->resource_count; i++) {
               if (!(dispatch->pipeline->used_descriptors & (UINT64_C(1) << i)))
                  continue;
               const struct rd_nir_resource_binding *resource = &dispatch->pipeline->resources[i];
               const struct rd_descriptor_set *set = dispatch->sets[resource->descriptor_set];
               indirect.descriptors[i] = set->descriptors[resource->logical_slot];
               struct rd_descriptor *descriptor = &indirect.descriptors[i];
               if (descriptor->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
                   descriptor->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
                  descriptor->offset += dispatch->dynamic_offsets[i];
            }
            if (dispatch->indirect)
               result = rd_resolve_indirect(device, &indirect);
            if (result == VK_SUCCESS)
               result = rd_prepare_sampling(device, &indirect);
            if (result == VK_SUCCESS)
               result = rd_execute_dispatch(device, dispatch,
                                            profile ? &submit_profile : NULL,
                                            deferred ? &window : NULL);
            if (result == VK_SUCCESS)
               rd_query_account_dispatch(device, dispatch);
            if (indirect.sampling_addresses)
               fd_bo_del(indirect.sampling_addresses);
         } else if (recorded->type == RD_CMD_IMAGE_COPY || recorded->type == RD_CMD_IMAGE_CLEAR) {
            result = rd_execute_image_transfer(device, &recorded->image_transfer,
                                                recorded->type == RD_CMD_IMAGE_CLEAR);
         } else if (recorded->type >= RD_CMD_COPY && recorded->type <= RD_CMD_UPDATE) {
            result = rd_execute_transfer(device, recorded->type, &recorded->transfer,
                                         profile ? &submit_profile : NULL);
         } else if (recorded->type >= RD_CMD_EVENT_SET && recorded->type <= RD_CMD_EVENT_WAIT) {
            result = rd_execute_event(device, recorded->type, &recorded->event);
         } else if (recorded->type >= RD_CMD_QUERY_RESET && recorded->type <= RD_CMD_QUERY_COPY) {
            result = rd_execute_query(device, recorded->type, &recorded->query);
         } else if (recorded->type == RD_CMD_RENDER) {
            result = rd_execute_render(device, recorded->render);
         } else {
            result = VK_ERROR_UNKNOWN;
         }
         if (device->trace) {
            struct rd_trace_record info = {
               .type = recorded->type,
            };
            if (recorded->type == RD_CMD_DISPATCH) {
               const struct ir3_shader_variant *v = dispatch->pipeline->variant;
               info.seq = device->trace->dispatch_seq++;
               info.pipeline = dispatch->pipeline;
               memcpy(info.grid, dispatch->grid, sizeof(info.grid));
               for (unsigned i = 0; i < 3; i++)
                  info.local[i] = v->local_size[i];
               info.instrlen = v->instrlen;
               info.max_reg = v->info.max_reg + 1;
               info.max_half_reg = v->info.max_half_reg + 1;
               info.barrier = v->has_barrier;
               info.shared_size = v->shared_size;
            } else if (recorded->type >= RD_CMD_COPY && recorded->type <= RD_CMD_UPDATE) {
               info.bytes = recorded->transfer.size;
            }
            rd_trace_command_end(device, &info);
         }
         if (profile) {
            int64_t elapsed = os_time_get_nano() - start;
            uint32_t chunks = device->seqno - seqno;
            uint64_t submits = device->submit_count - submit_count;
            if (recorded->type == RD_CMD_DISPATCH) {
               const struct rd_dispatch *d = dispatch;
               const struct ir3_shader_variant *v = d->pipeline->variant;
               fprintf(stderr, "RADISH_PROFILE dispatch pipeline=%p grid=%u,%u,%u local=%u,%u,%u barrier=%u loops=%u instrlen=%u chunks=%" PRIu64 " submits=%" PRIu64 " ns=%" PRId64 " submit_ns=%" PRId64 " submit_max_ns=%" PRId64 " result=%d\n",
                       (void *)d->pipeline, d->grid[0], d->grid[1], d->grid[2],
                       v->local_size[0], v->local_size[1], v->local_size[2],
                       (unsigned)v->has_barrier, v->loops, v->instrlen,
                       submit_profile.chunks, submits, elapsed, submit_profile.total_ns,
                       submit_profile.max_ns, result);
            } else if (recorded->type == RD_CMD_IMAGE_COPY || recorded->type == RD_CMD_IMAGE_CLEAR) {
               const struct rd_image_transfer *t = &recorded->image_transfer;
               fprintf(stderr, "RADISH_PROFILE image_transfer type=%u bytes=%" PRIu64 " ns=%" PRId64 " result=%d\n",
                       (unsigned)recorded->type, (uint64_t)t->row_bytes * t->rows * t->slices,
                       elapsed, result);
            } else if (recorded->type >= RD_CMD_COPY && recorded->type <= RD_CMD_UPDATE) {
               fprintf(stderr, "RADISH_PROFILE transfer type=%u bytes=%" PRIu64 " chunks=%u submits=%" PRIu64 " ns=%" PRId64 " submit_ns=%" PRId64 " submit_max_ns=%" PRId64 " result=%d\n",
                       (unsigned)recorded->type, (uint64_t)recorded->transfer.size,
                       chunks, submits, elapsed, submit_profile.total_ns,
                       submit_profile.max_ns, result);
            } else {
               fprintf(stderr, "RADISH_PROFILE synchronization type=%u submits=%" PRIu64 " ns=%" PRId64 " result=%d\n",
                       (unsigned)recorded->type, submits, elapsed, result);
            }
         }
         if (result != VK_SUCCESS) {
            rd_window_drain(device, &window, "error");
            return rd_queue_fail(queue);
         }
      }
   }
   result = rd_window_drain(device, &window, "final");
   if (result != VK_SUCCESS)
      return rd_queue_fail(queue);
   rd_trace_resolve(device, queue_start);
   result = vk_sync_signal_many(&device->vk, submit->signal_count, submit->signals);
   return result == VK_SUCCESS ? VK_SUCCESS : rd_queue_fail(queue);
}

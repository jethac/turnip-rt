#include "rd_trace.h"
#include "rd_private.h"

#include <inttypes.h>
#include <stdlib.h>

#include "common/fd5_hw.h"
#include "common/freedreno_common.h"
#include "drm/freedreno_ringbuffer.h"
#include "util/os_time.h"
#include <string.h>

void
rd_trace_init(struct rd_device *device)
{
   const char *path = getenv("RADISH_TRACE");
   if (!path || !*path)
      return;
   struct rd_trace *trace = calloc(1, sizeof(*trace));
   if (!trace)
      return;
   trace->file = fopen(path, "w");
   trace->bo = fd_bo_new(device->fd, RD_TRACE_SLOTS * 2 * sizeof(uint64_t), 0,
                         "radish-trace");
   trace->map = trace->bo ? fd_bo_map(trace->bo) : NULL;
   if (!trace->file || !trace->map) {
      if (trace->file)
         fclose(trace->file);
      if (trace->bo)
         fd_bo_del(trace->bo);
      free(trace);
      return;
   }
   util_dynarray_init(&trace->records, NULL);
   fprintf(trace->file, "# radish trace v1 tick_hz=19200000\n");
   device->trace = trace;
}

void
rd_trace_finish(struct rd_device *device)
{
   struct rd_trace *trace = device->trace;
   if (!trace)
      return;
   fclose(trace->file);
   fd_bo_del(trace->bo);
   util_dynarray_fini(&trace->records);
   free(trace);
   device->trace = NULL;
}

static void
emit_timestamp(struct fd_ringbuffer *ring, struct fd_bo *bo, uint32_t offset)
{
   OUT_PKT7(ring, CP_EVENT_WRITE, 4);
   OUT_RING(ring, CP_EVENT_WRITE_0_EVENT(RB_DONE_TS) | CP_EVENT_WRITE_0_TIMESTAMP);
   OUT_RELOC(ring, bo, offset, 0, 0);
   OUT_RING(ring, 0);
}

void
rd_trace_submit_begin(struct rd_device *device, struct fd_ringbuffer *ring)
{
   struct rd_trace *trace = device->trace;
   if (!trace || trace->open)
      return;
   uint32_t slot = trace->next_slot++ % RD_TRACE_SLOTS;
   trace->map[slot * 2] = 0;
   trace->map[slot * 2 + 1] = 0;
   if (!trace->current.slot_count)
      trace->current.slot_first = slot;
   trace->current.slot_count++;
   trace->pending_slots++;
   trace->open_slot = slot;
   trace->open = true;
   emit_timestamp(ring, trace->bo, slot * 16);
}

void
rd_trace_submit_end(struct rd_device *device, struct fd_ringbuffer *ring)
{
   struct rd_trace *trace = device->trace;
   if (!trace || !trace->open)
      return;
   OUT_WFI5(ring);
   emit_timestamp(ring, trace->bo, trace->open_slot * 16 + 8);
   trace->open = false;
}

bool
rd_trace_needs_resolve(struct rd_device *device)
{
   return device->trace && device->trace->pending_slots > RD_TRACE_SLOTS - 1024;
}

void
rd_trace_command_begin(struct rd_device *device)
{
   struct rd_trace *trace = device->trace;
   if (!trace)
      return;
   memset(&trace->current, 0, sizeof(trace->current));
   trace->current.host_start_ns = os_time_get_nano();
   trace->current.flush_ns = trace->flush_ns;
   trace->current.wait_ns = trace->wait_ns;
   trace->current.submits = device->submit_count;
   trace->current.chunks = trace->chunks;
}

void
rd_trace_command_end(struct rd_device *device, struct rd_trace_record *info)
{
   struct rd_trace *trace = device->trace;
   if (!trace)
      return;
   struct rd_trace_record *r = &trace->current;
   r->host_ns = os_time_get_nano() - r->host_start_ns;
   r->flush_ns = trace->flush_ns - r->flush_ns;
   r->wait_ns = trace->wait_ns - r->wait_ns;
   r->submits = device->submit_count - r->submits;
   r->seq = info->seq;
   r->type = info->type;
   r->chunks = trace->chunks - r->chunks;
   r->bytes = info->bytes;
   r->pipeline = info->pipeline;
   memcpy(r->grid, info->grid, sizeof(r->grid));
   memcpy(r->local, info->local, sizeof(r->local));
   r->instrlen = info->instrlen;
   r->max_reg = info->max_reg;
   r->max_half_reg = info->max_half_reg;
   r->barrier = info->barrier;
   r->shared_size = info->shared_size;
   util_dynarray_append_typed(&trace->records, struct rd_trace_record, *r);
}

void
rd_trace_resolve(struct rd_device *device, int64_t queue_start_ns)
{
   struct rd_trace *trace = device->trace;
   if (!trace)
      return;
   fd_bo_cpu_prep(trace->bo, device->pipe, FD_BO_PREP_READ);
   __sync_synchronize();
   fprintf(trace->file, "Q start_ns=%" PRId64 " resolve_ns=%" PRId64 " records=%zu\n",
           queue_start_ns, os_time_get_nano(),
           util_dynarray_num_elements(&trace->records, struct rd_trace_record));
   util_dynarray_foreach(&trace->records, struct rd_trace_record, r) {
      uint64_t gpu_ticks = 0, first = 0, last = 0;
      unsigned invalid = 0;
      for (uint32_t i = 0; i < r->slot_count; i++) {
         uint32_t slot = (r->slot_first + i) % RD_TRACE_SLOTS;
         uint64_t start = trace->map[slot * 2], end = trace->map[slot * 2 + 1];
         if (!start || !end || end < start) {
            invalid++;
            continue;
         }
         gpu_ticks += end - start;
         if (!first)
            first = start;
         last = end;
      }
      fprintf(trace->file,
              "C seq=%" PRIu64 " type=%u pipeline=%p grid=%u,%u,%u local=%u,%u,%u instrlen=%u regs=%u,%d barrier=%u shared=%u "
              "chunks=%" PRIu64 " submits=%" PRIu64 " bytes=%" PRIu64 " host_start=%" PRId64 " host=%" PRId64 " flush=%" PRId64 " wait=%" PRId64
              " gpu=%" PRIu64 " gpu_first=%" PRIu64 " gpu_last=%" PRIu64 " slots=%u invalid=%u\n",
              r->seq, r->type, r->pipeline, r->grid[0], r->grid[1], r->grid[2],
              r->local[0], r->local[1], r->local[2], r->instrlen, r->max_reg, (int)r->max_half_reg,
              r->barrier, r->shared_size, r->chunks, r->submits, r->bytes,
              r->host_start_ns, r->host_ns, r->flush_ns, r->wait_ns,
              fd_ticks_to_ns(gpu_ticks), first, last, r->slot_count, invalid);
   }
   util_dynarray_clear(&trace->records);
   trace->pending_slots = 0;
   fflush(trace->file);
}

/* Execution trace for profiling: RADISH_TRACE=<file> writes one line per
 * executed command with host time split into encode/flush/wait and GPU time
 * from CP timestamps around every kernel submit. */
#ifndef RD_TRACE_H
#define RD_TRACE_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include "util/u_dynarray.h"

struct rd_device;
struct fd_ringbuffer;
struct fd_bo;

#define RD_TRACE_SLOTS 16384

struct rd_trace_record {
   uint64_t seq;            /* dispatch sequence number (dispatches only) */
   int64_t host_start_ns, host_ns, flush_ns, wait_ns;
   uint64_t submits, chunks, bytes;
   uint32_t slot_first, slot_count;
   uint32_t type, grid[3], local[3], instrlen, max_reg, max_half_reg;
   uint32_t barrier, shared_size;
   const void *pipeline;
};

struct rd_trace {
   FILE *file;
   struct fd_bo *bo;
   uint64_t *map;
   uint32_t next_slot, open_slot, pending_slots;
   bool open;
   /* running host counters, sampled at command boundaries */
   int64_t flush_ns, wait_ns;
   uint64_t dispatch_seq, chunks;
   struct util_dynarray records;
   struct rd_trace_record current;
};

void rd_trace_init(struct rd_device *device);
void rd_trace_finish(struct rd_device *device);
/* Emit the start timestamp of a kernel submit; no-op when tracing is off. */
void rd_trace_submit_begin(struct rd_device *device, struct fd_ringbuffer *ring);
/* Emit the end timestamp, called from rd_flush after the completion wait. */
void rd_trace_submit_end(struct rd_device *device, struct fd_ringbuffer *ring);
bool rd_trace_needs_resolve(struct rd_device *device);
void rd_trace_command_begin(struct rd_device *device);
void rd_trace_command_end(struct rd_device *device, struct rd_trace_record *record);
/* Read back timestamps of all completed records and write them out. Only
 * call when every traced submit has retired. */
void rd_trace_resolve(struct rd_device *device, int64_t queue_start_ns);

#endif

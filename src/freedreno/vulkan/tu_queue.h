/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 * SPDX-License-Identifier: MIT
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 */

#ifndef TU_QUEUE_H
#define TU_QUEUE_H

#include "tu_common.h"

enum tu_queue_type
{
   TU_QUEUE_GFX,
   TU_QUEUE_SPARSE,
};

/* TU_LPAC_COMPUTE: submission ordering shared by an LPAC queue and its
 * hidden BR fallback queue (see msm_queue_submit). Protected by
 * device->submit_mutex.
 */
struct tu_queue_order
{
   uint32_t syncobj;            /* timeline drm syncobj */
   uint64_t point;              /* last signalled point */
   const struct tu_queue *last; /* queue that signalled it */
   uint64_t lpac_submits, gfx_submits, switches;
};

/* TU_RING_TS_LOG: per-submission ALWAYS_ON start/end stamps, logged once
 * landed. One slot per submission, reused round-robin.
 */
struct tu_ring_ts
{
   struct tu_bo *bo;
   uint32_t slots;
   uint32_t next;               /* next slot to use */
   uint32_t pending;            /* oldest slot not yet logged */
   uint64_t dropped;
};

struct tu_queue
{
   struct vk_queue vk;

   struct tu_device *device;

   enum tu_queue_type type;
   uint32_t msm_queue_id;
   uint32_t priority;

   uint32_t sparse_queue_id;

   uint32_t sparse_syncobj, gfx_syncobj;
   uint64_t sparse_timepoint, gfx_timepoint;

   unsigned render_pass_idx;

   int fence;           /* timestamp/fence of the last queue submission */

   /* TU_LPAC_COMPUTE only; all zero/NULL otherwise. */
   bool lpac;                    /* kernel submitqueue is the LPAC ring */
   struct tu_queue *gfx_fallback; /* on an LPAC queue: hidden BR queue */
   struct tu_queue_order *order; /* shared by an LPAC queue and its fallback */

   struct tu_ring_ts *ring_ts;   /* TU_RING_TS_LOG only, created at first submit */
   bool ring_ts_wanted;

   struct tu_lpac_watchdog *watchdog; /* on an LPAC queue, TU_LPAC_WATCHDOG_MS != 0 */
};
VK_DEFINE_HANDLE_CASTS(tu_queue, vk.base, VkQueue, VK_OBJECT_TYPE_QUEUE)

VkResult
tu_queue_init(struct tu_device *device,
              struct tu_queue *queue,
              enum tu_queue_type type,
              const VkQueueGlobalPriorityKHR global_priority,
              int idx,
              const VkDeviceQueueCreateInfo *create_info,
              struct tu_queue *shared_queue);

void
tu_queue_finish(struct tu_queue *queue);

#endif

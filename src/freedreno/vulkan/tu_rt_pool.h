/*
 * Copyright © 2026 Jetha Chan
 * SPDX-License-Identifier: MIT
 */

#ifndef TU_RT_POOL_H
#define TU_RT_POOL_H

#include "tu_common.h"

struct tu_rt_pool;
struct tu_rt_batch;

struct tu_rt_task {
   const uint8_t *key;
   uint64_t cost;

   const uint8_t *ckey;
   const uint8_t *env;
   uint32_t env_count;
   double ccost;
   const uint64_t *need;
   uint32_t need_count;
   const uint64_t *have;
   uint32_t have_count;
   bool coalesced;

   struct tu_shader *shader;
   VkResult result;

   int state;
   uint64_t seq;
   struct tu_rt_batch *batch;
   uint32_t index;
   void *entry;
};

struct tu_rt_batch {
   VkResult (*run)(struct tu_rt_batch *batch, uint32_t index, struct tu_shader **out);
   void *data;
   struct vk_pipeline_cache *cache;
   uint32_t count;
   struct tu_rt_task *tasks;

   uint32_t remaining;
   uint32_t running;
   uint32_t deduped;
   uint32_t coalesced;
   uint32_t retargeted;
   uint32_t helped;
   int64_t wait_ns;
};

bool tu_rt_pool_enabled(void);
int tu_rt_superset_mode(void);
uint32_t tu_rt_superset_policy(void);
void tu_rt_pool_run(struct tu_device *dev, struct tu_rt_batch *batch);
void tu_rt_pool_finish(struct tu_device *dev);

#endif

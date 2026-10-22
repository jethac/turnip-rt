/*
 * Copyright © 2026 Jetha Chan
 * SPDX-License-Identifier: MIT
 */

#include "tu_rt_pool.h"

#include <algorithm>
#include <condition_variable>
#include <malloc.h>
#include <mutex>
#include <pthread.h>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "util/mesa-blake3.h"
#include "util/os_time.h"
#include "util/u_debug.h"
#include "vk_pipeline_cache.h"

#include "tu_device.h"
#include "tu_shader.h"

enum {
   TU_RT_TASK_PENDING,
   TU_RT_TASK_RUNNING,
   TU_RT_TASK_WAITING,
   TU_RT_TASK_DONE,
};

struct tu_rt_pool_entry {
   struct tu_rt_task *owner;
   std::vector<struct tu_rt_task *> waiters;
};

struct tu_rt_pool {
   std::mutex lock;
   std::condition_variable work_cv;
   std::condition_variable done_cv;
   std::vector<struct tu_rt_task *> heap;
   std::unordered_map<std::string, struct tu_rt_pool_entry *> inflight;
   std::vector<pthread_t> threads;
   uint64_t seq;
   unsigned running;
   bool quit;
   size_t heap_budget;
   size_t heap_base;
   uint64_t throttled;
};

static bool
tu_rt_task_less(const struct tu_rt_task *a, const struct tu_rt_task *b)
{
   if (a->cost != b->cost)
      return a->cost < b->cost;
   return a->seq > b->seq;
}

bool
tu_rt_pool_enabled(void)
{
   static const bool on = debug_get_bool_option("TU_RT_POOL", false);
   return on;
}

static size_t
tu_rt_heap_used(void)
{
   struct mallinfo2 mi = mallinfo2();
   return mi.uordblks + mi.hblkhd;
}

static bool
tu_rt_pool_may_start(struct tu_rt_pool *pool)
{
   if (!pool->heap_budget)
      return true;
   if (!pool->running) {
      pool->heap_base = tu_rt_heap_used();
      return true;
   }
   if (tu_rt_heap_used() < pool->heap_base + pool->heap_budget)
      return true;
   pool->throttled++;
   return false;
}

static struct tu_rt_task *
tu_rt_pool_pop_locked(struct tu_rt_pool *pool)
{
   while (!pool->heap.empty()) {
      std::pop_heap(pool->heap.begin(), pool->heap.end(), tu_rt_task_less);
      struct tu_rt_task *t = pool->heap.back();
      pool->heap.pop_back();
      if (t->state == TU_RT_TASK_PENDING)
         return t;
   }
   return NULL;
}

static bool
tu_rt_pool_has_pending_locked(struct tu_rt_pool *pool)
{
   for (struct tu_rt_task *t : pool->heap)
      if (t->state == TU_RT_TASK_PENDING)
         return true;
   return false;
}

static void
tu_rt_pool_complete_locked(struct tu_device *dev, struct tu_rt_pool *pool,
                           struct tu_rt_task *t)
{
   t->state = TU_RT_TASK_DONE;
   t->batch->remaining--;
   struct tu_rt_pool_entry *e = (struct tu_rt_pool_entry *) t->entry;
   if (e) {
      for (struct tu_rt_task *w : e->waiters) {
         w->result = t->result;
         if (t->shader) {
            vk_pipeline_cache_object_ref(&t->shader->base);
            w->shader = t->shader;
            if (w->batch->cache != t->batch->cache) {
               struct vk_pipeline_cache_object *obj =
                  vk_pipeline_cache_add_object(w->batch->cache, &t->shader->base);
               w->shader = container_of(obj, struct tu_shader, base);
            }
         }
         w->state = TU_RT_TASK_DONE;
         w->entry = NULL;
         w->batch->remaining--;
      }
      pool->inflight.erase(std::string((const char *) t->key, BLAKE3_KEY_LEN));
      t->entry = NULL;
      delete e;
   }
   (void) dev;
   pool->done_cv.notify_all();
   pool->work_cv.notify_all();
}

static void
tu_rt_pool_execute(struct tu_device *dev, struct tu_rt_pool *pool,
                   std::unique_lock<std::mutex> &lk, struct tu_rt_task *t)
{
   t->state = TU_RT_TASK_RUNNING;
   pool->running++;
   lk.unlock();
   struct tu_shader *shader = NULL;
   VkResult r = t->batch->run(t->batch, t->index, &shader);
   lk.lock();
   pool->running--;
   t->shader = shader;
   t->result = r == VK_SUCCESS && !shader ? VK_ERROR_OUT_OF_HOST_MEMORY : r;
   tu_rt_pool_complete_locked(dev, pool, t);
}

struct tu_rt_worker_arg {
   struct tu_device *dev;
   struct tu_rt_pool *pool;
};

static void *
tu_rt_pool_worker(void *data)
{
   struct tu_rt_worker_arg *arg = (struct tu_rt_worker_arg *) data;
   struct tu_device *dev = arg->dev;
   struct tu_rt_pool *pool = arg->pool;
   delete arg;
   std::unique_lock<std::mutex> lk(pool->lock);
   for (;;) {
      pool->work_cv.wait(lk, [&] {
         return pool->quit ||
                (tu_rt_pool_has_pending_locked(pool) && tu_rt_pool_may_start(pool));
      });
      if (pool->quit)
         break;
      struct tu_rt_task *t = tu_rt_pool_pop_locked(pool);
      if (t)
         tu_rt_pool_execute(dev, pool, lk, t);
   }
   return NULL;
}

static struct tu_rt_pool *
tu_rt_pool_get(struct tu_device *dev)
{
   static std::mutex create_lock;
   std::lock_guard<std::mutex> g(create_lock);
   if (dev->rt_pool)
      return dev->rt_pool;

   struct tu_rt_pool *pool = new tu_rt_pool();
   pool->seq = 0;
   pool->running = 0;
   pool->quit = false;
   pool->heap_budget = (size_t) debug_get_num_option("TU_RT_HEAP_MB", 2048) << 20;
   pool->heap_base = 0;
   pool->throttled = 0;

   long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
   int64_t n = debug_get_num_option("TU_RT_POOL_THREADS",
                                    MAX2(MIN2(6, ncpu - 2), 1));
   n = CLAMP(n, 0, 32);
   for (int64_t i = 0; i < n; i++) {
      pthread_attr_t attr;
      pthread_attr_init(&attr);
      pthread_attr_setstacksize(&attr, 256u << 20);
      pthread_t th;
      struct tu_rt_worker_arg *arg = new tu_rt_worker_arg { dev, pool };
      if (pthread_create(&th, &attr, tu_rt_pool_worker, arg) == 0) {
         pthread_setname_np(th, "tu_rt_pool");
         pool->threads.push_back(th);
      } else {
         delete arg;
      }
      pthread_attr_destroy(&attr);
   }
   dev->rt_pool = pool;
   return pool;
}

void
tu_rt_pool_finish(struct tu_device *dev)
{
   struct tu_rt_pool *pool = dev->rt_pool;
   if (!pool)
      return;
   {
      std::lock_guard<std::mutex> g(pool->lock);
      pool->quit = true;
   }
   pool->work_cv.notify_all();
   for (pthread_t th : pool->threads)
      pthread_join(th, NULL);
   delete pool;
   dev->rt_pool = NULL;
}

static struct tu_rt_task *
tu_rt_pool_pick_own_locked(struct tu_rt_pool *pool, struct tu_rt_batch *batch)
{
   struct tu_rt_task *best = NULL;
   for (uint32_t i = 0; i < batch->count; i++) {
      struct tu_rt_task *t = batch->tasks + i;
      if (t->state == TU_RT_TASK_PENDING && (!best || tu_rt_task_less(best, t)))
         best = t;
   }
   if (best)
      return best;
   for (uint32_t i = 0; i < batch->count; i++) {
      struct tu_rt_task *t = batch->tasks + i;
      if (t->state != TU_RT_TASK_WAITING || !t->entry)
         continue;
      struct tu_rt_task *o = ((struct tu_rt_pool_entry *) t->entry)->owner;
      if (o->state == TU_RT_TASK_PENDING && (!best || tu_rt_task_less(best, o)))
         best = o;
   }
   if (best)
      batch->helped++;
   return best;
}

void
tu_rt_pool_run(struct tu_device *dev, struct tu_rt_batch *batch)
{
   struct tu_rt_pool *pool = tu_rt_pool_get(dev);
   std::unique_lock<std::mutex> lk(pool->lock);

   batch->remaining = batch->count;
   batch->deduped = 0;
   batch->helped = 0;
   batch->wait_ns = 0;
   for (uint32_t i = 0; i < batch->count; i++) {
      struct tu_rt_task *t = batch->tasks + i;
      t->batch = batch;
      t->index = i;
      t->seq = pool->seq++;
      t->entry = NULL;
      t->shader = NULL;
      t->result = VK_SUCCESS;
      if (t->key) {
         std::string k((const char *) t->key, BLAKE3_KEY_LEN);
         auto it = pool->inflight.find(k);
         if (it != pool->inflight.end()) {
            t->state = TU_RT_TASK_WAITING;
            t->entry = it->second;
            it->second->waiters.push_back(t);
            batch->deduped++;
            continue;
         }
         struct tu_rt_pool_entry *e = new tu_rt_pool_entry();
         e->owner = t;
         t->entry = e;
         pool->inflight.emplace(k, e);
      }
      t->state = TU_RT_TASK_PENDING;
      pool->heap.push_back(t);
      std::push_heap(pool->heap.begin(), pool->heap.end(), tu_rt_task_less);
   }
   pool->work_cv.notify_all();

   while (batch->remaining) {
      struct tu_rt_task *t = NULL;
      if (tu_rt_pool_may_start(pool))
         t = tu_rt_pool_pick_own_locked(pool, batch);
      if (t) {
         tu_rt_pool_execute(dev, pool, lk, t);
         continue;
      }
      int64_t w0 = os_time_get_nano();
      pool->done_cv.wait(lk);
      batch->wait_ns += os_time_get_nano() - w0;
   }

   pool->heap.erase(std::remove_if(pool->heap.begin(), pool->heap.end(),
                                   [&](struct tu_rt_task *t) { return t->batch == batch; }),
                    pool->heap.end());
   std::make_heap(pool->heap.begin(), pool->heap.end(), tu_rt_task_less);
}

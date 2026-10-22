/*
 * Copyright © 2026 Jetha Chan
 * SPDX-License-Identifier: MIT
 */

#include "tu_rt_pool.h"

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <iterator>
#include <malloc.h>
#include <mutex>
#include <pthread.h>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "util/mesa-blake3.h"
#include "util/os_time.h"
#include "util/u_atomic.h"
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

struct tu_rt_done_rec {
   std::vector<uint8_t> env;
   std::vector<uint64_t> need;
   double ccost;
   struct tu_shader *shader;
};

struct tu_rt_pool {
   std::unordered_map<std::string, std::vector<struct tu_rt_task *>> by_rg;
   std::unordered_map<std::string, std::vector<struct tu_rt_done_rec>> done_by_rg;
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
   size_t done_count;
   size_t done_pruned_at;
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

int
tu_rt_superset_mode(void)
{
   static const int mode = (int) debug_get_num_option("TU_RT_SUPERSET", 0);
   return mode;
}

static uint32_t
tu_rt_superset_ratio_pct(void)
{
   static const uint32_t pct =
      (uint32_t) debug_get_num_option("TU_RT_SUPERSET_RATIO_PCT", 125);
   return pct;
}

static double
tu_rt_superset_ratio(void)
{
   return tu_rt_superset_ratio_pct() / 100.0;
}

uint32_t
tu_rt_superset_policy(void)
{
   return tu_rt_superset_mode() >= 2 ? 0 : tu_rt_superset_ratio_pct();
}

static bool
tu_rt_sorted_subset(const uint64_t *a, uint32_t na, const uint64_t *b, uint32_t nb)
{
   return std::includes(b, b + nb, a, a + na);
}

static bool
tu_rt_need_ok(const uint64_t *need, uint32_t need_count, const struct tu_rt_task *t)
{
   return tu_rt_sorted_subset(need, need_count, t->have, t->have_count);
}

static bool
tu_rt_env_subset(const uint8_t *a, uint32_t na, const uint8_t *b, uint32_t nb)
{
   uint32_t j = 0;
   for (uint32_t i = 0; i < na; i++) {
      int c = 1;
      while (j < nb && (c = memcmp(b + j * BLAKE3_KEY_LEN, a + i * BLAKE3_KEY_LEN,
                                   BLAKE3_KEY_LEN)) < 0)
         j++;
      if (j == nb || c != 0)
         return false;
      j++;
   }
   return true;
}

static bool
tu_rt_cost_ok(double sub, double sup)
{
   if (tu_rt_superset_mode() >= 2)
      return true;
   return sup <= sub * tu_rt_superset_ratio();
}

static std::string
tu_rt_ckey(const struct tu_rt_task *t)
{
   return std::string((const char *) t->ckey, BLAKE3_KEY_LEN);
}

static void
tu_rt_by_rg_remove_locked(struct tu_rt_pool *pool, struct tu_rt_task *t)
{
   if (!t->ckey)
      return;
   auto it = pool->by_rg.find(tu_rt_ckey(t));
   if (it == pool->by_rg.end())
      return;
   auto &v = it->second;
   v.erase(std::remove(v.begin(), v.end(), t), v.end());
   if (v.empty())
      pool->by_rg.erase(it);
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

enum {
   TU_RT_SCHED_SRPT,
   TU_RT_SCHED_FAIR,
   TU_RT_SCHED_LARGEST,
};

static int
tu_rt_pool_sched_init(void)
{
   const char *v = debug_get_option("TU_RT_POOL_SCHED", NULL);
   if (v && !strcmp(v, "fair"))
      return TU_RT_SCHED_FAIR;
   if (v && !strcmp(v, "largest"))
      return TU_RT_SCHED_LARGEST;
   if (!v && debug_get_bool_option("TU_RT_POOL_FAIR", false))
      return TU_RT_SCHED_FAIR;
   return TU_RT_SCHED_SRPT;
}

static int
tu_rt_pool_sched(void)
{
   static const int sched = tu_rt_pool_sched_init();
   return sched;
}

static void
tu_rt_task_count_locked(struct tu_rt_task *t, bool on)
{
   if (on == t->counted)
      return;
   t->counted = on;
   if (on)
      t->batch->left += t->cost;
   else
      t->batch->left -= t->cost;
}

static uint64_t
tu_rt_task_rank(const struct tu_rt_task *t)
{
   uint64_t r = t->batch->left;
   const struct tu_rt_pool_entry *e = (const struct tu_rt_pool_entry *) t->entry;
   if (e)
      for (const struct tu_rt_task *w : e->waiters)
         r = MIN2(r, w->batch->left);
   return r;
}

static struct tu_rt_task *
tu_rt_pool_pop_scan_locked(struct tu_rt_pool *pool)
{
   const bool fair = tu_rt_pool_sched() == TU_RT_SCHED_FAIR;
   struct tu_rt_task *best = NULL;
   uint64_t best_rank = 0;
   for (struct tu_rt_task *t : pool->heap) {
      if (t->state != TU_RT_TASK_PENDING)
         continue;
      const uint64_t rank = fair ? t->batch->running : tu_rt_task_rank(t);
      if (!best || rank < best_rank || (rank == best_rank && tu_rt_task_less(best, t))) {
         best = t;
         best_rank = rank;
      }
   }
   return best;
}

static struct tu_rt_task *
tu_rt_pool_pop_locked(struct tu_rt_pool *pool)
{
   if (tu_rt_pool_sched() != TU_RT_SCHED_LARGEST)
      return tu_rt_pool_pop_scan_locked(pool);
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
tu_rt_pool_push_locked(struct tu_rt_pool *pool, struct tu_rt_task *t)
{
   t->state = TU_RT_TASK_PENDING;
   tu_rt_task_count_locked(t, true);
   pool->heap.push_back(t);
   std::push_heap(pool->heap.begin(), pool->heap.end(), tu_rt_task_less);
}

static void
tu_rt_pool_requeue_locked(struct tu_rt_pool *pool, struct tu_rt_task *w)
{
   w->coalesced = false;
   w->ckey = NULL;
   w->entry = NULL;
   w->batch->fallback++;
   if (!w->key) {
      tu_rt_pool_push_locked(pool, w);
      return;
   }
   std::string k((const char *) w->key, BLAKE3_KEY_LEN);
   auto it = pool->inflight.find(k);
   if (it != pool->inflight.end()) {
      w->state = TU_RT_TASK_WAITING;
      w->entry = it->second;
      it->second->waiters.push_back(w);
      return;
   }
   struct tu_rt_pool_entry *e = new tu_rt_pool_entry();
   e->owner = w;
   w->entry = e;
   pool->inflight.emplace(k, e);
   tu_rt_pool_push_locked(pool, w);
}

static void
tu_rt_pool_complete_locked(struct tu_device *dev, struct tu_rt_pool *pool,
                           struct tu_rt_task *t)
{
   t->state = TU_RT_TASK_DONE;
   t->batch->remaining--;
   tu_rt_task_count_locked(t, false);
   tu_rt_by_rg_remove_locked(pool, t);
   const bool failed = t->result != VK_SUCCESS || !t->shader;
   if (t->ckey && !failed && tu_rt_superset_mode() >= 1) {
      struct tu_rt_done_rec rec;
      rec.env.assign(t->env, t->env + t->env_count * BLAKE3_KEY_LEN);
      rec.need.assign(t->need, t->need + t->need_count);
      rec.ccost = t->ccost;
      vk_pipeline_cache_object_ref(&t->shader->base);
      rec.shader = t->shader;
      pool->done_by_rg[tu_rt_ckey(t)].push_back(std::move(rec));
      pool->done_count++;
   }
   struct tu_rt_pool_entry *e = (struct tu_rt_pool_entry *) t->entry;
   if (e) {
      std::vector<struct tu_rt_task *> requeue;
      for (struct tu_rt_task *w : e->waiters) {
         if (failed && w->coalesced) {
            requeue.push_back(w);
            continue;
         }
         w->result = t->result;
         if (t->shader) {
            vk_pipeline_cache_object_ref(&t->shader->base);
            w->shader = t->shader;
            if (!w->coalesced && w->batch->cache != t->batch->cache) {
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
      for (struct tu_rt_task *w : requeue)
         tu_rt_pool_requeue_locked(pool, w);
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
   t->batch->running++;
   lk.unlock();
   struct tu_shader *shader = NULL;
   VkResult r = t->batch->run(t->batch, t->index, &shader);
   lk.lock();
   pool->running--;
   t->batch->running--;
   t->ran = true;
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
   pool->done_count = 0;
   pool->done_pruned_at = 0;

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
   for (auto &kv : pool->done_by_rg)
      for (struct tu_rt_done_rec &rec : kv.second)
         vk_pipeline_cache_object_unref(&dev->vk, &rec.shader->base);
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

static bool
tu_rt_pool_coalesce_locked(struct tu_rt_pool *pool, struct tu_rt_task *t)
{
   const std::string ck = tu_rt_ckey(t);
   auto dit = pool->done_by_rg.find(ck);
   if (dit != pool->done_by_rg.end()) {
      struct tu_rt_done_rec *best = NULL;
      for (struct tu_rt_done_rec &rec : dit->second) {
         if (!tu_rt_need_ok(rec.need.data(), rec.need.size(), t) ||
             !tu_rt_cost_ok(t->ccost, rec.ccost) ||
             !tu_rt_env_subset(t->env, t->env_count, rec.env.data(),
                               rec.env.size() / BLAKE3_KEY_LEN))
            continue;
         if (!best || rec.ccost < best->ccost)
            best = &rec;
      }
      if (best) {
         vk_pipeline_cache_object_ref(&best->shader->base);
         t->shader = best->shader;
         t->result = VK_SUCCESS;
         t->coalesced = true;
         t->state = TU_RT_TASK_DONE;
         t->batch->remaining--;
         t->batch->coalesced++;
         return true;
      }
   }
   auto it = pool->by_rg.find(ck);
   if (it != pool->by_rg.end()) {
      struct tu_rt_task *best = NULL;
      for (struct tu_rt_task *s : it->second) {
         if (s == t || (s->state != TU_RT_TASK_PENDING && s->state != TU_RT_TASK_RUNNING) ||
             !s->entry || !tu_rt_cost_ok(t->ccost, s->ccost) ||
             !tu_rt_need_ok(s->need, s->need_count, t) ||
             !tu_rt_env_subset(t->env, t->env_count, s->env, s->env_count))
            continue;
         if (!best || s->ccost < best->ccost)
            best = s;
      }
      if (best) {
         t->state = TU_RT_TASK_WAITING;
         t->coalesced = true;
         t->entry = best->entry;
         ((struct tu_rt_pool_entry *) best->entry)->waiters.push_back(t);
         t->batch->coalesced++;
         return true;
      }
   }
   return false;
}

static void
tu_rt_pool_retarget_locked(struct tu_rt_pool *pool, struct tu_rt_task *sup)
{
   auto it = pool->by_rg.find(tu_rt_ckey(sup));
   if (it == pool->by_rg.end())
      return;
   std::vector<struct tu_rt_task *> moved;
   for (struct tu_rt_task *u : it->second) {
      if (u == sup || u->state != TU_RT_TASK_PENDING || !u->entry ||
          !tu_rt_need_ok(sup->need, sup->need_count, u) ||
          !tu_rt_env_subset(u->env, u->env_count, sup->env, sup->env_count))
         continue;
      struct tu_rt_pool_entry *ue = (struct tu_rt_pool_entry *) u->entry;
      bool ok = tu_rt_cost_ok(u->ccost, sup->ccost);
      for (struct tu_rt_task *w : ue->waiters)
         ok = ok && tu_rt_need_ok(sup->need, sup->need_count, w) &&
              (w->ccost <= 0 || tu_rt_cost_ok(w->ccost, sup->ccost));
      if (!ok)
         continue;
      moved.push_back(u);
   }
   if (moved.empty())
      return;
   struct tu_rt_pool_entry *se = (struct tu_rt_pool_entry *) sup->entry;
   for (struct tu_rt_task *u : moved) {
      struct tu_rt_pool_entry *ue = (struct tu_rt_pool_entry *) u->entry;
      for (struct tu_rt_task *w : ue->waiters) {
         w->entry = se;
         w->coalesced = true;
         se->waiters.push_back(w);
      }
      pool->inflight.erase(std::string((const char *) u->key, BLAKE3_KEY_LEN));
      delete ue;
      u->entry = se;
      u->state = TU_RT_TASK_WAITING;
      tu_rt_task_count_locked(u, false);
      u->coalesced = true;
      se->waiters.push_back(u);
      u->batch->retargeted++;
   }
   auto &v = it->second;
   v.erase(std::remove_if(v.begin(), v.end(), [&](struct tu_rt_task *x) {
              return std::find(moved.begin(), moved.end(), x) != moved.end();
           }), v.end());
}

static void
tu_rt_pool_prune_locked(struct tu_device *dev, struct tu_rt_pool *pool)
{
   if (pool->done_count < pool->done_pruned_at + 64)
      return;
   for (auto it = pool->done_by_rg.begin(); it != pool->done_by_rg.end();) {
      auto &v = it->second;
      v.erase(std::remove_if(v.begin(), v.end(), [&](struct tu_rt_done_rec &rec) {
                 if (p_atomic_read(&rec.shader->base.ref_cnt) > 1)
                    return false;
                 vk_pipeline_cache_object_unref(&dev->vk, &rec.shader->base);
                 pool->done_count--;
                 return true;
              }), v.end());
      it = v.empty() ? pool->done_by_rg.erase(it) : std::next(it);
   }
   pool->done_pruned_at = pool->done_count;
}

void
tu_rt_pool_run(struct tu_device *dev, struct tu_rt_batch *batch)
{
   struct tu_rt_pool *pool = tu_rt_pool_get(dev);
   std::unique_lock<std::mutex> lk(pool->lock);
   tu_rt_pool_prune_locked(dev, pool);

   batch->remaining = batch->count;
   batch->running = 0;
   batch->deduped = 0;
   batch->coalesced = 0;
   batch->retargeted = 0;
   batch->helped = 0;
   batch->fallback = 0;
   batch->left = 0;
   batch->wait_ns = 0;
   for (uint32_t i = 0; i < batch->count; i++) {
      struct tu_rt_task *t = batch->tasks + i;
      t->batch = batch;
      t->index = i;
      t->seq = pool->seq++;
      t->entry = NULL;
      t->shader = NULL;
      t->result = VK_SUCCESS;
      t->coalesced = false;
      t->ran = false;
      t->counted = false;
      if (!t->key || tu_rt_superset_mode() <= 0)
         t->ckey = NULL;
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
         if (t->ckey && tu_rt_pool_coalesce_locked(pool, t))
            continue;
         struct tu_rt_pool_entry *e = new tu_rt_pool_entry();
         e->owner = t;
         t->entry = e;
         pool->inflight.emplace(k, e);
         if (t->ckey) {
            tu_rt_pool_retarget_locked(pool, t);
            pool->by_rg[tu_rt_ckey(t)].push_back(t);
         }
      }
      tu_rt_pool_push_locked(pool, t);
   }
   pool->work_cv.notify_all();

   while (batch->remaining) {
      struct tu_rt_task *t = NULL;
      if (!batch->running || tu_rt_pool_may_start(pool))
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

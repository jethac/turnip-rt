/*
 * Copyright © 2026 Jetha Chan
 * SPDX-License-Identifier: MIT
 */

#include "tu_free_log.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <string>
#include <unordered_map>
#include <unordered_set>

#ifdef HAVE_DLADDR
#include <dlfcn.h>
#endif
#ifdef __GLIBC__
#include <execinfo.h>
#endif

#include "c11/threads.h"
#include "util/log.h"
#include "util/u_debug.h"

#define TU_FREE_LOG_BUF_SIZE (256 * 1024)
#define TU_FREE_LOG_LINE_SIZE 1536
#define TU_FREE_LOG_BT_DEPTH 14

bool tu_free_log_on;
bool tu_rt_label_on;

struct tu_free_log_as {
   uint64_t va;
   uint64_t size;
};

static once_flag fl_once = ONCE_FLAG_INIT;
static mtx_t fl_life;
static mtx_t fl_lock;
static cnd_t fl_cond;
static thrd_t fl_thread;
static bool fl_thread_running;
static bool fl_stop;
static const char *fl_prefix;
static int fl_fd = -1;
static unsigned fl_devices;
static char *fl_buf;
static size_t fl_len;
static std::unordered_set<std::string> *fl_names;
static std::unordered_map<uintptr_t, struct tu_free_log_as> *fl_as;
static std::unordered_map<uint32_t, std::string> *fl_rt_names;
static std::atomic<uint32_t> fl_rt_seq;

static void
fl_flush_locked(void)
{
   size_t off = 0;
   while (fl_fd >= 0 && off < fl_len) {
      ssize_t n = write(fl_fd, fl_buf + off, fl_len - off);
      if (n < 0) {
         if (errno == EINTR)
            continue;
         break;
      }
      off += n;
   }
   fl_len = 0;
}

static void
fl_append(const char *body, int len)
{
   struct timespec ts;
   char head[64];

   mtx_lock(&fl_lock);
   clock_gettime(CLOCK_REALTIME, &ts);
   int hlen = snprintf(head, sizeof(head), "%" PRIu64 " %ld ",
                       (uint64_t) ts.tv_sec * 1000000000 + (uint64_t) ts.tv_nsec,
                       (long) syscall(SYS_gettid));
   if (fl_len + hlen + len + 1 > TU_FREE_LOG_BUF_SIZE)
      fl_flush_locked();
   memcpy(fl_buf + fl_len, head, hlen);
   fl_len += hlen;
   memcpy(fl_buf + fl_len, body, len);
   fl_len += len;
   fl_buf[fl_len++] = '\n';
   mtx_unlock(&fl_lock);
}

static int
fl_vprintf(char *line, int size, int len, const char *fmt, va_list ap)
{
   if (len >= size - 1)
      return len;
   int n = vsnprintf(line + len, size - len, fmt, ap);
   if (n < 0)
      return len;
   return MIN2(len + n, size - 1);
}

static int
fl_printf(char *line, int size, int len, const char *fmt, ...)
{
   va_list ap;
   va_start(ap, fmt);
   len = fl_vprintf(line, size, len, fmt, ap);
   va_end(ap);
   return len;
}

static void
fl_line(const char *fmt, ...)
{
   char line[TU_FREE_LOG_LINE_SIZE];
   va_list ap;
   va_start(ap, fmt);
   int len = fl_vprintf(line, sizeof(line), 0, fmt, ap);
   va_end(ap);
   fl_append(line, len);
}

static int
fl_backtrace(char *line, int size, int len)
{
#ifdef __GLIBC__
   void *frames[TU_FREE_LOG_BT_DEPTH];
   int n = backtrace(frames, ARRAY_SIZE(frames));
   len = fl_printf(line, size, len, " bt=");
   for (int i = 1; i < n; i++) {
      uintptr_t pc = (uintptr_t) frames[i];
      len = fl_printf(line, size, len, "%s0x%" PRIxPTR, i > 1 ? "," : "", pc);
#ifdef HAVE_DLADDR
      Dl_info info;
      if (dladdr(frames[i], &info) && info.dli_fname) {
         const char *base = strrchr(info.dli_fname, '/');
         len = fl_printf(line, size, len, ":%s+0x%" PRIxPTR,
                         base ? base + 1 : info.dli_fname,
                         pc - (uintptr_t) info.dli_fbase);
      }
#endif
   }
#endif
   return len;
}

static int
fl_thread_main(void *)
{
   mtx_lock(&fl_lock);
   while (!fl_stop) {
      struct timespec ts;
      timespec_get(&ts, TIME_UTC);
      ts.tv_sec += 1;
      cnd_timedwait(&fl_cond, &fl_lock, &ts);
      fl_flush_locked();
   }
   mtx_unlock(&fl_lock);
   return 0;
}

static void
fl_init_once(void)
{
   const char *prefix = debug_get_option("TU_FREE_LOG", NULL);
   const char *rt_label = debug_get_option("TU_RT_LABEL", NULL);
   bool free_log = prefix && *prefix;
   bool label = rt_label && *rt_label;
   if (!free_log && !label)
      return;
   if (!free_log)
      prefix = rt_label;

   fl_prefix = strdup(prefix);
   fl_buf = static_cast<char *>(malloc(TU_FREE_LOG_BUF_SIZE));
   if (!fl_prefix || !fl_buf)
      return;

   fl_names = new std::unordered_set<std::string>;
   fl_as = new std::unordered_map<uintptr_t, struct tu_free_log_as>;
   fl_rt_names = new std::unordered_map<uint32_t, std::string>;
   mtx_init(&fl_life, mtx_plain);
   mtx_init(&fl_lock, mtx_plain);
   cnd_init(&fl_cond);

#ifdef __GLIBC__
   void *warm[2];
   backtrace(warm, ARRAY_SIZE(warm));
#endif

   tu_free_log_on = free_log;
   tu_rt_label_on = label;
}

void
tu_free_log_device_init(void)
{
   call_once(&fl_once, fl_init_once);
   if (!tu_free_log_on && !tu_rt_label_on)
      return;

   const char *so = "?";
   uintptr_t so_base = 0;
#ifdef HAVE_DLADDR
   Dl_info info;
   if (dladdr(reinterpret_cast<void *>(tu_free_log_device_init), &info) && info.dli_fname) {
      so = info.dli_fname;
      so_base = (uintptr_t) info.dli_fbase;
   }
#endif

   mtx_lock(&fl_life);
   mtx_lock(&fl_lock);
   bool first = fl_devices++ == 0;
   if (first) {
      char path[4096];
      snprintf(path, sizeof(path), "%s.%d.log", fl_prefix, (int) getpid());
      fl_fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
      if (fl_fd < 0)
         mesa_loge("TU_FREE_LOG: cannot open %s: %s", path, strerror(errno));
      fl_stop = false;
   }
   unsigned devices = fl_devices;
   mtx_unlock(&fl_lock);

   if (first)
      fl_thread_running =
         thrd_create(&fl_thread, fl_thread_main, NULL) == thrd_success;

   fl_line("open pid=%d devices=%u so=%s so_base=0x%" PRIxPTR, (int) getpid(),
           devices, so, so_base);
   mtx_unlock(&fl_life);
}

void
tu_free_log_device_finish(void)
{
   if (!tu_free_log_on && !tu_rt_label_on)
      return;

   mtx_lock(&fl_life);
   mtx_lock(&fl_lock);
   unsigned devices = --fl_devices;
   mtx_unlock(&fl_lock);

   fl_line("close devices=%u", devices);

   mtx_lock(&fl_lock);
   if (devices) {
      fl_flush_locked();
      mtx_unlock(&fl_lock);
      mtx_unlock(&fl_life);
      return;
   }
   fl_stop = true;
   cnd_signal(&fl_cond);
   mtx_unlock(&fl_lock);

   if (fl_thread_running) {
      thrd_join(fl_thread, NULL);
      fl_thread_running = false;
   }

   mtx_lock(&fl_lock);
   fl_flush_locked();
   if (fl_fd >= 0)
      close(fl_fd);
   fl_fd = -1;
   mtx_unlock(&fl_lock);
   mtx_unlock(&fl_life);
}

static const char *
fl_intern(const char *name)
{
   std::string s(name ? name : "?");
   for (char &c : s) {
      if (c == ' ' || c == '\t' || c == '\n')
         c = '_';
   }
   mtx_lock(&fl_lock);
   const char *ret = fl_names->insert(s).first->c_str();
   mtx_unlock(&fl_lock);
   return ret;
}

void
tu_free_log_bo_alloc(struct tu_bo *bo, uint32_t flags, const char *name)
{
   bo->free_log_name = fl_intern(name);
   bo->free_log_flags = flags;
   fl_line("bo_alloc iova=0x%" PRIx64 " size=0x%" PRIx64
           " end=0x%" PRIx64 " gem=%u name=%s flags=0x%x lazy=%u",
           bo->iova, bo->size, bo->iova + bo->size, bo->gem_handle,
           bo->free_log_name, flags, (unsigned) bo->lazy);
}

void
tu_free_log_bo_free(const struct tu_bo *bo)
{
   char line[TU_FREE_LOG_LINE_SIZE];
   int len = fl_printf(line, sizeof(line), 0,
                       "bo_free iova=0x%" PRIx64 " size=0x%" PRIx64
                       " end=0x%" PRIx64 " gem=%u name=%s flags=0x%x lazy=%u",
                       bo->iova, bo->size, bo->iova + bo->size,
                       bo->gem_handle,
                       bo->free_log_name ? bo->free_log_name : "?",
                       bo->free_log_flags, (unsigned) bo->lazy);
   if (bo->free_log_flags & TU_BO_ALLOC_INTERNAL_RESOURCE)
      len = fl_backtrace(line, sizeof(line), len);
   fl_append(line, len);
}

void
tu_free_log_vma(const char *op, uint64_t iova, uint64_t size, uint32_t flags)
{
   fl_line("%s iova=0x%" PRIx64 " size=0x%" PRIx64 " end=0x%" PRIx64
           " flags=0x%x",
           op, iova, size, iova + size, flags);
}

void
tu_free_log_free_memory(const void *mem, uint64_t iova, uint64_t size,
                        const struct tu_bo *bo, bool lazy)
{
   fl_line("free_memory mem=%p iova=0x%" PRIx64 " size=0x%" PRIx64
           " end=0x%" PRIx64 " bo_iova=0x%" PRIx64 " bo_size=0x%" PRIx64
           " gem=%u lazy=%u",
           mem, iova, size, iova + size, bo ? bo->iova : 0,
           bo ? bo->size : 0, bo ? bo->gem_handle : 0, (unsigned) lazy);
}

void
tu_free_log_destroy_buffer(const void *buffer, uint64_t va, uint64_t size,
                           bool sparse)
{
   fl_line("destroy_buffer buf=%p va=0x%" PRIx64 " size=0x%" PRIx64
           " end=0x%" PRIx64 " sparse=%u",
           buffer, va, size, va + size, (unsigned) sparse);
}

void
tu_free_log_create_as(const void *as, uint64_t va, uint64_t size,
                      uint32_t type)
{
   mtx_lock(&fl_lock);
   (*fl_as)[(uintptr_t) as] = tu_free_log_as { va, size };
   mtx_unlock(&fl_lock);
   fl_line("create_as as=%p va=0x%" PRIx64 " size=0x%" PRIx64
           " end=0x%" PRIx64 " type=%u",
           as, va, size, va + size, type);
}

void
tu_free_log_destroy_as(const void *as, uint64_t addr, uint64_t size)
{
   uint64_t va = addr;
   bool known = false;

   mtx_lock(&fl_lock);
   auto it = fl_as->find((uintptr_t) as);
   if (it != fl_as->end()) {
      va = it->second.va;
      size = it->second.size;
      known = true;
      fl_as->erase(it);
   }
   mtx_unlock(&fl_lock);

   fl_line("destroy_as as=%p va=0x%" PRIx64 " size=0x%" PRIx64
           " end=0x%" PRIx64 " known=%u",
           as, va, size, va + size, (unsigned) known);
}

static std::string
fl_spirv_name(const uint32_t *spirv, size_t size)
{
   size_t n = size / sizeof(uint32_t);
   std::string first;
   if (n < 5 || spirv[0] != 0x07230203)
      return first;
   for (size_t i = 5; i < n;) {
      uint32_t op = spirv[i] & 0xffff, len = spirv[i] >> 16;
      if (!len || i + len > n)
         break;
      if (op == 7 && len > 2) {
         const char *str = reinterpret_cast<const char *>(spirv + i + 2);
         std::string val(str, strnlen(str, (len - 2) * sizeof(uint32_t)));
         if (val.size() > 5 && val.compare(val.size() - 5, 5, ".dxil") == 0)
            return val;
         if (first.empty())
            first = val;
      } else if (op == 54) {
         break;
      }
      i += len;
   }
   return first;
}

void
tu_free_log_rt_stage(uint32_t handle, const char *stage, const uint32_t *spirv,
                     size_t size, const char *entry)
{
   std::string name = spirv ? fl_spirv_name(spirv, size) : std::string();
   if (name.empty())
      name = entry ? entry : "?";
   for (char &c : name) {
      if (c == ' ' || c == '\t' || c == '\n')
         c = '_';
   }
   mtx_lock(&fl_lock);
   (*fl_rt_names)[handle] = name;
   mtx_unlock(&fl_lock);
   fl_line("rt_stage handle=0x%x stage=%s name=%s entry=%s", handle, stage,
           name.c_str(), entry ? entry : "?");
}

void
tu_free_log_rt_pipeline(const void *pipeline, bool library,
                        uint32_t stage_count, uint32_t group_count,
                        uint32_t variant_count)
{
   fl_line("rt_pipeline pipe=%p kind=%s stages=%u groups=%u variants=%u",
           pipeline, library ? "library" : "final", stage_count, group_count,
           variant_count);
}

void
tu_free_log_rt_variant(const void *pipeline, uint32_t index, uint32_t count,
                       uint32_t handle, const char *blake3)
{
   std::string name = "?";
   mtx_lock(&fl_lock);
   auto it = fl_rt_names->find(handle);
   if (it != fl_rt_names->end())
      name = it->second;
   mtx_unlock(&fl_lock);
   fl_line("rt_variant pipe=%p v=%u/%u handle=0x%x blake3=%s name=%s",
           pipeline, index, count, handle, blake3 ? blake3 : "?",
           name.c_str());
}

uint32_t
tu_free_log_rt_dispatch(const void *cmd, const void *pipeline,
                        uint32_t variant_count, const uint32_t *dims,
                        uint64_t cmd_va, uint64_t raygen_va)
{
   uint32_t seq = fl_rt_seq.fetch_add(1, std::memory_order_relaxed);
   fl_line("rt_dispatch seq=%u cmd=%p pipe=%p variants=%u kind=%s "
           "dims=%ux%ux%u cmd_va=0x%" PRIx64 " raygen_va=0x%" PRIx64,
           seq, cmd, pipeline, variant_count,
           dims ? "direct" : (raygen_va ? "indirect" : "indirect2"),
           dims ? dims[0] : 0, dims ? dims[1] : 0, dims ? dims[2] : 0,
           cmd_va, raygen_va);
   return seq;
}

/*
 * Copyright © 2024 Igalia S.L.
 * SPDX-License-Identifier: MIT
 */

#include "freedreno_rd_output.h"

#include <assert.h>
#include <ctype.h>
#include <fcntl.h>
#include <inttypes.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <limits.h>
#include <unistd.h>

#include "c11/threads.h"
#include "util/detect_os.h"
#include "util/log.h"
#include "util/u_atomic.h"
#include "util/u_debug.h"

static const struct debug_control fd_rd_dump_options[] = {
   { "enable", FD_RD_DUMP_ENABLE },
   { "combine", FD_RD_DUMP_COMBINE },
   { "full", FD_RD_DUMP_FULL },
   { "trigger", FD_RD_DUMP_TRIGGER },
   { NULL, 0 }
};

struct fd_rd_dump_env fd_rd_dump_env;

struct fd_rd_dump_range {
   unsigned range_begin;
   unsigned range_end;
};

static void
fd_rd_dump_env_init_once(void)
{
   fd_rd_dump_env.flags = parse_debug_string(os_get_option("FD_RD_DUMP"),
                                             fd_rd_dump_options);

   /* If any of the more-detailed FD_RD_DUMP flags is enabled, the general
    * FD_RD_DUMP_ENABLE flag should also implicitly be set.
    */
   if (fd_rd_dump_env.flags & ~FD_RD_DUMP_ENABLE)
      fd_rd_dump_env.flags |= FD_RD_DUMP_ENABLE;

   const char *output_path_value = os_get_option("FD_RD_DUMP_PATH");
   if (!output_path_value) {
      output_path_value =
#if DETECT_OS_ANDROID
         "/data/local/tmp";
#else
         "/tmp";
#endif
   }
   snprintf(fd_rd_dump_env.output_path, sizeof(fd_rd_dump_env.output_path),
      "%s", output_path_value);
}

void
fd_rd_dump_env_init(void)
{
   static once_flag once = ONCE_FLAG_INIT;
   call_once(&once, fd_rd_dump_env_init_once);
}

static void
fd_rd_output_sanitize_name(char *name)
{
   /* The name string is null-terminated after being constructed via asprintf.
    * Sanitize it by reducing to an underscore anything that's not a hyphen,
    * underscore, dot or alphanumeric character.
    */
   for (char *s = name; *s; ++s) {
      if (isalnum(*s) || *s == '-' || *s == '_' || *s == '.')
         continue;
      *s = '_';
   }
}

static void
fd_rd_parse_dump_range(const char *option_name, struct util_dynarray *range_array)
{
   *range_array = UTIL_DYNARRAY_INIT;
   const char *range_value = os_get_option(option_name);
   if (!range_value)
      return;

   const char *p = range_value;
   while (p && *p) {
      char *ep = NULL;

      struct fd_rd_dump_range range;
      range.range_begin = strtol(p, &ep, 0);
      if (ep == p)
         break;
      p = ep;

      range.range_end = range.range_begin;
      if (*p == '-') {
         ep = NULL;
         range.range_end = strtol(++p, &ep, 0);
         if (ep == p)
            break;
         p = ep;
      }

      util_dynarray_append(range_array, range);

      if (*p == ',')
         ++p;
      if (!isdigit(*p) && !!*p)
         break;
   }

   if (*p == '\0') {
      mesa_logi("[fd_rd_output] %s specified %" PRIuPTR " dump ranges:",
                option_name, util_dynarray_num_elements(range_array, struct fd_rd_dump_range));
      util_dynarray_foreach(range_array, struct fd_rd_dump_range, range) {
         mesa_logi("[fd_rd_output]   [%u, %u]", range->range_begin, range->range_end);
      }
   } else {
      mesa_logi("[fd_rd_output] failed to parse dump range '%s' for %s",
                range_value, option_name);

      util_dynarray_clear(range_array);
      struct fd_rd_dump_range invalid_range = {
         .range_begin = UINT_MAX,
         .range_end = UINT_MAX,
      };
      util_dynarray_append(range_array, invalid_range);
   }
}

static bool
fd_rd_output_allowed(struct fd_rd_output *output, uint32_t frame, uint32_t submit)
{
   /* Allow output if no ranges were specified. */
   if (!util_dynarray_num_elements(&output->frame_ranges, struct fd_rd_dump_range) &&
       !util_dynarray_num_elements(&output->submit_ranges, struct fd_rd_dump_range))
      return true;

   util_dynarray_foreach(&output->frame_ranges, struct fd_rd_dump_range, range) {
      if (frame >= range->range_begin && frame <= range->range_end)
         return true;
   }

   util_dynarray_foreach(&output->submit_ranges, struct fd_rd_dump_range, range) {
      if (submit >= range->range_begin && submit <= range->range_end)
         return true;
   }

   return false;
}

static void
fd_rd_output_init_mode(struct fd_rd_output *output, const char* output_name, bool bounded)
{
   memset(output, 0, sizeof(*output));
   output->bounded = bounded;
   output->trigger_fd = -1;
   const char *test_name = os_get_option("FD_RD_DUMP_TESTNAME");
   ASSERTED int name_len;
   char *initial_name = NULL;
   if (test_name)
      name_len = asprintf(&initial_name, "%s_%s", test_name, output_name);
   else
      name_len = asprintf(&initial_name, "%s", output_name);
   if (name_len >= 0)
      output->name = initial_name;
   if (bounded && name_len < 0) {
      fd_rd_output_error(output, "namespace_name_allocation");
      return;
   }
   assert(name_len != -1);
   fd_rd_output_sanitize_name(output->name);

   if (bounded) {
      char marker[PATH_MAX];
      int length = snprintf(marker, sizeof(marker), "%s/%s_epochXXXXXX",
                            fd_rd_dump_env.output_path, output->name);
      if (length < 0 || (size_t)length >= sizeof(marker) - 64) {
         fd_rd_output_error(output, "namespace_path_length");
         return;
      }
      int fd = mkstemp(marker);
      if (fd < 0) {
         fd_rd_output_error(output, "namespace_reservation");
         return;
      }
      /* Retain the zero-byte reservation across device and DSO lifetimes. */
      if (close(fd)) {
         fd_rd_output_error(output, "namespace_close");
         return;
      }
      char *name = NULL;
      if (asprintf(&name, "%s", strrchr(marker, '/') + 1) < 0) {
         fd_rd_output_error(output, "namespace_name_allocation");
         return;
      }
      free(output->name);
      output->name = name;
   }

   output->combine = false;
   output->file = NULL;
   output->trigger_fd = -1;
   output->trigger_count = 0;

   if (bounded && (!FD_RD_DUMP(TRIGGER) || FD_RD_DUMP(FULL) || FD_RD_DUMP(COMBINE)))
      fd_rd_output_error(output, "requires_trigger_without_full_or_combine");

   if (!output->failed && FD_RD_DUMP(COMBINE)) {
      output->combine = true;

      char file_path[PATH_MAX];
      snprintf(file_path, sizeof(file_path), "%s/%s_combined.rd.gz",
               fd_rd_dump_env.output_path, output->name);
      output->file = gzopen(file_path, "w");
   }

   if (!output->failed && FD_RD_DUMP(TRIGGER)) {
      char file_path[PATH_MAX];
      snprintf(file_path, sizeof(file_path), "%s/%s_trigger",
               fd_rd_dump_env.output_path, output->name);
      output->trigger_fd = open(file_path, O_RDWR | O_CREAT | (bounded ? O_EXCL : O_TRUNC), 0600);
      if (output->trigger_fd < 0 && bounded)
         fd_rd_output_error(output, "trigger_open");
   }

   fd_rd_parse_dump_range("FD_RD_DUMP_FRAMES", &output->frame_ranges);
   fd_rd_parse_dump_range("FD_RD_DUMP_SUBMITS", &output->submit_ranges);
}

void
fd_rd_output_init(struct fd_rd_output *output, const char *name)
{
   fd_rd_output_init_mode(output, name, false);
}

void
fd_rd_output_init_bounded(struct fd_rd_output *output, const char *name)
{
   fd_rd_output_init_mode(output, name, true);
}

void
fd_rd_output_error(struct fd_rd_output *output, const char *reason)
{
   output->failed = true;
   mesa_loge("RD_CAPTURE_ERROR name=%s reason=%s", output->name ? output->name : "-", reason);
}

void
fd_rd_output_fini(struct fd_rd_output *output)
{
   if (output->file != NULL) {
      assert(output->combine);
      gzclose(output->file);
   }

   if (output->trigger_fd >= 0) {
      close(output->trigger_fd);

      /* Remove the trigger file. The filename is reconstructed here
       * instead of having to spend memory to store it in the struct.
       */
      char file_path[PATH_MAX];
      snprintf(file_path, sizeof(file_path), "%s/%s_trigger",
               fd_rd_dump_env.output_path, output->name);
      unlink(file_path);
   }

   free(output->name);
   util_dynarray_fini(&output->frame_ranges);
   util_dynarray_fini(&output->submit_ranges);
}

static void
fd_rd_output_update_trigger_count(struct fd_rd_output *output)
{
   assert(FD_RD_DUMP(TRIGGER));

   /* Retrieve the trigger file size, only attempt to update the trigger
    * value if anything was actually written to that file.
    */
   struct stat stat;
   if (fstat(output->trigger_fd, &stat) != 0) {
      mesa_loge("[fd_rd_output] failed to acccess the %s trigger file",
                output->name);
      if (output->bounded) fd_rd_output_error(output, "trigger_io");
      return;
   }

   if (stat.st_size == 0)
      return;

   if (output->bounded && (stat.st_size < 0 || stat.st_size >= 32)) {
      fd_rd_output_error(output, "trigger_size");
      return;
   }
   char trigger_data[32];
   int ret = read(output->trigger_fd, trigger_data, sizeof(trigger_data));
   if (ret < 0 || (output->bounded && ret != stat.st_size)) {
      mesa_loge("[fd_rd_output] failed to read from the %s trigger file",
                output->name);
      if (output->bounded) fd_rd_output_error(output, "trigger_io");
      return;
   }
   int num_read = MIN2(ret, sizeof(trigger_data) - 1);

   /* After reading from it, the trigger file should be reset, which means
    * moving the file offset to the start of the file as well as truncating
    * it to zero bytes.
    */
   if (lseek(output->trigger_fd, 0, SEEK_SET) < 0) {
      mesa_loge("[fd_rd_output] failed to reset the %s trigger file position",
                output->name);
      if (output->bounded) fd_rd_output_error(output, "trigger_io");
      return;
   }

   if (ftruncate(output->trigger_fd, 0) < 0) {
      mesa_loge("[fd_rd_output] failed to truncate the %s trigger file",
                output->name);
      if (output->bounded) fd_rd_output_error(output, "trigger_io");
      return;
   }

   /* Try to decode the count value through strtol. -1 translates to UINT_MAX
    * and keeps generating dumps until disabled. Any positive value will
    * allow generating dumps for that many submits. Any other value will
    * disable any further generation of RD dumps.
    */
   trigger_data[num_read] = '\0';
   char *end = NULL;
   errno = 0;
   long value = strtol(trigger_data, &end, 0);
   while (end && isspace((unsigned char)*end)) ++end;
   if (output->bounded && (errno || end == trigger_data || !end || *end ||
                           value < 0 || value > 8)) {
      fd_rd_output_error(output, "trigger_must_be_0_to_8");
      return;
   }

   if (value == -1) {
      output->trigger_count = UINT_MAX;
      mesa_logi("[fd_rd_output] %s trigger enabling RD dumps until disabled",
                output->name);
   } else if (value > 0) {
      output->trigger_count = (uint32_t) value;
      mesa_logi("[fd_rd_output] %s trigger enabling RD dumps for next %u submissions",
                output->name, output->trigger_count);
   } else {
      output->trigger_count = 0;
      mesa_logi("[fd_rd_output] %s trigger disabling RD dumps", output->name);
   }
}

bool
fd_rd_output_select(struct fd_rd_output *output, uint32_t frame, uint32_t submit)
{
   assert(output->combine ^ (output->file == NULL));

   if (output->failed)
      return false;
   if (FD_RD_DUMP(TRIGGER)) {
      fd_rd_output_update_trigger_count(output);
      if (output->failed)
         return false;

      if (output->trigger_count == 0)
         return false;
      /* UINT_MAX corresponds to generating dumps until disabled. */
      if (output->trigger_count != UINT_MAX)
          --output->trigger_count;
   }

   if (!fd_rd_output_allowed(output, frame, submit))
      return false;

   return true;
}

bool
fd_rd_output_begin(struct fd_rd_output *output, uint32_t frame, uint32_t submit)
{
   return fd_rd_output_select(output, frame, submit) &&
          fd_rd_output_begin_selected(output, frame, submit, 0);
}

bool
fd_rd_output_begin_selected(struct fd_rd_output *output, uint32_t frame,
                            uint32_t submit, uint64_t bytes)
{
   if (output->failed)
      return false;
   if (output->bounded) {
      const uint64_t gib = UINT64_C(1024) * 1024 * 1024;
      /* zlib compressBound plus ample gzip framing margin. */
      uint64_t bound = bytes + (bytes >> 12) + (bytes >> 14) + (bytes >> 25) + 1024;
      struct statvfs st;
      if (bytes > gib || output->captures >= 8 ||
          bound > 8 * gib - output->reserved_bytes ||
          statvfs(fd_rd_dump_env.output_path, &st) || !st.f_frsize ||
          st.f_bavail > UINT64_MAX / st.f_frsize) {
         fd_rd_output_error(output, "capture_budget_or_statvfs");
         return false;
      }
      uint64_t free_bytes = (uint64_t)st.f_bavail * st.f_frsize;
      if (free_bytes < bound + 2 * gib || (!output->captures && free_bytes < 12 * gib)) {
         fd_rd_output_error(output, "free_reserve");
         return false;
      }
      output->expected_bytes = bytes;
      output->written_bytes = 0;
      output->reserved_bytes += bound;
      ++output->captures;
   }
   if (output->combine)
      return true;

   char file_path[PATH_MAX];
   if (frame != UINT_MAX) {
      snprintf(file_path, sizeof(file_path), "%s/%s_frame%.5d_submit%.5d.rd",
               fd_rd_dump_env.output_path, output->name, frame, submit);
   } else {
      snprintf(file_path, sizeof(file_path), "%s/%s_submit%.5d.rd",
               fd_rd_dump_env.output_path, output->name, submit);
   }
   snprintf(output->capture_path, sizeof(output->capture_path), "%s", file_path);
   if (output->bounded) {
      int fd = open(file_path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
      output->file = fd < 0 ? NULL : gzdopen(fd, "w");
      if (fd >= 0 && !output->file) close(fd);
   } else {
      output->file = gzopen(file_path, "w");
   }
   if (output->file == NULL) {
      mesa_loge("[fd_rd_output] failed to open %s: %s", file_path,
                strerror(errno));
      if (output->bounded) fd_rd_output_error(output, "capture_open");
      return false;
   }
   return true;
}

static void
fd_rd_output_write(struct fd_rd_output *output, const void *buffer, int size)
{
   if (output->failed) return;
   if (output->bounded && (size < 0 || (uint64_t)size > output->expected_bytes - output->written_bytes)) {
      fd_rd_output_error(output, "unexpected_section_size");
      return;
   }
   output->written_bytes += size;
   const uint8_t *pos = (uint8_t *) buffer;
   while (size > 0) {
      int ret = gzwrite(output->file, pos, size);
      if (ret <= 0) {
         mesa_loge("[fd_rd_output] failed to write to compressed output: %s",
                   gzerror(output->file, NULL));
         if (output->bounded) fd_rd_output_error(output, "gzip_write");
         return;
      }
      pos += ret;
      size -= ret;
   }
}

void
fd_rd_output_write_section(struct fd_rd_output *output, enum rd_sect_type type,
                           const void *buffer, int size)
{
   fd_rd_output_write(output, &type, 4);
   fd_rd_output_write(output, &size, 4);
   fd_rd_output_write(output, buffer, size);
}

void
fd_rd_output_end(struct fd_rd_output *output)
{
   assert(output->file != NULL);

   /* When combining output, flush the gzip stream on each submit. This should
    * store all the data before any problem during the submit itself occurs.
    */
   if (output->combine) {
      gzflush(output->file, Z_FINISH);
      return;
   }

   int result = gzclose(output->file);
   output->file = NULL;
   if (output->bounded && (result != Z_OK || output->written_bytes != output->expected_bytes))
      fd_rd_output_error(output, "gzip_close_or_incomplete");
}

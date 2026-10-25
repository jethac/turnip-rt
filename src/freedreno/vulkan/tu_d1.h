/*
 * Copyright © 2026 Jetha Chan
 * SPDX-License-Identifier: MIT
 */

#ifndef TU_D1_H
#define TU_D1_H

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#define TU_D1_SLOTS 4
#define TU_D1_MAX_SIGS 32
#define TU_D1_REC_DWORDS 16
#define TU_D1_SCRATCH 3

#define TU_D1_STATS_OFFSET (TU_D1_SLOTS * TU_D1_MAX_SIGS * TU_D1_REC_DWORDS * 4)
#define TU_D1_STATS_ONE 0
#define TU_D1_STATS_HITS 1
#define TU_D1_STATS_DWORDS 16

enum tu_d1_rec {
   TU_D1_REC_BUF_INFO,
   TU_D1_REC_PITCH,
   TU_D1_REC_ARRAY_PITCH,
   TU_D1_REC_BASE_LO,
   TU_D1_REC_BASE_HI,
   TU_D1_REC_SP_PS_MRT,
   TU_D1_REC_FLAG_LO,
   TU_D1_REC_FLAG_HI,
   TU_D1_REC_FLAG_PITCH,
   TU_D1_REC_TAG,
   TU_D1_REC_KEY,
   TU_D1_REC_WRITTEN,
};

enum tu_d1_label_kind {
   TU_D1_LABEL_BAD,
   TU_D1_LABEL_PUB,
   TU_D1_LABEL_USE,
};

struct tu_d1_label {
   enum tu_d1_label_kind kind;
   uint32_t slot;
   uint32_t gen;
   uint64_t view;
   uint32_t a6, a7;
};

static inline bool
tu_d1_parse_lit(const char **p, const char *lit)
{
   size_t n = strlen(lit);
   if (strncmp(*p, lit, n) != 0)
      return false;
   *p += n;
   return true;
}

static inline bool
tu_d1_parse_dec(const char **p, uint64_t lo, uint64_t hi, uint32_t *out)
{
   const char *s = *p;
   uint64_t v = 0;
   unsigned n = 0;
   while (s[n] >= '0' && s[n] <= '9') {
      if (n == 10)
         return false;
      v = v * 10 + (uint64_t)(s[n] - '0');
      n++;
   }
   if (n == 0 || (n > 1 && s[0] == '0') || v < lo || v > hi)
      return false;
   *out = (uint32_t)v;
   *p = s + n;
   return true;
}

static inline bool
tu_d1_parse_hex16(const char **p, uint64_t *out)
{
   const char *s = *p;
   uint64_t v = 0;
   for (unsigned i = 0; i < 16; i++) {
      char c = s[i];
      uint64_t d;
      if (c >= '0' && c <= '9')
         d = (uint64_t)(c - '0');
      else if (c >= 'a' && c <= 'f')
         d = (uint64_t)(c - 'a' + 10);
      else
         return false;
      v = (v << 4) | d;
   }
   *out = v;
   *p = s + 16;
   return true;
}

static inline enum tu_d1_label_kind
tu_d1_parse_label(const char *s, struct tu_d1_label *l)
{
   const char *p = s;
   memset(l, 0, sizeof(*l));
   l->kind = TU_D1_LABEL_BAD;

   if (tu_d1_parse_lit(&p, "D1PUB s=")) {
      if (tu_d1_parse_dec(&p, 0, TU_D1_SLOTS - 1, &l->slot) &&
          tu_d1_parse_lit(&p, " g=") &&
          tu_d1_parse_dec(&p, 1, UINT32_MAX, &l->gen) &&
          tu_d1_parse_lit(&p, " v=") &&
          tu_d1_parse_hex16(&p, &l->view) && *p == '\0')
         l->kind = TU_D1_LABEL_PUB;
   } else if (tu_d1_parse_lit(&p, "D1USE g=")) {
      if (tu_d1_parse_dec(&p, 1, UINT32_MAX, &l->gen) &&
          tu_d1_parse_lit(&p, " a6=") &&
          tu_d1_parse_dec(&p, 0, TU_D1_SLOTS - 1, &l->a6) &&
          tu_d1_parse_lit(&p, " a7=") &&
          tu_d1_parse_dec(&p, 0, TU_D1_SLOTS - 1, &l->a7) && *p == '\0')
         l->kind = TU_D1_LABEL_USE;
   }

   if (l->kind == TU_D1_LABEL_BAD)
      memset(l, 0, sizeof(*l));
   return l->kind;
}

static inline uint32_t
tu_d1_rec_index(uint32_t slot, uint32_t sig_id)
{
   return slot * TU_D1_MAX_SIGS + (sig_id - 1);
}

struct tu_d1_cmd {
   bool armed;
   bool rp_active;
   uint32_t gen;
   uint32_t slot[2];
   struct {
      bool valid;
      uint32_t key;
      uint64_t rec_iova;
   } att[2];
};

#ifdef __cplusplus

#include "tu_common.h"

struct tu_d1_device;

uint32_t tu_d1_physical_device_mode(const struct tu_physical_device *pdevice);

void tu_d1_driver_info(const struct tu_physical_device *pdevice, char *info,
                       size_t size);

void tu_d1_device_init(struct tu_device *dev);
void tu_d1_device_finish(struct tu_device *dev);

void tu_d1_view_created(struct tu_device *dev, struct tu_image_view *iview);
void tu_d1_view_destroyed(struct tu_device *dev, struct tu_image_view *iview);

void tu_d1_label(struct tu_cmd_buffer *cmd, const char *name);
void tu_d1_begin_rendering(struct tu_cmd_buffer *cmd,
                           const VkRenderingInfo *info);
void tu_d1_disarm(struct tu_cmd_buffer *cmd, const char *why);

template <chip CHIP>
void tu_d1_emit_forward(struct tu_cmd_buffer *cmd, struct tu_cs *cs);

#endif

#endif

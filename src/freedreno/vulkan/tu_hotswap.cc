/*
 * Copyright © 2026 Local verification
 * SPDX-License-Identifier: MIT
 */

#include "tu_hotswap.h"

#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <map>
#include <set>
#include <string>
#include <vector>

#include "util/mesa-blake3.h"
#include "util/os_time.h"
#include "util/simple_mtx.h"
#include "util/u_atomic.h"
#include "util/u_debug.h"
#include "vk_shader_module.h"
#include "vk_util.h"

#include "tu_descriptor_set.h"
#include "tu_device.h"
#include "tu_foveate.h"
#include "tu_pipeline.h"
#include "tu_shader.h"

#define HS_MAXV TU_HOTSWAP_MAX_VARIANTS
#define HS_CTL_MAX 4096

struct hs_alt {
   bool defined;
   bool base;
   int same_as;
   std::string name;
   std::string path;
   std::vector<uint32_t> words;
};

struct hs_target {
   std::string label;
   bool by_src;
   uint64_t hash;
   int stage;
   bool relaxed;
   hs_alt alt[HS_MAXV];
   uint32_t poisoned;
   uint64_t binds[HS_MAXV];
   uint64_t fallback;
   uint32_t pipelines;
   uint32_t accepted[HS_MAXV];
   int last_group;
};

struct hs_group {
   std::string name;
   std::vector<int> members;
};

struct hs_state {
   bool on;
   bool any_src;
   bool epoch_submit;
   int log;
   std::string manifest;
   std::string ctl_path;
   std::vector<hs_target> targets;
   std::vector<hs_group> groups;
   uint64_t key;
   bool ctl_seen;
   bool ctl_present;
   char ctl_raw[HS_CTL_MAX];
   uint32_t ctl_changes;
   uint32_t poison_gen;
   uint32_t poison_seen;
   std::string seq;
   uint8_t sel[256];
   uint64_t last_stat_key;
   uint64_t switch_key;
};

static hs_state hs;
static simple_mtx_t hs_mtx = SIMPLE_MTX_INITIALIZER;
static uint64_t hs_fast_key = UINT64_MAX;
static uint32_t hs_sel_atomic[256];

static int64_t
hs_raw_ns(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
   return (int64_t) ts.tv_sec * 1000000000ll + ts.tv_nsec;
}

static bool
hs_parse_hex64(const char *s, uint64_t *out)
{
   size_t n = strlen(s);
   if (n < 1 || n > 16)
      return false;
   uint64_t v = 0;
   for (size_t i = 0; i < n; i++) {
      char c = s[i];
      unsigned d;
      if (c >= '0' && c <= '9')
         d = c - '0';
      else if (c >= 'a' && c <= 'f')
         d = c - 'a' + 10;
      else if (c >= 'A' && c <= 'F')
         d = c - 'A' + 10;
      else
         return false;
      v = (v << 4) | d;
   }
   if (n != 16)
      return false;
   *out = v;
   return true;
}

static bool
hs_read_spirv(const std::string &path, std::vector<uint32_t> &w)
{
   FILE *f = fopen(path.c_str(), "rb");
   if (!f)
      return false;
   fseek(f, 0, SEEK_END);
   long sz = ftell(f);
   fseek(f, 0, SEEK_SET);
   if (sz < 20 || (sz & 3) || sz > (64l << 20)) {
      fclose(f);
      return false;
   }
   w.resize(sz / 4);
   size_t r = fread(w.data(), 1, sz, f);
   fclose(f);
   return r == (size_t) sz && w[0] == 0x07230203;
}

static int
hs_find_target(const std::string &label)
{
   for (size_t i = 0; i < hs.targets.size(); i++)
      if (hs.targets[i].label == label)
         return (int) i;
   return -1;
}

static int
hs_find_group(const std::string &name)
{
   for (size_t i = 0; i < hs.groups.size(); i++)
      if (hs.groups[i].name == name)
         return (int) i;
   return -1;
}

static bool
hs_valid_label(const char *s)
{
   if (!*s || strlen(s) > 63)
      return false;
   for (; *s; s++)
      if (!((*s >= 'a' && *s <= 'z') || (*s >= 'A' && *s <= 'Z') ||
            (*s >= '0' && *s <= '9') || *s == '_' || *s == '-' || *s == '.'))
         return false;
   return true;
}

static bool
hs_load_manifest(const char *path)
{
   FILE *f = fopen(path, "r");
   if (!f) {
      mesa_loge("tu-hotswap: cannot open manifest %s; hot-swap disabled", path);
      return false;
   }
   std::string dir = path;
   size_t slash = dir.rfind('/');
   dir = slash == std::string::npos ? std::string(".") : dir.substr(0, slash);

   char line[4096];
   unsigned lineno = 0;
   bool ok = true;
   while (ok && fgets(line, sizeof(line), f)) {
      lineno++;
      char *hash = strchr(line, '#');
      if (hash)
         *hash = 0;
      char *tok[8];
      unsigned nt = 0;
      for (char *s = strtok(line, " \t\r\n"); s && nt < 8; s = strtok(NULL, " \t\r\n"))
         tok[nt++] = s;
      if (!nt)
         continue;
      if (!strcmp(tok[0], "target") && nt >= 3) {
         hs_target t = {};
         t.stage = -1;
         t.last_group = -1;
         t.label = tok[1];
         if (!hs_valid_label(tok[1]) || hs_find_target(t.label) >= 0 ||
             hs_find_group(t.label) >= 0 || !strcmp(tok[1], "seq") ||
             !strcmp(tok[1], "all")) {
            ok = false;
            break;
         }
         if (!strncmp(tok[2], "dxil=", 5))
            t.by_src = false;
         else if (!strncmp(tok[2], "src=", 4))
            t.by_src = true;
         else {
            ok = false;
            break;
         }
         if (!hs_parse_hex64(strchr(tok[2], '=') + 1, &t.hash)) {
            ok = false;
            break;
         }
         for (unsigned i = 3; i < nt; i++) {
            if (!strcmp(tok[i], "stage=cs"))
               t.stage = MESA_SHADER_COMPUTE;
            else if (!strcmp(tok[i], "stage=fs"))
               t.stage = MESA_SHADER_FRAGMENT;
            else if (!strcmp(tok[i], "stage=vs"))
               t.stage = MESA_SHADER_VERTEX;
            else if (!strcmp(tok[i], "iface=strict"))
               t.relaxed = false;
            else if (!strcmp(tok[i], "iface=relaxed"))
               t.relaxed = true;
            else
               ok = false;
         }
         for (const hs_target &o : hs.targets)
            if (o.by_src == t.by_src && o.hash == t.hash)
               ok = false;
         if (!ok || hs.targets.size() >= 255)
            break;
         hs.any_src |= t.by_src;
         hs.targets.push_back(t);
      } else if (!strcmp(tok[0], "alt") && nt == 5) {
         int ti = hs_find_target(tok[1]);
         char *end;
         long k = strtol(tok[2], &end, 10);
         if (ti < 0 || *end || k < 1 || k >= HS_MAXV || !hs_valid_label(tok[3]) ||
             hs.targets[ti].alt[k].defined) {
            ok = false;
            break;
         }
         hs_alt &a = hs.targets[ti].alt[k];
         for (unsigned j = 1; j < HS_MAXV; j++)
            if (hs.targets[ti].alt[j].defined && hs.targets[ti].alt[j].name == tok[3])
               ok = false;
         if (!ok)
            break;
         a.defined = true;
         a.name = tok[3];
         a.same_as = 0;
         if (!strcmp(tok[4], "base")) {
            a.base = true;
         } else if (tok[4][0] == '@') {
            a.base = true;
            for (unsigned j = 1; j < HS_MAXV; j++)
               if (hs.targets[ti].alt[j].defined && !hs.targets[ti].alt[j].base &&
                   hs.targets[ti].alt[j].name == tok[4] + 1)
                  a.same_as = (int) j;
            if (!a.same_as) {
               ok = false;
               break;
            }
         } else {
            a.path = tok[4][0] == '/' ? std::string(tok[4]) : dir + "/" + tok[4];
            if (!hs_read_spirv(a.path, a.words)) {
               mesa_loge("tu-hotswap: manifest %s:%u: cannot read SPIR-V %s",
                         path, lineno, a.path.c_str());
               ok = false;
               break;
            }
         }
      } else if (!strcmp(tok[0], "group") && nt >= 3) {
         hs_group g;
         g.name = tok[1];
         if (!hs_valid_label(tok[1]) || hs_find_group(g.name) >= 0 ||
             hs_find_target(g.name) >= 0 || !strcmp(tok[1], "seq") ||
             !strcmp(tok[1], "all")) {
            ok = false;
            break;
         }
         for (unsigned i = 2; i < nt; i++) {
            int ti = hs_find_target(tok[i]);
            if (ti < 0) {
               ok = false;
               break;
            }
            g.members.push_back(ti);
         }
         if (ok)
            hs.groups.push_back(g);
      } else {
         ok = false;
      }
      if (!ok)
         break;
   }
   fclose(f);
   if (!ok) {
      mesa_loge("tu-hotswap: manifest %s:%u malformed; hot-swap disabled", path, lineno);
      return false;
   }
   if (hs.targets.empty()) {
      mesa_loge("tu-hotswap: manifest %s has no targets; hot-swap disabled", path);
      return false;
   }
   return true;
}

static void
hs_init_once(void)
{
   const char *m = os_get_option("TU_HOTSWAP");
   hs.key = UINT64_MAX;
   if (!m || !*m)
      return;
   hs.manifest = m;
   const char *ctl = os_get_option("TU_HOTSWAP_CTL");
   hs.ctl_path = ctl && *ctl ? ctl : "/dev/shm/tu-hotswap";
   const char *ep = os_get_option("TU_HOTSWAP_EPOCH");
   hs.epoch_submit = ep && !strcmp(ep, "submit");
   hs.log = (int) debug_get_num_option("TU_HOTSWAP_LOG", 1);
   if (!hs_load_manifest(m)) {
      hs.targets.clear();
      hs.groups.clear();
      return;
   }
   hs.on = true;
   unsigned nalt = 0;
   for (const hs_target &t : hs.targets)
      for (unsigned k = 1; k < HS_MAXV; k++)
         nalt += t.alt[k].defined;
   mesa_logi("tu-hotswap: manifest %s targets=%zu alts=%u groups=%zu ctl=%s epoch=%s",
             m, hs.targets.size(), nalt, hs.groups.size(), hs.ctl_path.c_str(),
             hs.epoch_submit ? "submit" : "present");
   for (const hs_target &t : hs.targets) {
      for (unsigned k = 1; k < HS_MAXV; k++) {
         if (!t.alt[k].defined)
            continue;
         unsigned char h[BLAKE3_OUT_LEN] = { 0 };
         if (!t.alt[k].base)
            _mesa_blake3_compute(t.alt[k].words.data(), t.alt[k].words.size() * 4, h);
         uint32_t s0, s1;
         memcpy(&s0, h, 4);
         memcpy(&s1, h + 4, 4);
         mesa_logi("tu-hotswap: target=%s %s=%016" PRIx64 " alt=%u name=%s src=%08x%08x path=%s",
                   t.label.c_str(), t.by_src ? "src" : "dxil", t.hash, k,
                   t.alt[k].name.c_str(), s0, s1,
                   t.alt[k].same_as ? t.alt[t.alt[k].same_as].name.c_str() :
                   t.alt[k].base ? "base" : t.alt[k].path.c_str());
      }
   }
}

bool
tu_hotswap_enabled(void)
{
   static once_flag once = ONCE_FLAG_INIT;
   call_once(&once, hs_init_once);
   return hs.on;
}

static const uint32_t *
hs_stage_words(const VkPipelineShaderStageCreateInfo *stage, size_t *size)
{
   VK_FROM_HANDLE(vk_shader_module, module, stage->module);
   if (module) {
      *size = module->size;
      return (const uint32_t *) module->data;
   }
   const VkShaderModuleCreateInfo *mci =
      vk_find_struct_const(stage->pNext, SHADER_MODULE_CREATE_INFO);
   if (mci) {
      *size = mci->codeSize;
      return mci->pCode;
   }
   return NULL;
}

int
tu_hotswap_match(const VkPipelineShaderStageCreateInfo *stage,
                 mesa_shader_stage st)
{
   if (!tu_hotswap_enabled() || !stage)
      return -1;
   size_t size = 0;
   const uint32_t *words = hs_stage_words(stage, &size);
   if (!words)
      return -1;
   uint64_t dxil = tu_spirv_dxil_hash(words, size);
   uint64_t src = 0;
   if (hs.any_src) {
      unsigned char h[BLAKE3_OUT_LEN];
      _mesa_blake3_compute(words, size, h);
      uint32_t s0, s1;
      memcpy(&s0, h, 4);
      memcpy(&s1, h + 4, 4);
      src = ((uint64_t) s0 << 32) | s1;
   }
   for (size_t i = 0; i < hs.targets.size(); i++) {
      const hs_target &t = hs.targets[i];
      if (t.stage >= 0 && t.stage != (int) st)
         continue;
      if (st == MESA_SHADER_VERTEX && t.stage != MESA_SHADER_VERTEX)
         continue;
      if (t.by_src ? src == t.hash : (dxil && dxil == t.hash))
         return (int) i;
   }
   return -1;
}

struct hs_iface {
   bool ok;
   std::string err;
   bool has_entry;
   std::map<std::pair<uint32_t, uint32_t>, std::set<std::string>> res;
   std::string push;
   std::map<std::string, std::string> in, out;
   std::set<std::string> modes;
   bool gaze;
   uint32_t gaze_set, gaze_binding;
};

struct hs_spirv {
   const uint32_t *w;
   size_t n;
   std::map<uint32_t, size_t> def;
   std::map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>> deco;
   std::map<uint32_t, std::map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>>> mdeco;
};

static uint32_t
hs_deco(const hs_spirv &s, uint32_t id, uint32_t d, bool *found)
{
   auto it = s.deco.find(id);
   if (it != s.deco.end())
      for (auto &p : it->second)
         if (p.first == d) {
            *found = true;
            return p.second;
         }
   *found = false;
   return 0;
}

static std::string hs_type_sig(const hs_spirv &s, uint32_t id, int depth);

static std::string
hs_const_sig(const hs_spirv &s, uint32_t id)
{
   auto it = s.def.find(id);
   if (it == s.def.end())
      return "?";
   const uint32_t *ins = s.w + it->second;
   uint32_t op = ins[0] & 0xffff, len = ins[0] >> 16;
   char buf[64];
   if ((op == 43 || op == 50) && len >= 4) {
      bool f;
      uint32_t sid = hs_deco(s, id, 1, &f);
      if (op == 50)
         snprintf(buf, sizeof(buf), "spec%u:%u", f ? sid : ~0u, ins[3]);
      else
         snprintf(buf, sizeof(buf), "%u", ins[3]);
      return buf;
   }
   return "op" + std::to_string(op);
}

static std::string
hs_type_sig(const hs_spirv &s, uint32_t id, int depth)
{
   if (depth > 24)
      return "deep";
   auto it = s.def.find(id);
   if (it == s.def.end())
      return "?";
   const uint32_t *ins = s.w + it->second;
   uint32_t op = ins[0] & 0xffff, len = ins[0] >> 16;
   char buf[128];
   bool f;
   switch (op) {
   case 19: return "void";
   case 20: return "bool";
   case 21:
      snprintf(buf, sizeof(buf), "%c%u", ins[3] ? 'i' : 'u', ins[2]);
      return buf;
   case 22:
      snprintf(buf, sizeof(buf), "f%u", ins[2]);
      return buf;
   case 23:
   case 24:
      return std::string(op == 23 ? "v" : "m") + std::to_string(ins[3]) + "(" +
             hs_type_sig(s, ins[2], depth + 1) + ")";
   case 25:
      snprintf(buf, sizeof(buf), "img(dim%u,d%u,a%u,ms%u,s%u,fmt%u,", ins[3],
               ins[4], ins[5], ins[6], ins[7], ins[8]);
      return buf + hs_type_sig(s, ins[2], depth + 1) + ")";
   case 26: return "sampler";
   case 27: return "simg(" + hs_type_sig(s, ins[2], depth + 1) + ")";
   case 28:
   case 29: {
      uint32_t stride = hs_deco(s, id, 6, &f);
      std::string r = op == 28 ? "arr[" + hs_const_sig(s, ins[3]) + "]" : "rarr";
      if (f)
         r += "s" + std::to_string(stride);
      return r + "(" + hs_type_sig(s, ins[2], depth + 1) + ")";
   }
   case 30: {
      std::string r = "struct";
      if (s.deco.count(id))
         for (auto &p : s.deco.at(id))
            if (p.first == 2 || p.first == 3)
               r += p.first == 2 ? "B" : "BB";
      r += "{";
      for (uint32_t m = 2; m < len; m++) {
         if (m > 2)
            r += ",";
         auto md = s.mdeco.find(id);
         if (md != s.mdeco.end() && md->second.count(m - 2))
            for (auto &p : md->second.at(m - 2))
               if (p.first == 35 || p.first == 7 || p.first == 4 || p.first == 11)
                  r += "@" + std::to_string(p.first) + ":" + std::to_string(p.second);
         r += hs_type_sig(s, ins[m], depth + 1);
      }
      return r + "}";
   }
   case 32:
      if (ins[2] == 5349)
         return "psbptr";
      return "ptr" + std::to_string(ins[2]) + "(" + hs_type_sig(s, ins[3], depth + 1) + ")";
   case 39: return "fwdptr";
   case 5341: return "accel";
   case 4472: return "rayq";
   default:
      return "op" + std::to_string(op);
   }
}

static bool
hs_string_at(const uint32_t *w, size_t words, std::string &out)
{
   const char *c = (const char *) w;
   size_t l = strnlen(c, words * 4);
   if (l == words * 4)
      return false;
   out.assign(c, l);
   return true;
}

static void
hs_extract(const uint32_t *w, size_t size, const char *entry, mesa_shader_stage st,
           hs_iface &r)
{
   r.ok = false;
   r.has_entry = false;
   r.gaze = false;
   size_t n = size / 4;
   if (!w || n < 5 || w[0] != 0x07230203) {
      r.err = "not-spirv";
      return;
   }
   hs_spirv s;
   s.w = w;
   s.n = n;
   uint32_t want_model = st == MESA_SHADER_COMPUTE ? 5 :
                         st == MESA_SHADER_VERTEX ? 0 : 4;
   uint32_t entry_id = 0;
   std::vector<size_t> modes, vars;
   for (size_t i = 5; i < n;) {
      uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
      if (!len || len > n - i) {
         r.err = "truncated";
         return;
      }
      if (op == 54)
         break;
      switch (op) {
      case 15: {
         std::string name;
         if (len >= 4 && w[i + 1] == want_model && hs_string_at(w + i + 3, len - 3, name) &&
             name == entry) {
            entry_id = w[i + 2];
            r.has_entry = true;
         }
         break;
      }
      case 16:
      case 331:
         modes.push_back(i);
         break;
      case 330: {
         std::string m;
         if (len >= 2 && hs_string_at(w + i + 1, len - 1, m) &&
             !m.compare(0, 16, "turnip.gaze_ubo=")) {
            unsigned a, b;
            char tail;
            if (sscanf(m.c_str() + 16, "%u:%u%c", &a, &b, &tail) == 2) {
               r.gaze = true;
               r.gaze_set = a;
               r.gaze_binding = b;
            }
         }
         break;
      }
      case 71:
         if (len >= 3)
            s.deco[w[i + 1]].push_back({ w[i + 2], len >= 4 ? w[i + 3] : 0 });
         break;
      case 72:
         if (len >= 4)
            s.mdeco[w[i + 1]][w[i + 2]].push_back({ w[i + 3], len >= 5 ? w[i + 4] : 0 });
         break;
      case 59:
         if (len >= 4)
            s.def[w[i + 2]] = i;
         vars.push_back(i);
         break;
      case 19: case 20: case 21: case 22: case 23: case 24: case 25: case 26:
      case 27: case 28: case 29: case 30: case 32: case 39: case 5341: case 4472:
         if (len >= 2)
            s.def[w[i + 1]] = i;
         break;
      case 43: case 50:
         if (len >= 3)
            s.def[w[i + 2]] = i;
         break;
      default:
         break;
      }
      i += len;
   }
   if (!r.has_entry) {
      r.err = "no-entry-point";
      return;
   }
   for (size_t i : modes) {
      uint32_t len = w[i] >> 16;
      if (w[i + 1] != entry_id)
         continue;
      uint32_t mode = w[i + 2];
      if (st == MESA_SHADER_COMPUTE)
         continue;
      std::string m = "m" + std::to_string(mode);
      for (uint32_t j = 3; j < len; j++)
         m += ":" + std::to_string(w[i + j]);
      r.modes.insert(m);
   }
   for (size_t i : vars) {
      uint32_t len = w[i] >> 16;
      if (len < 4)
         continue;
      uint32_t type = w[i + 1], id = w[i + 2], sc = w[i + 3];
      std::string sig = hs_type_sig(s, type, 0);
      bool fs, fb;
      uint32_t set = hs_deco(s, id, 34, &fs);
      uint32_t binding = hs_deco(s, id, 33, &fb);
      if (fs || fb) {
         r.res[{ set, binding }].insert(sig);
         continue;
      }
      if (sc == 9) {
         if (!r.push.empty() && r.push != sig) {
            r.err = "multiple-push-blocks";
            return;
         }
         r.push = sig;
         continue;
      }
      if ((st == MESA_SHADER_FRAGMENT || st == MESA_SHADER_VERTEX) &&
          (sc == 1 || sc == 3)) {
         bool fl, fc, fbi, fi;
         uint32_t loc = hs_deco(s, id, 30, &fl);
         uint32_t comp = hs_deco(s, id, 31, &fc);
         uint32_t bi = hs_deco(s, id, 11, &fbi);
         uint32_t idx = hs_deco(s, id, 32, &fi);
         std::string key;
         if (fbi)
            key = "builtin" + std::to_string(bi);
         else if (fl)
            key = "loc" + std::to_string(loc) + "c" + std::to_string(fc ? comp : 0) +
                  "i" + std::to_string(fi ? idx : 0);
         else if (st == MESA_SHADER_VERTEX)
            key = "block:" + sig;
         else
            key = "block" + std::to_string(id);
         (sc == 1 ? r.in : r.out)[key] = sig;
      }
   }
   r.ok = true;
}

static bool
hs_layout_has(const struct tu_pipeline_layout *layout, uint32_t set, uint32_t binding)
{
   if (!layout || set >= layout->num_sets || !layout->set[set].layout)
      return false;
   const struct tu_descriptor_set_layout *sl = layout->set[set].layout;
   return binding < sl->binding_count;
}

static bool
hs_check(const hs_iface &b, const hs_iface &a, const struct tu_pipeline_layout *layout,
         bool relaxed, mesa_shader_stage st, std::string &reason, const char **mode)
{
   bool exact = true;
   for (auto &e : a.res) {
      bool gaze = a.gaze && e.first.first == a.gaze_set && e.first.second == a.gaze_binding;
      if (gaze) {
         if (!tu_gaze_block_enabled()) {
            reason = "gaze_ubo marker needs TU_GAZE_BLOCK=1";
            return false;
         }
         continue;
      }
      auto bi = b.res.find(e.first);
      char where[48];
      snprintf(where, sizeof(where), "set%u.binding%u", e.first.first, e.first.second);
      if (relaxed) {
         if (!hs_layout_has(layout, e.first.first, e.first.second)) {
            reason = std::string("binding not in pipeline layout: ") + where;
            return false;
         }
         if (bi == b.res.end() || bi->second != e.second)
            exact = false;
         continue;
      }
      if (bi == b.res.end()) {
         reason = std::string("new binding ") + where;
         return false;
      }
      for (auto &sig : e.second)
         if (!bi->second.count(sig)) {
            reason = std::string("binding type differs ") + where + " " + sig.substr(0, 120);
            return false;
         }
      if (bi->second != e.second)
         exact = false;
   }
   if (a.res.size() != b.res.size())
      exact = false;
   if (!a.push.empty()) {
      if (b.push.empty() && (!relaxed || !layout || !layout->push_constant_size)) {
         reason = "push constant block not in original";
         return false;
      }
      if (!b.push.empty() && a.push != b.push) {
         if (!relaxed) {
            reason = "push constant block differs";
            return false;
         }
         exact = false;
      }
   } else if (!b.push.empty()) {
      exact = false;
   }
   if (st == MESA_SHADER_FRAGMENT) {
      if (a.out != b.out) {
         reason = "fragment outputs differ";
         return false;
      }
      for (auto &e : a.in) {
         auto bi = b.in.find(e.first);
         if (bi == b.in.end() || bi->second != e.second) {
            reason = "fragment input not in original: " + e.first;
            return false;
         }
      }
      if (a.in.size() != b.in.size())
         exact = false;
      if (a.modes != b.modes) {
         reason = "fragment execution modes differ";
         return false;
      }
   }
   if (st == MESA_SHADER_VERTEX) {
      if (a.out != b.out) {
         reason = "vertex outputs differ";
         return false;
      }
      for (auto &e : a.in) {
         auto bi = b.in.find(e.first);
         if (bi == b.in.end() || bi->second != e.second) {
            reason = "vertex input not in original: " + e.first;
            return false;
         }
      }
      if (a.in.size() != b.in.size())
         exact = false;
   }
   *mode = exact ? "exact" : (relaxed ? "relaxed" : "subset");
   return true;
}

static void
hs_poison(int target, unsigned k)
{
   hs_target &t = hs.targets[target];
   if (!(t.poisoned & (1u << k))) {
      t.poisoned |= 1u << k;
      hs.poison_gen++;
   }
}

static const char *
hs_stage_name(mesa_shader_stage st)
{
   return st == MESA_SHADER_COMPUTE ? "cs" :
          st == MESA_SHADER_VERTEX ? "vs" : "fs";
}

void
tu_hotswap_build(struct tu_device *dev, struct tu_pipeline *base, int target,
                 const VkPipelineShaderStageCreateInfo *stage, mesa_shader_stage st,
                 const struct tu_pipeline_layout *layout, const void *feedback_pnext,
                 const VkAllocationCallbacks *alloc, tu_hotswap_create_cb create,
                 void *data)
{
   if (target < 0 || !base || base->hotswap)
      return;
   hs_target &t = hs.targets[target];
   size_t bsize = 0;
   const uint32_t *bw = hs_stage_words(stage, &bsize);
   hs_iface bif;
   hs_extract(bw, bsize, stage->pName, st, bif);

   struct tu_hotswap_pipe *hp = (struct tu_hotswap_pipe *)
      vk_zalloc2(&dev->vk.alloc, alloc, sizeof(*hp), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!hp)
      return;
   hp->target = target;
   hp->dxil = tu_spirv_dxil_hash(bw, bsize);

   const VkPipelineCreationFeedbackCreateInfo *fb =
      (const VkPipelineCreationFeedbackCreateInfo *)
         vk_find_struct_const(feedback_pnext, PIPELINE_CREATION_FEEDBACK_CREATE_INFO);
   VkPipelineCreationFeedback fb_pipe = {};
   std::vector<VkPipelineCreationFeedback> fb_stages;
   if (fb) {
      fb_pipe = *fb->pPipelineCreationFeedback;
      if (fb->pPipelineStageCreationFeedbacks)
         fb_stages.assign(fb->pPipelineStageCreationFeedbacks,
                          fb->pPipelineStageCreationFeedbacks +
                             fb->pipelineStageCreationFeedbackCount);
   }

   const char *base_bin = base->shaders[st] && base->shaders[st]->variant ?
                          base->shaders[st]->variant->blake3_str : "none";
   unsigned accepted = 0, refused = 0;
   for (unsigned k = 1; k < HS_MAXV; k++) {
      const hs_alt &a = t.alt[k];
      if (!a.defined)
         continue;
      if (a.base) {
         hp->alias |= 1u << k;
         accepted++;
         continue;
      }
      std::string reason;
      const char *mode = "";
      hs_iface aif;
      if (!bif.ok) {
         reason = "original interface unreadable: " + bif.err;
      } else {
         hs_extract(a.words.data(), a.words.size() * 4, stage->pName, st, aif);
         if (!aif.ok)
            reason = "alternative interface unreadable: " + aif.err;
         else
            hs_check(bif, aif, layout, t.relaxed, st, reason, &mode);
      }
      VkPipeline var = VK_NULL_HANDLE;
      if (reason.empty()) {
         VkShaderModuleCreateInfo mci = {};
         mci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
         mci.codeSize = a.words.size() * 4;
         mci.pCode = a.words.data();
         VkShaderModule mod = VK_NULL_HANDLE;
         VkResult r = vk_common_CreateShaderModule(tu_device_to_handle(dev), &mci, NULL, &mod);
         if (r == VK_SUCCESS) {
            r = create(data, mod, &var);
            vk_common_DestroyShaderModule(tu_device_to_handle(dev), mod, NULL);
         }
         if (r != VK_SUCCESS || !var) {
            reason = "variant pipeline creation failed (" + std::to_string((int) r) + ")";
            var = VK_NULL_HANDLE;
         }
      }
      if (var) {
         VK_FROM_HANDLE(tu_pipeline, vp, var);
         if (st == MESA_SHADER_COMPUTE) {
            struct tu_compute_pipeline *bc = tu_pipeline_to_compute(base);
            struct tu_compute_pipeline *vc = tu_pipeline_to_compute(vp);
            if (memcmp(bc->local_size, vc->local_size, sizeof(bc->local_size))) {
               char b[96];
               snprintf(b, sizeof(b), "local size %ux%ux%u != original %ux%ux%u",
                        vc->local_size[0], vc->local_size[1], vc->local_size[2],
                        bc->local_size[0], bc->local_size[1], bc->local_size[2]);
               reason = b;
            }
         } else if (base->type == TU_PIPELINE_GRAPHICS_LIB) {
            if (vp->type != TU_PIPELINE_GRAPHICS_LIB ||
                (!vp->shaders[st] && !tu_pipeline_to_graphics_lib(vp)->shaders[st].nir))
               reason = std::string("variant library has no ") + hs_stage_name(st) + " shader";
         } else if (vp->type != TU_PIPELINE_GRAPHICS || !vp->shaders[st]) {
            reason = std::string("variant has no ") + hs_stage_name(st) + " shader";
         }
         if (!reason.empty()) {
            tu_DestroyPipeline(tu_device_to_handle(dev), var, alloc);
            var = VK_NULL_HANDLE;
         }
      }
      simple_mtx_lock(&hs_mtx);
      if (var) {
         VK_FROM_HANDLE(tu_pipeline, vp, var);
         hp->variant[k] = var;
         t.accepted[k]++;
         accepted++;
         mesa_logi("tu-hotswap: create target=%s stage=%s dxil=%016" PRIx64
                   " pipeline=%p base_bin=%.12s variant=%u name=%s bin=%.12s iface=%s",
                   t.label.c_str(), hs_stage_name(st), hp->dxil, (void *) base,
                   base_bin, k, a.name.c_str(),
                   vp->shaders[st] && vp->shaders[st]->variant ?
                      vp->shaders[st]->variant->blake3_str : "none",
                   mode);
      } else {
         refused++;
         hs_poison(target, k);
         mesa_loge("tu-hotswap: refuse target=%s stage=%s dxil=%016" PRIx64
                   " pipeline=%p variant=%u name=%s reason=%s",
                   t.label.c_str(), hs_stage_name(st), hp->dxil, (void *) base, k,
                   a.name.c_str(), reason.c_str());
      }
      simple_mtx_unlock(&hs_mtx);
   }

   if (fb) {
      *fb->pPipelineCreationFeedback = fb_pipe;
      for (size_t i = 0; i < fb_stages.size(); i++)
         fb->pPipelineStageCreationFeedbacks[i] = fb_stages[i];
   }

   simple_mtx_lock(&hs_mtx);
   t.pipelines++;
   if (base->type == TU_PIPELINE_GRAPHICS_LIB)
      mesa_logi("tu-hotswap: armed target=%s library=%p stage=%s base_bin=%.12s accepted=%u refused=%u",
                t.label.c_str(), (void *) base, hs_stage_name(st), base_bin, accepted, refused);
   else
      mesa_logi("tu-hotswap: armed target=%s pipeline=%p base_bin=%.12s accepted=%u refused=%u",
                t.label.c_str(), (void *) base, base_bin, accepted, refused);
   simple_mtx_unlock(&hs_mtx);
   base->hotswap = hp;
}

void
tu_hotswap_build_linked(struct tu_device *dev, struct tu_pipeline *base,
                        const struct tu_pipeline *lib, unsigned lib_index,
                        const VkAllocationCallbacks *alloc,
                        tu_hotswap_link_cb create, void *data)
{
   const struct tu_hotswap_pipe *lp = lib->hotswap;
   if (!lp || !base || base->hotswap)
      return;
   hs_target &t = hs.targets[lp->target];
   struct tu_hotswap_pipe *hp = (struct tu_hotswap_pipe *)
      vk_zalloc2(&dev->vk.alloc, alloc, sizeof(*hp), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!hp)
      return;
   hp->target = lp->target;
   hp->dxil = lp->dxil;
   hp->alias = lp->alias;
   mesa_shader_stage st = t.stage == MESA_SHADER_VERTEX ? MESA_SHADER_VERTEX :
                          MESA_SHADER_FRAGMENT;
   const char *base_bin = base->shaders[st] && base->shaders[st]->variant ?
                          base->shaders[st]->variant->blake3_str : "none";
   unsigned accepted = util_bitcount(hp->alias), refused = 0;
   for (unsigned k = 1; k < HS_MAXV; k++) {
      if (!lp->variant[k])
         continue;
      VkPipeline var = VK_NULL_HANDLE;
      VkResult r = create(data, lib_index, lp->variant[k], &var);
      std::string reason;
      if (r != VK_SUCCESS || !var) {
         reason = "linked variant creation failed (" + std::to_string((int) r) + ")";
         var = VK_NULL_HANDLE;
      } else {
         VK_FROM_HANDLE(tu_pipeline, vp, var);
         if (vp->type != TU_PIPELINE_GRAPHICS || !vp->shaders[st]) {
            reason = std::string("linked variant has no ") + hs_stage_name(st) + " shader";
            tu_DestroyPipeline(tu_device_to_handle(dev), var, alloc);
            var = VK_NULL_HANDLE;
         }
      }
      simple_mtx_lock(&hs_mtx);
      if (var) {
         VK_FROM_HANDLE(tu_pipeline, vp, var);
         hp->variant[k] = var;
         t.accepted[k]++;
         accepted++;
         mesa_logi("tu-hotswap: create target=%s stage=%s dxil=%016" PRIx64
                   " pipeline=%p base_bin=%.12s variant=%u name=%s bin=%.12s iface=linked library=%p",
                   t.label.c_str(), hs_stage_name(st), hp->dxil, (void *) base,
                   base_bin, k, t.alt[k].name.c_str(),
                   vp->shaders[st]->variant ? vp->shaders[st]->variant->blake3_str : "none",
                   (void *) lib);
      } else {
         refused++;
         hs_poison(lp->target, k);
         mesa_loge("tu-hotswap: refuse target=%s stage=%s dxil=%016" PRIx64
                   " pipeline=%p variant=%u name=%s reason=%s",
                   t.label.c_str(), hs_stage_name(st), hp->dxil, (void *) base, k,
                   t.alt[k].name.c_str(), reason.c_str());
      }
      simple_mtx_unlock(&hs_mtx);
   }
   simple_mtx_lock(&hs_mtx);
   t.pipelines++;
   mesa_logi("tu-hotswap: armed target=%s pipeline=%p base_bin=%.12s accepted=%u refused=%u linked_from=%p",
             t.label.c_str(), (void *) base, base_bin, accepted, refused, (void *) lib);
   simple_mtx_unlock(&hs_mtx);
   base->hotswap = hp;
}

static bool
hs_resolve(int ti, const char *val, unsigned *k)
{
   const hs_target &t = hs.targets[ti];
   char *end;
   long v = strtol(val, &end, 10);
   if (*val && !*end) {
      if (v == 0) {
         *k = 0;
         return true;
      }
      if (v < 1 || v >= HS_MAXV || !t.alt[v].defined)
         return false;
      *k = t.alt[v].same_as ? (unsigned) t.alt[v].same_as : (unsigned) v;
      return true;
   }
   if (!strcmp(val, "base")) {
      *k = 0;
      return true;
   }
   for (unsigned j = 1; j < HS_MAXV; j++)
      if (t.alt[j].defined && t.alt[j].name == val) {
         *k = t.alt[j].same_as ? (unsigned) t.alt[j].same_as : j;
         return true;
      }
   return false;
}

static const char *
hs_vname(const hs_target &t, unsigned k)
{
   return k ? t.alt[k].name.c_str() : "orig";
}

static void
hs_log_binds(const char *why, uint64_t key)
{
   for (hs_target &t : hs.targets) {
      char buf[512];
      int o = 0;
      for (unsigned k = 0; k < HS_MAXV && o < (int) sizeof(buf) - 32; k++) {
         uint64_t c = p_atomic_xchg(&t.binds[k], 0);
         if (c || k == 0)
            o += snprintf(buf + o, sizeof(buf) - o, "%sv%u=%" PRIu64, o ? "," : "", k, c);
      }
      uint64_t fbk = p_atomic_xchg(&t.fallback, 0);
      mesa_logi("tu-hotswap: binds %s frame=%" PRIu64 " frames=%" PRIu64
                " target=%s sel=%u pipelines=%u %s fallback=%" PRIu64,
                why, key, key - hs.switch_key, t.label.c_str(),
                (unsigned) hs.sel[&t - &hs.targets[0]], t.pipelines, buf, fbk);
   }
}

static void
hs_update(uint64_t key)
{
   char raw[HS_CTL_MAX] = { 0 };
   bool present = false;
   int fd = open(hs.ctl_path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
   if (fd >= 0) {
      ssize_t n = read(fd, raw, sizeof(raw) - 1);
      close(fd);
      present = true;
      raw[n > 0 ? n : 0] = 0;
   }
   if (hs.log >= 2 && hs.last_stat_key + 600 <= key) {
      hs.last_stat_key = key;
      hs_log_binds("periodic", key);
   }
   if (hs.ctl_seen && present == hs.ctl_present && !strcmp(raw, hs.ctl_raw) &&
       hs.poison_gen == hs.poison_seen)
      return;
   bool ctl_changed = !hs.ctl_seen || present != hs.ctl_present || strcmp(raw, hs.ctl_raw);
   hs.ctl_seen = true;
   hs.ctl_present = present;
   memcpy(hs.ctl_raw, raw, sizeof(raw));
   hs.poison_seen = hs.poison_gen;

   size_t nt = hs.targets.size();
   std::vector<unsigned> want(nt, 0);
   std::vector<int> setter(nt, -1);
   std::string seq;
   bool err = false;
   std::string bad;
   char work[HS_CTL_MAX];
   memcpy(work, raw, sizeof(work));
   for (char *save = NULL, *tok = strtok_r(work, " \t\r\n,;", &save); tok;
        tok = strtok_r(NULL, " \t\r\n,;", &save)) {
      if (tok[0] == '#')
         break;
      char *eq = strchr(tok, '=');
      if (!eq) {
         err = true;
         bad = tok;
         break;
      }
      *eq = 0;
      const char *k = tok, *v = eq + 1;
      if (!strcmp(k, "seq")) {
         seq.assign(v, strnlen(v, 63));
         continue;
      }
      if (!strcmp(k, "all")) {
         if (strcmp(v, "0") && strcmp(v, "base")) {
            err = true;
            bad = std::string(k) + "=" + v;
            break;
         }
         std::fill(want.begin(), want.end(), 0);
         std::fill(setter.begin(), setter.end(), -1);
         continue;
      }
      int ti = hs_find_target(k);
      if (ti >= 0) {
         unsigned idx;
         if (!hs_resolve(ti, v, &idx)) {
            err = true;
            bad = std::string(k) + "=" + v;
            break;
         }
         want[ti] = idx;
         setter[ti] = -1;
         continue;
      }
      int gi = hs_find_group(k);
      if (gi >= 0) {
         for (int m : hs.groups[gi].members) {
            unsigned idx;
            if (!hs_resolve(m, v, &idx)) {
               err = true;
               bad = std::string(k) + "=" + v + " (member " + hs.targets[m].label + ")";
               break;
            }
            want[m] = idx;
            setter[m] = gi;
         }
         if (err)
            break;
         continue;
      }
      err = true;
      bad = std::string(k) + "=" + v;
      break;
   }

   if (ctl_changed) {
      hs.ctl_changes++;
      char shown[256];
      size_t j = 0;
      for (size_t i = 0; raw[i] && j < sizeof(shown) - 1; i++)
         shown[j++] = (raw[i] == '\n' || raw[i] == '\r') ? ' ' : raw[i];
      shown[j] = 0;
      mesa_logi("tu-hotswap: ctl change=%u frame=%" PRIu64 " raw_ns=%" PRId64
                " present=%d raw='%s' parse_error=%d%s%s",
                hs.ctl_changes, key, hs_raw_ns(), present, shown, err,
                err ? " bad=" : "", err ? bad.c_str() : "");
   }
   if (err)
      return;

   for (size_t i = 0; i < nt; i++) {
      if (!want[i] || !(hs.targets[i].poisoned & (1u << want[i])))
         continue;
      if (setter[i] >= 0) {
         int g = setter[i];
         mesa_loge("tu-hotswap: group=%s variant=%s refused: member %s variant %u was refused at pipeline creation; group stays at orig",
                   hs.groups[g].name.c_str(), hs_vname(hs.targets[i], want[i]),
                   hs.targets[i].label.c_str(), want[i]);
         for (size_t j = 0; j < nt; j++)
            if (setter[j] == g)
               want[j] = 0;
      } else {
         mesa_loge("tu-hotswap: target=%s variant=%u refused at pipeline creation; stays at orig",
                   hs.targets[i].label.c_str(), want[i]);
         want[i] = 0;
      }
   }

   bool any = false;
   for (size_t i = 0; i < nt; i++)
      any |= want[i] != hs.sel[i];
   if (!any) {
      if (!seq.empty())
         hs.seq = seq;
      return;
   }
   hs_log_binds("before-switch", key);
   int64_t now = hs_raw_ns();
   for (size_t i = 0; i < nt; i++) {
      if (want[i] == hs.sel[i])
         continue;
      const hs_target &t = hs.targets[i];
      mesa_logi("tu-hotswap: switch frame=%" PRIu64 " raw_ns=%" PRId64
                " seq=%s target=%s group=%s %u(%s)->%u(%s)",
                key, now, seq.empty() ? "-" : seq.c_str(), t.label.c_str(),
                setter[i] >= 0 ? hs.groups[setter[i]].name.c_str() : "-",
                (unsigned) hs.sel[i], hs_vname(t, hs.sel[i]), want[i],
                hs_vname(t, want[i]));
      hs.sel[i] = (uint8_t) want[i];
      p_atomic_set(&hs_sel_atomic[i], want[i]);
   }
   hs.seq = seq;
   hs.switch_key = key;
}

static uint64_t
hs_key(struct tu_device *dev)
{
   if (hs.epoch_submit)
      return p_atomic_read(&dev->submit_count);
   return p_atomic_read(&dev->vk.current_frame);
}

struct tu_pipeline *
tu_hotswap_bind(struct tu_device *dev, struct tu_pipeline *base)
{
   struct tu_hotswap_pipe *hp = base->hotswap;
   uint64_t key = hs_key(dev);
   if (key != p_atomic_read(&hs_fast_key)) {
      simple_mtx_lock(&hs_mtx);
      if (key != hs.key) {
         hs.key = key;
         hs_update(key);
         p_atomic_set(&hs_fast_key, key);
      }
      simple_mtx_unlock(&hs_mtx);
   }
   hs_target &t = hs.targets[hp->target];
   unsigned k = p_atomic_read(&hs_sel_atomic[hp->target]);
   if (!k || (hp->alias & (1u << k))) {
      p_atomic_inc(&t.binds[k]);
      return base;
   }
   if (!hp->variant[k]) {
      p_atomic_inc(&t.fallback);
      return base;
   }
   p_atomic_inc(&t.binds[k]);
   VK_FROM_HANDLE(tu_pipeline, vp, hp->variant[k]);
   return vp;
}

void
tu_hotswap_destroy(struct tu_device *dev, struct tu_pipeline *base,
                   const VkAllocationCallbacks *alloc)
{
   struct tu_hotswap_pipe *hp = base->hotswap;
   if (!hp)
      return;
   base->hotswap = NULL;
   for (unsigned k = 1; k < HS_MAXV; k++)
      if (hp->variant[k])
         tu_DestroyPipeline(tu_device_to_handle(dev), hp->variant[k], alloc);
   vk_free2(&dev->vk.alloc, alloc, hp);
}

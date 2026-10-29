/*
 * Copyright © 2026 Jetha Chan
 * SPDX-License-Identifier: MIT
 */

#include "nir/nir_builder.h"
#include "nir/nir_constant_expressions.h"

#include "tu_shader.h"

#include "ir3/ir3_nir.h"
#include "util/os_misc.h"
#include "util/u_dynarray.h"

#define TU_HW_WAVE 64
#define TU_LANE_SHFL_MAX_DELTAS 4
#define TU_LANE_SHFL_CACHE 4
#define TU_LANE_SHFL_MAX_DEPTH 16
#define TU_LANE_SHFL_BUDGET 64

struct emu_state {
   unsigned size;
   unsigned log2;
   bool direct; /* TU_SG32_EMU=v2 layer L2 */
};

/* TU_SG32_EMU selects the emulated-wave32 (requiredSubgroupSize < 64)
 * lowering. "v1" (default) is the original lowering, unchanged. "v2" adds,
 * for compute shaders:
 *   L1  uniform-first elimination (before emulation, API-subgroup level),
 *   L4  native wave64 when no subgroup-size-observable op remains,
 *   L2  half-uniform direct broadcasts (no divergent-index shuffle loop),
 *   L2b ALU pushed through the L2 half select onto the two uniform halves.
 * TU_SG32_EMU_DISABLE=l1,l2,l2b,l4 disables individual layers. The setting
 * is folded into the cache UUID only when it differs from the default.
 */
static bool
sg32_disabled(const char *list, const char *layer)
{
   if (!list)
      return false;
   size_t n = strlen(layer);
   for (const char *p = list; *p;) {
      const char *e = strchr(p, ',');
      size_t len = e ? (size_t)(e - p) : strlen(p);
      if (len == n && !strncmp(p, layer, n))
         return true;
      if (!e)
         break;
      p = e + 1;
   }
   return false;
}

static const struct tu_sg32_cfg *
tu_sg32_cfg(void)
{
   static struct tu_sg32_cfg cfg;
   static bool init;
   if (!init) {
      const char *v = os_get_option("TU_SG32_EMU");
      const char *d = os_get_option("TU_SG32_EMU_DISABLE");
      cfg.v2 = v && !strcmp(v, "v2");
      cfg.l1 = cfg.v2 && !sg32_disabled(d, "l1");
      cfg.l1u = cfg.l1 && !sg32_disabled(d, "l1u");
      cfg.l2 = cfg.v2 && !sg32_disabled(d, "l2");
      cfg.l2b = cfg.v2 && !sg32_disabled(d, "l2b");
      cfg.l4 = cfg.v2 && !sg32_disabled(d, "l4");
      init = true;
   }
   return &cfg;
}

bool
tu_sg32_v2_enabled(void)
{
   return tu_sg32_cfg()->v2;
}

/* Normalised non-default setting for the cache UUID, or NULL for v1. */
const char *
tu_sg32_uuid_string(void)
{
   static char buf[64];
   const struct tu_sg32_cfg *c = tu_sg32_cfg();
   if (!c->v2)
      return NULL;
   snprintf(buf, sizeof(buf), "TU_SG32_EMU=v2:%d%d%d%d:l1u%d", c->l1, c->l2,
            c->l2b, c->l4, c->l1u);
   return buf;
}

static nir_def *
emu_lane(nir_builder *b)
{
   return nir_load_subgroup_invocation(b);
}

static nir_def *
emu_mask(nir_builder *b, const struct emu_state *s)
{
   return nir_imm_int(b, s->size == 32 ? ~0u : (1u << s->size) - 1);
}

static nir_def *
emu_base(nir_builder *b, const struct emu_state *s)
{
   return nir_iand_imm(b, emu_lane(b), ~(s->size - 1));
}

static nir_def *
emu_sub(nir_builder *b, const struct emu_state *s)
{
   return nir_iand_imm(b, emu_lane(b), s->size - 1);
}

static nir_def *
emu_cluster_bits(nir_builder *b, const struct emu_state *s, nir_def *cond)
{
   nir_def *hw = nir_ballot(b, 1, 64, cond);
   nir_def *shifted = nir_ushr(b, hw, emu_base(b, s));
   return nir_iand(b, nir_u2u32(b, shifted), emu_mask(b, s));
}

static nir_def *
emu_format_mask(nir_builder *b, nir_def *v32, unsigned comps, unsigned bit_size)
{
   nir_def *c[NIR_MAX_VEC_COMPONENTS];
   c[0] = bit_size == 64 ? nir_u2u64(b, v32) : v32;
   for (unsigned i = 1; i < comps; i++)
      c[i] = nir_imm_intN_t(b, 0, bit_size);
   return nir_vec(b, c, comps);
}

static nir_def *
emu_first_lane(nir_builder *b, const struct emu_state *s)
{
   return nir_find_lsb(b, emu_cluster_bits(b, s, nir_imm_true(b)));
}

static nir_def *
emu_shuffle_to(nir_builder *b, const struct emu_state *s, nir_def *x, nir_def *sub_idx)
{
   nir_def *idx = nir_iadd(b, emu_base(b, s), nir_iand_imm(b, sub_idx, s->size - 1));
   return nir_shuffle(b, x, idx);
}

/* Direct per-half helpers (TU_SG32_EMU=v2 L2, size 32 on the 64-lane wave).
 * The hardware ballot of all active lanes is wave-uniform, so the first
 * active lane of the upper half is a wave-uniform index. An emulated
 * subgroup is exactly one 32-lane half (emu_base), and only active lanes
 * consume results.
 */
static nir_def *
emu_upper_half(nir_builder *b)
{
   return nir_uge_imm(b, emu_lane(b), 32);
}

struct emu_halves {
   nir_def *lo_any;     /* lower half has an active lane */
   nir_def *hi_any;     /* upper half has an active lane */
   nir_def *first;      /* first active lane of the wave */
   nir_def *upper;      /* first active lane of the upper half, else first */
};

/* All values are wave-uniform (derived from ballot(true)) and always name
 * an active lane, so they are valid read_invocation indices.
 */
static struct emu_halves
emu_halves(nir_builder *b)
{
   nir_def *hw = nir_ballot(b, 1, 64, nir_imm_true(b));
   nir_def *lo = nir_unpack_64_2x32_split_x(b, hw);
   nir_def *hi = nir_unpack_64_2x32_split_y(b, hw);
   struct emu_halves h;
   h.lo_any = nir_ine_imm(b, lo, 0);
   h.hi_any = nir_ine_imm(b, hi, 0);
   nir_def *lo_first = nir_find_lsb(b, lo);
   nir_def *hi_first = nir_iadd_imm(b, nir_find_lsb(b, hi), 32);
   h.first = nir_bcsel(b, h.lo_any, lo_first, hi_first);
   h.upper = nir_bcsel(b, h.hi_any, hi_first, lo_first);
   return h;
}

/* readFirstInvocation of each emulated 32-lane subgroup. The hardware
 * read_first_invocation returns the first active lane of the wave: for a
 * lower-half lane that is its own half's first active lane (it is active,
 * so the lower half is non-empty); for an upper-half lane it is that only if
 * the lower half is empty, otherwise use the upper half's first lane.
 */
static nir_def *
emu_direct_read_first(nir_builder *b, nir_def *x)
{
   nir_def *first = nir_read_first_invocation(b, x);
   nir_def *upper = nir_read_invocation(b, x, emu_halves(b).upper);
   return nir_bcsel(b, emu_upper_half(b), upper, first);
}

/* readInvocation(x, idx) with idx uniform within each emulated subgroup
 * (it may differ between the halves). Fetch each half's idx with a uniform
 * index, then x from the selected lane of that half.
 */
static nir_def *
emu_direct_read_invocation(nir_builder *b, nir_def *x, nir_def *idx)
{
   struct emu_halves h = emu_halves(b);
   nir_def *lo_lane, *hi_lane;
   if (nir_src_is_const(nir_src_for_ssa(idx))) {
      uint32_t i = nir_src_as_uint(nir_src_for_ssa(idx)) & 31;
      lo_lane = nir_imm_int(b, i);
      hi_lane = nir_imm_int(b, 32 + i);
   } else {
      /* read_first_invocation reads the lower half whenever it has an
       * active lane; h.upper is an upper-half lane whenever that half has
       * one. A half with no active lane has no consumer.
       */
      nir_def *idx_lo = nir_read_first_invocation(b, idx);
      nir_def *idx_hi = nir_read_invocation(b, idx, h.upper);
      lo_lane = nir_iand_imm(b, idx_lo, 31);
      hi_lane = nir_iadd_imm(b, nir_iand_imm(b, idx_hi, 31), 32);
   }
   /* Keep both indices on active lanes even when a half is empty. */
   lo_lane = nir_bcsel(b, h.lo_any, lo_lane, h.first);
   hi_lane = nir_bcsel(b, h.hi_any, hi_lane, h.first);
   nir_def *lo = nir_read_invocation(b, x, lo_lane);
   nir_def *hi = nir_read_invocation(b, x, hi_lane);
   return nir_bcsel(b, emu_upper_half(b), hi, lo);
}

static nir_def *
emu_vote_all(nir_builder *b, const struct emu_state *s, nir_def *cond)
{
   return nir_ieq_imm(b, emu_cluster_bits(b, s, nir_inot(b, cond)), 0);
}

static nir_def *
emu_ballot_low32(nir_builder *b, nir_def *v)
{
   nir_def *x = nir_channel(b, v, 0);
   return x->bit_size == 64 ? nir_u2u32(b, x) : x;
}

static nir_def *
emu_per_cluster_scan(nir_builder *b, const struct emu_state *s,
                     nir_intrinsic_instr *intr)
{
   nir_def *x = intr->src[0].ssa;
   nir_op op = (nir_op) nir_intrinsic_reduction_op(intr);
   nir_def *identity =
      nir_build_imm(b, 1, x->bit_size,
                    (nir_const_value[]) { nir_alu_binop_identity(op, x->bit_size) });
   nir_def *ids[NIR_MAX_VEC_COMPONENTS];
   for (unsigned c = 0; c < x->num_components; c++)
      ids[c] = identity;
   nir_def *idv = nir_vec(b, ids, x->num_components);

   nir_def *cluster = nir_ushr_imm(b, emu_lane(b), s->log2);
   nir_def *result = NULL;
   for (unsigned k = 0; k < TU_HW_WAVE / s->size; k++) {
      nir_def *in_k = nir_ieq_imm(b, cluster, k);
      nir_def *xk = nir_bcsel(b, in_k, x, idv);
      nir_def *scan = intr->intrinsic == nir_intrinsic_inclusive_scan
                         ? nir_inclusive_scan(b, xk, .reduction_op = op)
                         : nir_exclusive_scan(b, xk, .reduction_op = op);
      result = result ? nir_bcsel(b, in_k, scan, result) : scan;
   }
   return result;
}

static bool
lower_emulated_subgroups(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct emu_state *s = (const struct emu_state *) data;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *rep = NULL;

   switch (intr->intrinsic) {
   case nir_intrinsic_load_subgroup_size:
      rep = nir_imm_intN_t(b, s->size, intr->def.bit_size);
      break;
   case nir_intrinsic_load_subgroup_invocation:
      rep = nir_u2uN(b, emu_sub(b, s), intr->def.bit_size);
      break;
   case nir_intrinsic_load_subgroup_id:
      rep = nir_u2uN(b, nir_ushr_imm(b, nir_load_local_invocation_index(b), s->log2),
                     intr->def.bit_size);
      break;
   case nir_intrinsic_load_num_subgroups: {
      nir_def *ws = nir_load_workgroup_size(b);
      nir_def *total = nir_imul(b, nir_imul(b, nir_channel(b, ws, 0), nir_channel(b, ws, 1)),
                                nir_channel(b, ws, 2));
      rep = nir_u2uN(b, nir_ushr_imm(b, nir_iadd_imm(b, total, s->size - 1), s->log2),
                     intr->def.bit_size);
      break;
   }
   case nir_intrinsic_load_subgroup_eq_mask:
   case nir_intrinsic_load_subgroup_ge_mask:
   case nir_intrinsic_load_subgroup_gt_mask:
   case nir_intrinsic_load_subgroup_le_mask:
   case nir_intrinsic_load_subgroup_lt_mask: {
      nir_def *eq = nir_ishl(b, nir_imm_int(b, 1), emu_sub(b, s));
      nir_def *lt = nir_iadd_imm(b, eq, -1);
      nir_def *m = emu_mask(b, s);
      nir_def *v;
      switch (intr->intrinsic) {
      case nir_intrinsic_load_subgroup_eq_mask: v = eq; break;
      case nir_intrinsic_load_subgroup_lt_mask: v = lt; break;
      case nir_intrinsic_load_subgroup_le_mask: v = nir_ior(b, lt, eq); break;
      case nir_intrinsic_load_subgroup_ge_mask: v = nir_iand(b, m, nir_inot(b, lt)); break;
      default: v = nir_iand(b, m, nir_inot(b, nir_ior(b, lt, eq))); break;
      }
      rep = emu_format_mask(b, v, intr->def.num_components, intr->def.bit_size);
      break;
   }
   case nir_intrinsic_ballot:
   case nir_intrinsic_ballot_relaxed:
      rep = emu_format_mask(b, emu_cluster_bits(b, s, intr->src[0].ssa),
                            intr->def.num_components, intr->def.bit_size);
      break;
   case nir_intrinsic_vote_any:
      rep = nir_ine_imm(b, emu_cluster_bits(b, s, intr->src[0].ssa), 0);
      break;
   case nir_intrinsic_vote_all:
      rep = emu_vote_all(b, s, intr->src[0].ssa);
      break;
   case nir_intrinsic_vote_ieq:
   case nir_intrinsic_vote_feq: {
      nir_def *x = intr->src[0].ssa;
      nir_def *first = s->direct ? emu_direct_read_first(b, x)
                                 : emu_shuffle_to(b, s, x, emu_first_lane(b, s));
      nir_def *eq = intr->intrinsic == nir_intrinsic_vote_feq ? nir_feq(b, x, first)
                                                              : nir_ieq(b, x, first);
      nir_def *all = nir_channel(b, eq, 0);
      for (unsigned c = 1; c < eq->num_components; c++)
         all = nir_iand(b, all, nir_channel(b, eq, c));
      rep = emu_vote_all(b, s, all);
      break;
   }
   case nir_intrinsic_first_invocation:
      rep = emu_first_lane(b, s);
      break;
   case nir_intrinsic_last_invocation:
      rep = nir_ufind_msb(b, emu_cluster_bits(b, s, nir_imm_true(b)));
      break;
   case nir_intrinsic_elect:
      rep = nir_ieq(b, emu_sub(b, s), emu_first_lane(b, s));
      break;
   case nir_intrinsic_read_first_invocation:
      if (s->direct)
         rep = emu_direct_read_first(b, intr->src[0].ssa);
      else
         rep = emu_shuffle_to(b, s, intr->src[0].ssa, emu_first_lane(b, s));
      break;
   case nir_intrinsic_read_invocation:
      if (s->direct) {
         rep = emu_direct_read_invocation(b, intr->src[0].ssa, intr->src[1].ssa);
         break;
      }
      FALLTHROUGH;
   case nir_intrinsic_shuffle:
      rep = emu_shuffle_to(b, s, intr->src[0].ssa, intr->src[1].ssa);
      break;
   case nir_intrinsic_shuffle_xor:
      rep = nir_shuffle(b, intr->src[0].ssa,
                        nir_ixor(b, emu_lane(b), nir_iand_imm(b, intr->src[1].ssa, s->size - 1)));
      break;
   case nir_intrinsic_rotate:
   case nir_intrinsic_reduce: {
      unsigned c = nir_intrinsic_cluster_size(intr);
      if (c == 0 || c > s->size) {
         nir_intrinsic_set_cluster_size(intr, s->size);
         return true;
      }
      return false;
   }
   case nir_intrinsic_inclusive_scan:
   case nir_intrinsic_exclusive_scan:
      rep = emu_per_cluster_scan(b, s, intr);
      break;
   case nir_intrinsic_ballot_bit_count_exclusive:
   case nir_intrinsic_ballot_bit_count_inclusive: {
      nir_def *eq = nir_ishl(b, nir_imm_int(b, 1), emu_sub(b, s));
      nir_def *m = nir_iadd_imm(b, eq, -1);
      if (intr->intrinsic == nir_intrinsic_ballot_bit_count_inclusive)
         m = nir_ior(b, m, eq);
      rep = nir_u2uN(b, nir_bit_count(b, nir_iand(b, emu_ballot_low32(b, intr->src[0].ssa), m)),
                     intr->def.bit_size);
      break;
   }
   case nir_intrinsic_inverse_ballot: {
      nir_def *v = emu_ballot_low32(b, intr->src[0].ssa);
      rep = nir_i2b(b, nir_iand_imm(b, nir_ushr(b, v, emu_sub(b, s)), 1));
      break;
   }
   default:
      return false;
   }

   nir_def_replace(&intr->def, rep);
   return true;
}

bool
tu_nir_lower_emulated_subgroups(nir_shader *shader, unsigned subgroup_size)
{
   assert(subgroup_size >= 4 && subgroup_size <= 32 &&
          util_is_power_of_two_nonzero(subgroup_size));
   struct emu_state s = { subgroup_size, util_logbase2(subgroup_size), false };
   bool p = nir_shader_intrinsics_pass(shader, lower_emulated_subgroups,
                                       nir_metadata_none, &s);
   shader->info.api_subgroup_size = TU_HW_WAVE;
   shader->info.min_subgroup_size = TU_HW_WAVE;
   shader->info.max_subgroup_size = TU_HW_WAVE;
   return p;
}

/* ---- TU_SG32_EMU=v2 ---------------------------------------------------- */

/* L1: exact identities for subgroup ops whose operand is uniform within the
 * API (emulated, 32-lane) subgroup. Runs before emulation on API-level NIR,
 * so NIR divergence analysis' "uniform" means uniform within the emulated
 * subgroup. Only exact (bit-identical) rewrites: no float reductions.
 */
static bool
sg32_l1_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   unsigned *count = (unsigned *) data;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *rep = NULL;

   switch (intr->intrinsic) {
   case nir_intrinsic_read_first_invocation:
   case nir_intrinsic_read_invocation:
   case nir_intrinsic_shuffle:
   case nir_intrinsic_shuffle_xor:
   case nir_intrinsic_shuffle_up:
   case nir_intrinsic_shuffle_down:
   case nir_intrinsic_rotate:
   case nir_intrinsic_vote_any:
   case nir_intrinsic_vote_all:
      if (nir_src_is_divergent(&intr->src[0]))
         return false;
      rep = intr->src[0].ssa;
      break;
   case nir_intrinsic_vote_ieq:
      if (nir_src_is_divergent(&intr->src[0]))
         return false;
      rep = nir_imm_true(b);
      break;
   case nir_intrinsic_vote_feq: {
      if (nir_src_is_divergent(&intr->src[0]))
         return false;
      nir_def *x = intr->src[0].ssa;
      b->fp_math_ctrl = nir_fp_preserve_nan | nir_fp_preserve_inf;
      nir_def *eq = nir_feq(b, x, x);
      b->fp_math_ctrl = nir_fp_fast_math;
      rep = nir_channel(b, eq, 0);
      for (unsigned c = 1; c < eq->num_components; c++)
         rep = nir_iand(b, rep, nir_channel(b, eq, c));
      break;
   }
   case nir_intrinsic_reduce:
   case nir_intrinsic_inclusive_scan:
   case nir_intrinsic_exclusive_scan: {
      if (nir_src_is_divergent(&intr->src[0]))
         return false;
      if (nir_intrinsic_has_cluster_size(intr) &&
          nir_intrinsic_cluster_size(intr) != 0)
         return false;
      nir_op op = (nir_op) nir_intrinsic_reduction_op(intr);
      nir_def *x = intr->src[0].ssa;
      switch (op) {
      case nir_op_imin: case nir_op_umin: case nir_op_fmin:
      case nir_op_imax: case nir_op_umax: case nir_op_fmax:
      case nir_op_iand: case nir_op_ior:
         if (intr->intrinsic == nir_intrinsic_exclusive_scan)
            return false;
         rep = x;
         break;
      case nir_op_iadd:
      case nir_op_ixor: {
         nir_def *ballot = nir_ballot(b, 1, 32, nir_imm_true(b));
         if (intr->intrinsic == nir_intrinsic_inclusive_scan)
            ballot = nir_iand(b, ballot, nir_load_subgroup_le_mask(b, 1, 32));
         else if (intr->intrinsic == nir_intrinsic_exclusive_scan)
            ballot = nir_iand(b, ballot, nir_load_subgroup_lt_mask(b, 1, 32));
         nir_def *cnt = nir_bit_count(b, ballot);
         if (op == nir_op_ixor)
            cnt = nir_iand_imm(b, cnt, 1);
         rep = nir_imul(b, x, nir_u2uN(b, cnt, x->bit_size));
         break;
      }
      default:
         return false;
      }
      break;
   }
   default:
      return false;
   }

   nir_def_replace(&intr->def, rep);
   (*count)++;
   return true;
}

/* L4: an op whose result can depend on the subgroup size or on which lanes
 * share a subgroup. Conservative: any subgroup-flavoured intrinsic.
 */
static const char *
sg32_observable(nir_intrinsic_instr *intr)
{
   const char *n = nir_intrinsic_infos[intr->intrinsic].name;
   if (!strncmp(n, "load_local_invocation", 21) ||
       !strncmp(n, "load_global_invocation", 22) ||
       !strncmp(n, "load_base_global_invocation", 27))
      return NULL;
   static const char *const pat[] = {
      "subgroup", "ballot", "vote", "shuffle", "quad", "scan", "reduce",
      "invocation", "elect", "rotate", "shfl", "brcst", "swizzle", "lane",
   };
   for (unsigned i = 0; i < ARRAY_SIZE(pat); i++) {
      if (strstr(n, pat[i]))
         return n;
   }
   return NULL;
}

static const char *
sg32_first_observable(nir_shader *nir)
{
   nir_foreach_function_impl (impl, nir) {
      nir_foreach_block (block, impl) {
         nir_foreach_instr (instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            const char *n = sg32_observable(nir_instr_as_intrinsic(instr));
            if (n)
               return n;
         }
      }
   }
   return NULL;
}

static unsigned
sg32_count_direct(nir_shader *nir)
{
   unsigned n = 0;
   nir_foreach_function_impl (impl, nir) {
      nir_foreach_block (block, impl) {
         nir_foreach_instr (instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            switch (nir_instr_as_intrinsic(instr)->intrinsic) {
            case nir_intrinsic_read_first_invocation:
            case nir_intrinsic_read_invocation:
            case nir_intrinsic_vote_ieq:
            case nir_intrinsic_vote_feq:
               n++;
               break;
            default:
               break;
            }
         }
      }
   }
   return n;
}

/* L1u: ir3 runs divergence analysis with nir_divergence_uniform_load_tears,
 * so at L1 time (before explicit-io lowering) every UBO load_deref without
 * ACCESS_NON_WRITEABLE is treated as divergent, and L1 cannot see that
 * UBO-derived operands (e.g. vote_all(cbuffer flag)) are uniform. UBO memory
 * cannot be written by shader invocations, and ir3 itself treats the lowered
 * load_ubo as uniform whenever its block index and offset are. Mark UBO
 * load_derefs non-writeable for the L1 divergence analysis only, and restore
 * the original access afterwards so later lowering and codegen see exactly
 * the instructions they saw before.
 */
static unsigned
sg32_l1u_mark(nir_shader *nir, struct util_dynarray *marked)
{
   unsigned n = 0;
   nir_foreach_function_impl (impl, nir) {
      nir_foreach_block (block, impl) {
         nir_foreach_instr (instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_deref)
               continue;
            nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
            if (!nir_deref_mode_is(deref, nir_var_mem_ubo))
               continue;
            enum gl_access_qualifier access = nir_intrinsic_access(intr);
            if (access & (ACCESS_NON_WRITEABLE | ACCESS_VOLATILE | ACCESS_ATOMIC))
               continue;
            nir_intrinsic_set_access(intr, (enum gl_access_qualifier)
                                     (access | ACCESS_NON_WRITEABLE));
            util_dynarray_append_typed(marked, nir_intrinsic_instr *, intr);
            n++;
         }
      }
   }
   return n;
}

static void
sg32_l1u_restore(struct util_dynarray *marked)
{
   util_dynarray_foreach (marked, nir_intrinsic_instr *, p) {
      nir_intrinsic_instr *intr = *p;
      nir_intrinsic_set_access(intr, (enum gl_access_qualifier)
                               (nir_intrinsic_access(intr) &
                                ~ACCESS_NON_WRITEABLE));
   }
   util_dynarray_clear(marked);
}

bool
tu_nir_sg32_v2(nir_shader *nir, struct ir3_compiler *compiler,
               unsigned subgroup_size, struct tu_sg32_report *rep)
{
   const struct tu_sg32_cfg *cfg = tu_sg32_cfg();
   memset(rep, 0, sizeof(*rep));
   rep->active = true;

   if (cfg->l1 || cfg->l4) {
      /* Expose SSA values to divergence analysis (vars to SSA, copy-prop). */
      struct ir3_optimize_options o = {};
      ir3_optimize_loop(compiler, &o, nir);
   }

   if (cfg->l1) {
      struct util_dynarray marked;
      util_dynarray_init(&marked, NULL);
      if (cfg->l1u) {
         rep->l1u = sg32_l1u_mark(nir, &marked);
         /* Divergence must be recomputed with the marked access. */
         nir_metadata_invalidate(nir);
      }
      nir_divergence_analysis(nir);
      nir_shader_intrinsics_pass(nir, sg32_l1_instr, nir_metadata_control_flow,
                                 &rep->l1);
      if (cfg->l1u) {
         sg32_l1u_restore(&marked);
         nir_metadata_invalidate(nir);
      }
      util_dynarray_fini(&marked);
      if (rep->l1)
         nir_opt_dce(nir);
   }

   const char *obs = sg32_first_observable(nir);
   if (cfg->l4 && !obs) {
      /* Nothing can observe the 32-lane subgroup: compile as the native
       * wave64 shader (same as the emulation pass leaves info).
       */
      nir->info.api_subgroup_size = TU_HW_WAVE;
      nir->info.min_subgroup_size = TU_HW_WAVE;
      nir->info.max_subgroup_size = TU_HW_WAVE;
      rep->native = true;
      rep->reason = "no-observable-op";
      return rep->l1 != 0;
   }
   rep->reason = obs ? obs : "l4-disabled";

   bool direct = cfg->l2 && subgroup_size == 32;
   if (direct)
      rep->l2 = sg32_count_direct(nir);
   struct emu_state s = { subgroup_size, util_logbase2(subgroup_size), direct };
   nir_shader_intrinsics_pass(nir, lower_emulated_subgroups, nir_metadata_none, &s);
   nir->info.api_subgroup_size = TU_HW_WAVE;
   nir->info.min_subgroup_size = TU_HW_WAVE;
   nir->info.max_subgroup_size = TU_HW_WAVE;
   return true;
}

/* L2b: push pure ALU through the L2 half select. For a select
 * H = bcsel(lane >= 32, hi, lo) with uniform hi/lo, and an ALU A whose other
 * sources are uniform, A(.., H, ..) == bcsel(lane >= 32, A(.., hi, ..),
 * A(.., lo, ..)) exactly (A is a pure per-lane function), and both arms are
 * uniform, so they can stay scalar. Repeats along chains.
 */
static bool
sg32_is_upper_pred(nir_def *d)
{
   nir_alu_instr *a = nir_def_as_alu_or_null(d);
   if (!a || a->op != nir_op_uge || a->def.num_components != 1)
      return false;
   nir_scalar l = nir_scalar_chase_movs(nir_get_scalar(a->src[0].src.ssa, a->src[0].swizzle[0]));
   nir_scalar c = nir_scalar_chase_movs(nir_get_scalar(a->src[1].src.ssa, a->src[1].swizzle[0]));
   return nir_scalar_is_intrinsic(l) &&
          nir_scalar_intrinsic_op(l) == nir_intrinsic_load_subgroup_invocation &&
          nir_scalar_is_const(c) && nir_scalar_as_uint(c) == 32;
}

static nir_alu_instr *
sg32_half_select(nir_def *d)
{
   nir_alu_instr *a = nir_def_as_alu_or_null(d);
   if (!a || a->op != nir_op_bcsel || a->def.num_components != 1 ||
       a->src[0].swizzle[0] != 0 || !sg32_is_upper_pred(a->src[0].src.ssa) ||
       a->src[1].src.ssa->divergent || a->src[2].src.ssa->divergent)
      return NULL;
   return a;
}

#define TU_SG32_L2B_MAX 4096

/* phi(H_0, .., H_n) of half selects -> bcsel(P, phi(hi_i), phi(lo_i)). */
static bool
sg32_l2b_phi(nir_shader *nir, nir_phi_instr *phi)
{
   if (phi->def.num_components != 1)
      return false;
   /* Profitability: the new per-half phis are only wave-uniform when the
    * merge follows an if with a wave-uniform condition (not a loop header,
    * not a branch the two halves can take differently).
    */
   nir_cf_node *prev = nir_cf_node_prev(&phi->instr.block->cf_node);
   if (!prev || prev->type != nir_cf_node_if ||
       nir_cf_node_as_if(prev)->condition.ssa->divergent)
      return false;
   nir_alu_instr *first = NULL;
   nir_foreach_phi_src (src, phi) {
      nir_alu_instr *h = sg32_half_select(src->src.ssa);
      if (!h)
         return false;
      for (unsigned k = 1; k <= 2; k++) {
         if (h->src[k].swizzle[0] != 0 || h->src[k].src.ssa->num_components != 1)
            return false;
      }
      if (!first)
         first = h;
   }
   if (!first)
      return false;

   nir_def *arm[2];
   for (unsigned k = 0; k < 2; k++) {
      nir_phi_instr *n = nir_phi_instr_create(nir);
      nir_def_init(&n->instr, &n->def, 1, phi->def.bit_size);
      nir_foreach_phi_src (src, phi) {
         nir_alu_instr *h = sg32_half_select(src->src.ssa);
         nir_phi_instr_add_src(n, src->pred, h->src[1 + k].src.ssa);
      }
      nir_instr_insert_before(&phi->instr, &n->instr);
      arm[k] = &n->def;
   }
   /* The selects' predicates live in the predecessors; rebuild it here. */
   nir_builder b = nir_builder_at(nir_after_phis(phi->instr.block));
   nir_def *pred = nir_uge_imm(&b, nir_load_subgroup_invocation(&b), 32);
   nir_def *r = nir_bcsel(&b, pred, arm[0], arm[1]);
   nir_def_replace(&phi->def, r);
   return true;
}

unsigned
tu_nir_sg32_l2b(nir_shader *nir)
{
   if (!tu_sg32_cfg()->l2b)
      return 0;
   unsigned pushed = 0;
   nir_foreach_function_impl (impl, nir) {
      bool progress;
      do {
         progress = false;
         nir_metadata_require(impl, nir_metadata_block_index);
         nir_divergence_analysis_impl(impl, nir->options->divergence_analysis_options);
         nir_foreach_block (block, impl) {
            nir_foreach_phi_safe (phi, block) {
               if (pushed < TU_SG32_L2B_MAX && sg32_l2b_phi(nir, phi)) {
                  pushed++;
                  progress = true;
               }
            }
            nir_foreach_instr_safe (instr, block) {
               if (instr->type != nir_instr_type_alu || pushed >= TU_SG32_L2B_MAX)
                  continue;
               nir_alu_instr *alu = nir_instr_as_alu(instr);
               if (alu->def.num_components != 1 || sg32_half_select(&alu->def))
                  continue;
               const nir_op_info *info = &nir_op_infos[alu->op];
               nir_alu_instr *sel = NULL;
               bool ok = true;
               for (unsigned i = 0; i < info->num_inputs && ok; i++) {
                  nir_def *s = alu->src[i].src.ssa;
                  nir_alu_instr *h = sg32_half_select(s);
                  if (h) {
                     if (!sel)
                        sel = h;
                  } else if (s->divergent) {
                     ok = false;
                  }
               }
               if (!ok || !sel)
                  continue;

               nir_builder b = nir_builder_at(nir_before_instr(instr));
               nir_def *arm[2];
               for (unsigned k = 0; k < 2; k++) {
                  nir_alu_instr *c = nir_alu_instr_create(nir, alu->op);
                  c->no_signed_wrap = alu->no_signed_wrap;
                  c->no_unsigned_wrap = alu->no_unsigned_wrap;
                  c->fp_math_ctrl = alu->fp_math_ctrl;
                  for (unsigned i = 0; i < info->num_inputs; i++) {
                     nir_alu_instr *h = sg32_half_select(alu->src[i].src.ssa);
                     if (!h) {
                        nir_alu_src_copy(&c->src[i], &alu->src[i]);
                        continue;
                     }
                     /* src[1] = hi (lane >= 32), src[2] = lo. h is scalar. */
                     nir_alu_src *as = &h->src[1 + k];
                     c->src[i].src = nir_src_for_ssa(as->src.ssa);
                     for (unsigned j = 0; j < NIR_MAX_VEC_COMPONENTS; j++)
                        c->src[i].swizzle[j] = as->swizzle[0];
                  }
                  nir_def_init(&c->instr, &c->def, 1, alu->def.bit_size);
                  nir_builder_instr_insert(&b, &c->instr);
                  arm[k] = &c->def;
               }
               nir_def *r = nir_bcsel(&b, sel->src[0].src.ssa, arm[0], arm[1]);
               nir_def_replace(&alu->def, r);
               pushed++;
               progress = true;
            }
         }
      } while (progress && pushed < TU_SG32_L2B_MAX);
      nir_progress(pushed != 0, impl, nir_metadata_control_flow);
   }
   return pushed;
}

struct lane_shfl_class {
   nir_def *idx;
   bool ok;
   bool is_xor;
   uint32_t xor_mask;
   unsigned num_deltas;
   int32_t deltas[TU_LANE_SHFL_MAX_DELTAS];
};

struct lane_shfl_state {
   unsigned lanes;
   unsigned exec_mode;
   unsigned next;
   struct lane_shfl_class cache[TU_LANE_SHFL_CACHE];
};

static bool
lane_eval(const struct lane_shfl_state *st, nir_scalar s, uint32_t lane,
          unsigned depth, unsigned *budget, uint64_t *out)
{
   if (depth > TU_LANE_SHFL_MAX_DEPTH || *budget == 0)
      return false;
   (*budget)--;

   s = nir_scalar_chase_movs(s);

   if (nir_scalar_is_const(s)) {
      *out = nir_scalar_as_uint(s);
      return true;
   }

   if (nir_scalar_is_intrinsic(s)) {
      if (nir_scalar_intrinsic_op(s) != nir_intrinsic_load_subgroup_invocation)
         return false;
      *out = lane & BITFIELD64_MASK(s.def->bit_size);
      return true;
   }

   if (!nir_scalar_is_alu(s))
      return false;

   nir_alu_instr *alu = nir_scalar_as_alu(s);
   const nir_op_info *info = &nir_op_infos[alu->op];
   if (info->output_size != 0 ||
       nir_alu_type_get_base_type(info->output_type) == nir_type_float)
      return false;

   unsigned bit_size = 0;
   if (!nir_alu_type_get_type_size(info->output_type))
      bit_size = alu->def.bit_size;

   nir_const_value src[NIR_ALU_MAX_INPUTS];
   nir_const_value *srcs[NIR_ALU_MAX_INPUTS];
   for (unsigned i = 0; i < info->num_inputs; i++) {
      if (info->input_sizes[i] != 0 ||
          nir_alu_type_get_base_type(info->input_types[i]) == nir_type_float)
         return false;
      if (bit_size == 0 && !nir_alu_type_get_type_size(info->input_types[i]))
         bit_size = alu->src[i].src.ssa->bit_size;
      nir_scalar c = nir_scalar_chase_alu_src(s, i);
      uint64_t v;
      if (!lane_eval(st, c, lane, depth + 1, budget, &v))
         return false;
      src[i] = nir_const_value_for_raw_uint(v, c.def->bit_size);
      srcs[i] = &src[i];
   }

   if (bit_size == 0)
      bit_size = 32;

   nir_const_value dst[NIR_MAX_VEC_COMPONENTS];
   memset(dst, 0, sizeof(dst));
   nir_eval_const_opcode(alu->op, dst, NULL, 1, bit_size, srcs, st->exec_mode);
   *out = nir_const_value_as_uint(dst[0], alu->def.bit_size);
   return true;
}

static const struct lane_shfl_class *
lane_shfl_classify(struct lane_shfl_state *st, nir_def *idx)
{
   for (unsigned i = 0; i < TU_LANE_SHFL_CACHE; i++) {
      if (st->cache[i].idx == idx)
         return &st->cache[i];
   }

   struct lane_shfl_class *c = &st->cache[st->next];
   st->next = (st->next + 1) % TU_LANE_SHFL_CACHE;
   memset(c, 0, sizeof(*c));
   c->idx = idx;

   if (idx->num_components != 1 || idx->bit_size != 32)
      return c;

   nir_scalar s = nir_get_scalar(idx, 0);
   bool any = false, xor_ok = true, too_many = false;
   uint32_t mask = 0;
   for (uint32_t lane = 0; lane < st->lanes; lane++) {
      unsigned budget = TU_LANE_SHFL_BUDGET;
      uint64_t v;
      if (!lane_eval(st, s, lane, 0, &budget, &v))
         return c;

      uint32_t src = (uint32_t) v;
      if (src >= st->lanes)
         continue;

      if (!any) {
         mask = src ^ lane;
         any = true;
      } else if ((src ^ lane) != mask) {
         xor_ok = false;
      }

      int32_t d = (int32_t) src - (int32_t) lane;
      bool found = false;
      for (unsigned k = 0; k < c->num_deltas; k++)
         found |= c->deltas[k] == d;
      if (!found) {
         if (c->num_deltas < TU_LANE_SHFL_MAX_DELTAS)
            c->deltas[c->num_deltas++] = d;
         else
            too_many = true;
      }
   }

   if (!any)
      return c;

   if (xor_ok) {
      c->ok = true;
      c->is_xor = true;
      c->xor_mask = mask;
   } else if (!too_many) {
      c->ok = true;
   }

   return c;
}

static nir_def *
lane_shfl_by(nir_builder *b, nir_def *x, int32_t d)
{
   if (d == 0)
      return x;
   if (d > 0)
      return nir_shuffle_down(b, x, nir_imm_int(b, d));
   return nir_shuffle_up(b, x, nir_imm_int(b, -d));
}

static bool
opt_lane_shuffle(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct lane_shfl_state *st = (struct lane_shfl_state *) data;

   if (intr->intrinsic != nir_intrinsic_shuffle)
      return false;

   nir_def *x = intr->src[0].ssa;
   nir_def *idx = intr->src[1].ssa;
   if (x->bit_size != 16 && x->bit_size != 32)
      return false;

   const struct lane_shfl_class *c = lane_shfl_classify(st, idx);
   if (!c->ok)
      return false;

   b->cursor = nir_before_instr(&intr->instr);

   nir_def *rep;
   if (c->is_xor) {
      rep = c->xor_mask ? nir_shuffle_xor(b, x, nir_imm_int(b, c->xor_mask)) : x;
   } else {
      rep = lane_shfl_by(b, x, c->deltas[0]);
      if (c->num_deltas > 1) {
         nir_def *delta = nir_isub(b, idx, nir_load_subgroup_invocation(b));
         for (unsigned k = 1; k < c->num_deltas; k++) {
            rep = nir_bcsel(b, nir_ieq_imm(b, delta, (uint32_t) c->deltas[k]),
                            lane_shfl_by(b, x, c->deltas[k]), rep);
         }
      }
   }

   nir_def_replace(&intr->def, rep);
   return true;
}

bool
tu_nir_opt_lane_shuffles(nir_shader *shader, unsigned lanes)
{
   assert(util_is_power_of_two_nonzero(lanes) && lanes <= 128);
   struct lane_shfl_state st;
   memset(&st, 0, sizeof(st));
   st.lanes = lanes;
   st.exec_mode = shader->info.float_controls_execution_mode;
   return nir_shader_intrinsics_pass(shader, opt_lane_shuffle,
                                     nir_metadata_control_flow, &st);
}

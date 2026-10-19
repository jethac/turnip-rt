/*
 * Copyright © 2026 Jetha Chan
 * SPDX-License-Identifier: MIT
 */

#include "nir/nir_builder.h"
#include "nir/nir_constant_expressions.h"

#include "tu_shader.h"

#define TU_HW_WAVE 64
#define TU_LANE_SHFL_MAX_DELTAS 4
#define TU_LANE_SHFL_CACHE 4
#define TU_LANE_SHFL_MAX_DEPTH 16
#define TU_LANE_SHFL_BUDGET 64

struct emu_state {
   unsigned size;
   unsigned log2;
};

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
      nir_def *first = emu_shuffle_to(b, s, x, emu_first_lane(b, s));
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
      rep = emu_shuffle_to(b, s, intr->src[0].ssa, emu_first_lane(b, s));
      break;
   case nir_intrinsic_read_invocation:
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
   struct emu_state s = { subgroup_size, util_logbase2(subgroup_size) };
   bool p = nir_shader_intrinsics_pass(shader, lower_emulated_subgroups,
                                       nir_metadata_none, &s);
   shader->info.api_subgroup_size = TU_HW_WAVE;
   shader->info.min_subgroup_size = TU_HW_WAVE;
   shader->info.max_subgroup_size = TU_HW_WAVE;
   return p;
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

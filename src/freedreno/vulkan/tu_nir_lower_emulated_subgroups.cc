/*
 * Copyright © 2026 Jetha Chan
 * SPDX-License-Identifier: MIT
 */

#include "nir/nir_builder.h"

#include "tu_shader.h"

#define TU_HW_WAVE 64

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

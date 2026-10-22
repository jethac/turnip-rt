/*
 * Copyright © 2026 Jetha Chan
 * SPDX-License-Identifier: MIT
 */

#include "tu_rt_funnel.h"

#include "nir_builder.h"
#include "nir_control_flow.h"
extern "C" {
#include "nir_phi_builder.h"
}
#include "util/hash_table.h"

#include <map>
#include <set>
#include <vector>

#define TU_RT_FUNNEL_ARGS 10
#define TU_RT_FUNNEL_MAX_SITES 64

struct tu_rt_funnel_class {
   const struct glsl_type *ptype;
   unsigned first_site;
   uint32_t flags_known_mask;
   uint32_t flags_known_val;
   unsigned sites;
   bool seen;
   bool arg_const[TU_RT_FUNNEL_ARGS];
   nir_const_value arg_val[TU_RT_FUNNEL_ARGS][NIR_MAX_VEC_COMPONENTS];
   unsigned arg_comps[TU_RT_FUNNEL_ARGS];
   unsigned arg_bits[TU_RT_FUNNEL_ARGS];
   nir_variable *payload;
   nir_variable *arg_var[TU_RT_FUNNEL_ARGS];
   bool as_desc;
   int as_idx[2][NIR_INTRINSIC_MAX_CONST_INDEX];
   unsigned as_shape[2][3];
};

struct tu_rt_funnel_item {
   nir_instr *first;
   nir_instr *last;
   nir_cf_node *node;
   uint64_t own;
};

struct tu_rt_funnel_run {
   struct tu_rt_funnel_item begin;
   struct tu_rt_funnel_item end;
   uint64_t own;
};

struct tu_rt_funnel_state {
   nir_shader *s;
   struct tu_rt_funnel_class *classes;
   unsigned nclasses;
   unsigned nsites;
   bool slots_in_regs;
   nir_variable *state_var;
   nir_variable *cls_var;
   struct hash_table_u64 *slots;
   unsigned done;
   unsigned yield_base;
   unsigned resume_base;
   std::map<nir_cf_node *, uint64_t> masks;
   std::vector<std::pair<nir_if *, std::pair<uint64_t, uint64_t>>> forced;
   std::vector<struct tu_rt_funnel_run> runs;
   std::vector<nir_loop *> loops;
   nir_intrinsic_instr *resume[TU_RT_FUNNEL_MAX_SITES];
   const char *fail;
};

static unsigned
tu_rt_funnel_count(nir_shader *s)
{
   unsigned n = 0;
   nir_foreach_function_impl (impl, s)
      nir_foreach_block (block, impl)
         nir_foreach_instr (instr, block)
            n++;
   return n;
}

static const struct glsl_type *
tu_rt_funnel_type(unsigned bits, unsigned comps)
{
   enum glsl_base_type base;
   switch (bits) {
   case 1: base = GLSL_TYPE_BOOL; break;
   case 8: base = GLSL_TYPE_UINT8; break;
   case 16: base = GLSL_TYPE_UINT16; break;
   case 64: base = GLSL_TYPE_UINT64; break;
   default: base = GLSL_TYPE_UINT; break;
   }
   return glsl_vector_type(base, comps);
}

static bool
tu_rt_funnel_const_eq(nir_src src, const nir_const_value *vals, unsigned comps,
                      unsigned bits)
{
   if (!nir_src_is_const(src) || src.ssa->num_components != comps ||
       src.ssa->bit_size != bits)
      return false;
   const nir_const_value *v = nir_src_as_const_value(src);
   for (unsigned c = 0; c < comps; c++) {
      if (nir_const_value_as_uint(v[c], bits) != nir_const_value_as_uint(vals[c], bits))
         return false;
   }
   return true;
}

static void
tu_rt_funnel_known_bits(nir_scalar s, uint32_t *mask, uint32_t *val, unsigned depth)
{
   *mask = 0;
   *val = 0;
   if (s.def->bit_size != 32)
      return;
   if (nir_scalar_is_const(s)) {
      *mask = ~0u;
      *val = nir_scalar_as_uint(s);
      return;
   }
   if (!depth || !nir_scalar_is_alu(s))
      return;
   nir_op op = nir_scalar_alu_op(s);
   uint32_t ma, va, mb, vb;
   switch (op) {
   case nir_op_mov:
      tu_rt_funnel_known_bits(nir_scalar_chase_alu_src(s, 0), mask, val, depth - 1);
      return;
   case nir_op_ior:
   case nir_op_iand: {
      tu_rt_funnel_known_bits(nir_scalar_chase_alu_src(s, 0), &ma, &va, depth - 1);
      tu_rt_funnel_known_bits(nir_scalar_chase_alu_src(s, 1), &mb, &vb, depth - 1);
      uint32_t ones, zeros;
      if (op == nir_op_ior) {
         ones = (ma & va) | (mb & vb);
         zeros = (ma & ~va) & (mb & ~vb);
      } else {
         ones = (ma & va) & (mb & vb);
         zeros = (ma & ~va) | (mb & ~vb);
      }
      *mask = ones | zeros;
      *val = ones;
      return;
   }
   case nir_op_bcsel:
      tu_rt_funnel_known_bits(nir_scalar_chase_alu_src(s, 1), &ma, &va, depth - 1);
      tu_rt_funnel_known_bits(nir_scalar_chase_alu_src(s, 2), &mb, &vb, depth - 1);
      *mask = ma & mb & ~(va ^ vb);
      *val = va & *mask;
      return;
   default:
      return;
   }
}

static nir_variable *
tu_rt_funnel_slot(struct tu_rt_funnel_state *st, unsigned offset, unsigned bits)
{
   uint64_t key = ((uint64_t) offset << 8) | bits;
   nir_variable *var = (nir_variable *) _mesa_hash_table_u64_search(st->slots, key);
   if (!var) {
      var = nir_variable_create(st->s, nir_var_shader_temp,
                                tu_rt_funnel_type(bits, 1), "tu_rt_funnel_slot");
      _mesa_hash_table_u64_insert(st->slots, key, var);
   }
   return var;
}

static unsigned
tu_rt_funnel_scratch_offset(nir_intrinsic_instr *intr, bool *ok)
{
   nir_src *off = &intr->src[intr->intrinsic == nir_intrinsic_store_scratch ? 1 : 0];
   if (!nir_src_is_const(*off)) {
      *ok = false;
      return 0;
   }
   return nir_src_as_uint(*off);
}

static nir_def *
tu_rt_funnel_in(nir_builder *b, struct tu_rt_funnel_state *st, nir_def *s, unsigned base,
                uint64_t mask)
{
   nir_def *r = NULL;
   u_foreach_bit64 (j, mask) {
      nir_def *e = nir_ieq_imm(b, s, base + j);
      r = r ? nir_ior(b, r, e) : e;
   }
   return r ? r : nir_imm_false(b);
}

static uint64_t
tu_rt_funnel_list_mask(struct tu_rt_funnel_state *st, struct exec_list *list)
{
   uint64_t m = 0;
   foreach_list_typed (nir_cf_node, node, node, list) {
      auto it = st->masks.find(node);
      if (it != st->masks.end())
         m |= it->second;
   }
   return m;
}

static uint64_t
tu_rt_funnel_mask(struct tu_rt_funnel_state *st, nir_cf_node *node)
{
   auto it = st->masks.find(node);
   return it == st->masks.end() ? 0 : it->second;
}

static void
tu_rt_funnel_plan_list(struct tu_rt_funnel_state *st, struct exec_list *list)
{
   std::vector<struct tu_rt_funnel_item> items;
   foreach_list_typed (nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_block) {
         nir_block *block = nir_cf_node_as_block(node);
         struct tu_rt_funnel_item it = {};
         nir_foreach_instr (instr, block) {
            if (instr->type == nir_instr_type_phi)
               continue;
            if (!it.first && (instr->type == nir_instr_type_load_const ||
                              instr->type == nir_instr_type_undef))
               continue;
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_rt_resume) {
               if (it.first)
                  items.push_back(it);
               it = {};
               it.own = BITFIELD64_BIT(nir_intrinsic_call_idx(nir_instr_as_intrinsic(instr)));
            }
            if (!it.first)
               it.first = instr;
            it.last = instr;
         }
         if (it.first)
            items.push_back(it);
         continue;
      }

      struct tu_rt_funnel_item it = {};
      it.node = node;
      it.own = tu_rt_funnel_mask(st, node);
      if (it.own && node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         st->forced.push_back({ nif, { tu_rt_funnel_list_mask(st, &nif->then_list),
                                       tu_rt_funnel_list_mask(st, &nif->else_list) } });
         tu_rt_funnel_plan_list(st, &nif->then_list);
         tu_rt_funnel_plan_list(st, &nif->else_list);
      } else if (it.own && node->type == nir_cf_node_loop) {
         nir_loop *loop = nir_cf_node_as_loop(node);
         tu_rt_funnel_plan_list(st, &loop->body);
         st->loops.push_back(loop);
      }
      items.push_back(it);
   }

   for (unsigned i = 0; i < items.size();) {
      unsigned j = i;
      while (j + 1 < items.size() && items[j + 1].own == items[i].own)
         j++;
      st->runs.push_back({ items[i], items[j], items[i].own });
      i = j + 1;
   }
}

static bool
tu_rt_funnel_inside(nir_block *block, nir_cf_node *node)
{
   for (nir_cf_node *n = &block->cf_node; n; n = n->parent) {
      if (n == node)
         return true;
   }
   return false;
}

static nir_block *
tu_rt_funnel_use_block(nir_src *src)
{
   if (nir_src_is_if(src))
      return nir_cf_node_as_block(nir_cf_node_prev(&nir_src_use_if(src)->cf_node));
   nir_instr *use = nir_src_use_instr(src);
   if (use->type == nir_instr_type_phi)
      return exec_node_data(nir_phi_src, src, src)->pred;
   return use->block;
}

struct tu_rt_funnel_repair {
   nir_function_impl *impl;
   nir_cf_node *region;
   nir_block *merge;
   nir_block *skip_pred;
   std::map<nir_def *, nir_def *> phis;
   std::map<nir_def *, struct nir_phi_builder_value *> vals;
   struct nir_phi_builder *pb;
   BITSET_WORD *def_set;
};

static nir_def *
tu_rt_funnel_outside(struct tu_rt_funnel_repair *rp, nir_def *def, nir_block *use)
{
   nir_block *db = nir_def_block(def);
   if (!tu_rt_funnel_inside(db, rp->region) || nir_block_dominates(db, use))
      return def;
   bool merge_ok = nir_block_dominates(rp->merge, use);
   nir_foreach_pred (pred, rp->merge) {
      if (pred != rp->skip_pred && !nir_block_dominates(db, pred))
         merge_ok = false;
   }
   if (merge_ok) {
      auto it = rp->phis.find(def);
      if (it != rp->phis.end())
         return it->second;
      nir_builder ub = nir_builder_at(nir_before_impl(rp->impl));
      nir_def *undef = nir_undef(&ub, def->num_components, def->bit_size);
      nir_phi_instr *phi = nir_phi_instr_create(rp->impl->function->shader);
      nir_def_init(&phi->instr, &phi->def, def->num_components, def->bit_size);
      nir_foreach_pred (pred, rp->merge)
         nir_phi_instr_add_src(phi, pred, pred == rp->skip_pred ? undef : def);
      nir_instr_insert(nir_before_block(rp->merge), &phi->instr);
      rp->phis[def] = &phi->def;
      return &phi->def;
   }
   struct nir_phi_builder_value *val;
   auto it = rp->vals.find(def);
   if (it != rp->vals.end()) {
      val = it->second;
   } else {
      if (!rp->pb) {
         rp->pb = nir_phi_builder_create(rp->impl);
         rp->def_set = rzalloc_array(rp->pb, BITSET_WORD, BITSET_WORDS(rp->impl->num_blocks));
      }
      memset(rp->def_set, 0, BITSET_WORDS(rp->impl->num_blocks) * sizeof(BITSET_WORD));
      BITSET_SET(rp->def_set, db->index);
      val = nir_phi_builder_add_value(rp->pb, def->num_components, def->bit_size, rp->def_set);
      nir_phi_builder_value_set_block_def(val, db, def);
      rp->vals[def] = val;
   }
   return nir_phi_builder_value_get_block_def(val, use);
}

static nir_deref_instr *
tu_rt_funnel_clone_deref(nir_builder *b, struct tu_rt_funnel_repair *rp, nir_deref_instr *d,
                         nir_block *use)
{
   switch (d->deref_type) {
   case nir_deref_type_var:
      return nir_build_deref_var(b, d->var);
   case nir_deref_type_cast: {
      nir_deref_instr *parent = nir_src_as_deref(d->parent);
      nir_def *p = parent ? &tu_rt_funnel_clone_deref(b, rp, parent, use)->def :
                            tu_rt_funnel_outside(rp, d->parent.ssa, use);
      return nir_build_deref_cast_with_alignment(b, p, d->modes, d->type, d->cast.ptr_stride,
                                                 d->cast.align_mul, d->cast.align_offset);
   }
   case nir_deref_type_struct:
      return nir_build_deref_struct(b, tu_rt_funnel_clone_deref(b, rp, nir_deref_instr_parent(d), use),
                                    d->strct.index);
   case nir_deref_type_array:
      return nir_build_deref_array(b, tu_rt_funnel_clone_deref(b, rp, nir_deref_instr_parent(d), use),
                                   tu_rt_funnel_outside(rp, d->arr.index.ssa, use));
   case nir_deref_type_ptr_as_array:
      return nir_build_deref_ptr_as_array(b,
                                          tu_rt_funnel_clone_deref(b, rp, nir_deref_instr_parent(d), use),
                                          tu_rt_funnel_outside(rp, d->arr.index.ssa, use));
   case nir_deref_type_array_wildcard:
      return nir_build_deref_array_wildcard(b,
                                            tu_rt_funnel_clone_deref(b, rp, nir_deref_instr_parent(d), use));
   default:
      UNREACHABLE("unexpected deref");
   }
}

static void
tu_rt_funnel_repair_region(struct tu_rt_funnel_repair *rp)
{
   std::vector<nir_def *> defs;
   nir_foreach_block_in_cf_node (block, rp->region) {
      nir_foreach_instr (instr, block) {
         nir_def *def = nir_instr_def(instr);
         if (def)
            defs.push_back(def);
      }
   }
   nir_progress(true, rp->impl, nir_metadata_none);
   nir_metadata_require(rp->impl, nir_metadata_block_index | nir_metadata_dominance);
   rp->pb = NULL;
   for (nir_def *def : defs) {
      std::vector<nir_src *> outside;
      nir_foreach_use_including_if (src, def) {
         nir_block *ub = tu_rt_funnel_use_block(src);
         if (tu_rt_funnel_inside(ub, rp->region) || nir_block_dominates(nir_def_block(def), ub))
            continue;
         if (!nir_src_is_if(src) && nir_src_use_instr(src)->type == nir_instr_type_phi &&
             nir_src_use_instr(src)->block == rp->merge)
            continue;
         outside.push_back(src);
      }
      if (outside.empty())
         continue;
      nir_instr *dinstr = nir_def_instr(def);
      if (dinstr->type == nir_instr_type_deref) {
         for (nir_src *src : outside) {
            nir_builder b;
            nir_block *ub = tu_rt_funnel_use_block(src);
            if (nir_src_is_if(src) || nir_src_use_instr(src)->type == nir_instr_type_phi)
               b = nir_builder_at(nir_after_block_before_jump(ub));
            else
               b = nir_builder_at(nir_before_instr(nir_src_use_instr(src)));
            nir_src_rewrite(src, &tu_rt_funnel_clone_deref(&b, rp, nir_instr_as_deref(dinstr), ub)->def);
         }
         continue;
      }
      for (nir_src *src : outside)
         nir_src_rewrite(src, tu_rt_funnel_outside(rp, def, tu_rt_funnel_use_block(src)));
   }
   if (rp->pb)
      nir_phi_builder_finish(rp->pb);
   rp->pb = NULL;
}

static void
tu_rt_funnel_guard_run(nir_builder *b, struct tu_rt_funnel_state *st,
                       const struct tu_rt_funnel_run &run)
{
   nir_cursor at = run.begin.node ? nir_before_cf_node(run.begin.node) :
                                    nir_before_instr(run.begin.first);
   nir_block *next = run.end.node ? nir_cf_node_as_block(nir_cf_node_next(run.end.node)) : NULL;
   b->cursor = at;
   nir_def *s = nir_load_var(b, st->state_var);
   nir_def *g = nir_ieq_imm(b, s, 0);
   if (run.own)
      g = nir_ior(b, g, tu_rt_funnel_in(b, st, s, st->resume_base, run.own));
   nir_if *nif = nir_push_if(b, g);
   nir_pop_if(b, NULL);
   nir_cursor begin = run.begin.node ? nir_before_cf_node(run.begin.node) :
                                       nir_before_instr(run.begin.first);
   nir_cursor end = run.end.node ? nir_before_block_after_phis(next) :
                                   nir_after_instr(run.end.last);
   nir_cf_list list;
   nir_cf_extract(&list, begin, end);
   nir_cf_reinsert(&list, nir_before_block(nir_if_first_then_block(nif)));

   struct tu_rt_funnel_repair rp = {};
   rp.impl = b->impl;
   rp.region = &nif->cf_node;
   rp.merge = nir_cf_node_as_block(nir_cf_node_next(&nif->cf_node));
   rp.skip_pred = nir_if_last_else_block(nif);
   tu_rt_funnel_repair_region(&rp);
}

static void
tu_rt_funnel_fix_phis(nir_function_impl *impl)
{
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_foreach_block (block, impl) {
      nir_foreach_phi (phi, block) {
         nir_foreach_pred (pred, block) {
            bool found = false;
            nir_foreach_phi_src (src, phi) {
               if (src->pred == pred) {
                  found = true;
                  break;
               }
            }
            if (!found) {
               b.cursor = nir_before_impl(impl);
               nir_def *undef = nir_undef(&b, phi->def.num_components, phi->def.bit_size);
               nir_phi_src *src = nir_phi_instr_add_src(phi, pred, undef);
               list_addtail(&src->src.use_link, &undef->uses);
            }
         }
      }
   }
}

static bool
tu_rt_funnel_inplace(struct tu_rt_funnel_state *st, nir_function_impl *impl,
                     std::vector<nir_jump_instr *> &returns)
{
   for (unsigned j = 0; j < st->nsites; j++)
      st->resume[j] = NULL;

   nir_foreach_block (block, impl) {
      nir_foreach_instr (instr, block) {
         if (instr->type == nir_instr_type_jump) {
            nir_jump_instr *jump = nir_instr_as_jump(instr);
            if (jump->type == nir_jump_return)
               returns.push_back(jump);
            else if (jump->type == nir_jump_halt)
               return false;
            continue;
         }
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_rt_resume)
            continue;
         unsigned j = nir_intrinsic_call_idx(intr);
         if (j >= st->nsites || st->resume[j])
            return false;
         nir_instr *prev = nir_instr_prev(instr);
         if (!prev || prev->type != nir_instr_type_intrinsic ||
             nir_instr_as_intrinsic(prev)->intrinsic != nir_intrinsic_rt_trace_ray ||
             nir_intrinsic_call_idx(nir_instr_as_intrinsic(prev)) != j)
            return false;
         st->resume[j] = intr;
      }
   }
   for (unsigned j = 0; j < st->nsites; j++) {
      if (!st->resume[j])
         return false;
      for (nir_cf_node *n = &st->resume[j]->instr.block->cf_node;
           n && n->type != nir_cf_node_function; n = n->parent)
         st->masks[n] |= BITFIELD64_BIT(j);
   }

   nir_foreach_block_safe (block, impl) {
      nir_foreach_instr_safe (instr, block) {
         if (instr->type == nir_instr_type_load_const || instr->type == nir_instr_type_undef) {
            nir_instr_remove(instr);
            nir_instr_insert(nir_before_impl(impl), instr);
         }
      }
   }

   tu_rt_funnel_plan_list(st, &impl->body);

   nir_builder b = nir_builder_create(impl);
   for (auto &f : st->forced) {
      nir_if *nif = f.first;
      b.cursor = nir_before_cf_node(&nif->cf_node);
      nir_def *s = nir_load_var(&b, st->state_var);
      nir_def *cond = nif->condition.ssa;
      if (f.second.first)
         cond = nir_ior(&b, cond, tu_rt_funnel_in(&b, st, s, st->resume_base, f.second.first));
      if (f.second.second)
         cond = nir_iand(&b, cond, nir_inot(&b, tu_rt_funnel_in(&b, st, s, st->resume_base,
                                                                f.second.second)));
      nir_src_rewrite(&nif->condition, cond);
   }
   for (auto &run : st->runs)
      tu_rt_funnel_guard_run(&b, st, run);
   for (nir_loop *loop : st->loops) {
      nir_block *tail = nir_loop_last_block(loop);
      if (nir_block_ends_in_jump(tail))
         return false;
      b.cursor = nir_after_block(tail);
      nir_if *nif = nir_push_if(&b, nir_uge_imm(&b, nir_load_var(&b, st->state_var),
                                                st->yield_base));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, nif);
      nir_block *brk = nir_if_last_then_block(nif);
      nir_block *exit = nir_cf_node_as_block(nir_cf_node_next(&loop->cf_node));
      tu_rt_funnel_fix_phis(impl);
      nir_progress(true, impl, nir_metadata_none);
      struct tu_rt_funnel_repair rp = {};
      rp.impl = impl;
      rp.region = &loop->cf_node;
      rp.merge = exit;
      rp.skip_pred = brk;
      tu_rt_funnel_repair_region(&rp);
   }

   tu_rt_funnel_fix_phis(impl);
   nir_progress(true, impl, nir_metadata_none);
   if (nir_repair_ssa_impl(impl)) {
      st->fail = "repair";
      return false;
   }
   return true;
}

static bool
tu_rt_funnel_lower(struct tu_rt_funnel_state *st, nir_function_impl *impl,
                   std::vector<nir_jump_instr *> &returns)
{
   nir_builder b = nir_builder_create(impl);
   for (nir_jump_instr *jump : returns) {
      b.cursor = nir_before_instr(&jump->instr);
      nir_store_var(&b, st->state_var, nir_imm_int(&b, st->done), 0x1);
   }
   nir_block *last = nir_impl_last_block(impl);
   if (nir_block_ends_in_jump(last)) {
      st->fail = "tail";
      return false;
   }
   b.cursor = nir_after_block(last);
   {
      nir_def *s = nir_load_var(&b, st->state_var);
      nir_store_var(&b, st->state_var,
                    nir_bcsel(&b, nir_ieq_imm(&b, s, 0), nir_imm_int(&b, st->done), s), 0x1);
   }

   nir_foreach_block_safe (block, impl) {
      nir_foreach_instr_safe (instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         b.cursor = nir_before_instr(instr);
         switch (intr->intrinsic) {
         case nir_intrinsic_rt_resume:
            nir_store_var(&b, st->state_var, nir_imm_int(&b, 0), 0x1);
            nir_instr_remove(instr);
            break;
         case nir_intrinsic_rt_trace_ray: {
            nir_deref_instr *pd = nir_src_as_deref(intr->src[10]);
            nir_variable *pv = pd ? nir_deref_instr_get_variable(pd) : NULL;
            unsigned c = 0;
            while (c < st->nclasses && st->classes[c].payload != pv)
               c++;
            if (!pv || c == st->nclasses) {
               st->fail = "class";
               return false;
            }
            struct tu_rt_funnel_class *cls = st->classes + c;
            for (unsigned k = 0; k < TU_RT_FUNNEL_ARGS; k++) {
               if (cls->arg_const[k])
                  continue;
               nir_def *v = intr->src[k].ssa;
               if (v->num_components != cls->arg_comps[k] || v->bit_size != cls->arg_bits[k]) {
                  st->fail = "arg";
                  return false;
               }
               nir_store_var(&b, cls->arg_var[k], v, nir_component_mask(v->num_components));
            }
            if (st->nclasses > 1)
               nir_store_var(&b, st->cls_var, nir_imm_int(&b, c), 0x1);
            nir_store_var(&b, st->state_var,
                          nir_imm_int(&b, st->yield_base + nir_intrinsic_call_idx(intr)), 0x1);
            nir_instr_remove(instr);
            break;
         }
         case nir_intrinsic_load_scratch: {
            if (!st->slots_in_regs)
               break;
            bool ok = true;
            unsigned off = tu_rt_funnel_scratch_offset(intr, &ok);
            if (!ok) {
               st->fail = "scratch-offset";
               return false;
            }
            unsigned bits = intr->def.bit_size;
            nir_def *comps[NIR_MAX_VEC_COMPONENTS];
            for (unsigned c = 0; c < intr->def.num_components; c++)
               comps[c] = nir_load_var(&b, tu_rt_funnel_slot(st, off + c * bits / 8, bits));
            nir_def_replace(&intr->def, nir_vec(&b, comps, intr->def.num_components));
            break;
         }
         case nir_intrinsic_store_scratch: {
            if (!st->slots_in_regs)
               break;
            bool ok = true;
            unsigned off = tu_rt_funnel_scratch_offset(intr, &ok);
            if (!ok) {
               st->fail = "scratch-offset";
               return false;
            }
            nir_def *v = intr->src[0].ssa;
            unsigned mask = nir_intrinsic_write_mask(intr);
            for (unsigned c = 0; c < v->num_components; c++) {
               if (mask & (1u << c))
                  nir_store_var(&b, tu_rt_funnel_slot(st, off + c * v->bit_size / 8, v->bit_size),
                                nir_channel(&b, v, c), 0x1);
            }
            nir_instr_remove(instr);
            break;
         }
         default:
            break;
         }
      }
   }

   b.cursor = nir_before_impl(impl);
   if (st->nclasses > 1)
      nir_store_var(&b, st->cls_var, nir_undef(&b, 1, 32), 0x1);
   for (unsigned c = 0; c < st->nclasses; c++) {
      struct tu_rt_funnel_class *cls = st->classes + c;
      for (unsigned k = 0; k < TU_RT_FUNNEL_ARGS; k++) {
         if (cls->arg_var[k])
            nir_store_var(&b, cls->arg_var[k],
                          nir_undef(&b, cls->arg_comps[k], cls->arg_bits[k]),
                          nir_component_mask(cls->arg_comps[k]));
      }
   }
   nir_progress(true, impl, nir_metadata_none);
   return true;
}

struct tu_rt_funnel_field {
   bool dynamic;
   unsigned comps;
   unsigned bits;
   uint32_t mask[NIR_MAX_VEC_COMPONENTS];
   uint32_t val[NIR_MAX_VEC_COMPONENTS];
   nir_const_value cval[NIR_MAX_VEC_COMPONENTS];
};

static bool
tu_rt_funnel_path(nir_deref_instr *d, nir_variable *var, std::vector<uint32_t> &key)
{
   key.clear();
   for (; d->deref_type != nir_deref_type_var; d = nir_deref_instr_parent(d)) {
      if (d->deref_type == nir_deref_type_struct)
         key.push_back(d->strct.index * 2);
      else if (d->deref_type == nir_deref_type_array && nir_src_is_const(d->arr.index))
         key.push_back(nir_src_as_uint(d->arr.index) * 2 + 1);
      else
         return false;
   }
   return d->var == var;
}

static nir_deref_instr *
tu_rt_funnel_build_path(nir_builder *b, nir_variable *var, const std::vector<uint32_t> &key)
{
   nir_deref_instr *d = nir_build_deref_var(b, var);
   for (auto it = key.rbegin(); it != key.rend(); ++it) {
      if (*it & 1)
         d = nir_build_deref_array_imm(b, d, *it >> 1);
      else
         d = nir_build_deref_struct(b, d, *it >> 1);
   }
   return d;
}

static void
tu_rt_funnel_pin_payloads(struct tu_rt_funnel_state *st, nir_function_impl *gi)
{
   for (unsigned c = 0; c < st->nclasses; c++) {
      nir_variable *var = st->classes[c].payload;
      std::map<std::vector<uint32_t>, struct tu_rt_funnel_field> fields;
      std::vector<uint32_t> key;
      std::vector<nir_intrinsic_instr *> traces;
      bool ok = true;
      nir_foreach_block (block, gi) {
         nir_foreach_instr (instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_trace_ray) {
               nir_deref_instr *pd = nir_src_as_deref(intr->src[10]);
               if (pd && nir_deref_instr_get_variable(pd) == var)
                  traces.push_back(intr);
               continue;
            }
            if (intr->intrinsic != nir_intrinsic_store_deref)
               continue;
            nir_deref_instr *d = nir_src_as_deref(intr->src[0]);
            if (nir_deref_instr_get_variable(d) != var)
               continue;
            if (!tu_rt_funnel_path(d, var, key)) {
               ok = false;
               continue;
            }
            nir_src v = intr->src[1];
            if (nir_src_is_undef(v))
               continue;
            bool full = nir_intrinsic_write_mask(intr) == nir_component_mask(v.ssa->num_components);
            auto it = fields.find(key);
            if (it == fields.end()) {
               struct tu_rt_funnel_field f = {};
               f.comps = v.ssa->num_components;
               f.bits = v.ssa->bit_size;
               f.dynamic = !full;
               for (unsigned i = 0; i < f.comps && f.bits == 32; i++) {
                  tu_rt_funnel_known_bits(nir_get_scalar(v.ssa, i), &f.mask[i], &f.val[i], 8);
                  f.val[i] &= f.mask[i];
               }
               if (f.bits != 32) {
                  f.dynamic |= !nir_src_is_const(v);
                  if (!f.dynamic)
                     memcpy(f.cval, nir_src_as_const_value(v), sizeof(nir_const_value) * f.comps);
               }
               fields[key] = f;
               continue;
            }
            struct tu_rt_funnel_field &f = it->second;
            if (!full || v.ssa->num_components != f.comps || v.ssa->bit_size != f.bits) {
               f.dynamic = true;
               continue;
            }
            if (f.bits == 32) {
               for (unsigned i = 0; i < f.comps; i++) {
                  uint32_t m, vv;
                  tu_rt_funnel_known_bits(nir_get_scalar(v.ssa, i), &m, &vv, 8);
                  f.mask[i] &= m & ~(f.val[i] ^ vv);
                  f.val[i] &= f.mask[i];
               }
            } else if (!f.dynamic && !tu_rt_funnel_const_eq(v, f.cval, f.comps, f.bits)) {
               f.dynamic = true;
            }
         }
      }
      if (!ok)
         continue;
      for (nir_intrinsic_instr *intr : traces) {
         nir_builder b = nir_builder_at(nir_before_instr(&intr->instr));
         for (auto &f : fields) {
            const struct tu_rt_funnel_field &fd = f.second;
            if (fd.dynamic)
               continue;
            if (fd.bits != 32) {
               nir_store_deref(&b, tu_rt_funnel_build_path(&b, var, f.first),
                               nir_build_imm(&b, fd.comps, fd.bits, fd.cval),
                               nir_component_mask(fd.comps));
               continue;
            }
            bool any = false, all = true;
            for (unsigned i = 0; i < fd.comps; i++) {
               any |= fd.mask[i] != 0;
               all &= fd.mask[i] == ~0u;
            }
            if (!any)
               continue;
            nir_deref_instr *d = tu_rt_funnel_build_path(&b, var, f.first);
            nir_def *cur = all ? NULL : nir_load_deref(&b, d);
            nir_def *comp[NIR_MAX_VEC_COMPONENTS];
            for (unsigned i = 0; i < fd.comps; i++) {
               if (fd.mask[i] == ~0u)
                  comp[i] = nir_imm_int(&b, fd.val[i]);
               else if (fd.mask[i])
                  comp[i] = nir_ior_imm(&b, nir_iand_imm(&b, nir_channel(&b, cur, i), ~fd.mask[i]),
                                        fd.val[i]);
               else
                  comp[i] = nir_channel(&b, cur, i);
            }
            nir_store_deref(&b, d, nir_vec(&b, comp, fd.comps), nir_component_mask(fd.comps));
         }
      }
      nir_progress(true, gi, nir_metadata_control_flow);
   }
}

template <typename F>
static void
tu_rt_funnel_ladder(nir_builder *b, nir_def *sel, unsigned lo, unsigned hi, F &emit)
{
   if (hi - lo == 1) {
      emit(lo);
      return;
   }
   if (hi - lo == 2) {
      nir_push_if(b, nir_ieq_imm(b, sel, lo));
      emit(lo);
      nir_push_else(b, NULL);
      emit(lo + 1);
      nir_pop_if(b, NULL);
      return;
   }
   unsigned mid = (lo + hi) / 2;
   nir_push_if(b, nir_ult(b, sel, nir_imm_int(b, mid)));
   tu_rt_funnel_ladder(b, sel, lo, mid, emit);
   nir_push_else(b, NULL);
   tu_rt_funnel_ladder(b, sel, mid, hi, emit);
   nir_pop_if(b, NULL);
}

static bool
tu_rt_funnel_is_desc(nir_def *def)
{
   nir_instr *instr = nir_def_instr(def);
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   switch (nir_instr_as_intrinsic(instr)->intrinsic) {
   case nir_intrinsic_vulkan_resource_index:
   case nir_intrinsic_vulkan_resource_reindex:
   case nir_intrinsic_load_vulkan_descriptor:
      return true;
   default:
      return false;
   }
}

static bool
tu_rt_funnel_from_desc(nir_def *def, std::set<nir_def *> &seen)
{
   if (!seen.insert(def).second)
      return false;
   if (tu_rt_funnel_is_desc(def))
      return true;
   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_phi) {
      nir_foreach_phi_src (src, nir_instr_as_phi(instr)) {
         if (tu_rt_funnel_from_desc(src->src.ssa, seen))
            return true;
      }
   } else if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
         if (tu_rt_funnel_from_desc(alu->src[i].src.ssa, seen))
            return true;
      }
   }
   return false;
}

static nir_intrinsic_instr *
tu_rt_funnel_as_chain(nir_def *def)
{
   nir_instr *instr = nir_def_instr(def);
   if (instr->type != nir_instr_type_intrinsic ||
       nir_instr_as_intrinsic(instr)->intrinsic != nir_intrinsic_load_vulkan_descriptor)
      return NULL;
   nir_instr *ri = nir_def_instr(nir_instr_as_intrinsic(instr)->src[0].ssa);
   if (ri->type != nir_instr_type_intrinsic ||
       nir_instr_as_intrinsic(ri)->intrinsic != nir_intrinsic_vulkan_resource_index)
      return NULL;
   return nir_instr_as_intrinsic(instr);
}

static bool
tu_rt_funnel_as_phi(nir_function_impl *impl)
{
   nir_foreach_block (block, impl) {
      nir_foreach_phi (phi, block) {
         nir_foreach_phi_src (src, phi) {
            nir_instr *p = nir_def_instr(src->src.ssa);
            if (p->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *i = nir_instr_as_intrinsic(p);
            if ((i->intrinsic == nir_intrinsic_vulkan_resource_index ||
                 i->intrinsic == nir_intrinsic_load_vulkan_descriptor) &&
                nir_intrinsic_desc_type(i) == nir_descriptor_type_acceleration_structure)
               return true;
         }
      }
   }
   return false;
}

static nir_shader *
tu_rt_funnel_fail(struct tu_rt_funnel_info *info, const char *why)
{
   info->fallback = why;
   return NULL;
}

nir_shader *
tu_rt_funnel_raygen(const nir_shader *raygen, void *mem_ctx,
                    const struct tu_rt_funnel_options *options,
                    struct tu_rt_funnel_info *info)
{
   memset(info, 0, sizeof(*info));
   nir_shader *r = nir_shader_clone(mem_ctx, raygen);
   info->instrs_before = tu_rt_funnel_count(r);

   unsigned nimpl = 0;
   nir_foreach_function_impl (impl, r)
      nimpl++;
   if (nimpl != 1)
      return tu_rt_funnel_fail(info, "functions");
   if (r->scratch_size)
      return tu_rt_funnel_fail(info, "scratch");

   nir_function_impl *impl = nir_shader_get_entrypoint(r);
   unsigned nsites = 0;
   nir_foreach_block (block, impl) {
      nir_foreach_instr (instr, block) {
         if (instr->type == nir_instr_type_jump &&
             nir_instr_as_jump(instr)->type == nir_jump_halt)
            return tu_rt_funnel_fail(info, "halt");
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         switch (nir_instr_as_intrinsic(instr)->intrinsic) {
         case nir_intrinsic_trace_ray:
            nsites++;
            break;
         case nir_intrinsic_execute_callable:
         case nir_intrinsic_report_ray_intersection:
            return tu_rt_funnel_fail(info, "callable");
         case nir_intrinsic_load_scratch:
         case nir_intrinsic_store_scratch:
         case nir_intrinsic_rt_resume:
         case nir_intrinsic_rt_trace_ray:
            return tu_rt_funnel_fail(info, "scratch");
         default:
            break;
         }
      }
   }
   info->sites = nsites;
   if (nsites < 2)
      return tu_rt_funnel_fail(info, "sites");
   if (nsites > TU_RT_FUNNEL_MAX_SITES)
      return tu_rt_funnel_fail(info, "too-many-sites");

   nir_intrinsic_instr **sites = ralloc_array(mem_ctx, nir_intrinsic_instr *, nsites);
   unsigned *site_cls = ralloc_array(mem_ctx, unsigned, nsites);
   struct tu_rt_funnel_class *classes =
      rzalloc_array(mem_ctx, struct tu_rt_funnel_class, nsites);
   unsigned nclasses = 0;
   {
      unsigned j = 0;
      nir_foreach_block (block, impl) {
         nir_foreach_instr (instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_trace_ray)
               sites[j++] = nir_instr_as_intrinsic(instr);
         }
      }
   }

   nir_variable **site_var = ralloc_array(mem_ctx, nir_variable *, nsites);
   std::map<const struct glsl_type *, nir_variable *> undef_vars;
   {
      nir_builder b = nir_builder_create(impl);
      for (unsigned j = 0; j < nsites; j++) {
         nir_intrinsic_instr *intr = sites[j];
         nir_deref_instr *pd = nir_src_as_deref(intr->src[10]);
         if (!pd || !nir_deref_mode_is_one_of(pd, nir_var_function_temp | nir_var_shader_temp))
            return tu_rt_funnel_fail(info, "payload");
         site_var[j] = nir_local_variable_create(impl, pd->type, "tu_rt_funnel_payload");
         nir_variable *&undef = undef_vars[pd->type];
         if (!undef)
            undef = nir_local_variable_create(impl, pd->type, "tu_rt_funnel_undef");
         b.cursor = nir_before_instr(&intr->instr);
         nir_deref_instr *cd = nir_build_deref_var(&b, site_var[j]);
         nir_copy_deref(&b, cd, pd);
         nir_src_rewrite(&intr->src[10], &cd->def);
         b.cursor = nir_after_instr(&intr->instr);
         nir_copy_deref(&b, pd, nir_build_deref_var(&b, site_var[j]));
         nir_copy_deref(&b, nir_build_deref_var(&b, site_var[j]),
                        nir_build_deref_var(&b, undef));
      }
      nir_progress(true, impl, nir_metadata_none);
      NIR_PASS(_, r, nir_split_var_copies);
      NIR_PASS(_, r, nir_lower_var_copies);
      NIR_PASS(_, r, nir_lower_vars_to_ssa);
      NIR_PASS(_, r, nir_opt_copy_prop);
   }

   struct tu_rt_funnel_sig {
      const struct glsl_type *ptype;
      uint32_t flags_mask;
      uint32_t flags_val;
      bool unique;
      std::map<std::vector<uint32_t>, std::vector<uint64_t>> fields;
      std::vector<int> as_key;
   };
   std::vector<struct tu_rt_funnel_sig> sigs(nsites);
   std::vector<nir_intrinsic_instr *> as_ld(nsites);
   {
      for (unsigned j = 0; j < nsites; j++) {
         as_ld[j] = tu_rt_funnel_as_chain(sites[j]->src[0].ssa);
         for (unsigned k = as_ld[j] ? 1 : 0; k < TU_RT_FUNNEL_ARGS; k++) {
            std::set<nir_def *> seen;
            if (tu_rt_funnel_from_desc(sites[j]->src[k].ssa, seen))
               return tu_rt_funnel_fail(info, "as-desc");
         }
         if (!as_ld[j])
            continue;
         nir_intrinsic_instr *ri = nir_def_as_intrinsic(as_ld[j]->src[0].ssa);
         if (!nir_src_is_const(ri->src[0]))
            return tu_rt_funnel_fail(info, "as-desc");
         sigs[j].as_key.push_back(1);
         sigs[j].as_key.push_back(nir_src_as_uint(ri->src[0]));
         for (nir_intrinsic_instr *i : { ri, as_ld[j] }) {
            sigs[j].as_key.push_back(i->num_components);
            sigs[j].as_key.push_back(i->def.num_components);
            sigs[j].as_key.push_back(i->def.bit_size);
            for (unsigned n = 0; n < nir_intrinsic_infos[i->intrinsic].num_indices; n++)
               sigs[j].as_key.push_back(i->const_index[n]);
         }
      }
   }
   {
      std::vector<uint32_t> key;
      for (unsigned j = 0; j < nsites; j++) {
         sigs[j].ptype = site_var[j]->type;
         tu_rt_funnel_known_bits(nir_get_scalar(sites[j]->src[1].ssa, 0), &sigs[j].flags_mask,
                                 &sigs[j].flags_val, 8);
         sigs[j].unique = false;
      }
      nir_foreach_block (block, impl) {
         nir_foreach_instr (instr, block) {
            if (instr->type != nir_instr_type_intrinsic ||
                nir_instr_as_intrinsic(instr)->intrinsic != nir_intrinsic_store_deref)
               continue;
            nir_intrinsic_instr *st_intr = nir_instr_as_intrinsic(instr);
            nir_deref_instr *d = nir_src_as_deref(st_intr->src[0]);
            nir_variable *var = nir_deref_instr_get_variable(d);
            unsigned j = 0;
            while (j < nsites && site_var[j] != var)
               j++;
            if (j == nsites)
               continue;
            nir_src v = st_intr->src[1];
            if (nir_src_is_undef(v))
               continue;
            if (!tu_rt_funnel_path(d, var, key) ||
                nir_intrinsic_write_mask(st_intr) != nir_component_mask(v.ssa->num_components)) {
               sigs[j].unique = true;
               continue;
            }
            std::vector<uint64_t> val;
            val.push_back(v.ssa->num_components | (v.ssa->bit_size << 8));
            if (nir_src_is_const(v)) {
               const nir_const_value *cv = nir_src_as_const_value(v);
               for (unsigned c = 0; c < v.ssa->num_components; c++)
                  val.push_back(nir_const_value_as_uint(cv[c], v.ssa->bit_size));
            } else {
               val.push_back(~0ull);
               val.push_back(~0ull);
            }
            if (sigs[j].fields.count(key))
               sigs[j].unique = true;
            sigs[j].fields[key] = val;
         }
      }
   }

   for (unsigned j = 0; j < nsites; j++) {
      unsigned c = 0;
      if (!sigs[j].unique) {
         while (c < nclasses) {
            const struct tu_rt_funnel_sig &o = sigs[classes[c].first_site];
            bool same_flags = o.flags_mask == sigs[j].flags_mask &&
                              o.flags_val == sigs[j].flags_val;
            bool same_fields = o.fields == sigs[j].fields;
            if (!o.unique && o.ptype == sigs[j].ptype && o.as_key == sigs[j].as_key &&
                ((options->merge_mode == 0 && same_flags && same_fields) ||
                 options->merge_mode == 1 ||
                 (options->merge_mode == 2 && same_fields) ||
                 (options->merge_mode == 3 && same_flags)))
               break;
            c++;
         }
      } else {
         c = nclasses;
      }
      if (c == nclasses) {
         classes[c].ptype = sigs[j].ptype;
         classes[c].first_site = j;
         classes[c].payload = site_var[j];
         classes[c].flags_known_mask = sigs[j].flags_mask;
         classes[c].flags_known_val = sigs[j].flags_val & sigs[j].flags_mask;
         if (as_ld[j]) {
            nir_intrinsic_instr *ri = nir_def_as_intrinsic(as_ld[j]->src[0].ssa);
            classes[c].as_desc = true;
            for (unsigned n = 0; n < 2; n++) {
               nir_intrinsic_instr *i = n ? as_ld[j] : ri;
               memcpy(classes[c].as_idx[n], i->const_index, sizeof(i->const_index));
               classes[c].as_shape[n][0] = i->num_components;
               classes[c].as_shape[n][1] = i->def.num_components;
               classes[c].as_shape[n][2] = i->def.bit_size;
            }
         }
         nclasses++;
      } else {
         struct tu_rt_funnel_class *cls = classes + c;
         cls->flags_known_mask &= sigs[j].flags_mask &
                                  ~(cls->flags_known_val ^ sigs[j].flags_val);
         cls->flags_known_val &= cls->flags_known_mask;
      }
      site_cls[j] = c;
      struct tu_rt_funnel_class *cls = classes + c;
      cls->sites++;
      nir_intrinsic_instr *intr = sites[j];
      if (as_ld[j])
         nir_src_rewrite(&intr->src[0], nir_def_as_intrinsic(as_ld[j]->src[0].ssa)->src[0].ssa);
      for (unsigned k = 0; k < TU_RT_FUNNEL_ARGS; k++) {
         nir_src s = intr->src[k];
         if (!cls->seen) {
            cls->arg_comps[k] = s.ssa->num_components;
            cls->arg_bits[k] = s.ssa->bit_size;
            cls->arg_const[k] = nir_src_is_const(s);
            if (cls->arg_const[k])
               memcpy(cls->arg_val[k], nir_src_as_const_value(s),
                      sizeof(nir_const_value) * s.ssa->num_components);
         } else if (s.ssa->num_components != cls->arg_comps[k] ||
                    s.ssa->bit_size != cls->arg_bits[k]) {
            return tu_rt_funnel_fail(info, "arg-shape");
         } else if (cls->arg_const[k] &&
                    !tu_rt_funnel_const_eq(s, cls->arg_val[k], cls->arg_comps[k],
                                           cls->arg_bits[k])) {
            cls->arg_const[k] = false;
         }
      }
      cls->seen = true;
   }

   unsigned max_sites = 0;
   unsigned desc_len = 0;
   for (unsigned c = 0; c < nclasses; c++) {
      max_sites = MAX2(max_sites, classes[c].sites);
      unsigned size, align;
      glsl_get_natural_size_align_bytes(classes[c].ptype, &size, &align);
      if (desc_len < sizeof(info->classes_desc)) {
         int n = snprintf(info->classes_desc + desc_len, sizeof(info->classes_desc) - desc_len,
                          "%s%ux%uB/%x:%x", c ? "," : "", classes[c].sites, size,
                          classes[c].flags_known_mask, classes[c].flags_known_val);
         desc_len += MAX2(n, 0);
      }
   }
   info->classes = nclasses;
   if (max_sites < 2)
      return tu_rt_funnel_fail(info, "no-shared-class");

   for (unsigned j = 0; j < nsites; j++) {
      nir_variable *rep = classes[site_cls[j]].payload;
      if (site_var[j] == rep)
         continue;
      nir_foreach_block (block, impl) {
         nir_foreach_instr (instr, block) {
            if (instr->type != nir_instr_type_deref)
               continue;
            nir_deref_instr *d = nir_instr_as_deref(instr);
            if (d->deref_type == nir_deref_type_var && d->var == site_var[j])
               d->var = rep;
         }
      }
      exec_node_remove(&site_var[j]->node);
   }

   struct tu_rt_funnel_state st = {};
   st.s = r;
   st.classes = classes;
   st.nclasses = nclasses;
   st.nsites = nsites;
   st.slots_in_regs = options->slots_in_regs;
   st.done = 1;
   st.yield_base = 2;
   st.resume_base = 2 + nsites;
   st.slots = _mesa_hash_table_u64_create(mem_ctx);
   st.state_var = nir_variable_create(r, nir_var_shader_temp, glsl_uint_type(),
                                      "tu_rt_funnel_state");
   st.cls_var = nir_variable_create(r, nir_var_shader_temp, glsl_uint_type(),
                                    "tu_rt_funnel_class");

   for (unsigned c = 0; c < nclasses; c++) {
      for (unsigned k = 0; k < TU_RT_FUNNEL_ARGS; k++) {
         if (!classes[c].arg_const[k])
            classes[c].arg_var[k] =
               nir_variable_create(r, nir_var_shader_temp,
                                   tu_rt_funnel_type(classes[c].arg_bits[k],
                                                     classes[c].arg_comps[k]),
                                   "tu_rt_funnel_arg");
      }
   }
   nir_foreach_function_temp_variable_safe (var, impl) {
      exec_node_remove(&var->node);
      var->data.mode = nir_var_shader_temp;
      nir_shader_add_variable(r, var);
   }
   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, r, nir_fixup_deref_modes);
   NIR_PASS(_, r, nir_lower_continue_constructs);

   nir_lower_shader_calls_options lsc = {};
   lsc.address_format = nir_address_format_32bit_offset;
   lsc.stack_alignment = 4;
   lsc.keep_resumes_inline = true;
   nir_shader **resumes = NULL;
   uint32_t nres = 0;
   if (!nir_lower_shader_calls(r, &lsc, &resumes, &nres, mem_ctx))
      return tu_rt_funnel_fail(info, "split");
   impl = nir_shader_get_entrypoint(r);

   bool ok = true;
   unsigned stack = 0;
   nir_foreach_block (block, impl) {
      nir_foreach_instr (instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         nir_def *v;
         if (intr->intrinsic == nir_intrinsic_load_scratch)
            v = &intr->def;
         else if (intr->intrinsic == nir_intrinsic_store_scratch)
            v = intr->src[0].ssa;
         else
            continue;
         unsigned off = tu_rt_funnel_scratch_offset(intr, &ok);
         stack = MAX2(stack, off + v->num_components * MAX2(v->bit_size / 8, 1));
      }
   }
   info->stack_bytes = stack;
   if (!ok)
      return tu_rt_funnel_fail(info, "scratch-offset");
   if (stack > options->max_stack_bytes)
      return tu_rt_funnel_fail(info, "stack");
   r->scratch_size = options->slots_in_regs ? 0 : align(stack, 4);

   std::vector<nir_jump_instr *> returns;
   if (!tu_rt_funnel_inplace(&st, impl, returns))
      return tu_rt_funnel_fail(info, st.fail ? st.fail : "inplace");
   if (!tu_rt_funnel_lower(&st, impl, returns))
      return tu_rt_funnel_fail(info, st.fail);
   NIR_PASS(_, r, nir_opt_remove_phis);

   nir_function *entry = impl->function;
   nir_function *body = nir_function_create(r, "tu_rt_funnel_body");
   entry->impl = NULL;
   nir_function_set_impl(body, impl);
   nir_function_impl *gi = nir_function_impl_create(entry);
   {
      nir_builder b = nir_builder_at(nir_after_impl(gi));
      nir_store_var(&b, st.state_var, nir_imm_int(&b, 0), 0x1);
      nir_loop *loop = nir_push_loop(&b);
      {
         nir_build_call(&b, body, 0, NULL);
         nir_def *s = nir_load_var(&b, st.state_var);
         nir_break_if(&b, nir_ieq_imm(&b, s, st.done));
         nir_store_var(&b, st.state_var, nir_iadd_imm(&b, s, nsites), 0x1);
         auto trace = [&](unsigned c) {
            struct tu_rt_funnel_class *cls = classes + c;
            nir_def *src[TU_RT_FUNNEL_ARGS + 1];
            for (unsigned k = 0; k < TU_RT_FUNNEL_ARGS; k++) {
               src[k] = cls->arg_const[k] ?
                  nir_build_imm(&b, cls->arg_comps[k], cls->arg_bits[k], cls->arg_val[k]) :
                  nir_load_var(&b, cls->arg_var[k]);
            }
            if (!cls->arg_const[1] && cls->flags_known_mask)
               src[1] = nir_ior_imm(&b, nir_iand_imm(&b, src[1], ~cls->flags_known_mask),
                                    cls->flags_known_val & cls->flags_known_mask);
            if (cls->as_desc) {
               for (unsigned n = 0; n < 2; n++) {
                  nir_intrinsic_instr *i =
                     nir_intrinsic_instr_create(r, n ? nir_intrinsic_load_vulkan_descriptor :
                                                       nir_intrinsic_vulkan_resource_index);
                  i->src[0] = nir_src_for_ssa(src[0]);
                  memcpy(i->const_index, cls->as_idx[n], sizeof(i->const_index));
                  i->num_components = cls->as_shape[n][0];
                  nir_def_init(&i->instr, &i->def, cls->as_shape[n][1], cls->as_shape[n][2]);
                  nir_builder_instr_insert(&b, &i->instr);
                  src[0] = &i->def;
               }
            }
            src[TU_RT_FUNNEL_ARGS] = &nir_build_deref_var(&b, cls->payload)->def;
            nir_trace_ray(&b, src[0], src[1], src[2], src[3], src[4], src[5], src[6],
                          src[7], src[8], src[9], src[10]);
         };
         if (nclasses == 1)
            trace(0);
         else
            tu_rt_funnel_ladder(&b, nir_load_var(&b, st.cls_var), 0, nclasses, trace);
      }
      nir_pop_loop(&b, loop);
   }

   NIR_PASS(_, r, nir_lower_returns);
   NIR_PASS(_, r, nir_inline_functions);
   nir_remove_non_entrypoints(r);
   NIR_PASS(_, r, nir_lower_global_vars_to_local);
   NIR_PASS(_, r, nir_split_var_copies);
   NIR_PASS(_, r, nir_lower_var_copies);
   NIR_PASS(_, r, nir_lower_vars_to_ssa);
   NIR_PASS(_, r, nir_opt_copy_prop);
   NIR_PASS(_, r, nir_opt_remove_phis);
   NIR_PASS(_, r, nir_opt_dce);
   NIR_PASS(_, r, nir_opt_dead_cf);
   NIR_PASS(_, r, nir_opt_cse);
   NIR_PASS(_, r, nir_opt_dce);
   NIR_PASS(_, r, nir_remove_dead_variables, nir_var_function_temp | nir_var_shader_temp, NULL);
   tu_rt_funnel_pin_payloads(&st, nir_shader_get_entrypoint(r));
   NIR_PASS(_, r, nir_opt_dce);
   if (tu_rt_funnel_as_phi(nir_shader_get_entrypoint(r)))
      return tu_rt_funnel_fail(info, "as-desc");

   info->instrs_after = tu_rt_funnel_count(r);
   return r;
}

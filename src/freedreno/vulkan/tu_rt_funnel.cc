/*
 * Copyright © 2026 Jetha Chan
 * SPDX-License-Identifier: MIT
 */

#include "tu_rt_funnel.h"

#include "nir_builder.h"
#include "nir_control_flow.h"
#include "util/hash_table.h"

#include <map>
#include <vector>

#define TU_RT_FUNNEL_ARGS 10
#define TU_RT_FUNNEL_MAX_SITES 64

struct tu_rt_funnel_class {
   const struct glsl_type *ptype;
   bool flags_const;
   uint32_t flags;
   unsigned sites;
   bool seen;
   bool arg_const[TU_RT_FUNNEL_ARGS];
   nir_const_value arg_val[TU_RT_FUNNEL_ARGS][NIR_MAX_VEC_COMPONENTS];
   unsigned arg_comps[TU_RT_FUNNEL_ARGS];
   unsigned arg_bits[TU_RT_FUNNEL_ARGS];
   nir_variable *payload;
   nir_variable *payload_undef;
   nir_variable *arg_var[TU_RT_FUNNEL_ARGS];
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
   bool dom_check;
   std::map<nir_def *, nir_def *> phis;
};

static nir_def *
tu_rt_funnel_outside(struct tu_rt_funnel_repair *rp, nir_def *def)
{
   if (!tu_rt_funnel_inside(nir_def_block(def), rp->region))
      return def;
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

static nir_deref_instr *
tu_rt_funnel_clone_deref(nir_builder *b, struct tu_rt_funnel_repair *rp, nir_deref_instr *d)
{
   switch (d->deref_type) {
   case nir_deref_type_var:
      return nir_build_deref_var(b, d->var);
   case nir_deref_type_cast: {
      nir_deref_instr *parent = nir_src_as_deref(d->parent);
      nir_def *p = parent ? &tu_rt_funnel_clone_deref(b, rp, parent)->def :
                            tu_rt_funnel_outside(rp, d->parent.ssa);
      return nir_build_deref_cast_with_alignment(b, p, d->modes, d->type, d->cast.ptr_stride,
                                                 d->cast.align_mul, d->cast.align_offset);
   }
   case nir_deref_type_struct:
      return nir_build_deref_struct(b, tu_rt_funnel_clone_deref(b, rp, nir_deref_instr_parent(d)),
                                    d->strct.index);
   case nir_deref_type_array:
      return nir_build_deref_array(b, tu_rt_funnel_clone_deref(b, rp, nir_deref_instr_parent(d)),
                                   tu_rt_funnel_outside(rp, d->arr.index.ssa));
   case nir_deref_type_ptr_as_array:
      return nir_build_deref_ptr_as_array(b,
                                          tu_rt_funnel_clone_deref(b, rp, nir_deref_instr_parent(d)),
                                          tu_rt_funnel_outside(rp, d->arr.index.ssa));
   case nir_deref_type_array_wildcard:
      return nir_build_deref_array_wildcard(b,
                                            tu_rt_funnel_clone_deref(b, rp, nir_deref_instr_parent(d)));
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
   if (rp->dom_check)
      nir_metadata_require(rp->impl, nir_metadata_dominance);
   for (nir_def *def : defs) {
      if (rp->dom_check && nir_block_dominates(nir_def_block(def), rp->skip_pred))
         continue;
      std::vector<nir_src *> outside;
      nir_foreach_use_including_if (src, def) {
         nir_block *ub = tu_rt_funnel_use_block(src);
         if (tu_rt_funnel_inside(ub, rp->region))
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
            if (nir_src_is_if(src) || nir_src_use_instr(src)->type == nir_instr_type_phi)
               b = nir_builder_at(nir_after_block_before_jump(tu_rt_funnel_use_block(src)));
            else
               b = nir_builder_at(nir_before_instr(nir_src_use_instr(src)));
            nir_src_rewrite(src, &tu_rt_funnel_clone_deref(&b, rp, nir_instr_as_deref(dinstr))->def);
         }
         continue;
      }
      nir_def *phi = tu_rt_funnel_outside(rp, def);
      for (nir_src *src : outside)
         nir_src_rewrite(src, phi);
   }
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

   struct tu_rt_funnel_repair rp;
   rp.impl = b->impl;
   rp.region = &nif->cf_node;
   rp.merge = nir_cf_node_as_block(nir_cf_node_next(&nif->cf_node));
   rp.skip_pred = nir_if_last_else_block(nif);
   rp.dom_check = false;
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
      struct tu_rt_funnel_repair rp;
      rp.impl = impl;
      rp.region = &loop->cf_node;
      rp.merge = exit;
      rp.skip_pred = brk;
      rp.dom_check = true;
      tu_rt_funnel_repair_region(&rp);
   }

   tu_rt_funnel_fix_phis(impl);
   nir_progress(true, impl, nir_metadata_none);
   nir_repair_ssa_impl(impl);
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
   nir_const_value val[NIR_MAX_VEC_COMPONENTS];
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
            auto it = fields.find(key);
            bool full = nir_intrinsic_write_mask(intr) == nir_component_mask(v.ssa->num_components);
            if (it == fields.end()) {
               struct tu_rt_funnel_field f = {};
               f.comps = v.ssa->num_components;
               f.bits = v.ssa->bit_size;
               f.dynamic = !full || !nir_src_is_const(v);
               if (!f.dynamic)
                  memcpy(f.val, nir_src_as_const_value(v), sizeof(nir_const_value) * f.comps);
               fields[key] = f;
            } else if (!it->second.dynamic &&
                       (!full || !tu_rt_funnel_const_eq(v, it->second.val, it->second.comps,
                                                        it->second.bits))) {
               it->second.dynamic = true;
            }
         }
      }
      if (!ok)
         continue;
      for (nir_intrinsic_instr *intr : traces) {
         nir_builder b = nir_builder_at(nir_before_instr(&intr->instr));
         for (auto &f : fields) {
            if (f.second.dynamic)
               continue;
            nir_store_deref(&b, tu_rt_funnel_build_path(&b, var, f.first),
                            nir_build_imm(&b, f.second.comps, f.second.bits, f.second.val),
                            nir_component_mask(f.second.comps));
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

   for (unsigned j = 0; j < nsites; j++) {
      nir_intrinsic_instr *intr = sites[j];
      nir_deref_instr *pd = nir_src_as_deref(intr->src[10]);
      if (!pd || !nir_deref_mode_is_one_of(pd, nir_var_function_temp | nir_var_shader_temp))
         return tu_rt_funnel_fail(info, "payload");
      bool fc = options->split_flags && nir_src_is_const(intr->src[1]);
      uint32_t fv = fc ? nir_src_as_uint(intr->src[1]) : 0;
      unsigned c = 0;
      while (c < nclasses &&
             !(classes[c].ptype == pd->type && classes[c].flags_const == fc &&
               classes[c].flags == fv))
         c++;
      if (c == nclasses) {
         classes[c].ptype = pd->type;
         classes[c].flags_const = fc;
         classes[c].flags = fv;
         nclasses++;
      }
      site_cls[j] = c;
      struct tu_rt_funnel_class *cls = classes + c;
      cls->sites++;
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
   for (unsigned c = 0; c < nclasses; c++)
      max_sites = MAX2(max_sites, classes[c].sites);
   info->classes = nclasses;
   if (max_sites < 2)
      return tu_rt_funnel_fail(info, "no-shared-class");

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

   {
      nir_builder b = nir_builder_create(impl);
      for (unsigned c = 0; c < nclasses; c++) {
         classes[c].payload = nir_variable_create(r, nir_var_shader_temp, classes[c].ptype,
                                                  "tu_rt_funnel_payload");
         classes[c].payload_undef = nir_variable_create(r, nir_var_shader_temp,
                                                        classes[c].ptype,
                                                        "tu_rt_funnel_undef");
         for (unsigned k = 0; k < TU_RT_FUNNEL_ARGS; k++) {
            if (!classes[c].arg_const[k])
               classes[c].arg_var[k] =
                  nir_variable_create(r, nir_var_shader_temp,
                                      tu_rt_funnel_type(classes[c].arg_bits[k],
                                                        classes[c].arg_comps[k]),
                                      "tu_rt_funnel_arg");
         }
      }
      for (unsigned j = 0; j < nsites; j++) {
         nir_intrinsic_instr *intr = sites[j];
         nir_deref_instr *pd = nir_src_as_deref(intr->src[10]);
         struct tu_rt_funnel_class *cls = classes + site_cls[j];
         b.cursor = nir_before_instr(&intr->instr);
         nir_deref_instr *cd = nir_build_deref_var(&b, cls->payload);
         nir_copy_deref(&b, cd, pd);
         nir_src_rewrite(&intr->src[10], &cd->def);
         b.cursor = nir_after_instr(&intr->instr);
         nir_copy_deref(&b, pd, nir_build_deref_var(&b, cls->payload));
         nir_copy_deref(&b, nir_build_deref_var(&b, cls->payload),
                        nir_build_deref_var(&b, cls->payload_undef));
      }
      nir_foreach_function_temp_variable_safe (var, impl) {
         exec_node_remove(&var->node);
         var->data.mode = nir_var_shader_temp;
         nir_shader_add_variable(r, var);
      }
      nir_progress(true, impl, nir_metadata_none);
      NIR_PASS(_, r, nir_fixup_deref_modes);
      NIR_PASS(_, r, nir_lower_continue_constructs);
   }

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
      return tu_rt_funnel_fail(info, "inplace");
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

   info->funneled_sites = nsites;
   info->instrs_after = tu_rt_funnel_count(r);
   return r;
}

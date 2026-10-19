/*
 * Copyright © 2024 Igalia
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "nir.h"
#include "nir_builder.h"
#include "nir_builder_opcodes.h"
#include "nir_intrinsics.h"

/*
 * Convert atomic arithmetic to regular arithmetic along with cmpxchg
 * by repeating the operation until the result is expected.
 *
 * eg:
 * atomicAdd(a[0], 1) ->
 *
 * uint expected = atomicLoad(a[0]);
 * while (true) {
 *    uint before = expected;
 *    expected += 1;
 *    uint original = atomicCompareExchange(a[0], before, expected);
 *    if (original == before) {break;}
 *    expected = original;
 * }
 */

static nir_def *
build_atomic(nir_builder *b, nir_intrinsic_instr *intr, bool early_out)
{
   nir_def *load;
   switch (intr->intrinsic) {
   case nir_intrinsic_ssbo_atomic:
      load = nir_load_ssbo(b, 1, intr->def.bit_size, intr->src[0].ssa,
                           intr->src[1].ssa,
                           .align_mul = intr->def.bit_size / 8,
                           .align_offset = 0,
                           .offset_shift = nir_intrinsic_offset_shift(intr),
                           .access = ACCESS_ATOMIC | ACCESS_COHERENT);
      break;
   case nir_intrinsic_shared_atomic:
      load = nir_load_shared(b, 1, intr->def.bit_size,
                             intr->src[0].ssa,
                             .align_mul = intr->def.bit_size / 8,
                             .align_offset = 0,
                             .access = nir_intrinsic_access(intr) |
                                       ACCESS_ATOMIC);
      break;
   case nir_intrinsic_global_atomic:
      load = nir_load_global(b, 1, intr->def.bit_size, intr->src[0].ssa,
                             .access = nir_intrinsic_access(intr) |
                                       ACCESS_ATOMIC | ACCESS_COHERENT);
      break;
   case nir_intrinsic_bindless_image_atomic:
      load = nir_bindless_image_load(
         b, 1, intr->def.bit_size, intr->src[0].ssa, intr->src[1].ssa,
         nir_imm_zero(b, 1, 32), intr->src[2].ssa,
         .access = ACCESS_ATOMIC | ACCESS_COHERENT,
         .format = nir_intrinsic_format(intr),
         .image_dim = nir_intrinsic_image_dim(intr),
         .image_array = nir_intrinsic_image_array(intr));
      break;
   default:
      UNREACHABLE("unsupported atomic type");
   }

   nir_def *data = nir_get_io_data_src(intr)->ssa;
   nir_loop *loop = nir_push_loop(b);
   nir_def *before, *xchg;
   {
      nir_phi_instr *phi = nir_phi_instr_create(b->shader);
      nir_def_init(&phi->instr, &phi->def, 1, intr->def.bit_size);
      nir_phi_instr_add_src(phi, nir_def_block(load), load);
      before = &phi->def;
      nir_def *expected = nir_build_alu2(
         b, nir_atomic_op_to_alu(nir_intrinsic_atomic_op(intr)), before, data);
      nir_alu_instr *op = nir_def_as_alu(expected);
      op->fp_math_ctrl = nir_op_valid_fp_math_ctrl(op->op, nir_fp_no_fast_math);
      if (early_out)
         nir_break_if(b, nir_ieq(b, expected, before));
      switch (intr->intrinsic) {
      case nir_intrinsic_ssbo_atomic:
         xchg = nir_ssbo_atomic_swap(b, intr->def.bit_size,
                                     intr->src[0].ssa,
                                     intr->src[1].ssa,
                                     before, expected,
                                     .atomic_op = nir_atomic_op_cmpxchg,
                                     .offset_shift = nir_intrinsic_offset_shift(intr));
         break;
      case nir_intrinsic_shared_atomic:
         xchg = nir_shared_atomic_swap(b, intr->def.bit_size,
                                       intr->src[0].ssa,
                                       before, expected,
                                       .atomic_op = nir_atomic_op_cmpxchg,
                                       .access = nir_intrinsic_access(intr));
         break;
      case nir_intrinsic_global_atomic:
         xchg = nir_global_atomic_swap(b, intr->def.bit_size,
                                       intr->src[0].ssa,
                                       before, expected,
                                       .atomic_op = nir_atomic_op_cmpxchg,
                                       .access = nir_intrinsic_access(intr));
         break;
      case nir_intrinsic_bindless_image_atomic:
         xchg = nir_bindless_image_atomic_swap(
            b, intr->def.bit_size, intr->src[0].ssa, intr->src[1].ssa,
            intr->src[2].ssa, before, expected,
            .atomic_op = nir_atomic_op_cmpxchg,
            .format = nir_intrinsic_format(intr),
            .image_dim = nir_intrinsic_image_dim(intr),
            .image_array = nir_intrinsic_image_array(intr));
         break;
      default:
         UNREACHABLE("unsupported atomic type");
      }
      nir_break_if(b, nir_ieq(b, xchg, before));
      nir_phi_instr_add_src(phi, nir_loop_last_block(loop), xchg);
      b->cursor = nir_before_block(nir_loop_first_block(loop));
      nir_builder_instr_insert(b, &phi->instr);
   }
   nir_pop_loop(b, loop);
   return early_out ? before : xchg;
}

struct lower_atomics_state {
   nir_instr_filter_cb supported;
   bool early_out;
   nir_variable_mode release_modes;
};

static nir_variable_mode
atomic_modes(const nir_intrinsic_instr *intr)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_global_atomic:
      return nir_var_mem_ssbo | nir_var_mem_global;
   case nir_intrinsic_shared_atomic:
      return nir_var_mem_shared;
   case nir_intrinsic_bindless_image_atomic:
      return nir_var_image;
   default:
      UNREACHABLE("unsupported atomic type");
   }
}

static bool
can_early_out(const struct lower_atomics_state *state,
              nir_intrinsic_instr *intr)
{
   if (!state->early_out || !nir_def_is_unused(&intr->def) ||
       (nir_intrinsic_access(intr) & ACCESS_VOLATILE) ||
       (state->release_modes & atomic_modes(intr)))
      return false;

   switch (nir_intrinsic_atomic_op(intr)) {
   case nir_atomic_op_imin:
   case nir_atomic_op_umin:
   case nir_atomic_op_imax:
   case nir_atomic_op_umax:
   case nir_atomic_op_iand:
   case nir_atomic_op_ior:
      return true;
   default:
      return false;
   }
}

static nir_variable_mode
gather_release_modes(nir_shader *shader)
{
   nir_variable_mode modes = 0;

   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_end_invocation_interlock) {
               modes |= nir_var_all;
            } else if (intr->intrinsic == nir_intrinsic_barrier &&
                       (nir_intrinsic_memory_semantics(intr) &
                        NIR_MEMORY_RELEASE)) {
               modes |= nir_intrinsic_memory_modes(intr);
            }
         }
      }
   }

   return modes;
}

static bool
lower_atomics(struct nir_builder *b, nir_intrinsic_instr *intr,
              void *data)
{
   const struct lower_atomics_state *state = data;

   if (intr->intrinsic != nir_intrinsic_ssbo_atomic &&
       intr->intrinsic != nir_intrinsic_shared_atomic &&
       intr->intrinsic != nir_intrinsic_global_atomic &&
       intr->intrinsic != nir_intrinsic_bindless_image_atomic)
      return false;
   if (state->supported(&intr->instr, NULL))
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   switch (nir_intrinsic_atomic_op(intr)) {
   case nir_atomic_op_imin:
   case nir_atomic_op_umin:
   case nir_atomic_op_imax:
   case nir_atomic_op_umax:
   case nir_atomic_op_iand:
   case nir_atomic_op_ior:
   case nir_atomic_op_ixor:
   case nir_atomic_op_fadd:
   case nir_atomic_op_fmin:
   case nir_atomic_op_fmax:
   case nir_atomic_op_iadd: {
      bool early_out = can_early_out(state, intr);
      nir_def_replace(&intr->def, build_atomic(b, intr, early_out));
      return true;
   }
   case nir_atomic_op_cmpxchg:
   case nir_atomic_op_xchg:
      return false;
   case nir_atomic_op_fcmpxchg: /* unimplemented */
   default:
      UNREACHABLE("Invalid nir_atomic_op");
   }
}

bool
nir_lower_atomics_with_options(nir_shader *shader,
                               const nir_lower_atomics_options *options)
{
   struct lower_atomics_state state = {
      .supported = options->supported,
      .early_out = options->early_out_unused_idempotent,
      .release_modes = options->early_out_unused_idempotent ?
                       gather_release_modes(shader) : 0,
   };

   return nir_shader_intrinsics_pass(shader, lower_atomics,
                                     nir_metadata_none, &state);
}

bool
nir_lower_atomics(nir_shader *shader, nir_instr_filter_cb supported)
{
   const nir_lower_atomics_options options = {
      .supported = supported,
   };

   return nir_lower_atomics_with_options(shader, &options);
}

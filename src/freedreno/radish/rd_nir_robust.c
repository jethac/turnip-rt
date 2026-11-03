#include "rd_nir_resources.h"
#include "compiler/nir/nir_builder.h"

struct rd_robust_state { bool failed; };

static nir_def *
rd_access_valid(nir_builder *b, nir_def *offset, nir_def *size,
                  uint32_t component_offset, uint32_t bytes)
{
   nir_def *part = nir_iadd_imm(b, offset, component_offset);
   nir_def *valid = nir_iand(b, nir_uge_imm(b, size, bytes),
      nir_uge(b, nir_iadd_imm(b, size, -(int32_t)bytes), part));
   if (component_offset)
      valid = nir_iand(b, valid, nir_ule_imm(b, offset, UINT32_MAX - component_offset));
   return valid;
}

static bool
rd_robust_load(nir_builder *b, nir_intrinsic_instr *intr,
                 struct rd_robust_state *state)
{
   if ((intr->def.bit_size != 8 && intr->def.bit_size != 16 &&
        intr->def.bit_size != 32) ||
       intr->src[1].ssa->bit_size != 32 ||
       nir_intrinsic_offset_shift(intr)) {
      state->failed = true;
      return false;
   }
   nir_def *index = intr->src[0].ssa, *offset = intr->src[1].ssa;
   nir_def *size = nir_get_ssbo_size(b, 32, index);
   nir_def *parts[NIR_MAX_VEC_COMPONENTS];
   uint32_t bytes = intr->def.bit_size / 8;
   nir_def *zero = nir_imm_intN_t(b, 0, intr->def.bit_size);
   uint32_t alignment = MAX2(1, nir_intrinsic_align_mul(intr));
   for (uint32_t i = 0; i < intr->def.num_components; i++) {
      uint32_t added = i * bytes;
      nir_def *valid = rd_access_valid(b, offset, size, added, bytes);
      nir_push_if(b, valid);
      nir_def *value = nir_load_ssbo(b, 1, intr->def.bit_size, index,
         nir_iadd_imm(b, offset, added),
         .access = nir_intrinsic_access(intr) & ~(ACCESS_CAN_REORDER | ACCESS_CAN_SPECULATE),
         .align_mul = alignment,
         .align_offset = (nir_intrinsic_align_offset(intr) + added) % alignment);
      nir_pop_if(b, NULL);
      parts[i] = nir_if_phi(b, value, zero);
   }
   nir_def_replace(&intr->def, nir_vec(b, parts, intr->def.num_components));
   return true;
}

static bool
rd_robust_store(nir_builder *b, nir_intrinsic_instr *intr,
                  struct rd_robust_state *state)
{
   uint32_t bits = nir_src_bit_size(intr->src[0]);
   if ((bits != 8 && bits != 16 && bits != 32) || intr->src[2].ssa->bit_size != 32) {
      state->failed = true;
      return false;
   }
   nir_def *index = intr->src[1].ssa, *offset = intr->src[2].ssa;
   nir_def *size = nir_get_ssbo_size(b, 32, index);
   uint32_t bytes = bits / 8;
   uint32_t mask = nir_intrinsic_write_mask(intr);
   uint32_t alignment = MAX2(1, nir_intrinsic_align_mul(intr));
   for (uint32_t i = 0; i < intr->num_components; i++) {
      if (!(mask & (1u << i)))
         continue;
      uint32_t added = i * bytes;
      nir_def *valid = rd_access_valid(b, offset, size, added, bytes);
      nir_push_if(b, valid);
      nir_store_ssbo(b, nir_channel(b, intr->src[0].ssa, i), index,
         nir_iadd_imm(b, offset, added), .write_mask = 1,
         .access = nir_intrinsic_access(intr), .align_mul = alignment,
         .align_offset = (nir_intrinsic_align_offset(intr) + added) % alignment);
      nir_pop_if(b, NULL);
   }
   nir_instr_remove(&intr->instr);
   return true;
}

static void
rd_robust_clone(nir_builder *b, nir_intrinsic_instr *intr, nir_def *valid)
{
   bool destination = nir_intrinsic_infos[intr->intrinsic].has_dest;
   nir_def *zero = destination ? nir_imm_zero(b, intr->def.num_components, intr->def.bit_size) : NULL;
   nir_push_if(b, valid);
   nir_instr *clone = nir_instr_clone(b->shader, &intr->instr);
   if (nir_intrinsic_has_access(nir_instr_as_intrinsic(clone)))
      nir_intrinsic_set_access(nir_instr_as_intrinsic(clone),
         nir_intrinsic_access(nir_instr_as_intrinsic(clone)) &
         ~(ACCESS_CAN_REORDER | ACCESS_CAN_SPECULATE));
   nir_builder_instr_insert(b, clone);
   nir_pop_if(b, NULL);
   if (destination) {
      nir_def_replace(&intr->def, nir_if_phi(b, nir_instr_def(clone), zero));
   } else {
      nir_instr_remove(&intr->instr);
   }
}

static bool
rd_robust_atomic(nir_builder *b, nir_intrinsic_instr *intr,
                   struct rd_robust_state *state)
{
   if (intr->def.bit_size != 32 || intr->src[1].ssa->bit_size != 32) {
      state->failed = true;
      return false;
   }
   nir_def *size = nir_get_ssbo_size(b, 32, intr->src[0].ssa);
   rd_robust_clone(b, intr, rd_access_valid(b, intr->src[1].ssa, size, 0, 4));
   return true;
}

static bool
rd_robust_image_buffer(nir_builder *b, nir_intrinsic_instr *intr)
{
   if (nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_BUF)
      return false;
   nir_def *size = nir_image_size(b, 1, 32, intr->src[0].ssa, nir_imm_int(b, 0),
      .image_dim = GLSL_SAMPLER_DIM_BUF, .image_array = false);
   nir_def *valid = nir_ult(b, nir_channel(b, intr->src[1].ssa, 0), size);
   rd_robust_clone(b, intr, valid);
   return true;
}

static bool
rd_robust_texture_buffer(nir_builder *b, nir_tex_instr *tex,
                           struct rd_robust_state *state)
{
   if (tex->sampler_dim != GLSL_SAMPLER_DIM_BUF || tex->op != nir_texop_txf)
      return false;
   int coordinate = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (coordinate < 0 || tex->src[coordinate].src.ssa->bit_size != 32) {
      state->failed = true;
      return false;
   }
   nir_def *coord = tex->src[coordinate].src.ssa;
   nir_tex_instr *size = nir_tex_instr_create(b->shader, 1);
   size->op = nir_texop_txs;
   size->sampler_dim = GLSL_SAMPLER_DIM_BUF;
   size->dest_type = nir_type_uint32;
   size->texture_index = tex->texture_index;
   size->src[0].src_type = nir_tex_src_lod;
   size->src[0].src = nir_src_for_ssa(nir_imm_int(b, 0));
   nir_def_init(&size->instr, &size->def, 1, 32);
   nir_builder_instr_insert(b, &size->instr);
   nir_def *valid = nir_ult(b, coord, &size->def);
   nir_def *safe_coord = nir_bcsel(b, valid, coord, nir_imm_int(b, 0));
   nir_tex_instr *clone = nir_instr_as_tex(nir_instr_clone(b->shader, &tex->instr));
   nir_builder_instr_insert(b, &clone->instr);
   nir_src_rewrite(&clone->src[coordinate].src, safe_coord);
   nir_def *zero = nir_imm_zero(b, tex->def.num_components, tex->def.bit_size);
   nir_def_replace(&tex->def, nir_bcsel(b, valid, &clone->def, zero));
   return true;
}

static bool
rd_robust_instruction(nir_builder *b, nir_instr *instr, void *data)
{
   struct rd_robust_state *state = data;
   if (state->failed)
      return false;
   b->cursor = nir_before_instr(instr);
   if (instr->type == nir_instr_type_tex)
      return rd_robust_texture_buffer(b, nir_instr_as_tex(instr), state);
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_load_ssbo: return rd_robust_load(b, intr, state);
   case nir_intrinsic_store_ssbo: return rd_robust_store(b, intr, state);
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap: return rd_robust_atomic(b, intr, state);
   case nir_intrinsic_image_load:
   case nir_intrinsic_image_store:
   case nir_intrinsic_image_atomic:
   case nir_intrinsic_image_atomic_swap: return rd_robust_image_buffer(b, intr);
   default: return false;
   }
}

bool
rd_nir_lower_robust_buffers(nir_shader *shader)
{
   struct rd_robust_state state = {0};
   nir_shader_instructions_pass(shader, rd_robust_instruction, nir_metadata_none, &state);
   return !state.failed;
}

static nir_mem_access_size_align
rd_subword_storage_access(nir_intrinsic_op intrin, uint8_t bytes, uint8_t bit_size,
                    uint32_t align_mul, uint32_t align_offset,
                    bool offset_is_const, enum gl_access_qualifier access,
                    const void *data)
{
   (void)intrin;
   (void)align_mul;
   (void)align_offset;
   (void)offset_is_const;
   (void)access;
   (void)data;
   return (nir_mem_access_size_align) {
      .num_components = (bit_size == 8 || bit_size == 16) ? 1 : bytes / (bit_size / 8),
      .bit_size = (bit_size == 8 || bit_size == 16) ? 32 : bit_size,
      .align = (bit_size == 8 || bit_size == 16) ? 4 : 1,
      .shift = nir_mem_access_shift_method_scalar,
   };
}

bool
rd_nir_lower_subword_storage(nir_shader *shader)
{
   bool found = false;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            unsigned bits, offset;
            if (intr->intrinsic == nir_intrinsic_load_ssbo) {
               bits = intr->def.bit_size;
               offset = 1;
            } else if (intr->intrinsic == nir_intrinsic_store_ssbo) {
               bits = nir_src_bit_size(intr->src[0]);
               offset = 2;
            } else {
               continue;
            }
            if (bits != 8 && bits != 16)
               continue;
            if (intr->src[offset].ssa->bit_size != 32 ||
                nir_intrinsic_align(intr) < bits / 8 ||
                (nir_intrinsic_has_offset_shift(intr) &&
                 nir_intrinsic_offset_shift(intr)))
               return false;
            found = true;
         }
      }
   }
   if (found) {
      const nir_lower_mem_access_bit_sizes_options options = {
         .modes = nir_var_mem_ssbo,
         .callback = rd_subword_storage_access,
         .may_lower_unaligned_stores_to_atomics = true,
      };
      NIR_PASS(_, shader, nir_lower_mem_access_bit_sizes, &options);
   }
   return true;
}

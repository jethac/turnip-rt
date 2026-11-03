#include "rd_nir_resources.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_format_convert.h"
#include "util/format/u_format.h"
#include <string.h>

struct rd_resource_lower {
   rd_nir_resource_resolver resolve;
   void *data;
   struct rd_nir_resource_usage *usage;
   uint32_t textures_by_txf, texture_buffers, image_buffers;
   bool failed;
};

static const struct rd_nir_resource_binding *
rd_resource_binding(struct rd_resource_lower *state, nir_deref_instr *deref,
                    uint32_t *element)
{
   uint64_t index = 0, stride = 1;
   while (deref && deref->deref_type == nir_deref_type_array) {
      nir_deref_instr *parent = nir_deref_instr_parent(deref);
      if (!parent || !glsl_type_is_array(parent->type) ||
          !nir_src_is_const(deref->arr.index))
         return NULL;
      uint32_t length = glsl_array_size(parent->type);
      uint32_t item = nir_src_as_uint(deref->arr.index);
      if (!length || item >= length || stride > UINT32_MAX / length)
         return NULL;
      index += (uint64_t)item * stride;
      stride *= length;
      deref = parent;
   }
   if (!deref || deref->deref_type != nir_deref_type_var || index > UINT32_MAX)
      return NULL;
   const nir_variable *var = deref->var;
   *element = 0;
   return state->resolve(state->data, var->data.descriptor_set, var->data.binding, index);
}

static bool
rd_mark_resource(struct rd_resource_lower *state,
                 const struct rd_nir_resource_binding *entry, uint32_t element,
                 uint32_t base, uint32_t *mask, uint32_t *count)
{
   if (base >= 16 || element >= 16 - base || entry->descriptor_slot >= 64 ||
       element >= 64 - entry->descriptor_slot)
      return false;
   *mask |= 1u << (base + element);
   *count = MAX2(*count, base + element + 1);
   state->usage->descriptors |= UINT64_C(1) << (entry->descriptor_slot + element);
   return true;
}

static bool
rd_lower_texture(nir_tex_instr *tex, struct rd_resource_lower *state)
{
   if (tex->sampler_dim != GLSL_SAMPLER_DIM_1D &&
       tex->sampler_dim != GLSL_SAMPLER_DIM_2D &&
       tex->sampler_dim != GLSL_SAMPLER_DIM_3D &&
       tex->sampler_dim != GLSL_SAMPLER_DIM_CUBE &&
       tex->sampler_dim != GLSL_SAMPLER_DIM_BUF &&
       tex->sampler_dim != GLSL_SAMPLER_DIM_MS)
      return false;
   if ((tex->is_array && tex->sampler_dim != GLSL_SAMPLER_DIM_1D &&
        tex->sampler_dim != GLSL_SAMPLER_DIM_2D && tex->sampler_dim != GLSL_SAMPLER_DIM_CUBE &&
        tex->sampler_dim != GLSL_SAMPLER_DIM_MS) ||
       tex->is_sparse ||
       tex->def.bit_size != 32 ||
       nir_tex_instr_src_index(tex, nir_tex_src_texture_offset) >= 0 ||
       nir_tex_instr_src_index(tex, nir_tex_src_sampler_offset) >= 0 ||
       nir_tex_instr_src_index(tex, nir_tex_src_texture_handle) >= 0 ||
       nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) >= 0)
      return false;
   switch (tex->op) {
   case nir_texop_txl:
   case nir_texop_txd:
   case nir_texop_tg4:
   case nir_texop_txf:
   case nir_texop_txf_ms:
   case nir_texop_texture_samples:
   case nir_texop_txs:
   case nir_texop_query_levels:
      break;
   default:
      return false;
   }
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_MS) {
      if (tex->is_shadow || (tex->op != nir_texop_txf_ms &&
          tex->op != nir_texop_txs && tex->op != nir_texop_texture_samples))
         return false;
   } else if (tex->op == nir_texop_txf_ms || tex->op == nir_texop_texture_samples) {
      return false;
   }
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_BUF &&
       tex->op != nir_texop_txf && tex->op != nir_texop_txs)
      return false;
   int texture = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   if (texture < 0)
      return false;
   uint32_t element;
   const struct rd_nir_resource_binding *entry = rd_resource_binding(state,
      nir_src_as_deref(tex->src[texture].src), &element);
   if (!entry || (entry->type != VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE &&
                  entry->type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
                  entry->type != VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER))
      return false;
   if ((tex->sampler_dim == GLSL_SAMPLER_DIM_BUF) !=
       (entry->type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER))
      return false;
   if (!rd_mark_resource(state, entry, element, entry->texture_slot,
                        &state->usage->textures, &state->usage->texture_count))
      return false;
   tex->texture_index = entry->texture_slot + element;
   if (tex->op == nir_texop_txf)
      state->textures_by_txf |= 1u << tex->texture_index;
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_BUF)
      state->texture_buffers |= 1u << tex->texture_index;
   nir_tex_instr_remove_src(tex, texture);
   int sampler = nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref);
   if (sampler >= 0) {
      entry = rd_resource_binding(state,
         nir_src_as_deref(tex->src[sampler].src), &element);
      if (!entry || (entry->type != VK_DESCRIPTOR_TYPE_SAMPLER &&
                     entry->type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) ||
          !rd_mark_resource(state, entry, element, entry->sampler_slot,
                            &state->usage->samplers, &state->usage->sampler_count))
         return false;
      tex->sampler_index = entry->sampler_slot + element;
      nir_tex_instr_remove_src(tex, sampler);
   } else if (tex->op == nir_texop_txl || tex->op == nir_texop_txd ||
              tex->op == nir_texop_tg4) {
      return false;
   }
   return true;
}

static bool
rd_storage_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R8_UNORM:
   case PIPE_FORMAT_R8_SNORM:
   case PIPE_FORMAT_R8_UINT:
   case PIPE_FORMAT_R8_SINT:
   case PIPE_FORMAT_R8G8_UNORM:
   case PIPE_FORMAT_R8G8_SNORM:
   case PIPE_FORMAT_R8G8_UINT:
   case PIPE_FORMAT_R8G8_SINT:
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R8G8B8A8_SNORM:
   case PIPE_FORMAT_R8G8B8A8_UINT:
   case PIPE_FORMAT_R8G8B8A8_SINT:
   case PIPE_FORMAT_R16_UNORM:
   case PIPE_FORMAT_R16_SNORM:
   case PIPE_FORMAT_R16_UINT:
   case PIPE_FORMAT_R16_SINT:
   case PIPE_FORMAT_R16_FLOAT:
   case PIPE_FORMAT_R16G16_UNORM:
   case PIPE_FORMAT_R16G16_SNORM:
   case PIPE_FORMAT_R16G16_UINT:
   case PIPE_FORMAT_R16G16_SINT:
   case PIPE_FORMAT_R16G16_FLOAT:
   case PIPE_FORMAT_R16G16B16A16_UNORM:
   case PIPE_FORMAT_R16G16B16A16_SNORM:
   case PIPE_FORMAT_R16G16B16A16_UINT:
   case PIPE_FORMAT_R16G16B16A16_SINT:
   case PIPE_FORMAT_R16G16B16A16_FLOAT:
   case PIPE_FORMAT_R10G10B10A2_UNORM:
   case PIPE_FORMAT_R10G10B10A2_UINT:
   case PIPE_FORMAT_R11G11B10_FLOAT:
   case PIPE_FORMAT_R32_UINT:
   case PIPE_FORMAT_R32_SINT:
   case PIPE_FORMAT_R32_FLOAT:
   case PIPE_FORMAT_R32G32_UINT:
   case PIPE_FORMAT_R32G32_SINT:
   case PIPE_FORMAT_R32G32_FLOAT:
   case PIPE_FORMAT_R32G32B32A32_UINT:
   case PIPE_FORMAT_R32G32B32A32_SINT:
   case PIPE_FORMAT_R32G32B32A32_FLOAT:
      return true;
   default:
      return false;
   }
}

static void
rd_lower_snorm_storage(nir_builder *b, nir_intrinsic_instr *intr)
{
   if (intr->intrinsic != nir_intrinsic_image_load &&
       intr->intrinsic != nir_intrinsic_image_store)
      return;
   enum pipe_format format = nir_intrinsic_format(intr);
   enum pipe_format integer_format;
   switch (format) {
   case PIPE_FORMAT_R8_SNORM: integer_format = PIPE_FORMAT_R8_SINT; break;
   case PIPE_FORMAT_R8G8_SNORM: integer_format = PIPE_FORMAT_R8G8_SINT; break;
   case PIPE_FORMAT_R8G8B8A8_SNORM: integer_format = PIPE_FORMAT_R8G8B8A8_SINT; break;
   case PIPE_FORMAT_R16_SNORM: integer_format = PIPE_FORMAT_R16_SINT; break;
   case PIPE_FORMAT_R16G16_SNORM: integer_format = PIPE_FORMAT_R16G16_SINT; break;
   case PIPE_FORMAT_R16G16B16A16_SNORM: integer_format = PIPE_FORMAT_R16G16B16A16_SINT; break;
   default: return;
   }
   const struct util_format_description *desc = util_format_description(format);
   unsigned channels = desc->nr_channels;
   unsigned bits[4] = {0};
   for (unsigned i = 0; i < channels; i++)
      bits[i] = desc->channel[i].size;
   nir_intrinsic_set_format(intr, integer_format);
   if (intr->intrinsic == nir_intrinsic_image_load) {
      nir_intrinsic_set_dest_type(intr, nir_type_int32);
      b->cursor = nir_after_instr(&intr->instr);
      nir_def *value = nir_format_snorm_to_float(b,
         nir_trim_vector(b, &intr->def, channels), bits);
      nir_def *components[4];
      for (unsigned i = 0; i < intr->num_components; i++)
         components[i] = i < channels ? nir_channel(b, value, i) :
            nir_i2f32(b, nir_channel(b, &intr->def, i));
      value = nir_vec(b, components, intr->num_components);
      nir_def_rewrite_uses_after(&intr->def, value);
   } else {
      b->cursor = nir_before_instr(&intr->instr);
      nir_def *value = nir_format_float_to_snorm(b,
         nir_trim_vector(b, intr->src[3].ssa, channels), bits);
      nir_src_rewrite(&intr->src[3],
         nir_pad_vector_imm_int(b, value, 0, intr->num_components));
      nir_intrinsic_set_src_type(intr, nir_type_int32);
   }
}

static bool
rd_lower_resource(nir_builder *b, nir_instr *instr, void *data)
{
   struct rd_resource_lower *state = data;
   if (state->failed)
      return false;
   if (instr->type == nir_instr_type_tex) {
      if (!rd_lower_texture(nir_instr_as_tex(instr), state))
         state->failed = true;
      return !state->failed;
   }
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   bool access = false;
   switch (intr->intrinsic) {
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
      access = true;
      if (nir_intrinsic_format(intr) == PIPE_FORMAT_NONE) {
         nir_variable *variable = nir_deref_instr_get_variable(nir_src_as_deref(intr->src[0]));
         if (variable && (variable->data.image.format == PIPE_FORMAT_R32_UINT ||
                          variable->data.image.format == PIPE_FORMAT_R32_SINT))
            nir_intrinsic_set_format(intr, variable->data.image.format);
      }
      if ((nir_intrinsic_format(intr) != PIPE_FORMAT_R32_UINT &&
           nir_intrinsic_format(intr) != PIPE_FORMAT_R32_SINT) || intr->def.bit_size != 32) {
         state->failed = true;
         return false;
      }
      switch (nir_intrinsic_atomic_op(intr)) {
      case nir_atomic_op_iadd:
      case nir_atomic_op_imin:
      case nir_atomic_op_umin:
      case nir_atomic_op_imax:
      case nir_atomic_op_umax:
      case nir_atomic_op_iand:
      case nir_atomic_op_ior:
      case nir_atomic_op_ixor:
      case nir_atomic_op_xchg:
      case nir_atomic_op_cmpxchg:
         break;
      default:
         state->failed = true;
         return false;
      }
      break;
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_store:
      access = true;
      if (!rd_storage_format(nir_intrinsic_format(intr)) ||
          (intr->intrinsic == nir_intrinsic_image_deref_load &&
           intr->def.bit_size != 32) ||
          (intr->intrinsic == nir_intrinsic_image_deref_store &&
           nir_src_bit_size(intr->src[3]) != 32)) {
         state->failed = true;
         return false;
      }
      break;
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_image_deref_samples:
      break;
   default:
      if (!strncmp(nir_intrinsic_infos[intr->intrinsic].name, "image_", 6) ||
          !strncmp(nir_intrinsic_infos[intr->intrinsic].name, "bindless_image_", 15))
         state->failed = true;
      return false;
   }
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   if ((dim != GLSL_SAMPLER_DIM_1D && dim != GLSL_SAMPLER_DIM_2D &&
        dim != GLSL_SAMPLER_DIM_3D && dim != GLSL_SAMPLER_DIM_CUBE &&
        dim != GLSL_SAMPLER_DIM_BUF) ||
       (nir_intrinsic_image_array(intr) && dim != GLSL_SAMPLER_DIM_1D &&
        dim != GLSL_SAMPLER_DIM_2D)) {
      state->failed = true;
      return false;
   }
   uint32_t element;
   const struct rd_nir_resource_binding *entry = rd_resource_binding(state,
      nir_src_as_deref(intr->src[0]), &element);
   if (!entry || (entry->type != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE &&
                  entry->type != VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) ||
       ((dim == GLSL_SAMPLER_DIM_BUF) !=
        (entry->type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER)) ||
       !rd_mark_resource(state, entry, element, entry->image_slot,
                         &state->usage->images, &state->usage->image_count)) {
      state->failed = true;
      return false;
   }
   if (dim == GLSL_SAMPLER_DIM_BUF)
      state->image_buffers |= 1u << (entry->image_slot + element);
   b->cursor = nir_before_instr(instr);
   if (intr->intrinsic == nir_intrinsic_image_deref_samples) {
      nir_def_replace(&intr->def, nir_imm_intN_t(b, 1, intr->def.bit_size));
      return true;
   }
   if (access)
      nir_intrinsic_set_access(intr,
         (nir_intrinsic_access(intr) & ~ACCESS_CAN_REORDER) | ACCESS_COHERENT);
   nir_rewrite_image_intrinsic(intr, nir_imm_int(b, entry->image_slot + element),
                               nir_image_intrinsic_type_default);
   rd_lower_snorm_storage(b, intr);
   return true;
}

bool
rd_nir_lower_compute_resources(nir_shader *shader,
   rd_nir_resource_resolver resolve, void *data,
   struct rd_nir_resource_usage *usage)
{
   if (!shader || shader->info.stage != MESA_SHADER_COMPUTE || !usage || !resolve)
      return false;
   memset(usage, 0, sizeof(*usage));
   struct rd_resource_lower state = {
      .resolve = resolve, .data = data, .usage = usage,
   };
   nir_shader_instructions_pass(shader, rd_lower_resource,
                                nir_metadata_control_flow, &state);
   if (state.failed)
      return false;
   BITSET_ZERO(shader->info.textures_used);
   BITSET_ZERO(shader->info.textures_used_by_txf);
   BITSET_ZERO(shader->info.samplers_used);
   BITSET_ZERO(shader->info.texture_buffers);
   BITSET_ZERO(shader->info.images_used);
   BITSET_ZERO(shader->info.image_buffers);
   BITSET_ZERO(shader->info.msaa_images);
   for (uint32_t i = 0; i < 16; i++) {
      if (usage->textures & (1u << i))
         BITSET_SET(shader->info.textures_used, i);
      if (state.textures_by_txf & (1u << i))
         BITSET_SET(shader->info.textures_used_by_txf, i);
      if (usage->samplers & (1u << i))
         BITSET_SET(shader->info.samplers_used, i);
      if (state.texture_buffers & (1u << i))
         BITSET_SET(shader->info.texture_buffers, i);
      if (usage->images & (1u << i))
         BITSET_SET(shader->info.images_used, i);
      if (state.image_buffers & (1u << i))
         BITSET_SET(shader->info.image_buffers, i);
   }
   shader->info.num_images = usage->image_count;
   shader->info.num_textures = usage->texture_count;
   return true;
}

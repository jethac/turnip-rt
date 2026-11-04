#include "util/u_debug.h"
#include <stdio.h>
#include <math.h>
#include "rd_nir_resources.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_format_convert.h"
#include "util/format/u_format.h"
#include <string.h>

struct rd_resource_lower {
   bool fragment;
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
   if ((tex->op == nir_texop_tex || tex->op == nir_texop_txb || tex->op == nir_texop_lod) &&
       !state->fragment)
      return false; /* implicit derivatives exist only in fragment shaders */
   switch (tex->op) {
   case nir_texop_tex:
   case nir_texop_txb:
   case nir_texop_lod:
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
                  entry->type != VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER &&
                  entry->type != VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT))
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
              tex->op == nir_texop_tg4 || tex->op == nir_texop_tex ||
              tex->op == nir_texop_txb || tex->op == nir_texop_lod) {
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

/* Storage formats that ldib/stib convert correctly (SNORM ones are handled
 * above as SINT). Every other storage format is accessed as an unsigned
 * integer format of the same texel size, converted in the shader. The
 * hardware path returned wrong values for two-component formats, UNORM16
 * and the packed 10/11-bit formats, and faulted on some of them. The
 * descriptor side (rd_storage_resource_format) applies the same rule.
 */
static bool
rd_storage_format_native(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R8_UNORM:
   case PIPE_FORMAT_R8_UINT:
   case PIPE_FORMAT_R8_SINT:
   case PIPE_FORMAT_R8_SNORM:
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R8G8B8A8_UINT:
   case PIPE_FORMAT_R8G8B8A8_SINT:
   case PIPE_FORMAT_R8G8B8A8_SNORM:
   case PIPE_FORMAT_R16_UINT:
   case PIPE_FORMAT_R16_SINT:
   case PIPE_FORMAT_R16_FLOAT:
   case PIPE_FORMAT_R16_SNORM:
   case PIPE_FORMAT_R16G16B16A16_UINT:
   case PIPE_FORMAT_R16G16B16A16_SINT:
   case PIPE_FORMAT_R16G16B16A16_FLOAT:
   case PIPE_FORMAT_R16G16B16A16_SNORM:
   case PIPE_FORMAT_R32_UINT:
   case PIPE_FORMAT_R32_SINT:
   case PIPE_FORMAT_R32_FLOAT:
   case PIPE_FORMAT_R32G32B32A32_UINT:
   case PIPE_FORMAT_R32G32B32A32_SINT:
   case PIPE_FORMAT_R32G32B32A32_FLOAT:
      return true;
   default:
      return false;
   }
}

/* Unsigned 5-bit-exponent float (10/11-bit) to fp32 bits, in integer
 * arithmetic so denormals, infinities and NaNs survive (the 16-bit float
 * conversion path flushes denormals). */
static nir_def *
rd_unpack_ufloat(nir_builder *b, nir_def *value, unsigned mantissa_bits)
{
   nir_def *mantissa = nir_iand_imm(b, value, (1u << mantissa_bits) - 1);
   nir_def *exponent = nir_ushr_imm(b, value, mantissa_bits);
   nir_def *shifted = nir_ishl_imm(b, mantissa, 23 - mantissa_bits);
   nir_def *normal = nir_ior(b, nir_ishl_imm(b, nir_iadd_imm(b, exponent, 127 - 15), 23), shifted);
   nir_def *special = nir_ior_imm(b, shifted, 0x7f800000);
   nir_def *denormal = nir_fmul_imm(b, nir_u2f32(b, mantissa),
                                    ldexp(1.0, -14 - (int)mantissa_bits));
   return nir_bcsel(b, nir_ieq_imm(b, exponent, 0), denormal,
                    nir_bcsel(b, nir_ieq_imm(b, exponent, 31), special, normal));
}

/* fp32 to an unsigned 5-bit-exponent float, in integer arithmetic for the
 * same reason. Negative values become 0, overflow becomes infinity, NaN
 * stays NaN; the mantissa is truncated. */
static nir_def *
rd_pack_ufloat(nir_builder *b, nir_def *value, unsigned mantissa_bits)
{
   nir_def *bits = value;
   nir_def *exponent32 = nir_ubitfield_extract_imm(b, bits, 23, 8);
   nir_def *mantissa32 = nir_iand_imm(b, bits, 0x7fffff);
   nir_def *mantissa = nir_ushr_imm(b, mantissa32, 23 - mantissa_bits);
   nir_def *exponent = nir_iadd_imm(b, exponent32, 15 - 127);
   nir_def *infinity = nir_imm_int(b, 31u << mantissa_bits);
   nir_def *nan = nir_ior(b, infinity, nir_umax(b, mantissa, nir_imm_int(b, 1)));
   nir_def *normal = nir_ior(b, nir_ishl_imm(b, exponent, mantissa_bits), mantissa);
   nir_def *denormal = nir_f2u32(b, nir_fmul_imm(b, value, ldexp(1.0, 14 + (int)mantissa_bits)));
   nir_def *result = nir_bcsel(b, nir_ilt_imm(b, exponent, 1), denormal,
                               nir_bcsel(b, nir_ige_imm(b, exponent, 31), infinity, normal));
   result = nir_bcsel(b, nir_ieq_imm(b, exponent32, 255),
                      nir_bcsel(b, nir_ine_imm(b, mantissa32, 0), nan, infinity), result);
   nir_def *negative = nir_iand(b, nir_ine_imm(b, nir_ushr_imm(b, bits, 31), 0),
      nir_inot(b, nir_iand(b, nir_ieq_imm(b, exponent32, 255), nir_ine_imm(b, mantissa32, 0))));
   return nir_bcsel(b, negative, nir_imm_int(b, 0), result);
}

static void
rd_lower_raw_storage(nir_builder *b, nir_intrinsic_instr *intr)
{
   if (intr->intrinsic != nir_intrinsic_image_load &&
       intr->intrinsic != nir_intrinsic_image_store)
      return;
   enum pipe_format format = nir_intrinsic_format(intr);
   if (format == PIPE_FORMAT_NONE || rd_storage_format_native(format))
      return;
   const unsigned bytes = util_format_get_blocksize(format);
   enum pipe_format raw;
   switch (bytes) {
   case 1: raw = PIPE_FORMAT_R8_UINT; break;
   case 2: raw = PIPE_FORMAT_R16_UINT; break;
   case 4: raw = PIPE_FORMAT_R32_UINT; break;
   case 8: raw = PIPE_FORMAT_R16G16B16A16_UINT; break;
   case 16: raw = PIPE_FORMAT_R32G32B32A32_UINT; break;
   default: return;
   }
   nir_intrinsic_set_format(intr, raw);
   if (intr->intrinsic == nir_intrinsic_image_load) {
      nir_intrinsic_set_dest_type(intr, nir_type_uint32);
      b->cursor = nir_after_instr(&intr->instr);
      nir_def *words;
      if (bytes <= 4) {
         words = nir_channel(b, &intr->def, 0);
      } else if (bytes == 8) {
         nir_def *halves[2];
         for (unsigned i = 0; i < 2; i++)
            halves[i] = nir_ior(b, nir_iand_imm(b, nir_channel(b, &intr->def, 2 * i), 0xffff),
                                nir_ishl_imm(b, nir_channel(b, &intr->def, 2 * i + 1), 16));
         words = nir_vec(b, halves, 2);
      } else {
         words = nir_trim_vector(b, &intr->def, 4);
      }
      nir_def *rgba;
      if (format == PIPE_FORMAT_R11G11B10_FLOAT) {
         rgba = nir_vec4(b,
            rd_unpack_ufloat(b, nir_ubitfield_extract_imm(b, words, 0, 11), 6),
            rd_unpack_ufloat(b, nir_ubitfield_extract_imm(b, words, 11, 11), 6),
            rd_unpack_ufloat(b, nir_ubitfield_extract_imm(b, words, 22, 10), 5),
            nir_imm_float(b, 1.0));
      } else {
         rgba = nir_format_unpack_rgba(b, words, format);
      }
      nir_def *components[4];
      for (unsigned i = 0; i < intr->num_components; i++)
         components[i] = nir_channel(b, rgba, MIN2(i, 3));
      nir_def_rewrite_uses_after(&intr->def, nir_vec(b, components, intr->num_components));
   } else {
      b->cursor = nir_before_instr(&intr->instr);
      nir_def *value = intr->src[3].ssa;
      nir_def *words;
      if (format == PIPE_FORMAT_R11G11B10_FLOAT) {
         words = nir_ior(b, rd_pack_ufloat(b, nir_channel(b, value, 0), 6),
                 nir_ior(b, nir_ishl_imm(b, rd_pack_ufloat(b, nir_channel(b, value, 1), 6), 11),
                            nir_ishl_imm(b, rd_pack_ufloat(b, nir_channel(b, value, 2), 5), 22)));
      } else {
         words = nir_format_pack_rgba(b, format,
            nir_trim_vector(b, value, MIN2(value->num_components, 4)));
      }
      nir_def *raw_value;
      if (bytes <= 4) {
         raw_value = nir_pad_vector_imm_int(b, nir_channel(b, words, 0), 0, intr->num_components);
      } else if (bytes == 8) {
         nir_def *parts[4];
         for (unsigned i = 0; i < 2; i++) {
            nir_def *w = nir_channel(b, words, i);
            parts[2 * i] = nir_iand_imm(b, w, 0xffff);
            parts[2 * i + 1] = nir_ushr_imm(b, w, 16);
         }
         raw_value = nir_pad_vector_imm_int(b, nir_vec(b, parts, 4), 0, intr->num_components);
      } else {
         raw_value = nir_pad_vector_imm_int(b, words, 0, intr->num_components);
      }
      nir_src_rewrite(&intr->src[3], raw_value);
      nir_intrinsic_set_src_type(intr, nir_type_uint32);
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
      /* Atomic loads/stores of an image texel arrive without a format;
       * take the image variable's. */
      if (nir_intrinsic_format(intr) == PIPE_FORMAT_NONE) {
         nir_variable *variable = nir_deref_instr_get_variable(nir_src_as_deref(intr->src[0]));
         if (variable)
            nir_intrinsic_set_format(intr, variable->data.image.format);
      }
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
   rd_lower_raw_storage(b, intr);
   rd_lower_snorm_storage(b, intr);
   return true;
}

/* Atomic loads and stores of image texels (OpAtomicLoad/OpAtomicStore on an
 * image) become atomic add-zero and exchange: a plain ldib/stib pair is not
 * ordered against another wave's access by a fence (memory_model
 * message_passing with an image guard and atomicwrite failed across waves),
 * while the atomics go to memory. */
static bool
rd_lower_atomic_image_access(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if ((intr->intrinsic != nir_intrinsic_image_deref_load &&
        intr->intrinsic != nir_intrinsic_image_deref_store) ||
       !(nir_intrinsic_access(intr) & ACCESS_ATOMIC))
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_variable *variable = nir_deref_instr_get_variable(nir_src_as_deref(intr->src[0]));
   enum pipe_format format = nir_intrinsic_format(intr);
   if (format == PIPE_FORMAT_NONE && variable)
      format = variable->data.image.format;
   if (format != PIPE_FORMAT_R32_UINT && format != PIPE_FORMAT_R32_SINT)
      return false;
   if (intr->intrinsic == nir_intrinsic_image_deref_load) {
      if (intr->def.bit_size != 32)
         return false;
      nir_def *value = nir_image_deref_atomic(b, 32, intr->src[0].ssa, intr->src[1].ssa,
         intr->src[2].ssa, nir_imm_int(b, 0), .image_dim = nir_intrinsic_image_dim(intr),
         .image_array = nir_intrinsic_image_array(intr), .format = format,
         .access = nir_intrinsic_access(intr), .atomic_op = nir_atomic_op_iadd);
      nir_def *result = intr->def.num_components == 1 ? value :
         nir_vec4(b, value, nir_imm_int(b, 0), nir_imm_int(b, 0), nir_imm_int(b, 1));
      nir_def_replace(&intr->def, nir_trim_vector(b, result, intr->def.num_components));
   } else {
      if (nir_src_bit_size(intr->src[3]) != 32)
         return false;
      nir_image_deref_atomic(b, 32, intr->src[0].ssa, intr->src[1].ssa, intr->src[2].ssa,
         nir_channel(b, intr->src[3].ssa, 0), .image_dim = nir_intrinsic_image_dim(intr),
         .image_array = nir_intrinsic_image_array(intr), .format = format,
         .access = nir_intrinsic_access(intr), .atomic_op = nir_atomic_op_xchg);
      nir_instr_remove(&intr->instr);
   }
   return true;
}

bool
rd_nir_lower_compute_resources(nir_shader *shader,
   rd_nir_resource_resolver resolve, void *data,
   struct rd_nir_resource_usage *usage)
{
   if (!shader || (shader->info.stage != MESA_SHADER_COMPUTE &&
                   shader->info.stage != MESA_SHADER_VERTEX &&
                   shader->info.stage != MESA_SHADER_FRAGMENT) || !usage || !resolve)
      return false;
   memset(usage, 0, sizeof(*usage));
   struct rd_resource_lower state = {
      .fragment = shader->info.stage == MESA_SHADER_FRAGMENT,
      .resolve = resolve, .data = data, .usage = usage,
   };
   nir_shader_intrinsics_pass(shader, rd_lower_atomic_image_access,
                              nir_metadata_control_flow, NULL);
   nir_shader_instructions_pass(shader, rd_lower_resource,
                                nir_metadata_control_flow, &state);
   if (state.failed && debug_get_bool_option("RADISH_DEBUG_ERRORS", false)) {
      nir_foreach_function_impl(impl, shader) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_intrinsic &&
                   !strncmp(nir_intrinsic_infos[nir_instr_as_intrinsic(instr)->intrinsic].name, "image", 5))
                  fprintf(stderr, "radish: resource lowering left %s\n",
                          nir_intrinsic_infos[nir_instr_as_intrinsic(instr)->intrinsic].name);
               else if (instr->type == nir_instr_type_tex)
                  fprintf(stderr, "radish: resource lowering left tex op %d dim %d array %d\n",
                          nir_instr_as_tex(instr)->op, nir_instr_as_tex(instr)->sampler_dim,
                          nir_instr_as_tex(instr)->is_array);
            }
         }
      }
   }
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

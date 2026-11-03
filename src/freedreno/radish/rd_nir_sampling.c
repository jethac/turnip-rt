#include "rd_sampling.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_format_convert.h"
#include "vk_format.h"

struct rd_fetch_state { const struct rd_sampling_key *key; uint32_t buffer; bool failed; };

static nir_def *
rd_level_value(nir_builder *b, nir_def *lod, const struct rd_sampling_view *view,
                unsigned member)
{
   nir_def *value = nir_imm_int(b, 0);
   for (unsigned i = 0; i < view->levels; i++) {
      const struct rd_sampling_level *level = &view->level[i];
      const uint32_t fields[] = {level->offset, level->row_pitch, level->array_pitch,
                                 level->width, level->height};
      value = nir_bcsel(b, nir_ieq_imm(b, lod, i), nir_imm_int(b, fields[member]), value);
   }
   return value;
}

static nir_def *
rd_fetch_swizzle(nir_builder *b, nir_def *rgba, VkComponentMapping swizzle,
                  enum pipe_format format)
{
   VkComponentSwizzle components[] = {swizzle.r, swizzle.g, swizzle.b, swizzle.a};
   nir_def *result[4];
   for (unsigned i = 0; i < 4; i++) {
      VkComponentSwizzle s = components[i];
      if (s == VK_COMPONENT_SWIZZLE_IDENTITY)
         result[i] = nir_channel(b, rgba, i);
      else if (s >= VK_COMPONENT_SWIZZLE_R && s <= VK_COMPONENT_SWIZZLE_A)
         result[i] = nir_channel(b, rgba, s - VK_COMPONENT_SWIZZLE_R);
      else if (s == VK_COMPONENT_SWIZZLE_ONE)
         result[i] = util_format_is_pure_integer(format) ? nir_imm_int(b, 1) : nir_imm_float(b, 1);
      else
         result[i] = nir_imm_int(b, 0);
   }
   return nir_vec(b, result, 4);
}

static enum pipe_format
rd_sample_format(const struct rd_sampling_view *view)
{
   return view->format == VK_FORMAT_D16_UNORM ? PIPE_FORMAT_R16_UNORM :
      view->format == VK_FORMAT_D32_SFLOAT ? PIPE_FORMAT_R32_FLOAT :
      view->format == VK_FORMAT_X8_D24_UNORM_PACK32 ? PIPE_FORMAT_R32_FLOAT :
      view->format == VK_FORMAT_S8_UINT ? PIPE_FORMAT_R8_UINT : vk_format_to_pipe_format(view->format);
}

static nir_def *
rd_raw_sample(nir_builder *b, struct rd_fetch_state *state, unsigned texture,
               nir_def *lod, nir_def *layer, nir_def *x, nir_def *y, nir_def *sample)
{
   const struct rd_sampling_view *view = &state->key->views[texture];
   nir_def *valid = nir_iand(b, nir_ult_imm(b, lod, view->levels),
      nir_iand(b, nir_ult_imm(b, layer, view->layers), nir_iand(b,
         nir_ult(b, x, rd_level_value(b, lod, view, 3)),
         nir_ult(b, y, rd_level_value(b, lod, view, 4)))));
   valid = nir_iand(b, valid, nir_ult_imm(b, sample, view->samples));
   nir_def *zero = nir_imm_zero(b, 4, 32);
   nir_push_if(b, valid);
   nir_def *pair = nir_load_ssbo(b, 2, 32, nir_imm_int(b, state->buffer),
      nir_imm_int(b, texture * 8), .align_mul = 8);
   nir_def *offset = nir_iadd(b, rd_level_value(b, lod, view, 0),
      nir_iadd(b, nir_imul(b, layer, rd_level_value(b, lod, view, 2)),
         nir_iadd(b, nir_imul(b, y, rd_level_value(b, lod, view, 1)),
            nir_iadd(b, nir_imul_imm(b, x, view->cpp * view->samples), nir_imul_imm(b, sample, view->cpp)))));
   nir_def *address = nir_iadd(b, nir_pack_64_2x32(b, pair),
      nir_u2u64(b, nir_iand_imm(b, offset, ~3u)));
   nir_def *packed = nir_load_global(b, MAX2(view->cpp, 4) / 4, 32, address, .align_mul = 4);
   if (view->cpp < 4)
      packed = nir_ushr(b, packed, nir_imul_imm(b, nir_iand_imm(b, offset, 3), 8));
   if (view->format == VK_FORMAT_S8_UINT && view->cpp == 4)
      packed = nir_ushr_imm(b, packed, 24);
   nir_def *rgba;
   if (view->format == VK_FORMAT_X8_D24_UNORM_PACK32) {
      unsigned bits[1] = {24};
      nir_def *depth = nir_format_unorm_to_float(b, nir_iand_imm(b, packed, 0x00ffffff), bits);
      rgba = nir_vec4(b, depth, nir_imm_float(b, 0), nir_imm_float(b, 0), nir_imm_float(b, 1));
   } else {
      rgba = nir_format_unpack_rgba(b, packed, rd_sample_format(view));
   }
   nir_pop_if(b, NULL);
   return nir_if_phi(b, rgba, zero);
}

static nir_def *
rd_raw_pixel(nir_builder *b, struct rd_fetch_state *state, unsigned texture,
               nir_def *lod, nir_def *layer, nir_def *x, nir_def *y)
{
   return rd_raw_sample(b, state, texture, lod, layer, x, y, nir_imm_int(b, 0));
}

static nir_def *
rd_reduce_coordinate(nir_builder *b, nir_def *value, unsigned wrap)
{
   if (wrap == 0)
      return nir_ffract(b, value);
   if (wrap == 2)
      return nir_fmul_imm(b, nir_ffract(b, nir_fmul_imm(b, value, 0.5)), 2.0);
   if (wrap == 4) /* mirror once, then clamp to edge */
      return nir_fmin(b, nir_fabs(b, value), nir_imm_float(b, 17));
   return nir_fmin(b, nir_fmax(b, value, nir_imm_float(b, -16)), nir_imm_float(b, 17));
}

static nir_def *
rd_wrap_coordinate(nir_builder *b, nir_def *coordinate, nir_def *size,
                     unsigned wrap, nir_def **inside)
{
   if (wrap == 0 || wrap == 2) {
      nir_def *period = wrap == 2 ? nir_imul_imm(b, size, 2) : size;
      nir_def *value = nir_irem(b, coordinate, period);
      value = nir_bcsel(b, nir_ilt_imm(b, value, 0), nir_iadd(b, value, period), value);
      return wrap == 2 ? nir_bcsel(b, nir_ige(b, value, size),
         nir_isub(b, nir_iadd_imm(b, period, -1), value), value) : value;
   }
   if (wrap == 3)
      *inside = nir_iand(b, *inside, nir_ult(b, coordinate, size));
   if (wrap == 4)
      coordinate = nir_bcsel(b, nir_ilt_imm(b, coordinate, 0),
                             nir_isub(b, nir_imm_int(b, -1), coordinate), coordinate);
   return nir_imin(b, nir_imax(b, coordinate, nir_imm_int(b, 0)), nir_iadd_imm(b, size, -1));
}

static nir_def *
rd_depth_compare(nir_builder *b, unsigned compare, nir_def *reference, nir_def *value)
{
   nir_def *pass;
   switch (compare) {
   case VK_COMPARE_OP_NEVER: return nir_imm_float(b, 0);
   case VK_COMPARE_OP_LESS: pass = nir_flt(b, reference, value); break;
   case VK_COMPARE_OP_EQUAL: pass = nir_feq(b, reference, value); break;
   case VK_COMPARE_OP_LESS_OR_EQUAL: pass = nir_fge(b, value, reference); break;
   case VK_COMPARE_OP_GREATER: pass = nir_flt(b, value, reference); break;
   case VK_COMPARE_OP_NOT_EQUAL: pass = nir_fneu(b, reference, value); break;
   case VK_COMPARE_OP_GREATER_OR_EQUAL: pass = nir_fge(b, reference, value); break;
   default: return nir_imm_float(b, 1);
   }
   return nir_b2f32(b, pass);
}

struct rd_cube_axes { nir_def *face, *sc, *tc, *major; };

static nir_def *
rd_cube_select(nir_builder *b, nir_def *face, nir_def *values[6])
{
   nir_def *value = values[5];
   for (unsigned i = 0; i < 5; i++)
      value = nir_bcsel(b, nir_ieq_imm(b, face, i), values[i], value);
   return value;
}

static struct rd_cube_axes
rd_cube_axes(nir_builder *b, nir_def *direction)
{
   nir_def *x = nir_channel(b, direction, 0), *y = nir_channel(b, direction, 1), *z = nir_channel(b, direction, 2);
   nir_def *ax = nir_fabs(b, x), *ay = nir_fabs(b, y), *az = nir_fabs(b, z);
   nir_def *z_major = nir_iand(b, nir_fge(b, az, ax), nir_fge(b, az, ay));
   nir_def *y_major = nir_fge(b, ay, ax);
   nir_def *face = nir_bcsel(b, z_major,
      nir_bcsel(b, nir_flt_imm(b, z, 0), nir_imm_int(b, 5), nir_imm_int(b, 4)),
      nir_bcsel(b, y_major,
         nir_bcsel(b, nir_flt_imm(b, y, 0), nir_imm_int(b, 3), nir_imm_int(b, 2)),
         nir_bcsel(b, nir_flt_imm(b, x, 0), nir_imm_int(b, 1), nir_imm_int(b, 0))));
   nir_def *sc[] = {nir_fneg(b, z), z, x, x, x, nir_fneg(b, x)};
   nir_def *tc[] = {nir_fneg(b, y), nir_fneg(b, y), z, nir_fneg(b, z), nir_fneg(b, y), nir_fneg(b, y)};
   nir_def *major = nir_bcsel(b, z_major, az, nir_bcsel(b, y_major, ay, ax));
   major = nir_bcsel(b, nir_flt(b, nir_imm_float(b, 0), major), major, nir_imm_float(b, 1));
   return (struct rd_cube_axes){face, rd_cube_select(b, face, sc), rd_cube_select(b, face, tc), major};
}

static nir_def *
rd_cube_coordinate(nir_builder *b, nir_def *direction)
{
   struct rd_cube_axes axis = rd_cube_axes(b, direction);
   return nir_vec3(b, nir_fadd_imm(b, nir_fmul_imm(b, nir_fdiv(b, axis.sc, axis.major), 0.5), 0.5),
      nir_fadd_imm(b, nir_fmul_imm(b, nir_fdiv(b, axis.tc, axis.major), 0.5), 0.5), nir_u2f32(b, axis.face));
}

static nir_def *
rd_cube_derivative(nir_builder *b, nir_def *direction, nir_def *gradient)
{
   struct rd_cube_axes axis = rd_cube_axes(b, direction);
   nir_def *x = nir_channel(b, gradient, 0), *y = nir_channel(b, gradient, 1), *z = nir_channel(b, gradient, 2);
   nir_def *sc[] = {nir_fneg(b, z), z, x, x, x, nir_fneg(b, x)};
   nir_def *tc[] = {nir_fneg(b, y), nir_fneg(b, y), z, nir_fneg(b, z), nir_fneg(b, y), nir_fneg(b, y)};
   nir_def *ma[] = {x, nir_fneg(b, x), y, nir_fneg(b, y), z, nir_fneg(b, z)};
   nir_def *relative = nir_fdiv(b, rd_cube_select(b, axis.face, ma), axis.major);
   nir_def *ds = nir_fsub(b, nir_fdiv(b, rd_cube_select(b, axis.face, sc), axis.major),
      nir_fmul(b, nir_fdiv(b, axis.sc, axis.major), relative));
   nir_def *dt = nir_fsub(b, nir_fdiv(b, rd_cube_select(b, axis.face, tc), axis.major),
      nir_fmul(b, nir_fdiv(b, axis.tc, axis.major), relative));
   return nir_fmul_imm(b, nir_vec2(b, ds, dt), 0.5);
}

static nir_def *
rd_cube_pixel(nir_builder *b, struct rd_fetch_state *state, unsigned texture,
                 nir_def *lod, nir_def *face, nir_def *x, nir_def *y)
{
   const struct rd_sampling_view *view = &state->key->views[texture];
   nir_def *size = rd_level_value(b, lod, view, 3);
   nir_def *inside = nir_iand(b, nir_ult(b, x, size), nir_ult(b, y, size));
   nir_def *u = nir_fadd_imm(b, nir_fdiv(b, nir_fadd_imm(b, nir_fmul_imm(b, nir_i2f32(b, x), 2), 1), nir_u2f32(b, size)), -1);
   nir_def *v = nir_fadd_imm(b, nir_fdiv(b, nir_fadd_imm(b, nir_fmul_imm(b, nir_i2f32(b, y), 2), 1), nir_u2f32(b, size)), -1);
   nir_def *one = nir_imm_float(b, 1), *minus_one = nir_imm_float(b, -1);
   nir_def *directions[] = {
      nir_vec3(b, one, nir_fneg(b, v), nir_fneg(b, u)), nir_vec3(b, minus_one, nir_fneg(b, v), u),
      nir_vec3(b, u, one, v), nir_vec3(b, u, minus_one, nir_fneg(b, v)),
      nir_vec3(b, u, nir_fneg(b, v), one), nir_vec3(b, nir_fneg(b, u), nir_fneg(b, v), minus_one),
   };
   nir_def *projected = rd_cube_coordinate(b, rd_cube_select(b, face, directions));
   nir_def *new_x = nir_f2u32(b, nir_ffloor(b, nir_fmul(b, nir_fmax(b, nir_channel(b, projected, 0), nir_imm_float(b, 0)), nir_u2f32(b, size))));
   nir_def *new_y = nir_f2u32(b, nir_ffloor(b, nir_fmul(b, nir_fmax(b, nir_channel(b, projected, 1), nir_imm_float(b, 0)), nir_u2f32(b, size))));
   new_x = nir_umin(b, new_x, nir_iadd_imm(b, size, -1));
   new_y = nir_umin(b, new_y, nir_iadd_imm(b, size, -1));
   return rd_raw_pixel(b, state, texture, lod,
      nir_bcsel(b, inside, face, nir_f2u32(b, nir_channel(b, projected, 2))),
      nir_bcsel(b, inside, x, new_x), nir_bcsel(b, inside, y, new_y));
}

static nir_def *
rd_sample_pixel(nir_builder *b, struct rd_fetch_state *state, const nir_tex_instr *tex,
                  nir_def *lod, nir_def *layer, nir_def *x, nir_def *y,
                  nir_def *reference)
{
   const struct rd_sampling_view *view = &state->key->views[tex->texture_index];
   const uint32_t *sampler = state->key->samplers[tex->sampler_index];
   nir_def *inside = nir_imm_true(b);
   nir_def *rgba;
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE) {
      rgba = rd_cube_pixel(b, state, tex->texture_index, lod, layer, x, y);
   } else {
      x = rd_wrap_coordinate(b, x, rd_level_value(b, lod, view, 3), (sampler[0] >> 5) & 7, &inside);
      y = rd_wrap_coordinate(b, y, rd_level_value(b, lod, view, 4), (sampler[0] >> 8) & 7, &inside);
      rgba = rd_raw_pixel(b, state, tex->texture_index, lod, layer, x, y);
   }
   unsigned border = sampler[2] >> 7;
   bool integer = util_format_is_pure_integer(rd_sample_format(view));
   nir_def *one = integer ? nir_imm_int(b, 1) : nir_imm_float(b, 1);
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *rgb = border >= 4 ? one : zero;
   nir_def *alpha = border >= 2 ? one : zero;
   rgba = nir_bcsel(b, inside, rgba, nir_vec4(b, rgb, rgb, rgb, alpha));
   if (tex->is_shadow) {
      if (view->format == VK_FORMAT_D16_UNORM || view->format == VK_FORMAT_X8_D24_UNORM_PACK32)
         reference = nir_fsat(b, reference);
      nir_def *comparison = rd_depth_compare(b, (sampler[1] >> 1) & 7,
         reference, nir_channel(b, rgba, 0));
      rgba = nir_vector_insert_imm(b, rgba, comparison, 0);
   }
   return rd_fetch_swizzle(b, rgba, view->swizzle, rd_sample_format(view));
}

static nir_def *
rd_filter_level(nir_builder *b, struct rd_fetch_state *state, const nir_tex_instr *tex,
                  nir_def *lod, nir_def *layer, nir_def *s, nir_def *t,
                  nir_def *dx, nir_def *dy, nir_def *reference, bool linear)
{
   const struct rd_sampling_view *view = &state->key->views[tex->texture_index];
   nir_def *u = nir_fadd(b, nir_fmul(b, s, nir_u2f32(b, rd_level_value(b, lod, view, 3))), nir_i2f32(b, dx));
   nir_def *v = nir_fadd(b, nir_fmul(b, t, nir_u2f32(b, rd_level_value(b, lod, view, 4))), nir_i2f32(b, dy));
   if (!linear) {
      nir_def *x = nir_f2i32(b, nir_ffloor(b, u)), *y = nir_f2i32(b, nir_ffloor(b, v));
      if (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE) {
         x = nir_imin(b, nir_imax(b, x, nir_imm_int(b, 0)), nir_iadd_imm(b, rd_level_value(b, lod, view, 3), -1));
         y = nir_imin(b, nir_imax(b, y, nir_imm_int(b, 0)), nir_iadd_imm(b, rd_level_value(b, lod, view, 4), -1));
      }
      return rd_sample_pixel(b, state, tex, lod, layer, x, y, reference);
   }
   u = nir_fadd_imm(b, u, -0.5);
   v = nir_fadd_imm(b, v, -0.5);
   nir_def *x = nir_f2i32(b, nir_ffloor(b, u)), *y = nir_f2i32(b, nir_ffloor(b, v));
   nir_def *a = rd_sample_pixel(b, state, tex, lod, layer, x, y, reference);
   nir_def *c = rd_sample_pixel(b, state, tex, lod, layer, x, nir_iadd_imm(b, y, 1), reference);
   nir_def *d = rd_sample_pixel(b, state, tex, lod, layer, nir_iadd_imm(b, x, 1), nir_iadd_imm(b, y, 1), reference);
   nir_def *e = rd_sample_pixel(b, state, tex, lod, layer, nir_iadd_imm(b, x, 1), y, reference);
   return nir_flrp(b, nir_flrp(b, a, e, nir_ffract(b, u)),
      nir_flrp(b, c, d, nir_ffract(b, u)), nir_ffract(b, v));
}

static nir_def *
rd_filter_mips(nir_builder *b, struct rd_fetch_state *state, const nir_tex_instr *tex,
                 nir_def *level, nir_def *layer, nir_def *s, nir_def *t,
                 nir_def *dx, nir_def *dy, nir_def *reference, bool linear)
{
   const uint32_t *sampler = state->key->samplers[tex->sampler_index];
   if (!(sampler[0] & 1))
      return rd_filter_level(b, state, tex, nir_f2u32(b, nir_ffloor(b, nir_fadd_imm(b, level, 0.5))),
         layer, s, t, dx, dy, reference, linear);
   nir_def *lo = nir_f2u32(b, nir_ffloor(b, level));
   nir_def *hi = nir_umin(b, nir_iadd_imm(b, lo, 1),
      nir_imm_int(b, state->key->views[tex->texture_index].levels - 1));
   return nir_flrp(b, rd_filter_level(b, state, tex, lo, layer, s, t, dx, dy, reference, linear),
      rd_filter_level(b, state, tex, hi, layer, s, t, dx, dy, reference, linear), nir_ffract(b, level));
}

static nir_def *
rd_filtered_sample(nir_builder *b, struct rd_fetch_state *state, nir_tex_instr *tex,
                     nir_def *coord, nir_def *dx, nir_def *dy)
{
   const struct rd_sampling_view *view = &state->key->views[tex->texture_index];
   const uint32_t *sampler = state->key->samplers[tex->sampler_index];
   nir_def *direction = coord;
   bool cube = tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE;
   if (cube) coord = rd_cube_coordinate(b, direction);
   nir_def *s = cube ? nir_channel(b, coord, 0) : rd_reduce_coordinate(b, nir_channel(b, coord, 0), (sampler[0] >> 5) & 7);
   nir_def *t = cube ? nir_channel(b, coord, 1) : rd_reduce_coordinate(b, nir_channel(b, coord, 1), (sampler[0] >> 8) & 7);
   nir_def *layer = nir_f2u32(b, nir_fmin(b, nir_fmax(b,
      nir_fround_even(b, nir_channel(b, coord, 2)), nir_imm_float(b, 0)),
      nir_imm_float(b, view->layers - 1)));
   int comparator = nir_tex_instr_src_index(tex, nir_tex_src_comparator);
   nir_def *reference = comparator >= 0 ? tex->src[comparator].src.ssa : NULL;
   if (tex->op == nir_texop_tg4) {
      nir_def *lod = nir_imm_int(b, 0);
      nir_def *u = nir_fadd_imm(b, nir_fadd(b, nir_fmul_imm(b, s, view->level[0].width), nir_i2f32(b, dx)), -0.5);
      nir_def *v = nir_fadd_imm(b, nir_fadd(b, nir_fmul_imm(b, t, view->level[0].height), nir_i2f32(b, dy)), -0.5);
      nir_def *x = nir_f2i32(b, nir_ffloor(b, u)), *y = nir_f2i32(b, nir_ffloor(b, v));
      nir_def *values[4] = {
         rd_sample_pixel(b, state, tex, lod, layer, x, nir_iadd_imm(b, y, 1), reference),
         rd_sample_pixel(b, state, tex, lod, layer, nir_iadd_imm(b, x, 1), nir_iadd_imm(b, y, 1), reference),
         rd_sample_pixel(b, state, tex, lod, layer, nir_iadd_imm(b, x, 1), y, reference),
         rd_sample_pixel(b, state, tex, lod, layer, x, y, reference),
      };
      for (unsigned i = 0; i < 4; i++)
         values[i] = nir_channel(b, values[i], tex->is_shadow ? 0 : tex->component);
      return nir_vec(b, values, 4);
   }
   nir_def *lambda;
   if (tex->op == nir_texop_txd) {
      int xi = nir_tex_instr_src_index(tex, nir_tex_src_ddx);
      int yi = nir_tex_instr_src_index(tex, nir_tex_src_ddy);
      if (xi < 0 || yi < 0) { state->failed = true; return nir_imm_zero(b, 4, 32); }
      nir_def *gx = tex->src[xi].src.ssa, *gy = tex->src[yi].src.ssa;
      if (cube) { gx = rd_cube_derivative(b, direction, gx); gy = rd_cube_derivative(b, direction, gy); }
      nir_def *xx = nir_fmul_imm(b, nir_channel(b, gx, 0), view->level[0].width);
      nir_def *xy = nir_fmul_imm(b, nir_channel(b, gx, 1), view->level[0].height);
      nir_def *yx = nir_fmul_imm(b, nir_channel(b, gy, 0), view->level[0].width);
      nir_def *yy = nir_fmul_imm(b, nir_channel(b, gy, 1), view->level[0].height);
      lambda = nir_fmul_imm(b, nir_flog2(b, nir_fmax(b,
         nir_fadd(b, nir_fmul(b, xx, xx), nir_fmul(b, xy, xy)),
         nir_fadd(b, nir_fmul(b, yx, yx), nir_fmul(b, yy, yy)))), 0.5);
   } else {
      int li = nir_tex_instr_src_index(tex, nir_tex_src_lod);
      lambda = li >= 0 ? tex->src[li].src.ssa : nir_imm_float(b, 0);
   }
   int32_t bias_bits = (sampler[0] >> 19) & 0x1fff;
   if (bias_bits & 0x1000) bias_bits -= 0x2000;
   nir_def *bias = nir_imm_float(b, bias_bits / 256.0f);
   int bi = nir_tex_instr_src_index(tex, nir_tex_src_bias);
   if (bi >= 0) bias = nir_fadd(b, bias, tex->src[bi].src.ssa);
   bias = nir_fmin(b, nir_fmax(b, bias, nir_imm_float(b, -16)), nir_imm_float(b, 16));
   nir_def *minimum = nir_imm_float(b, ((sampler[1] >> 20) & 0xfff) / 256.0f);
   int mi = nir_tex_instr_src_index(tex, nir_tex_src_min_lod);
   if (mi >= 0) minimum = nir_fmax(b, minimum, tex->src[mi].src.ssa);
   lambda = nir_fmin(b, nir_fmax(b, nir_fadd(b, lambda, bias), minimum),
      nir_imm_float(b, ((sampler[1] >> 8) & 0xfff) / 256.0f));
   nir_def *level = nir_fmin(b, nir_fmax(b, lambda, nir_imm_float(b, 0)),
      nir_imm_float(b, view->levels - 1));
   bool magnify = ((sampler[0] >> 1) & 3) == 1;
   bool minify = ((sampler[0] >> 3) & 3) == 1;
   if (magnify == minify)
      return rd_filter_mips(b, state, tex, level, layer, s, t, dx, dy, reference, minify);
   nir_push_if(b, nir_flt(b, nir_imm_float(b, 0), lambda));
   nir_def *small = rd_filter_mips(b, state, tex, level, layer, s, t, dx, dy, reference, minify);
   nir_push_else(b, NULL);
   nir_def *large = rd_filter_mips(b, state, tex, level, layer, s, t, dx, dy, reference, magnify);
   nir_pop_if(b, NULL);
   return nir_if_phi(b, small, large);
}

static bool
rd_lower_fetch(nir_builder *b, nir_instr *instr, void *data)
{
   struct rd_fetch_state *state = data;
   if (state->failed || instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->texture_index >= 16 || !(state->key->mask & (1u << tex->texture_index)))
      return false;
   if (tex->op == nir_texop_txs || tex->op == nir_texop_query_levels)
      return false;
   const struct rd_sampling_view *view = &state->key->views[tex->texture_index];
   bool ms = tex->sampler_dim == GLSL_SAMPLER_DIM_MS && tex->is_array;
   if (ms && tex->op == nir_texop_texture_samples) {
      b->cursor = nir_before_instr(instr);
      nir_def_replace(&tex->def, nir_imm_intN_t(b, view->samples, tex->def.bit_size));
      return true;
   }
   if ((ms ? tex->op != nir_texop_txf_ms :
        (tex->op != nir_texop_txf && tex->op != nir_texop_txl && tex->op != nir_texop_txd && tex->op != nir_texop_tg4)) ||
       !((tex->sampler_dim == GLSL_SAMPLER_DIM_2D && tex->is_array) || ms ||
         (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE && !tex->is_array)) ||
       (!ms && view->samples != 1) || tex->def.bit_size != 32) {
      state->failed = true;
      return false;
   }
   int ci = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   int oi = nir_tex_instr_src_index(tex, nir_tex_src_offset);
   if (ci < 0 || tex->src[ci].src.ssa->num_components != 3 || tex->src[ci].src.ssa->bit_size != 32 ||
       nir_tex_instr_src_index(tex, nir_tex_src_projector) >= 0 ||
       (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE && (tex->op == nir_texop_txf || oi >= 0)) ||
       (tex->op == nir_texop_tg4 &&
        (tex->is_gather_implicit_lod || nir_tex_instr_has_explicit_tg4_offsets(tex)))) {
      state->failed = true;
      return false;
   }
   b->cursor = nir_before_instr(instr);
   nir_def *coord = tex->src[ci].src.ssa;
   nir_def *dx = oi >= 0 ? nir_channel(b, tex->src[oi].src.ssa, 0) : nir_imm_int(b, 0);
   nir_def *dy = oi >= 0 ? nir_channel(b, tex->src[oi].src.ssa, 1) : nir_imm_int(b, 0);
   nir_def *rgba;
   if (ms) {
      int si = nir_tex_instr_src_index(tex, nir_tex_src_ms_index);
      if (tex->is_shadow || si < 0 || tex->src[si].src.ssa->num_components != 1 ||
          tex->src[si].src.ssa->bit_size != 32) {
         state->failed = true;
         return false;
      }
      rgba = rd_raw_sample(b, state, tex->texture_index, nir_imm_int(b, 0),
         nir_channel(b, coord, 2), nir_iadd(b, nir_channel(b, coord, 0), dx),
         nir_iadd(b, nir_channel(b, coord, 1), dy), tex->src[si].src.ssa);
      rgba = rd_fetch_swizzle(b, rgba, view->swizzle, rd_sample_format(view));
   } else if (tex->op == nir_texop_txf) {
      if (tex->is_shadow) { state->failed = true; return false; }
      int li = nir_tex_instr_src_index(tex, nir_tex_src_lod);
      rgba = rd_raw_pixel(b, state, tex->texture_index,
         li >= 0 ? tex->src[li].src.ssa : nir_imm_int(b, 0), nir_channel(b, coord, 2),
         nir_iadd(b, nir_channel(b, coord, 0), dx), nir_iadd(b, nir_channel(b, coord, 1), dy));
      rgba = rd_fetch_swizzle(b, rgba, view->swizzle, rd_sample_format(view));
   } else {
      if (tex->sampler_index >= 16) { state->failed = true; return false; }
      const uint32_t *sampler = state->key->samplers[tex->sampler_index];
      bool integer = util_format_is_pure_integer(rd_sample_format(view));
      if ((sampler[1] & (1u << 5)) || ((sampler[0] >> 5) & 7) > 3 ||
          ((sampler[0] >> 8) & 7) > 3 || (sampler[2] >> 7) > 5 ||
          ((sampler[0] >> 1) & 3) > 1 || ((sampler[0] >> 3) & 3) > 1 ||
          (integer && tex->op != nir_texop_tg4 &&
           ((sampler[0] & 1) || ((sampler[0] >> 1) & 3) || ((sampler[0] >> 3) & 3))) ||
          (tex->is_shadow && (!vk_format_has_depth(view->format) ||
           nir_tex_instr_src_index(tex, nir_tex_src_comparator) < 0))) {
         state->failed = true;
         return false;
      }
      rgba = rd_filtered_sample(b, state, tex, coord, dx, dy);
   }
   nir_def_replace(&tex->def, nir_trim_vector(b, rgba, tex->def.num_components));
   return true;
}

bool
rd_nir_lower_array_fetch(nir_shader *nir, const struct rd_sampling_key *key,
                          uint32_t metadata_buffer)
{
   struct rd_fetch_state state = {.key = key, .buffer = metadata_buffer};
   nir_shader_instructions_pass(nir, rd_lower_fetch, nir_metadata_none, &state);
   return !state.failed;
}

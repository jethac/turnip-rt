/*
 * Copyright © 2026 Jetha Chan
 * SPDX-License-Identifier: MIT
 */

#include "tu_pipeline.h"

#include "compiler/spirv/spirv.h"
#include "nir/nir_builder.h"
#include "nir/nir_deref.h"
#include "util/hash_table.h"
#include "util/mesa-blake3.h"
#include "util/u_atomic.h"
#include "util/u_debug.h"
#include "util/os_time.h"
#include "util/os_misc.h"
#include "util/simple_mtx.h"
#include "util/blob.h"
#include "nir/nir_serialize.h"
#include <pthread.h>
#include <atomic>
#include <malloc.h>
#include "vk_pipeline.h"
#include "vk_shader_module.h"

#include "tu_device.h"
#include "tu_shader.h"
#include "tu_rt_prof.h"
#include "tu_foveate.h"
#include "ir3/ir3_nir.h"
#include "bvh/tu_bvh_defines.h"

static VkResult
tu_rt_compile_stages(struct tu_device *dev,
                     struct vk_pipeline_cache *cache,
                     struct tu_ray_tracing_pipeline *pipeline,
                     const VkRayTracingPipelineCreateInfoKHR *create_info,
                     VkPipelineCreateFlags2KHR flags,
                     const struct tu_shader_key *key);

static void
tu_rt_prebuild_parts(struct tu_device *dev, struct vk_pipeline_cache *cache,
                     const VkRayTracingPipelineCreateInfoKHR *create_info,
                     VkPipelineCreateFlags2KHR flags,
                     const struct tu_shader_key *key,
                     const unsigned char *stage_hash, nir_shader *nir);

static void
tu_keylog_hex(char *out, const unsigned char *p, unsigned n)
{
   static const char d[] = "0123456789abcdef";
   for (unsigned i = 0; i < n; i++) {
      out[2 * i] = d[p[i] >> 4];
      out[2 * i + 1] = d[p[i] & 15];
   }
   out[2 * n] = 0;
}

static int
tu_rt_group_key_cmp(const void *a, const void *b, void *data)
{
   const struct tu_ray_tracing_pipeline *pipeline = (const struct tu_ray_tracing_pipeline *) data;
   const struct tu_rt_group *ga = pipeline->groups + *(const uint32_t *) a;
   const struct tu_rt_group *gb = pipeline->groups + *(const uint32_t *) b;
   if (ga->handle != gb->handle)
      return ga->handle < gb->handle ? -1 : 1;
   uint32_t ia[3] = { ga->recursive_index, ga->ahit_index, ga->isec_index };
   uint32_t ib[3] = { gb->recursive_index, gb->ahit_index, gb->isec_index };
   for (unsigned k = 0; k < 3; k++) {
      bool ua = ia[k] == VK_SHADER_UNUSED_KHR, ub = ib[k] == VK_SHADER_UNUSED_KHR;
      if (ua != ub)
         return ua ? -1 : 1;
      if (!ua) {
         int c = memcmp(pipeline->stage_hashes + ia[k] * BLAKE3_KEY_LEN,
                        pipeline->stage_hashes + ib[k] * BLAKE3_KEY_LEN, BLAKE3_KEY_LEN);
         if (c)
            return c;
      }
   }
   return 0;
}

static bool
tu_keylog_enabled(void)
{
   static const bool on = debug_get_bool_option("TU_KEYLOG", false);
   return on;
}

static void
tu_rt_init_groups(struct tu_device *dev,
                  struct tu_ray_tracing_pipeline *pipeline,
                  const VkRayTracingPipelineCreateInfoKHR *create_info)
{
   uint32_t i = 0;
   for (; i < create_info->groupCount; i++) {
      const VkRayTracingShaderGroupCreateInfoKHR *group_info =
         create_info->pGroups + i;
      struct tu_rt_group *dst = pipeline->groups + i;

      dst->recursive_index = VK_SHADER_UNUSED_KHR;
      dst->ahit_index = VK_SHADER_UNUSED_KHR;
      dst->isec_index = VK_SHADER_UNUSED_KHR;

      switch (group_info->type) {
      case VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR:
         dst->recursive_index = group_info->generalShader;
         break;
      case VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR:
         dst->recursive_index = group_info->closestHitShader;
         dst->ahit_index = group_info->anyHitShader;
         break;
      case VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR:
         dst->recursive_index = group_info->closestHitShader;
         dst->isec_index = group_info->intersectionShader;
         if (dst->isec_index != VK_SHADER_UNUSED_KHR)
            dst->ahit_index = group_info->anyHitShader;
         break;
      default:
         UNREACHABLE("Unimplemented VkRayTracingShaderGroupTypeKHR");
      }

      uint32_t idx[3] = { dst->recursive_index, dst->ahit_index, dst->isec_index };
      struct mesa_blake3 h;
      unsigned char out[BLAKE3_KEY_LEN];
      _mesa_blake3_init(&h);
      _mesa_blake3_update(&h, &group_info->type, sizeof(group_info->type));
      for (unsigned k = 0; k < 3; k++) {
         if (idx[k] == VK_SHADER_UNUSED_KHR) {
            uint32_t none = ~0u;
            _mesa_blake3_update(&h, &none, sizeof(none));
         } else {
            _mesa_blake3_update(&h, pipeline->stage_hashes + idx[k] * BLAKE3_KEY_LEN,
                                BLAKE3_KEY_LEN);
         }
      }
      _mesa_blake3_final(&h, out);
      memcpy(&dst->handle, out, sizeof(dst->handle));
      if (!dst->handle)
         dst->handle = 1;
   }

   if (!create_info->pLibraryInfo)
      return;

   uint32_t stage_base = create_info->stageCount;
   for (uint32_t l = 0; l < create_info->pLibraryInfo->libraryCount; l++) {
      VK_FROM_HANDLE(tu_pipeline, lib_base,
                     create_info->pLibraryInfo->pLibraries[l]);
      struct tu_ray_tracing_pipeline *lib =
         tu_pipeline_to_ray_tracing(lib_base);

      for (uint32_t g = 0; g < lib->group_count; g++) {
         const struct tu_rt_group *src = lib->groups + g;
         struct tu_rt_group *dst = pipeline->groups + i;

         dst->handle = src->handle;
         dst->recursive_index = src->recursive_index == VK_SHADER_UNUSED_KHR ?
            VK_SHADER_UNUSED_KHR : stage_base + src->recursive_index;
         dst->ahit_index = src->ahit_index == VK_SHADER_UNUSED_KHR ?
            VK_SHADER_UNUSED_KHR : stage_base + src->ahit_index;
         dst->isec_index = src->isec_index == VK_SHADER_UNUSED_KHR ?
            VK_SHADER_UNUSED_KHR : stage_base + src->isec_index;
         i++;
      }
      stage_base += lib->stage_count;
   }
}

static bool
tu_rt_retag_constant_derefs(nir_shader *shader)
{
   bool progress = false;
   nir_foreach_function_impl (impl, shader) {
      nir_foreach_block (block, impl) {
         nir_foreach_instr (instr, block) {
            if (instr->type != nir_instr_type_deref)
               continue;
            nir_deref_instr *deref = nir_instr_as_deref(instr);
            if (deref->modes == nir_var_mem_constant) {
               deref->modes = nir_var_mem_global;
               progress = true;
            }
         }
      }
      nir_progress(progress, impl, nir_metadata_all);
   }
   return progress;
}

static bool
tu_rt_move_intrinsic(nir_builder *b, nir_intrinsic_instr *instr, void *data)
{
   switch (instr->intrinsic) {
   case nir_intrinsic_load_shader_record_ptr:
   case nir_intrinsic_load_ray_flags:
   case nir_intrinsic_load_ray_object_origin:
   case nir_intrinsic_load_ray_world_origin:
   case nir_intrinsic_load_ray_t_min:
   case nir_intrinsic_load_ray_object_direction:
   case nir_intrinsic_load_ray_world_direction:
   case nir_intrinsic_load_ray_t_max:
      nir_instr_move(nir_before_impl(b->impl), &instr->instr);
      return true;
   default:
      return false;
   }
}

#define TU_RT_HIT_WORDS (TU_RT_HIT_ATTRIBS_SIZE / 4)
static const char tu_rt_hit_words_name[] = "tu_rt_hit_words";

static nir_variable *
tu_rt_hit_words_var(nir_shader *s)
{
   nir_foreach_variable_with_modes (var, s, nir_var_shader_temp) {
      if (var->name && !strcmp(var->name, tu_rt_hit_words_name))
         return var;
   }
   return nir_variable_create(s, nir_var_shader_temp,
                              glsl_array_type(glsl_uint_type(), TU_RT_HIT_WORDS, 4),
                              tu_rt_hit_words_name);
}

static nir_deref_instr *
tu_rt_hw(nir_builder *b, nir_variable *hw, nir_def *word)
{
   return nir_build_deref_array(b, nir_build_deref_var(b, hw), word);
}

static unsigned
tu_rt_elem_size(unsigned bit_size)
{
   return bit_size == 1 ? 4 : bit_size / 8;
}

static nir_def *
tu_rt_hw_load_comp(nir_builder *b, nir_variable *hw, nir_def *off,
                   unsigned c, unsigned bs)
{
   unsigned esz = tu_rt_elem_size(bs);
   nir_def *coff = nir_iadd_imm(b, off, c * esz);
   nir_def *word = nir_ushr_imm(b, coff, 2);
   switch (esz) {
   case 4: {
      nir_def *v = nir_load_deref(b, tu_rt_hw(b, hw, word));
      return bs == 1 ? nir_ine_imm(b, v, 0) : v;
   }
   case 8:
      return nir_pack_64_2x32_split(b, nir_load_deref(b, tu_rt_hw(b, hw, word)),
                                    nir_load_deref(b, tu_rt_hw(b, hw, nir_iadd_imm(b, word, 1))));
   default: {
      nir_def *w = nir_load_deref(b, tu_rt_hw(b, hw, word));
      nir_def *shift = nir_imul_imm(b, nir_iand_imm(b, coff, 3), 8);
      return nir_u2uN(b, nir_ubitfield_extract(b, w, shift, nir_imm_int(b, bs)), bs);
   }
   }
}

static void
tu_rt_hw_store_comp(nir_builder *b, nir_variable *hw, nir_def *off,
                    unsigned c, nir_def *v)
{
   unsigned bs = v->bit_size;
   unsigned esz = tu_rt_elem_size(bs);
   nir_def *coff = nir_iadd_imm(b, off, c * esz);
   nir_def *word = nir_ushr_imm(b, coff, 2);
   switch (esz) {
   case 4:
      nir_store_deref(b, tu_rt_hw(b, hw, word), bs == 1 ? nir_b2i32(b, v) : v, 0x1);
      break;
   case 8:
      nir_store_deref(b, tu_rt_hw(b, hw, word), nir_unpack_64_2x32_split_x(b, v), 0x1);
      nir_store_deref(b, tu_rt_hw(b, hw, nir_iadd_imm(b, word, 1)),
                      nir_unpack_64_2x32_split_y(b, v), 0x1);
      break;
   default: {
      nir_deref_instr *d = tu_rt_hw(b, hw, word);
      nir_def *shift = nir_imul_imm(b, nir_iand_imm(b, coff, 3), 8);
      nir_store_deref(b, d, nir_bitfield_insert(b, nir_load_deref(b, d), nir_u2u32(b, v),
                                                shift, nir_imm_int(b, bs)), 0x1);
      break;
   }
   }
}

static bool
tu_rt_lower_hit_attribs(nir_shader *nir)
{
   bool has = false;
   nir_foreach_variable_with_modes (var, nir, nir_var_ray_hit_attrib)
      has = true;
   if (!has)
      return false;

   NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, nir_var_ray_hit_attrib,
            glsl_get_natural_size_align_bytes);

   nir_variable *hw = tu_rt_hit_words_var(nir);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_create(impl);

   nir_foreach_block_safe (block, impl) {
      nir_foreach_instr_safe (instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_deref &&
             intr->intrinsic != nir_intrinsic_store_deref)
            continue;
         nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
         if (!nir_deref_mode_is(deref, nir_var_ray_hit_attrib))
            continue;

         b.cursor = nir_before_instr(instr);
         nir_def *off = nir_build_deref_offset(&b, deref, glsl_get_natural_size_align_bytes);

         if (intr->intrinsic == nir_intrinsic_load_deref) {
            nir_def *vals[NIR_MAX_VEC_COMPONENTS];
            for (unsigned c = 0; c < intr->def.num_components; c++)
               vals[c] = tu_rt_hw_load_comp(&b, hw, off, c, intr->def.bit_size);
            nir_def_replace(&intr->def, nir_vec(&b, vals, intr->def.num_components));
         } else {
            nir_def *v = intr->src[1].ssa;
            unsigned mask = nir_intrinsic_write_mask(intr);
            for (unsigned c = 0; c < v->num_components; c++) {
               if (mask & (1u << c))
                  tu_rt_hw_store_comp(&b, hw, off, c, nir_channel(&b, v, c));
            }
            nir_instr_remove(instr);
         }
      }
   }

   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_remove_dead_derefs);
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_ray_hit_attrib, NULL);
   return true;
}

static struct tu_rt_stage_blob *
tu_rt_blob_from_nir(const nir_shader *nir)
{
   struct blob b;
   blob_init(&b);
   nir_serialize(&b, nir, false);
   struct tu_rt_stage_blob *sb = NULL;
   if (!b.out_of_memory) {
      sb = (struct tu_rt_stage_blob *) malloc(sizeof(*sb) + b.size);
      if (sb) {
         sb->refcnt = 1;
         sb->size = b.size;
         memcpy(sb->data, b.data, b.size);
      }
   }
   blob_finish(&b);
   return sb;
}

static nir_shader *
tu_rt_nir_from_blob(struct tu_device *dev, void *mem_ctx,
                    const struct tu_rt_stage_blob *sb)
{
   struct blob_reader r;
   blob_reader_init(&r, sb->data, sb->size);
   return nir_deserialize(mem_ctx, ir3_get_compiler_options(dev->compiler), &r);
}

static void
tu_rt_stage_set_destroy(void *p)
{
   struct tu_rt_stage_set *set = (struct tu_rt_stage_set *) p;
   for (uint32_t i = 0; i < set->count; i++) {
      struct tu_rt_stage_blob *sb = set->blobs[i];
      if (sb && p_atomic_dec_zero(&sb->refcnt))
         free(sb);
   }
}

static struct tu_rt_stage_set *
tu_rt_stage_set_create(void *mem_ctx, uint32_t count)
{
   struct tu_rt_stage_set *set = (struct tu_rt_stage_set *)
      rzalloc_size(mem_ctx, sizeof(*set) + MAX2(count, 1) * sizeof(set->blobs[0]));
   if (!set)
      return NULL;
   set->count = count;
   ralloc_set_destructor(set, tu_rt_stage_set_destroy);
   return set;
}

static VkResult
tu_rt_compile_stages(struct tu_device *dev,
                     struct vk_pipeline_cache *cache,
                     struct tu_ray_tracing_pipeline *pipeline,
                     const VkRayTracingPipelineCreateInfoKHR *create_info,
                     VkPipelineCreateFlags2KHR flags,
                     const struct tu_shader_key *key,
                     void *nir_ctx)
{
   struct tu_rt_stage_set *set = pipeline->stage_set;
   const nir_shader_compiler_options *nir_options =
      ir3_get_compiler_options(dev->compiler);
   cache = cache ? cache : dev->mem_cache;
   bool can_cache = !(flags & VK_PIPELINE_CREATE_2_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR);

   uint32_t i = 0;
   for (; i < create_info->stageCount; i++) {
      const VkPipelineShaderStageCreateInfo *stage_info =
         create_info->pStages + i;
      mesa_shader_stage stage = vk_to_mesa_shader_stage(stage_info->stage);

      static const char version[] = "tu-rt-stage-3";
      unsigned char nir_key[BLAKE3_KEY_LEN];
      struct mesa_blake3 h;
      _mesa_blake3_init(&h);
      _mesa_blake3_update(&h, version, sizeof(version));
      _mesa_blake3_update(&h, pipeline->stage_hashes + i * BLAKE3_KEY_LEN, BLAKE3_KEY_LEN);
      _mesa_blake3_update(&h, key, sizeof(*key));
      enum ir3_shader_debug ir3_debug_key = ir3_shader_debug_hash_key();
      _mesa_blake3_update(&h, &ir3_debug_key, sizeof(ir3_debug_key));
      _mesa_blake3_final(&h, nir_key);

      void *tmp_ctx = nir_ctx ? NULL : ralloc_context(NULL);
      void *ctx = nir_ctx ? nir_ctx : tmp_ctx;
      nir_shader *nir = NULL;
      if (can_cache)
         nir = vk_pipeline_cache_lookup_nir(cache, nir_key, sizeof(nir_key),
                                            nir_options, NULL, ctx);
      if (!nir) {
         nir = tu_spirv_to_nir(dev, ctx, flags, stage_info, key, stage);
         if (!nir) {
            ralloc_free(tmp_ctx);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
         }

         NIR_PASS(_, nir, tu_rt_retag_constant_derefs);
         NIR_PASS(_, nir, tu_rt_lower_hit_attribs);
         NIR_PASS(_, nir, nir_shader_intrinsics_pass, tu_rt_move_intrinsic,
                  nir_metadata_control_flow, NULL);

         if (can_cache)
            vk_pipeline_cache_add_nir(cache, nir_key, sizeof(nir_key), nir);
      }

      if (!nir_ctx)
         tu_rt_prebuild_parts(dev, cache, create_info, flags, key,
                              pipeline->stage_hashes + i * BLAKE3_KEY_LEN, nir);

      if (set) {
         set->blobs[i] = tu_rt_blob_from_nir(nir);
         if (!set->blobs[i]) {
            ralloc_free(tmp_ctx);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
         }
      }
      pipeline->stages[i] = nir_ctx ? nir : NULL;
      ralloc_free(tmp_ctx);
   }

   if (!create_info->pLibraryInfo)
      return VK_SUCCESS;

   for (uint32_t l = 0; l < create_info->pLibraryInfo->libraryCount; l++) {
      VK_FROM_HANDLE(tu_pipeline, lib_base,
                     create_info->pLibraryInfo->pLibraries[l]);
      struct tu_ray_tracing_pipeline *lib =
         tu_pipeline_to_ray_tracing(lib_base);
      for (uint32_t s = 0; s < lib->stage_count; s++, i++) {
         struct tu_rt_stage_blob *sb = lib->stage_set->blobs[s];
         if (set) {
            p_atomic_inc(&sb->refcnt);
            set->blobs[i] = sb;
         }
         if (nir_ctx) {
            pipeline->stages[i] = tu_rt_nir_from_blob(dev, nir_ctx, sb);
            if (!pipeline->stages[i])
               return VK_ERROR_OUT_OF_HOST_MEMORY;
         }
      }
   }

   return VK_SUCCESS;
}

enum {
   TU_RT_USE_INSTANCE_ID = 1 << 0,
   TU_RT_USE_CUSTOM_INDEX = 1 << 1,
   TU_RT_USE_PRIMITIVE_ID = 1 << 2,
   TU_RT_USE_GEOMETRY_INDEX = 1 << 3,
   TU_RT_USE_OBJECT_ORIGIN = 1 << 4,
   TU_RT_USE_OBJECT_DIR = 1 << 5,
   TU_RT_USE_WTO = 1 << 6,
   TU_RT_USE_OTW = 1 << 7,
};

struct tu_rt_state {
   nir_variable *flags;
   nir_variable *cull_mask;
   nir_variable *sbt_offset;
   nir_variable *sbt_stride;
   nir_variable *origin;
   nir_variable *tmin;
   nir_variable *dir;
   nir_variable *tmax;

   nir_variable *instance_id;
   nir_variable *custom_index;
   nir_variable *primitive_id;
   nir_variable *geometry_index;
   nir_variable *hit_kind;
   nir_variable *object_origin;
   nir_variable *object_dir;
   nir_variable *wto[4];
   nir_variable *otw[4];

   nir_variable *shader_record_ptr;

   nir_variable *accept;
   nir_variable *terminate;
   nir_variable *opaque;
};

enum tu_rt_part_kind {
   TU_RT_PART_NONE = 0,
   TU_RT_PART_WHOLE,
   TU_RT_PART_RAYGEN,
   TU_RT_PART_TRAVERSAL,
   TU_RT_PART_CALLEE,
};

#define TU_RT_MAX_CLASSES 8

struct tu_rt_compiler {
   struct tu_ray_tracing_pipeline *pipeline;
   VkPipelineCreateFlags2KHR flags;
   struct tu_rt_state state;
   nir_variable *hit_words;

   enum tu_rt_part_kind part;
   unsigned part_payload_size;
   uint32_t part_hit_used;
   const char *part_fail;
   unsigned part_nclasses;
   uint32_t part_classes[TU_RT_MAX_CLASSES];
   nir_variable *part_dummy;
   bool part_has_chit;
   bool part_has_miss;

   struct hash_table *fn_ctx;
   nir_function **fn_cache;
   nir_function **isec_cache;

   uint32_t used;
   bool has_ahit;
   bool has_isec;
   bool nohit;
   int only_raygen;
   unsigned max_call_depth;
   uint32_t fov_mode;
   nir_variable *fov_sel;
};

static uint32_t
tu_rt_fov_mode(struct tu_device *dev)
{
   return dev->physical_device->info->chip >= 7 ? tu_foveate_rt_mode() : 0;
}

static void
tu_rt_state_init(nir_shader *nir, struct tu_rt_state *s)
{
#define VAR(name, type) \
   s->name = nir_variable_create(nir, nir_var_shader_temp, type, #name)
   VAR(flags, glsl_uint_type());
   VAR(cull_mask, glsl_uint_type());
   VAR(sbt_offset, glsl_uint_type());
   VAR(sbt_stride, glsl_uint_type());
   VAR(origin, glsl_vec_type(3));
   VAR(tmin, glsl_float_type());
   VAR(dir, glsl_vec_type(3));
   VAR(tmax, glsl_float_type());
   VAR(instance_id, glsl_uint_type());
   VAR(custom_index, glsl_uint_type());
   VAR(primitive_id, glsl_uint_type());
   VAR(geometry_index, glsl_uint_type());
   VAR(hit_kind, glsl_uint_type());
   VAR(object_origin, glsl_vec_type(3));
   VAR(object_dir, glsl_vec_type(3));
   VAR(shader_record_ptr, glsl_uint64_t_type());
   VAR(accept, glsl_bool_type());
   VAR(terminate, glsl_bool_type());
   VAR(opaque, glsl_bool_type());
#undef VAR
   for (unsigned c = 0; c < 4; c++) {
      s->wto[c] = nir_variable_create(nir, nir_var_shader_temp,
                                      glsl_vec_type(3), "wto");
      s->otw[c] = nir_variable_create(nir, nir_var_shader_temp,
                                      glsl_vec_type(3), "otw");
   }
}

static uint32_t
tu_rt_scan_used(nir_shader *nir)
{
   uint32_t used = 0;
   nir_foreach_function_impl (impl, nir) {
      nir_foreach_block (block, impl) {
         nir_foreach_instr (instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            switch (nir_instr_as_intrinsic(instr)->intrinsic) {
            case nir_intrinsic_load_instance_id: used |= TU_RT_USE_INSTANCE_ID; break;
            case nir_intrinsic_load_ray_instance_custom_index: used |= TU_RT_USE_CUSTOM_INDEX; break;
            case nir_intrinsic_load_primitive_id: used |= TU_RT_USE_PRIMITIVE_ID; break;
            case nir_intrinsic_load_ray_geometry_index: used |= TU_RT_USE_GEOMETRY_INDEX; break;
            case nir_intrinsic_load_ray_object_origin: used |= TU_RT_USE_OBJECT_ORIGIN; break;
            case nir_intrinsic_load_ray_object_direction: used |= TU_RT_USE_OBJECT_DIR; break;
            case nir_intrinsic_load_ray_world_to_object: used |= TU_RT_USE_WTO; break;
            case nir_intrinsic_load_ray_object_to_world: used |= TU_RT_USE_OTW; break;
            default: break;
            }
         }
      }
   }
   return used;
}

static nir_def *
tu_rt_load_cmd(nir_builder *b, uint32_t offset, unsigned comps)
{
   nir_def *addr = nir_pack_64_2x32(b, nir_load_rt_cmd_addr_ir3(b));
   return nir_load_global(b, comps, 32, nir_iadd_imm(b, addr, offset),
                          .access = (gl_access_qualifier) (ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                          .align_mul = 4);
}

static nir_def *
tu_rt_load_cmd_addr(nir_builder *b, uint32_t offset)
{
   return nir_pack_64_2x32(b, tu_rt_load_cmd(b, offset, 2));
}

enum {
   TU_RT_FRAME_SEL = 0,
   TU_RT_FRAME_RET = 4,
   TU_RT_FRAME_LAUNCH_ID = 8,
   TU_RT_FRAME_RGIDX = 20,
   TU_RT_FRAME_CMD = 24,
   TU_RT_FRAME_ACCEL = 32,
   TU_RT_FRAME_FLAGS = 40,
   TU_RT_FRAME_CULL = 44,
   TU_RT_FRAME_SBT_OFFSET = 48,
   TU_RT_FRAME_SBT_STRIDE = 52,
   TU_RT_FRAME_MISS_INDEX = 56,
   TU_RT_FRAME_TMIN = 60,
   TU_RT_FRAME_ORIGIN = 64,
   TU_RT_FRAME_DIR = 76,
   TU_RT_FRAME_TMAX = 88,
   TU_RT_FRAME_HIT_KIND = 92,
   TU_RT_FRAME_INSTANCE_ID = 96,
   TU_RT_FRAME_CUSTOM_INDEX = 100,
   TU_RT_FRAME_PRIMITIVE_ID = 104,
   TU_RT_FRAME_GEOMETRY_INDEX = 108,
   TU_RT_FRAME_RECORD = 112,
   TU_RT_FRAME_HANDLE = 120,
   TU_RT_FRAME_OBJECT_ORIGIN = 124,
   TU_RT_FRAME_OBJECT_DIR = 136,
   TU_RT_FRAME_WTO = 148,
   TU_RT_FRAME_OTW = 196,
   TU_RT_FRAME_HIT_WORDS = 244,
   TU_RT_FRAME_PAYLOAD = 288,
   TU_RT_PAYLOAD_MAX = 256,
   TU_RT_FRAME_STACK = TU_RT_FRAME_PAYLOAD + TU_RT_PAYLOAD_MAX,
   TU_RT_STACK_MAX = 1024,
   TU_RT_FRAME_CALLEE = TU_RT_FRAME_STACK + TU_RT_STACK_MAX,
   TU_RT_FRAME_WHOLE = 32,
};

#define TU_RT_SEL_DONE 0u
#define TU_RT_SEL_RG(k) (1u + (k))
#define TU_RT_SEL_TRAV(c) (0x1000000u | (c))
#define TU_RT_SEL_CHIT(c) (0x2000000u | (c))
#define TU_RT_SEL_MISS(c) (0x3000000u | (c))
#define TU_RT_CLASS_SKIP_CHIT 0x10000u

static inline unsigned
tu_rt_class_size(uint32_t cls)
{
   return cls & 0xffff;
}

static void
tu_rt_emit_blob(nir_builder *b, uint32_t idx)
{
   nir_rt_blob_ir3(b, .base = idx);
}

static nir_def *
tu_rt_fload(nir_builder *b, unsigned off, unsigned comps)
{
   return nir_load_scratch(b, comps, 32, nir_imm_int(b, off), .align_mul = 4);
}

static void
tu_rt_fstore(nir_builder *b, unsigned off, nir_def *v)
{
   nir_store_scratch(b, v, nir_imm_int(b, off), .align_mul = 4,
                     .write_mask = nir_component_mask(v->num_components));
}


struct tu_rt_sbt_entry {
   nir_def *value;
   nir_def *shader_record_ptr;
};

static struct tu_rt_sbt_entry
tu_rt_load_sbt_entry(nir_builder *b, nir_def *index, uint32_t addr_offset)
{
   nir_def *addr = tu_rt_load_cmd_addr(b, addr_offset);

   if (index) {
      nir_def *stride =
         tu_rt_load_cmd(b, addr_offset + 2 * sizeof(VkDeviceSize), 1);
      addr = nir_iadd(b, addr, nir_u2u64(b, nir_imul(b, index, stride)));
   }

   return (struct tu_rt_sbt_entry) {
      .value = nir_load_global(b, 1, 32, addr,
                               .access = (gl_access_qualifier) (ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                               .align_mul = 4),
      .shader_record_ptr = nir_iadd_imm(b, addr, TU_RT_GROUP_HANDLE_SIZE),
   };
}

#define TU_RT_MAX_CALL_DEPTH 8

struct tu_rt_fn_ctx {
   mesa_shader_stage stage;
   unsigned ray_depth;
   unsigned call_depth;
};

static const struct tu_rt_fn_ctx tu_rt_main_ctx = { MESA_SHADER_COMPUTE, 0, 0 };

static const struct tu_rt_fn_ctx *
tu_rt_ctx(struct tu_rt_compiler *c, nir_function *function)
{
   struct hash_entry *e = _mesa_hash_table_search(c->fn_ctx, function);
   return e ? (const struct tu_rt_fn_ctx *) e->data : &tu_rt_main_ctx;
}

static void
tu_rt_set_ctx(struct tu_rt_compiler *c, nir_function *function,
              mesa_shader_stage stage, unsigned ray_depth, unsigned call_depth)
{
   struct tu_rt_fn_ctx *ctx = rzalloc(c->fn_ctx, struct tu_rt_fn_ctx);
   ctx->stage = stage;
   ctx->ray_depth = ray_depth;
   ctx->call_depth = call_depth;
   _mesa_hash_table_insert(c->fn_ctx, function, ctx);
}

static uint32_t
tu_rt_stage_index(struct tu_rt_compiler *c, nir_shader *stage)
{
   for (uint32_t i = 0; i < c->pipeline->stage_count; i++) {
      if (c->pipeline->stages[i] == stage)
         return i;
   }
   UNREACHABLE("unknown ray tracing stage");
}

static void
tu_rt_bind_call_data_param(nir_function_impl *impl)
{
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_def *param = nir_load_param(&b, 0);

   nir_foreach_block (block, impl) {
      nir_foreach_instr_safe (instr, block) {
         if (instr->type != nir_instr_type_deref)
            continue;
         nir_deref_instr *deref = nir_instr_as_deref(instr);
         if (!nir_deref_mode_is(deref, nir_var_shader_call_data))
            continue;
         deref->modes = nir_var_function_temp;
         if (deref->deref_type == nir_deref_type_var) {
            b.cursor = nir_before_instr(&deref->instr);
            nir_deref_instr *cast =
               nir_build_deref_cast(&b, param, nir_var_function_temp,
                                    deref->var->type, 0);
            nir_def_replace(&deref->def, &cast->def);
         }
      }
   }
   nir_progress(true, impl, nir_metadata_none);
}

static nir_function *
tu_rt_clone_stage(struct tu_rt_compiler *c, nir_shader *dst, nir_shader *stage)
{
   nir_function_impl *copy =
      nir_function_impl_clone(dst, nir_shader_get_entrypoint(stage));

   struct hash_table *var_remap = _mesa_pointer_hash_table_create(NULL);
   nir_foreach_block (block, copy) {
      nir_foreach_instr_safe (instr, block) {
         if (instr->type != nir_instr_type_deref)
            continue;
         nir_deref_instr *deref = nir_instr_as_deref(instr);
         if (deref->deref_type != nir_deref_type_var ||
             deref->var->data.mode == nir_var_function_temp ||
             deref->var->data.mode == nir_var_shader_call_data)
            continue;
         struct hash_entry *entry =
            _mesa_hash_table_search(var_remap, deref->var);
         if (!entry) {
            nir_variable *new_var;
            if (deref->var->name &&
                !strcmp(deref->var->name, tu_rt_hit_words_name)) {
               new_var = c->hit_words;
            } else {
               new_var = nir_variable_clone(deref->var, dst);
               nir_shader_add_variable(dst, new_var);
            }
            entry = _mesa_hash_table_insert(var_remap, deref->var, new_var);
         }
         deref->var = (nir_variable *) entry->data;
      }
   }
   ralloc_free(var_remap);

   nir_function *function =
      nir_function_create(dst, _mesa_shader_stage_to_string(stage->info.stage));
   function->num_params = 1;
   function->params = rzalloc_array(dst, nir_parameter, 1);
   function->params[0].num_components = 1;
   function->params[0].bit_size = 32;
   nir_function_set_impl(function, copy);
   tu_rt_bind_call_data_param(copy);
   return function;
}

static void
tu_rt_call_stage(nir_builder *b, struct tu_rt_compiler *c, nir_shader *stage,
                 nir_def *arg, unsigned ray_depth, unsigned call_depth)
{
   uint32_t slot = (tu_rt_stage_index(c, stage) * 2 + ray_depth) *
                   (TU_RT_MAX_CALL_DEPTH + 1) + call_depth;
   nir_function *function = c->fn_cache[slot];
   if (!function) {
      function = tu_rt_clone_stage(c, b->shader, stage);
      tu_rt_set_ctx(c, function, stage->info.stage, ray_depth, call_depth);
      c->fn_cache[slot] = function;
   }
   nir_build_call(b, function, 1, &arg);
}

static bool
tu_rt_handle_seen(struct tu_rt_compiler *c, uint32_t upto, uint32_t handle,
                  uint32_t mask_kind)
{
   for (uint32_t j = 0; j < upto; j++) {
      const struct tu_rt_group *g = c->pipeline->groups + j;
      if (g->handle != handle)
         continue;
      if (mask_kind == 0 && g->recursive_index != VK_SHADER_UNUSED_KHR)
         return true;
      if (mask_kind == 1 && g->ahit_index != VK_SHADER_UNUSED_KHR &&
          g->isec_index == VK_SHADER_UNUSED_KHR)
         return true;
      if (mask_kind == 2 && g->isec_index != VK_SHADER_UNUSED_KHR)
         return true;
   }
   return false;
}

static nir_shader *
tu_rt_group_stage(struct tu_rt_compiler *c, uint32_t index)
{
   return index == VK_SHADER_UNUSED_KHR ? NULL : c->pipeline->stages[index];
}

static bool
tu_rt_payload_compatible(nir_shader *stage, nir_def *arg)
{
   if (!nir_def_is_deref(arg))
      return true;
   const struct glsl_type *pt = nir_def_as_deref(arg)->type;
   nir_foreach_variable_with_modes (var, stage, nir_var_shader_call_data) {
      if (var->type == pt)
         return true;
      unsigned sa, aa, sb, ab;
      glsl_get_natural_size_align_bytes(var->type, &sa, &aa);
      glsl_get_natural_size_align_bytes(pt, &sb, &ab);
      return sa == sb;
   }
   return true;
}

static unsigned
tu_rt_call_data_size(nir_shader *stage, bool *has)
{
   nir_foreach_variable_with_modes (var, stage, nir_var_shader_call_data) {
      unsigned size, al;
      glsl_get_natural_size_align_bytes(var->type, &size, &al);
      *has = true;
      return size;
   }
   *has = false;
   return 0;
}

static bool
tu_rt_compat(struct tu_rt_compiler *c, nir_shader *stage, nir_def *arg)
{
   if (c->part == TU_RT_PART_TRAVERSAL) {
      bool has;
      unsigned size = tu_rt_call_data_size(stage, &has);
      return !has || size == tu_rt_class_size(c->part_payload_size);
   }
   return tu_rt_payload_compatible(stage, arg);
}

struct tu_rt_case {
   uint32_t handle;
   uint32_t group;
};

static int
tu_rt_case_cmp(const void *a, const void *b)
{
   uint32_t ha = ((const struct tu_rt_case *) a)->handle;
   uint32_t hb = ((const struct tu_rt_case *) b)->handle;
   return ha < hb ? -1 : (ha > hb ? 1 : 0);
}

static bool
tu_rt_use_case_tree(void)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = debug_get_bool_option("TU_RT_BST", true);
   return enabled;
}

template <typename F>
static void
tu_rt_emit_case_chain(nir_builder *b, nir_def *handle, const struct tu_rt_case *cases,
                      unsigned count, F &emit)
{
   for (unsigned i = 0; i < count; i++) {
      nir_push_if(b, nir_ieq_imm(b, handle, cases[i].handle));
      emit(cases[i]);
      if (i + 1 < count)
         nir_push_else(b, NULL);
   }
   for (unsigned i = 0; i < count; i++)
      nir_pop_if(b, NULL);
}

template <typename F>
static void
tu_rt_emit_case_tree(nir_builder *b, nir_def *handle, const struct tu_rt_case *cases,
                     unsigned count, F &emit)
{
   if (count <= 3) {
      tu_rt_emit_case_chain(b, handle, cases, count, emit);
      return;
   }
   unsigned mid = count / 2;
   nir_push_if(b, nir_ult(b, handle, nir_imm_int(b, cases[mid].handle)));
   tu_rt_emit_case_tree(b, handle, cases, mid, emit);
   nir_push_else(b, NULL);
   tu_rt_emit_case_tree(b, handle, cases + mid, count - mid, emit);
   nir_pop_if(b, NULL);
}

template <typename F, typename G>
static void
tu_rt_emit_case_chain_or(nir_builder *b, nir_def *handle, const struct tu_rt_case *cases,
                         unsigned count, F &emit, G &dflt)
{
   for (unsigned i = 0; i < count; i++) {
      nir_push_if(b, nir_ieq_imm(b, handle, cases[i].handle));
      emit(cases[i]);
      nir_push_else(b, NULL);
   }
   dflt();
   for (unsigned i = 0; i < count; i++)
      nir_pop_if(b, NULL);
}

template <typename F, typename G>
static void
tu_rt_emit_case_tree_or(nir_builder *b, nir_def *handle, const struct tu_rt_case *cases,
                        unsigned count, F &emit, G &dflt)
{
   if (count <= 3) {
      tu_rt_emit_case_chain_or(b, handle, cases, count, emit, dflt);
      return;
   }
   unsigned mid = count / 2;
   nir_push_if(b, nir_ult(b, handle, nir_imm_int(b, cases[mid].handle)));
   tu_rt_emit_case_tree_or(b, handle, cases, mid, emit, dflt);
   nir_push_else(b, NULL);
   tu_rt_emit_case_tree_or(b, handle, cases + mid, count - mid, emit, dflt);
   nir_pop_if(b, NULL);
}

template <typename F, typename G>
static void
tu_rt_emit_cases_or(nir_builder *b, nir_def *handle, struct tu_rt_case *cases,
                    unsigned count, F &&emit, G &&dflt)
{
   qsort(cases, count, sizeof(*cases), tu_rt_case_cmp);
   tu_rt_emit_case_tree_or(b, handle, cases, count, emit, dflt);
}

template <typename F>
static void
tu_rt_emit_cases(nir_builder *b, nir_def *handle, struct tu_rt_case *cases,
                 unsigned count, F &&emit)
{
   if (!count)
      return;
   if (!tu_rt_use_case_tree()) {
      for (unsigned i = 0; i < count; i++) {
         nir_push_if(b, nir_ieq_imm(b, handle, cases[i].handle));
         emit(cases[i]);
         nir_pop_if(b, NULL);
      }
      return;
   }
   qsort(cases, count, sizeof(*cases), tu_rt_case_cmp);
   tu_rt_emit_case_tree(b, handle, cases, count, emit);
}

static void
tu_rt_dispatch_recursive(nir_builder *b, struct tu_rt_compiler *c,
                         nir_def *handle, mesa_shader_stage want, nir_def *arg,
                         unsigned ray_depth, unsigned call_depth)
{
   if (c->nohit && want == MESA_SHADER_CLOSEST_HIT)
      return;
   struct tu_rt_case *cases =
      (struct tu_rt_case *) malloc(sizeof(struct tu_rt_case) * MAX2(c->pipeline->group_count, 1));
   unsigned count = 0;
   for (uint32_t i = 0; i < c->pipeline->group_count; i++) {
      if (want == MESA_SHADER_RAYGEN && c->only_raygen >= 0 &&
          i != (uint32_t) c->only_raygen)
         continue;
      struct tu_rt_group *group = c->pipeline->groups + i;
      nir_shader *stage = tu_rt_group_stage(c, group->recursive_index);
      if (!stage || stage->info.stage != want ||
          tu_rt_handle_seen(c, i, group->handle, 0) ||
          !tu_rt_compat(c, stage, arg))
         continue;
      cases[count++] = (struct tu_rt_case) { group->handle, i };
   }
   tu_rt_emit_cases(b, handle, cases, count, [&](const struct tu_rt_case &cs) {
      nir_shader *stage = tu_rt_group_stage(c, c->pipeline->groups[cs.group].recursive_index);
      tu_rt_call_stage(b, c, stage, arg, ray_depth, call_depth);
   });
   free(cases);
}

static void
tu_rt_load_hit_state(nir_builder *b, struct tu_rt_compiler *c,
                     nir_def *rq, bool committed)
{
   struct tu_rt_state *s = &c->state;
   uint32_t u = c->used;

#define RQ(n, bits, val, ...) \
   nir_rq_load(b, n, bits, rq, .ray_query_value = nir_ray_query_value_##val, \
               .committed = committed, ##__VA_ARGS__)

   if (u & TU_RT_USE_INSTANCE_ID)
      nir_store_var(b, s->instance_id, RQ(1, 32, intersection_instance_id), 0x1);
   if (u & TU_RT_USE_CUSTOM_INDEX)
      nir_store_var(b, s->custom_index,
                    RQ(1, 32, intersection_instance_custom_index), 0x1);
   if (u & TU_RT_USE_PRIMITIVE_ID)
      nir_store_var(b, s->primitive_id,
                    RQ(1, 32, intersection_primitive_index), 0x1);
   if (u & TU_RT_USE_GEOMETRY_INDEX)
      nir_store_var(b, s->geometry_index,
                    RQ(1, 32, intersection_geometry_index), 0x1);
   if (u & TU_RT_USE_OBJECT_ORIGIN)
      nir_store_var(b, s->object_origin,
                    RQ(3, 32, intersection_object_ray_origin), 0x7);
   if (u & TU_RT_USE_OBJECT_DIR)
      nir_store_var(b, s->object_dir,
                    RQ(3, 32, intersection_object_ray_direction), 0x7);
   for (unsigned col = 0; col < 4; col++) {
      if (u & TU_RT_USE_WTO)
         nir_store_var(b, s->wto[col],
                       RQ(3, 32, intersection_world_to_object, .column = col), 0x7);
      if (u & TU_RT_USE_OTW)
         nir_store_var(b, s->otw[col],
                       RQ(3, 32, intersection_object_to_world, .column = col), 0x7);
   }
#undef RQ
}

static nir_def *
tu_rt_sbt_index(nir_builder *b, struct tu_rt_compiler *c, nir_def *rq,
                bool committed)
{
   struct tu_rt_state *s = &c->state;
   nir_def *instance_sbt =
      nir_rq_load(b, 1, 32, rq,
                  .ray_query_value = nir_ray_query_value_intersection_instance_sbt_index,
                  .committed = committed);
   nir_def *geometry =
      nir_rq_load(b, 1, 32, rq,
                  .ray_query_value = nir_ray_query_value_intersection_geometry_index,
                  .committed = committed);
   return nir_iadd(b, nir_iadd(b, nir_load_var(b, s->sbt_offset), instance_sbt),
                   nir_imul(b, nir_load_var(b, s->sbt_stride), geometry));
}

static nir_def *
tu_rt_triangle_hit_kind(nir_builder *b, nir_def *rq, bool committed)
{
   nir_def *front =
      nir_rq_load(b, 1, 1, rq,
                  .ray_query_value = nir_ray_query_value_intersection_front_face,
                  .committed = committed);
   return nir_bcsel(b, front, nir_imm_int(b, 0xFE), nir_imm_int(b, 0xFF));
}

static void
tu_rt_store_bary(nir_builder *b, struct tu_rt_compiler *c, nir_def *rq,
                 bool committed)
{
   nir_def *bary =
      nir_rq_load(b, 2, 32, rq,
                  .ray_query_value = nir_ray_query_value_intersection_barycentrics,
                  .committed = committed);
   for (unsigned i = 0; i < 2; i++)
      nir_store_deref(b, tu_rt_hw(b, c->hit_words, nir_imm_int(b, i)),
                      nir_channel(b, bary, i), 0x1);
}

static nir_function *tu_rt_get_isec_function(nir_builder *b,
                                             struct tu_rt_compiler *c,
                                             uint32_t group,
                                             unsigned call_depth);

struct tu_rt_ray {
   nir_def *accel, *flags, *cull_mask, *sbt_offset, *sbt_stride, *miss_index;
   nir_def *origin, *tmin, *dir, *tmax, *payload;
};

static void tu_rt_traversal_tail(nir_builder *b, struct tu_rt_compiler *c,
                                 nir_def *rq, nir_def *committed_type,
                                 nir_def *committed_hit_kind,
                                 const struct tu_rt_ray *r);

static nir_def *
tu_rt_fov_launch_id(nir_builder *b, struct tu_rt_compiler *c)
{
   nir_def *size = tu_rt_load_cmd(b, offsetof(VkTraceRaysIndirectCommand2KHR, width), 3);
   nir_def *wg = nir_load_workgroup_id(b);
   nir_def *li = nir_load_local_invocation_index(b);
   nir_def *id1d = nir_vec3(b, nir_iadd(b, nir_ishl_imm(b, nir_channel(b, wg, 0), 6), li),
                            nir_imm_int(b, 0), nir_channel(b, wg, 2));
   nir_def *id2d;
   if (c->fov_mode & TU_RT_FOV_SHARE) {
      nir_def *lx = nir_ior(b, nir_iand_imm(b, li, 1),
                            nir_ior(b, nir_iand_imm(b, nir_ushr_imm(b, li, 1), 2),
                                    nir_iand_imm(b, nir_ushr_imm(b, li, 2), 4)));
      nir_def *ly = nir_ior(b, nir_iand_imm(b, nir_ushr_imm(b, li, 1), 1),
                            nir_ior(b, nir_iand_imm(b, nir_ushr_imm(b, li, 2), 2),
                                    nir_iand_imm(b, nir_ushr_imm(b, li, 3), 4)));
      id2d = nir_vec3(b, nir_iadd(b, nir_ishl_imm(b, nir_channel(b, wg, 0), 3), lx),
                      nir_iadd(b, nir_ishl_imm(b, nir_channel(b, wg, 1), 3), ly),
                      nir_channel(b, wg, 2));
   } else {
      id2d = nir_load_global_invocation_id(b, 32);
   }
   return nir_bcsel(b, nir_ieq_imm(b, nir_channel(b, size, 1), 1), id1d, id2d);
}

struct tu_rt_fov_site {
   nir_def *off;
   nir_def *take[4];
};

static nir_def *
tu_rt_fov_level(nir_builder *b, struct tu_rt_compiler *c)
{
   nir_def *size = tu_rt_load_cmd(b, offsetof(VkTraceRaysIndirectCommand2KHR, width), 3);
   nir_def *p0 = tu_rt_load_cmd(b, TU_RT_FOV_CMD_OFFSET, 2);
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *on = nir_ine_imm(b, nir_channel(b, p0, 0), 0);
   nir_def *flat = nir_bcsel(b, on, nir_channel(b, p0, 1), zero);
   nir_push_if(b, nir_iand(b, on, nir_ine_imm(b, nir_channel(b, size, 1), 1)));

   nir_def *p2 = tu_rt_load_cmd(b, TU_RT_FOV_CMD_OFFSET + 32, 4);
   nir_def *p3 = tu_rt_load_cmd(b, TU_RT_FOV_CMD_OFFSET + 48, 3);
   nir_def *w = nir_u2f32(b, nir_channel(b, size, 0));
   nir_def *h = nir_u2f32(b, nir_channel(b, size, 1));
   nir_def *sbs = nir_iand(b, nir_fge(b, w, nir_fmul(b, nir_channel(b, p2, 0), h)),
                           nir_fge(b, nir_fmul(b, nir_channel(b, p3, 0), h), w));
   nir_def *eye = nir_iand(b, nir_fge(b, w, nir_fmul(b, nir_channel(b, p3, 1), h)),
                           nir_fge(b, nir_fmul(b, nir_channel(b, p3, 2), h), w));
   nir_def *big = nir_iand(b, nir_uge(b, nir_channel(b, size, 0), nir_channel(b, p2, 2)),
                           nir_uge(b, nir_channel(b, size, 1), nir_channel(b, p2, 3)));
   nir_def *max_lvl = nir_channel(b, p2, 1);
   nir_push_if(b, nir_iand(b, big, nir_ior(b, sbs, eye)));

   nir_def *rr = tu_rt_load_cmd(b, TU_RT_FOV_CMD_OFFSET + 8, 2);
   nir_def *gz = tu_rt_load_cmd(b, TU_RT_FOV_CMD_OFFSET + 16, 4);
   nir_def *id = tu_rt_fov_launch_id(b, c);
   nir_def *ew = nir_bcsel(b, sbs, nir_fmul_imm(b, w, 0.5), w);
   nir_def *px = nir_fadd_imm(b, nir_u2f32(b, nir_channel(b, id, 0)), 0.5);
   nir_def *py = nir_fadd_imm(b, nir_u2f32(b, nir_channel(b, id, 1)), 0.5);
   nir_def *eye1 = nir_iand(b, sbs, nir_fge(b, px, ew));
   nir_def *ex0 = nir_bcsel(b, eye1, ew, nir_imm_float(b, 0.0f));
   nir_def *gx = nir_bcsel(b, eye1, nir_channel(b, gz, 2), nir_channel(b, gz, 0));
   nir_def *gy = nir_bcsel(b, eye1, nir_channel(b, gz, 3), nir_channel(b, gz, 1));
   nir_def *dx = nir_fsub(b, px, nir_fadd(b, nir_fmul(b, gx, ew), ex0));
   nir_def *dy = nir_fsub(b, py, nir_fmul(b, gy, h));
   nir_def *d2 = nir_fadd(b, nir_fmul(b, dx, dx), nir_fmul(b, dy, dy));
   nir_def *rr0 = nir_fmul(b, nir_channel(b, rr, 0), h);
   nir_def *rr1 = nir_fmul(b, nir_channel(b, rr, 1), h);
   nir_def *ring =
      nir_bcsel(b, nir_fge(b, d2, nir_fmul(b, rr1, rr1)), nir_imm_int(b, 4),
                nir_bcsel(b, nir_fge(b, d2, nir_fmul(b, rr0, rr0)), nir_imm_int(b, 2), zero));
   ring = nir_umin(b, ring, max_lvl);

   nir_pop_if(b, NULL);
   nir_def *lvl2d = nir_if_phi(b, ring, zero);
   nir_pop_if(b, NULL);
   return nir_if_phi(b, lvl2d, flat);
}

static nir_def *
tu_rt_fov_select(nir_builder *b, nir_def *lvl)
{
   nir_def *lane = nir_load_subgroup_invocation(b);
   nir_def *word0 = nir_ieq_imm(b, nir_ushr_imm(b, lane, 5), 0);
   nir_def *sh = nir_iand_imm(b, lane, 31);
   nir_def *sel = nir_imm_int(b, 0);
   for (unsigned k = 1; k <= 4; k++) {
      nir_def *bal = nir_ballot(b, 2, 32, nir_uge_imm(b, lvl, k));
      nir_def *bits = nir_bcsel(b, word0, nir_channel(b, bal, 0), nir_channel(b, bal, 1));
      const unsigned bs = 1u << k;
      const uint32_t m = (1u << bs) - 1;
      nir_def *base = nir_iand_imm(b, sh, ~(bs - 1) & 31);
      nir_def *full = nir_ieq_imm(b, nir_iand_imm(b, nir_ushr(b, bits, base), m), m);
      sel = nir_bcsel(b, full, nir_imm_int(b, k), sel);
   }
   return sel;
}

static void
tu_rt_fov_begin(nir_builder *b, struct tu_rt_compiler *c, struct tu_rt_fov_site *fs)
{
   nir_def *sel = nir_load_var(b, c->fov_sel);
   nir_def *lane = nir_load_subgroup_invocation(b);
   nir_def *bs = nir_ishl(b, nir_imm_int(b, 1), sel);
   nir_def *act = nir_ballot(b, 2, 32, nir_imm_true(b));
   nir_def *bits = nir_bcsel(b, nir_ieq_imm(b, nir_ushr_imm(b, lane, 5), 0),
                             nir_channel(b, act, 0), nir_channel(b, act, 1));
   nir_def *base = nir_iand(b, nir_iand_imm(b, lane, 31), nir_inot(b, nir_iadd_imm(b, bs, -1)));
   nir_def *m = nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1), bs), -1);
   nir_def *full = nir_ieq(b, nir_iand(b, nir_ushr(b, bits, base), m), m);
   fs->off = nir_bcsel(b, full, nir_iand(b, lane, nir_iadd_imm(b, bs, -1)), nir_imm_int(b, 0));
}

static nir_def *
tu_rt_fov_bcast_val(nir_builder *b, nir_def *v, const struct tu_rt_fov_site *fs)
{
   const unsigned bits = v->bit_size;
   nir_def *x;
   if (bits == 1)
      x = nir_b2i32(b, v);
   else if (bits < 32)
      x = nir_u2u32(b, v);
   else if (bits == 64)
      x = nir_bitcast_vector(b, v, 32);
   else
      x = v;
   for (unsigned i = 0; i < 4; i++) {
      nir_def *t = nir_shuffle_xor(b, x, nir_imm_int(b, 1u << i));
      x = nir_bcsel(b, fs->take[i], t, x);
   }
   if (bits == 1)
      return nir_ine_imm(b, x, 0);
   if (bits < 32)
      return nir_u2uN(b, x, bits);
   if (bits == 64)
      return nir_bitcast_vector(b, x, 64);
   return x;
}

static void
tu_rt_fov_bcast_deref(nir_builder *b, nir_deref_instr *d, const struct tu_rt_fov_site *fs)
{
   const struct glsl_type *t = d->type;
   if (glsl_type_is_vector_or_scalar(t)) {
      nir_def *v = nir_load_deref(b, d);
      nir_store_deref(b, d, tu_rt_fov_bcast_val(b, v, fs),
                      nir_component_mask(v->num_components));
   } else if (glsl_type_is_struct_or_ifc(t)) {
      for (unsigned i = 0; i < glsl_get_length(t); i++)
         tu_rt_fov_bcast_deref(b, nir_build_deref_struct(b, d, i), fs);
   } else if (glsl_type_is_matrix(t)) {
      for (unsigned i = 0; i < glsl_get_matrix_columns(t); i++)
         tu_rt_fov_bcast_deref(b, nir_build_deref_array_imm(b, d, i), fs);
   } else if (glsl_type_is_array(t)) {
      for (unsigned i = 0; i < glsl_get_length(t); i++)
         tu_rt_fov_bcast_deref(b, nir_build_deref_array_imm(b, d, i), fs);
   }
}

static void
tu_rt_trace_ray_emit(nir_builder *b, struct tu_rt_compiler *c,
                     const struct tu_rt_ray *r)
{
   struct tu_rt_state *s = &c->state;
   const uint32_t hit_sbt =
      offsetof(VkTraceRaysIndirectCommand2KHR, hitShaderBindingTableAddress);

   nir_def *accel_struct = r->accel;
   nir_def *flags = r->flags;
   nir_def *cull_mask = r->cull_mask;
   nir_def *sbt_offset = nir_iand_imm(b, r->sbt_offset, 0xF);
   nir_def *sbt_stride = nir_iand_imm(b, r->sbt_stride, 0xF);
   nir_def *miss_index = nir_iand_imm(b, r->miss_index, 0xFFFF);
   nir_def *origin = r->origin;
   nir_def *tmin = r->tmin;
   nir_def *dir = r->dir;
   nir_def *tmax = r->tmax;
   nir_def *payload = r->payload;
   const unsigned cd = tu_rt_ctx(c, b->impl->function)->call_depth;

   struct tu_rt_fov_site fov_site = {};
   const bool fov = (c->fov_mode & TU_RT_FOV_SHARE) && nir_def_is_deref(payload);
   if (fov) {
      tu_rt_fov_begin(b, c, &fov_site);
      nir_push_if(b, nir_ieq_imm(b, fov_site.off, 0));
   }

   nir_store_var(b, s->flags, flags, 0x1);
   nir_store_var(b, s->cull_mask, cull_mask, 0x1);
   nir_store_var(b, s->sbt_offset, sbt_offset, 0x1);
   nir_store_var(b, s->sbt_stride, sbt_stride, 0x1);
   nir_store_var(b, s->origin, origin, 0x7);
   nir_store_var(b, s->tmin, tmin, 0x1);
   nir_store_var(b, s->dir, dir, 0x7);
   nir_store_var(b, s->tmax, tmax, 0x1);

   nir_variable *rq_var =
      nir_local_variable_create(b->impl, glsl_uint64_t_type(), "trace_rq");
   rq_var->data.ray_query = true;
   nir_def *rq = &nir_build_deref_var(b, rq_var)->def;

   nir_variable *committed_hit_kind =
      nir_local_variable_create(b->impl, glsl_uint_type(), "committed_hit_kind");
   nir_store_var(b, committed_hit_kind, nir_imm_int(b, 0), 0x1);

   nir_def *rq_flags =
      nir_iand_imm(b, flags, ~SpvRayFlagsSkipClosestHitShaderKHRMask);
   if (c->flags & VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_TRIANGLES_BIT_KHR) {
      rq_flags = nir_ior_imm(b, nir_iand_imm(b, rq_flags,
                                             ~(SpvRayFlagsCullBackFacingTrianglesKHRMask |
                                               SpvRayFlagsCullFrontFacingTrianglesKHRMask)),
                             SpvRayFlagsSkipTrianglesKHRMask);
   }
   if (c->flags & VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_AABBS_BIT_KHR)
      rq_flags = nir_ior_imm(b, rq_flags, SpvRayFlagsSkipAABBsKHRMask);
   if (!c->has_isec) {
      rq_flags = nir_bcsel(b, nir_test_mask(b, rq_flags, SpvRayFlagsSkipTrianglesKHRMask),
                           rq_flags,
                           nir_ior_imm(b, rq_flags, SpvRayFlagsSkipAABBsKHRMask));
   }
   if (!c->has_ahit) {
      nir_def *forced =
         nir_ior_imm(b, nir_iand_imm(b, rq_flags, ~SpvRayFlagsNoOpaqueKHRMask),
                     SpvRayFlagsOpaqueKHRMask);
      nir_def *culls_opacity =
         nir_test_mask(b, flags, SpvRayFlagsCullOpaqueKHRMask |
                                 SpvRayFlagsCullNoOpaqueKHRMask);
      rq_flags = nir_bcsel(b, culls_opacity, rq_flags, forced);
   }

   nir_rq_initialize(b, rq, accel_struct, rq_flags, cull_mask, origin, tmin,
                     dir, tmax);

   nir_loop *loop = nir_push_loop(b);
   {
      nir_break_if(b, nir_inot(b, nir_rq_proceed(b, 1, rq)));

      nir_def *ctype =
         nir_rq_load(b, 1, 32, rq,
                     .ray_query_value = nir_ray_query_value_intersection_type,
                     .committed = false);

      nir_push_if(b, nir_ieq_imm(b, ctype, 0));
      if (!c->has_ahit) {
         nir_rq_confirm_intersection(b, rq);
      } else {
         tu_rt_load_hit_state(b, c, rq, false);
         struct tu_rt_sbt_entry entry =
            tu_rt_load_sbt_entry(b, tu_rt_sbt_index(b, c, rq, false), hit_sbt);
         nir_store_var(b, s->shader_record_ptr, entry.shader_record_ptr, 0x1);

         nir_def *t = nir_rq_load(b, 1, 32, rq,
                                  .ray_query_value = nir_ray_query_value_intersection_t,
                                  .committed = false);
         nir_def *prev_tmax = nir_load_var(b, s->tmax);
         nir_def *prev_w0 = nir_load_deref(b, tu_rt_hw(b, c->hit_words, nir_imm_int(b, 0)));
         nir_def *prev_w1 = nir_load_deref(b, tu_rt_hw(b, c->hit_words, nir_imm_int(b, 1)));

         nir_store_var(b, s->accept, nir_imm_true(b), 0x1);
         nir_store_var(b, s->terminate, nir_imm_false(b), 0x1);
         nir_store_var(b, s->tmax, t, 0x1);
         nir_store_var(b, s->hit_kind, tu_rt_triangle_hit_kind(b, rq, false), 0x1);
         tu_rt_store_bary(b, c, rq, false);

         {
            struct tu_rt_case *cases = (struct tu_rt_case *)
               malloc(sizeof(struct tu_rt_case) * MAX2(c->pipeline->group_count, 1));
            unsigned count = 0;
            for (uint32_t i = 0; i < c->pipeline->group_count; i++) {
               nir_shader *ahit = tu_rt_group_stage(c, c->pipeline->groups[i].ahit_index);
               if (!ahit || c->pipeline->groups[i].isec_index != VK_SHADER_UNUSED_KHR ||
                   tu_rt_handle_seen(c, i, c->pipeline->groups[i].handle, 1) ||
                   !tu_rt_compat(c, ahit, payload))
                  continue;
               cases[count++] = (struct tu_rt_case) { c->pipeline->groups[i].handle, i };
            }
            tu_rt_emit_cases(b, entry.value, cases, count, [&](const struct tu_rt_case &cs) {
               nir_shader *ahit = tu_rt_group_stage(c, c->pipeline->groups[cs.group].ahit_index);
               tu_rt_call_stage(b, c, ahit, payload, 1, cd);
            });
            free(cases);
         }

         nir_push_if(b, nir_load_var(b, s->accept));
         {
            nir_rq_confirm_intersection(b, rq);
            nir_push_if(b, nir_load_var(b, s->terminate));
            nir_rq_terminate(b, rq);
            nir_pop_if(b, NULL);
         }
         nir_push_else(b, NULL);
         {
            nir_store_deref(b, tu_rt_hw(b, c->hit_words, nir_imm_int(b, 0)), prev_w0, 0x1);
            nir_store_deref(b, tu_rt_hw(b, c->hit_words, nir_imm_int(b, 1)), prev_w1, 0x1);
         }
         nir_pop_if(b, NULL);
         nir_store_var(b, s->tmax, prev_tmax, 0x1);
      }
      nir_push_else(b, NULL);
      if (c->has_isec) {
         tu_rt_load_hit_state(b, c, rq, false);
         struct tu_rt_sbt_entry entry =
            tu_rt_load_sbt_entry(b, tu_rt_sbt_index(b, c, rq, false), hit_sbt);
         nir_store_var(b, s->shader_record_ptr, entry.shader_record_ptr, 0x1);

         nir_def *prev_words[TU_RT_HIT_WORDS];
         for (unsigned i = 0; i < TU_RT_HIT_WORDS; i++)
            prev_words[i] = nir_load_deref(b, tu_rt_hw(b, c->hit_words, nir_imm_int(b, i)));

         nir_def *prev_tmax = nir_load_var(b, s->tmax);
         nir_store_var(b, s->accept, nir_imm_false(b), 0x1);
         nir_store_var(b, s->terminate,
                       nir_test_mask(b, flags, SpvRayFlagsTerminateOnFirstHitKHRMask), 0x1);
         nir_store_var(b, s->opaque,
                       nir_rq_load(b, 1, 1, rq,
                                   .ray_query_value = nir_ray_query_value_intersection_candidate_aabb_opaque,
                                   .committed = false),
                       0x1);
         nir_store_var(b, s->tmax,
                       nir_rq_load(b, 1, 32, rq,
                                   .ray_query_value = nir_ray_query_value_intersection_t,
                                   .committed = true),
                       0x1);

         {
            struct tu_rt_case *cases = (struct tu_rt_case *)
               malloc(sizeof(struct tu_rt_case) * MAX2(c->pipeline->group_count, 1));
            unsigned count = 0;
            for (uint32_t i = 0; i < c->pipeline->group_count; i++) {
               if (c->pipeline->groups[i].isec_index == VK_SHADER_UNUSED_KHR ||
                   tu_rt_handle_seen(c, i, c->pipeline->groups[i].handle, 2))
                  continue;
               cases[count++] = (struct tu_rt_case) { c->pipeline->groups[i].handle, i };
            }
            tu_rt_emit_cases(b, entry.value, cases, count, [&](const struct tu_rt_case &cs) {
               nir_build_call(b, tu_rt_get_isec_function(b, c, cs.group, cd), 1, &payload);
            });
            free(cases);
         }

         nir_push_if(b, nir_load_var(b, s->accept));
         {
            nir_rq_generate_intersection(b, rq, nir_load_var(b, s->tmax));
            nir_store_var(b, committed_hit_kind, nir_load_var(b, s->hit_kind), 0x1);
            nir_push_if(b, nir_load_var(b, s->terminate));
            nir_rq_terminate(b, rq);
            nir_pop_if(b, NULL);
         }
         nir_push_else(b, NULL);
         {
            for (unsigned i = 0; i < TU_RT_HIT_WORDS; i++)
               nir_store_deref(b, tu_rt_hw(b, c->hit_words, nir_imm_int(b, i)),
                               prev_words[i], 0x1);
         }
         nir_pop_if(b, NULL);
         nir_store_var(b, s->tmax, prev_tmax, 0x1);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_loop(b, loop);

   nir_def *committed_type =
      nir_rq_load(b, 1, 32, rq,
                  .ray_query_value = nir_ray_query_value_intersection_type,
                  .committed = true);

   if (c->part == TU_RT_PART_TRAVERSAL) {
      struct tu_rt_ray masked = *r;
      masked.miss_index = miss_index;
      tu_rt_traversal_tail(b, c, rq, committed_type,
                           nir_load_var(b, committed_hit_kind), &masked);
      return;
   }

   nir_push_if(b, nir_ine_imm(b, committed_type, 0));
   {
      nir_def *skip_chit =
         nir_test_mask(b, flags, SpvRayFlagsSkipClosestHitShaderKHRMask);
      nir_push_if(b, nir_inot(b, skip_chit));
      {
         tu_rt_load_hit_state(b, c, rq, true);
         nir_store_var(b, s->tmax,
                       nir_rq_load(b, 1, 32, rq,
                                   .ray_query_value = nir_ray_query_value_intersection_t,
                                   .committed = true),
                       0x1);

         if (c->has_isec) {
            nir_push_if(b, nir_ieq_imm(b, committed_type, 1));
            {
               nir_store_var(b, s->hit_kind, tu_rt_triangle_hit_kind(b, rq, true), 0x1);
               tu_rt_store_bary(b, c, rq, true);
            }
            nir_push_else(b, NULL);
            {
               nir_store_var(b, s->hit_kind, nir_load_var(b, committed_hit_kind), 0x1);
            }
            nir_pop_if(b, NULL);
         } else {
            nir_store_var(b, s->hit_kind, tu_rt_triangle_hit_kind(b, rq, true), 0x1);
            tu_rt_store_bary(b, c, rq, true);
         }

         struct tu_rt_sbt_entry chit =
            tu_rt_load_sbt_entry(b, tu_rt_sbt_index(b, c, rq, true), hit_sbt);
         nir_store_var(b, s->shader_record_ptr, chit.shader_record_ptr, 0x1);
         tu_rt_dispatch_recursive(b, c, chit.value, MESA_SHADER_CLOSEST_HIT, payload, 1, cd);
      }
      nir_pop_if(b, NULL);
   }
   nir_push_else(b, NULL);
   {
      struct tu_rt_sbt_entry miss = tu_rt_load_sbt_entry(
         b, miss_index,
         offsetof(VkTraceRaysIndirectCommand2KHR, missShaderBindingTableAddress));
      nir_store_var(b, s->shader_record_ptr, miss.shader_record_ptr, 0x1);
      tu_rt_dispatch_recursive(b, c, miss.value, MESA_SHADER_MISS, payload, 1, cd);
   }
   nir_pop_if(b, NULL);

   if (fov) {
      nir_pop_if(b, NULL);
      nir_push_if(b, nir_vote_any(b, 1, nir_ine_imm(b, fov_site.off, 0)));
      for (unsigned i = 0; i < 4; i++)
         fov_site.take[i] = nir_ine_imm(b, nir_iand_imm(b, fov_site.off, 1u << i), 0);
      tu_rt_fov_bcast_deref(b, nir_def_as_deref(payload), &fov_site);
      nir_pop_if(b, NULL);
   }
}

static void
tu_rt_fstore_vec(nir_builder *b, unsigned off, nir_def *v)
{
   for (unsigned i = 0; i < v->num_components; i++)
      tu_rt_fstore(b, off + 4 * i, nir_channel(b, v, i));
}

static nir_def *
tu_rt_fload_vec(nir_builder *b, unsigned off, unsigned comps)
{
   nir_def *v[4];
   for (unsigned i = 0; i < comps; i++)
      v[i] = tu_rt_fload(b, off + 4 * i, 1);
   return nir_vec(b, v, comps);
}

static void
tu_rt_traversal_miss(nir_builder *b, struct tu_rt_compiler *c, nir_def *miss_index)
{
   if (!c->part_has_miss) {
      tu_rt_fstore(b, TU_RT_FRAME_SEL, tu_rt_fload(b, TU_RT_FRAME_RET, 1));
      return;
   }
   struct tu_rt_sbt_entry miss = tu_rt_load_sbt_entry(
      b, miss_index,
      offsetof(VkTraceRaysIndirectCommand2KHR, missShaderBindingTableAddress));
   tu_rt_fstore(b, TU_RT_FRAME_HANDLE, miss.value);
   tu_rt_fstore_vec(b, TU_RT_FRAME_RECORD, nir_unpack_64_2x32(b, miss.shader_record_ptr));
   tu_rt_fstore(b, TU_RT_FRAME_SEL, nir_imm_int(b, TU_RT_SEL_MISS(c->part_payload_size)));
}

static void
tu_rt_traversal_tail(nir_builder *b, struct tu_rt_compiler *c, nir_def *rq,
                     nir_def *committed_type, nir_def *committed_hit_kind,
                     const struct tu_rt_ray *r)
{
   struct tu_rt_state *s = &c->state;
   const uint32_t hit_sbt =
      offsetof(VkTraceRaysIndirectCommand2KHR, hitShaderBindingTableAddress);
   const unsigned cls = c->part_payload_size;
   const uint32_t u = c->part_hit_used;

   nir_push_if(b, nir_ine_imm(b, committed_type, 0));
   if (!c->part_has_chit) {
      tu_rt_fstore(b, TU_RT_FRAME_SEL, tu_rt_fload(b, TU_RT_FRAME_RET, 1));
   } else {
      nir_push_if(b, nir_test_mask(b, r->flags, SpvRayFlagsSkipClosestHitShaderKHRMask));
      {
         tu_rt_fstore(b, TU_RT_FRAME_SEL, tu_rt_fload(b, TU_RT_FRAME_RET, 1));
      }
      nir_push_else(b, NULL);
      {
         uint32_t saved = c->used;
         c->used = u;
         tu_rt_load_hit_state(b, c, rq, true);
         c->used = saved;

         tu_rt_fstore(b, TU_RT_FRAME_TMAX,
                      nir_rq_load(b, 1, 32, rq,
                                  .ray_query_value = nir_ray_query_value_intersection_t,
                                  .committed = true));
         if (c->has_isec) {
            nir_push_if(b, nir_ieq_imm(b, committed_type, 1));
            {
               nir_store_var(b, s->hit_kind, tu_rt_triangle_hit_kind(b, rq, true), 0x1);
               tu_rt_store_bary(b, c, rq, true);
            }
            nir_push_else(b, NULL);
            {
               nir_store_var(b, s->hit_kind, committed_hit_kind, 0x1);
            }
            nir_pop_if(b, NULL);
         } else {
            nir_store_var(b, s->hit_kind, tu_rt_triangle_hit_kind(b, rq, true), 0x1);
            tu_rt_store_bary(b, c, rq, true);
         }
         tu_rt_fstore(b, TU_RT_FRAME_HIT_KIND, nir_load_var(b, s->hit_kind));
         for (unsigned i = 0; i < (c->has_isec ? TU_RT_HIT_WORDS : 2); i++)
            tu_rt_fstore(b, TU_RT_FRAME_HIT_WORDS + 4 * i,
                         nir_load_deref(b, tu_rt_hw(b, c->hit_words, nir_imm_int(b, i))));

         if (u & TU_RT_USE_INSTANCE_ID)
            tu_rt_fstore(b, TU_RT_FRAME_INSTANCE_ID, nir_load_var(b, s->instance_id));
         if (u & TU_RT_USE_CUSTOM_INDEX)
            tu_rt_fstore(b, TU_RT_FRAME_CUSTOM_INDEX, nir_load_var(b, s->custom_index));
         if (u & TU_RT_USE_PRIMITIVE_ID)
            tu_rt_fstore(b, TU_RT_FRAME_PRIMITIVE_ID, nir_load_var(b, s->primitive_id));
         if (u & TU_RT_USE_GEOMETRY_INDEX)
            tu_rt_fstore(b, TU_RT_FRAME_GEOMETRY_INDEX, nir_load_var(b, s->geometry_index));
         if (u & TU_RT_USE_OBJECT_ORIGIN)
            tu_rt_fstore_vec(b, TU_RT_FRAME_OBJECT_ORIGIN, nir_load_var(b, s->object_origin));
         if (u & TU_RT_USE_OBJECT_DIR)
            tu_rt_fstore_vec(b, TU_RT_FRAME_OBJECT_DIR, nir_load_var(b, s->object_dir));
         for (unsigned col = 0; col < 4; col++) {
            if (u & TU_RT_USE_WTO)
               tu_rt_fstore_vec(b, TU_RT_FRAME_WTO + 12 * col, nir_load_var(b, s->wto[col]));
            if (u & TU_RT_USE_OTW)
               tu_rt_fstore_vec(b, TU_RT_FRAME_OTW + 12 * col, nir_load_var(b, s->otw[col]));
         }

         struct tu_rt_sbt_entry chit =
            tu_rt_load_sbt_entry(b, tu_rt_sbt_index(b, c, rq, true), hit_sbt);
         tu_rt_fstore(b, TU_RT_FRAME_HANDLE, chit.value);
         tu_rt_fstore_vec(b, TU_RT_FRAME_RECORD, nir_unpack_64_2x32(b, chit.shader_record_ptr));
         tu_rt_fstore(b, TU_RT_FRAME_SEL, nir_imm_int(b, TU_RT_SEL_CHIT(cls)));
      }
      nir_pop_if(b, NULL);
   }
   nir_push_else(b, NULL);
   tu_rt_traversal_miss(b, c, r->miss_index);
   nir_pop_if(b, NULL);
}

static void
tu_rt_trace_ray(nir_builder *b, struct tu_rt_compiler *c,
                nir_intrinsic_instr *instr)
{
   struct tu_rt_ray r = {
      .accel = instr->src[0].ssa,
      .flags = instr->src[1].ssa,
      .cull_mask = instr->src[2].ssa,
      .sbt_offset = instr->src[3].ssa,
      .sbt_stride = instr->src[4].ssa,
      .miss_index = instr->src[5].ssa,
      .origin = instr->src[6].ssa,
      .tmin = instr->src[7].ssa,
      .dir = instr->src[8].ssa,
      .tmax = instr->src[9].ssa,
      .payload = instr->src[10].ssa,
   };
   tu_rt_trace_ray_emit(b, c, &r);
}

static void
tu_rt_execute_callable(nir_builder *b, struct tu_rt_compiler *c,
                       nir_intrinsic_instr *instr)
{
   struct tu_rt_sbt_entry entry = tu_rt_load_sbt_entry(
      b, instr->src[0].ssa,
      offsetof(VkTraceRaysIndirectCommand2KHR, callableShaderBindingTableAddress));
   nir_def *prev_record = nir_load_var(b, c->state.shader_record_ptr);
   nir_store_var(b, c->state.shader_record_ptr, entry.shader_record_ptr, 0x1);
   const struct tu_rt_fn_ctx *ctx = tu_rt_ctx(c, b->impl->function);
   tu_rt_dispatch_recursive(b, c, entry.value, MESA_SHADER_CALLABLE, instr->src[1].ssa,
                            ctx->ray_depth, ctx->call_depth + 1);
   nir_store_var(b, c->state.shader_record_ptr, prev_record, 0x1);
}

struct tu_rt_isec_state {
   struct tu_rt_compiler *compiler;
   nir_shader *ahit;
   unsigned call_depth;
};

static bool
tu_rt_lower_isec_intrinsic(nir_builder *b, nir_intrinsic_instr *instr, void *data)
{
   if (instr->intrinsic != nir_intrinsic_report_ray_intersection)
      return false;

   struct tu_rt_isec_state *isec = (struct tu_rt_isec_state *) data;
   struct tu_rt_compiler *c = isec->compiler;
   struct tu_rt_state *s = &c->state;

   b->cursor = nir_after_instr(&instr->instr);

   nir_def *t = instr->src[0].ssa;
   nir_def *hit_kind = instr->src[1].ssa;

   nir_def *prev_accept = nir_load_var(b, s->accept);
   nir_def *prev_tmax = nir_load_var(b, s->tmax);
   nir_def *prev_hit_kind = nir_load_var(b, s->hit_kind);

   nir_variable *commit =
      nir_local_variable_create(b->impl, glsl_bool_type(), "commit");
   nir_store_var(b, commit, nir_imm_false(b), 0x1);

   nir_def *in_range = nir_iand(b, nir_fge(b, t, nir_load_var(b, s->tmin)),
                                nir_fge(b, nir_load_var(b, s->tmax), t));
   nir_def *terminated = nir_iand(b, nir_load_var(b, s->terminate),
                                  nir_load_var(b, s->accept));
   nir_push_if(b, nir_iand(b, in_range, nir_inot(b, terminated)));
   {
      nir_store_var(b, s->accept, nir_imm_true(b), 0x1);
      nir_store_var(b, s->tmax, t, 0x1);
      nir_store_var(b, s->hit_kind, hit_kind, 0x1);

      if (isec->ahit) {
         nir_def *prev_terminate = nir_load_var(b, s->terminate);
         nir_store_var(b, s->terminate, nir_imm_false(b), 0x1);

         nir_push_if(b, nir_inot(b, nir_load_var(b, s->opaque)));
         tu_rt_call_stage(b, c, isec->ahit, nir_load_param(b, 0), 1, isec->call_depth);
         nir_pop_if(b, NULL);

         nir_def *terminate = nir_load_var(b, s->terminate);
         nir_store_var(b, s->terminate, nir_ior(b, terminate, prev_terminate), 0x1);

         nir_push_if(b, terminate);
         nir_jump(b, nir_jump_return);
         nir_pop_if(b, NULL);
      }

      nir_push_if(b, nir_load_var(b, s->accept));
      {
         nir_store_var(b, commit, nir_imm_true(b), 0x1);
      }
      nir_push_else(b, NULL);
      {
         nir_store_var(b, s->accept, prev_accept, 0x1);
         nir_store_var(b, s->tmax, prev_tmax, 0x1);
         nir_store_var(b, s->hit_kind, prev_hit_kind, 0x1);
      }
      nir_pop_if(b, NULL);

      nir_push_if(b, nir_iand(b, nir_load_var(b, commit), nir_load_var(b, s->terminate)));
      nir_jump(b, nir_jump_return);
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);

   nir_def_replace(&instr->def, nir_load_var(b, commit));
   return true;
}

static nir_function *
tu_rt_get_isec_function(nir_builder *b, struct tu_rt_compiler *c, uint32_t group,
                        unsigned call_depth)
{
   uint32_t slot = group * (TU_RT_MAX_CALL_DEPTH + 1) + call_depth;
   if (c->isec_cache[slot])
      return c->isec_cache[slot];

   struct tu_rt_group *g = c->pipeline->groups + group;
   nir_function *function =
      tu_rt_clone_stage(c, b->shader, c->pipeline->stages[g->isec_index]);
   c->isec_cache[slot] = function;
   tu_rt_set_ctx(c, function, MESA_SHADER_INTERSECTION, 1, call_depth);

   struct tu_rt_isec_state isec = {
      .compiler = c,
      .ahit = tu_rt_group_stage(c, g->ahit_index),
      .call_depth = call_depth,
   };
   nir_function_intrinsics_pass(function->impl, tu_rt_lower_isec_intrinsic,
                                nir_metadata_none, &isec);
   return function;
}

static unsigned tu_rt_natural_field_offset(const struct glsl_type *st,
                                           unsigned field);

template <typename F>
static bool
tu_rt_walk_leaves(nir_builder *b, nir_deref_instr *deref, unsigned offset, F &cb)
{
   const struct glsl_type *t = deref->type;
   if (glsl_type_is_struct_or_ifc(t)) {
      for (unsigned i = 0; i < glsl_get_length(t); i++) {
         if (!tu_rt_walk_leaves(b, nir_build_deref_struct(b, deref, i),
                                offset + tu_rt_natural_field_offset(t, i), cb))
            return false;
      }
      return true;
   }
   if (glsl_type_is_array(t) || glsl_type_is_matrix(t)) {
      const struct glsl_type *et = glsl_type_is_array(t) ?
         glsl_get_array_element(t) : glsl_get_column_type(t);
      unsigned n = glsl_type_is_array(t) ? glsl_get_length(t) :
         glsl_get_matrix_columns(t);
      unsigned size, al;
      glsl_get_natural_size_align_bytes(et, &size, &al);
      unsigned stride = align(size, al);
      for (unsigned i = 0; i < n; i++) {
         if (!tu_rt_walk_leaves(b, nir_build_deref_array_imm(b, deref, i),
                                offset + i * stride, cb))
            return false;
      }
      return true;
   }
   if (!glsl_type_is_vector_or_scalar(t))
      return false;
   cb(deref, offset);
   return true;
}

static bool
tu_rt_copy_payload(nir_builder *b, nir_deref_instr *deref, unsigned size,
                   bool to_frame)
{
   unsigned nwords = DIV_ROUND_UP(size, 4);
   if (!nwords)
      return true;
   nir_variable *w =
      nir_local_variable_create(b->impl, glsl_array_type(glsl_uint_type(), nwords, 4),
                                "tu_rt_pw");
   if (to_frame) {
      auto cb = [&](nir_deref_instr *leaf, unsigned off) {
         nir_def *v = nir_load_deref(b, leaf);
         for (unsigned comp = 0; comp < v->num_components; comp++)
            tu_rt_hw_store_comp(b, w, nir_imm_int(b, off), comp, nir_channel(b, v, comp));
      };
      if (!tu_rt_walk_leaves(b, deref, 0, cb))
         return false;
      for (unsigned i = 0; i < nwords; i++)
         tu_rt_fstore(b, TU_RT_FRAME_PAYLOAD + 4 * i,
                      nir_load_deref(b, tu_rt_hw(b, w, nir_imm_int(b, i))));
   } else {
      for (unsigned i = 0; i < nwords; i++)
         nir_store_deref(b, tu_rt_hw(b, w, nir_imm_int(b, i)),
                         tu_rt_fload(b, TU_RT_FRAME_PAYLOAD + 4 * i, 1), 0x1);
      auto cb = [&](nir_deref_instr *leaf, unsigned off) {
         unsigned nc = glsl_get_vector_elements(leaf->type);
         unsigned bs = glsl_get_bit_size(leaf->type);
         nir_def *vals[NIR_MAX_VEC_COMPONENTS];
         for (unsigned comp = 0; comp < nc; comp++)
            vals[comp] = tu_rt_hw_load_comp(b, w, nir_imm_int(b, off), comp, bs);
         nir_store_deref(b, leaf, nir_vec(b, vals, nc), nir_component_mask(nc));
      };
      if (!tu_rt_walk_leaves(b, deref, 0, cb))
         return false;
   }
   return true;
}

static void
tu_rt_part_add_class(struct tu_rt_compiler *c, unsigned size)
{
   for (unsigned i = 0; i < c->part_nclasses; i++) {
      if (c->part_classes[i] == size)
         return;
   }
   if (c->part_nclasses == TU_RT_MAX_CLASSES) {
      c->part_fail = "too many payload classes";
      return;
   }
   c->part_classes[c->part_nclasses++] = size;
}

static bool
tu_rt_part_raygen_trace(nir_builder *b, struct tu_rt_compiler *c,
                        nir_intrinsic_instr *intr)
{
   nir_def *payload = intr->src[10].ssa;
   nir_deref_instr *pd = NULL;
   unsigned size = 0;
   if (nir_def_is_deref(payload)) {
      unsigned al;
      pd = nir_def_as_deref(payload);
      glsl_get_natural_size_align_bytes(pd->type, &size, &al);
   }
   if (size > TU_RT_PAYLOAD_MAX) {
      c->part_fail = "payload too large";
      return false;
   }
   uint32_t cls = size;
   if (nir_src_is_const(intr->src[1]) &&
       (nir_src_as_uint(intr->src[1]) & SpvRayFlagsSkipClosestHitShaderKHRMask))
      cls |= TU_RT_CLASS_SKIP_CHIT;
   tu_rt_part_add_class(c, cls);

   b->cursor = nir_before_instr(&intr->instr);
   if (pd && !tu_rt_copy_payload(b, pd, size, true))
      c->part_fail = "payload type";
   tu_rt_fstore(b, TU_RT_FRAME_SEL, nir_imm_int(b, TU_RT_SEL_TRAV(cls)));
   if (!c->part_dummy)
      c->part_dummy = nir_variable_create(b->shader, nir_var_shader_call_data,
                                          glsl_uint_type(), "tu_rt_no_payload");
   nir_src_rewrite(&intr->src[10], &nir_build_deref_var(b, c->part_dummy)->def);

   b->cursor = nir_after_instr(&intr->instr);
   if (pd && !tu_rt_copy_payload(b, pd, size, false))
      c->part_fail = "payload type";
   return true;
}

static bool
tu_rt_lower_instr(nir_builder *b, nir_instr *instr, void *data)
{
   struct tu_rt_compiler *c = (struct tu_rt_compiler *) data;
   struct tu_rt_state *s = &c->state;

   if (instr->type == nir_instr_type_jump) {
      nir_jump_instr *jump = nir_instr_as_jump(instr);
      if (jump->type == nir_jump_halt) {
         jump->type = nir_jump_return;
         return true;
      }
      return false;
   } else if (instr->type != nir_instr_type_intrinsic) {
      return false;
   }

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   nir_def *def = NULL;
   b->cursor = nir_before_instr(instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_execute_callable:
      if (c->part != TU_RT_PART_NONE && c->part != TU_RT_PART_WHOLE)
         c->part_fail = "callable";
      else if (tu_rt_ctx(c, b->impl->function)->call_depth < c->max_call_depth)
         tu_rt_execute_callable(b, c, intr);
      break;
   case nir_intrinsic_trace_ray:
      if (tu_rt_ctx(c, b->impl->function)->ray_depth == 0) {
         if (c->part == TU_RT_PART_RAYGEN)
            return tu_rt_part_raygen_trace(b, c, intr);
         tu_rt_trace_ray(b, c, intr);
      }
      break;
   case nir_intrinsic_ignore_ray_intersection:
      nir_store_var(b, s->accept, nir_imm_false(b), 0x1);
      nir_push_if(b, nir_imm_true(b));
      nir_jump(b, nir_jump_return);
      nir_pop_if(b, NULL);
      break;
   case nir_intrinsic_terminate_ray:
      nir_store_var(b, s->accept, nir_imm_true(b), 0x1);
      nir_store_var(b, s->terminate, nir_imm_true(b), 0x1);
      nir_push_if(b, nir_imm_true(b));
      nir_jump(b, nir_jump_return);
      nir_pop_if(b, NULL);
      break;
   case nir_intrinsic_load_ray_launch_id:
      if (c->part != TU_RT_PART_NONE)
         return false;
      def = c->fov_mode ? tu_rt_fov_launch_id(b, c) : nir_load_global_invocation_id(b, 32);
      break;
   case nir_intrinsic_load_ray_launch_size:
      def = tu_rt_load_cmd(b, offsetof(VkTraceRaysIndirectCommand2KHR, width), 3);
      break;
   case nir_intrinsic_load_shader_record_ptr:
      def = nir_load_var(b, s->shader_record_ptr);
      break;
   case nir_intrinsic_load_ray_t_min:
      def = nir_load_var(b, s->tmin);
      break;
   case nir_intrinsic_load_ray_t_max:
      def = nir_load_var(b, s->tmax);
      break;
   case nir_intrinsic_load_ray_world_origin:
      def = nir_load_var(b, s->origin);
      break;
   case nir_intrinsic_load_ray_world_direction:
      def = nir_load_var(b, s->dir);
      break;
   case nir_intrinsic_load_ray_object_origin:
      def = nir_load_var(b, s->object_origin);
      break;
   case nir_intrinsic_load_ray_object_direction:
      def = nir_load_var(b, s->object_dir);
      break;
   case nir_intrinsic_load_ray_instance_custom_index:
      def = nir_load_var(b, s->custom_index);
      break;
   case nir_intrinsic_load_instance_id:
      def = nir_load_var(b, s->instance_id);
      break;
   case nir_intrinsic_load_primitive_id:
      def = nir_load_var(b, s->primitive_id);
      break;
   case nir_intrinsic_load_ray_geometry_index:
      def = nir_load_var(b, s->geometry_index);
      break;
   case nir_intrinsic_load_ray_flags:
      def = nir_load_var(b, s->flags);
      break;
   case nir_intrinsic_load_ray_hit_kind:
      def = nir_load_var(b, s->hit_kind);
      break;
   case nir_intrinsic_load_cull_mask:
      def = nir_iand_imm(b, nir_load_var(b, s->cull_mask), 0xFF);
      break;
   case nir_intrinsic_load_ray_world_to_object:
      def = nir_load_var(b, s->wto[nir_intrinsic_column(intr)]);
      break;
   case nir_intrinsic_load_ray_object_to_world:
      def = nir_load_var(b, s->otw[nir_intrinsic_column(intr)]);
      break;
   default:
      return false;
   }

   if (def)
      nir_def_rewrite_uses(&intr->def, def);
   nir_instr_remove(instr);
   return true;
}

static unsigned
tu_rt_natural_field_offset(const struct glsl_type *st, unsigned field)
{
   unsigned off = 0;
   for (unsigned i = 0; i <= field; i++) {
      unsigned size, al;
      glsl_get_natural_size_align_bytes(glsl_get_struct_field(st, i), &size, &al);
      off = align(off, al);
      if (i == field)
         return off;
      off += size;
   }
   return off;
}

static nir_def *
tu_rt_pun_offset(nir_builder *b, nir_deref_instr *deref)
{
   switch (deref->deref_type) {
   case nir_deref_type_var:
      return nir_imm_int(b, 0);
   case nir_deref_type_cast:
      return tu_rt_pun_offset(b, nir_deref_instr_parent(deref));
   case nir_deref_type_struct: {
      nir_deref_instr *parent = nir_deref_instr_parent(deref);
      return nir_iadd_imm(b, tu_rt_pun_offset(b, parent),
                          tu_rt_natural_field_offset(parent->type, deref->strct.index));
   }
   case nir_deref_type_array:
   case nir_deref_type_ptr_as_array: {
      nir_deref_instr *parent = nir_deref_instr_parent(deref);
      unsigned size, al;
      glsl_get_natural_size_align_bytes(deref->type, &size, &al);
      unsigned stride = glsl_type_is_vector(parent->type) ? size : align(size, al);
      nir_def *idx = nir_u2u32(b, deref->arr.index.ssa);
      return nir_iadd(b, tu_rt_pun_offset(b, parent), nir_imul_imm(b, idx, stride));
   }
   default:
      UNREACHABLE("unsupported deref in ray tracing pun");
   }
}

static nir_variable *
tu_rt_deref_root_var(nir_deref_instr *deref)
{
   while (deref->deref_type != nir_deref_type_var) {
      deref = nir_deref_instr_parent(deref);
      if (!deref)
         return NULL;
   }
   return deref->var;
}

static void
tu_rt_pun_mismatched_vars(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   struct hash_table *punned = _mesa_pointer_hash_table_create(NULL);

   nir_foreach_block (block, impl) {
      nir_foreach_instr (instr, block) {
         if (instr->type != nir_instr_type_deref)
            continue;
         nir_deref_instr *deref = nir_instr_as_deref(instr);
         if (deref->deref_type != nir_deref_type_cast ||
             !nir_deref_mode_may_be(deref, nir_var_function_temp))
            continue;
         nir_variable *var = tu_rt_deref_root_var(deref);
         if (!var || _mesa_hash_table_search(punned, var))
            continue;
         unsigned size, al;
         glsl_get_natural_size_align_bytes(var->type, &size, &al);
         nir_variable *words =
            nir_local_variable_create(impl, glsl_array_type(glsl_uint_type(),
                                                            MAX2(DIV_ROUND_UP(size, 4), 1), 4),
                                      "tu_rt_punned");
         _mesa_hash_table_insert(punned, var, words);
      }
   }

   if (!punned->entries) {
      _mesa_hash_table_destroy(punned, NULL);
      return;
   }

   mesa_logw("tu: ray tracing payload type mismatch, punning %u variable(s)",
             punned->entries);

   nir_builder b = nir_builder_create(impl);
   nir_foreach_block_safe (block, impl) {
      nir_foreach_instr_safe (instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_deref &&
             intr->intrinsic != nir_intrinsic_store_deref)
            continue;
         nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
         nir_variable *root = tu_rt_deref_root_var(deref);
         struct hash_entry *he = root ? _mesa_hash_table_search(punned, root) : NULL;
         if (!he)
            continue;
         nir_variable *words = (nir_variable *) he->data;

         b.cursor = nir_before_instr(instr);
         nir_def *off = tu_rt_pun_offset(&b, deref);
         if (intr->intrinsic == nir_intrinsic_load_deref) {
            nir_def *vals[NIR_MAX_VEC_COMPONENTS];
            for (unsigned c = 0; c < intr->def.num_components; c++)
               vals[c] = tu_rt_hw_load_comp(&b, words, off, c, intr->def.bit_size);
            nir_def_replace(&intr->def, nir_vec(&b, vals, intr->def.num_components));
         } else {
            nir_def *v = intr->src[1].ssa;
            unsigned mask = nir_intrinsic_write_mask(intr);
            for (unsigned c = 0; c < v->num_components; c++) {
               if (mask & (1u << c))
                  tu_rt_hw_store_comp(&b, words, off, c, nir_channel(&b, v, c));
            }
            nir_instr_remove(instr);
         }
      }
   }

   nir_progress(true, impl, nir_metadata_none);
   _mesa_hash_table_destroy(punned, NULL);
   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_remove_dead_derefs);
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_function_temp, NULL);
}

static nir_shader *
tu_rt_build_shader(struct tu_device *dev,
                   struct tu_ray_tracing_pipeline *pipeline,
                   const VkRayTracingPipelineCreateInfoKHR *create_info,
                   VkPipelineCreateFlags2KHR flags, int only_raygen,
                   enum tu_rt_part_kind part = TU_RT_PART_NONE)
{
   nir_builder _b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, ir3_get_compiler_options(dev->compiler),
      "tu ray tracing pipeline");
   nir_builder *b = &_b;

   b->shader->info.workgroup_size[0] = 8;
   b->shader->info.workgroup_size[1] = 8;
   b->shader->info.workgroup_size[2] = 1;

   struct tu_rt_compiler c = {};
   c.pipeline = pipeline;
   c.flags = flags;
   c.only_raygen = only_raygen;
   c.part = part;
   c.fov_mode = tu_rt_fov_mode(dev);
   c.fn_ctx = _mesa_pointer_hash_table_create(NULL);
   c.fn_cache = (nir_function **)
      calloc(MAX2(pipeline->stage_count, 1) * 2 * (TU_RT_MAX_CALL_DEPTH + 1),
             sizeof(nir_function *));
   c.isec_cache = (nir_function **)
      calloc(MAX2(pipeline->group_count, 1) * (TU_RT_MAX_CALL_DEPTH + 1),
             sizeof(nir_function *));
   tu_rt_state_init(b->shader, &c.state);
   c.hit_words = tu_rt_hit_words_var(b->shader);

   for (uint32_t i = 0; i < pipeline->group_count; i++) {
      const struct tu_rt_group *g = pipeline->groups + i;
      if (g->ahit_index != VK_SHADER_UNUSED_KHR)
         c.has_ahit = true;
      if (g->isec_index != VK_SHADER_UNUSED_KHR)
         c.has_isec = true;
   }
   {
      uint64_t total = 0;
      const uint32_t keep_rg = only_raygen >= 0 ?
         pipeline->groups[only_raygen].recursive_index : VK_SHADER_UNUSED_KHR;
      for (uint32_t i = 0; i < pipeline->stage_count; i++)
         if (only_raygen < 0 || i == keep_rg ||
             pipeline->stages[i]->info.stage != MESA_SHADER_RAYGEN)
         nir_foreach_function_impl (impl, pipeline->stages[i])
            nir_foreach_block (block, impl)
               nir_foreach_instr (instr, block)
                  total++;
      static const uint64_t budget =
         debug_get_num_option("TU_RT_HIT_BUDGET", 250000);
      if (total > budget || debug_get_bool_option("TU_RT_LOG", false)) {
         uint64_t by_stage[32] = {};
         unsigned n_by_stage[32] = {};
         unsigned trace_sites = 0, call_sites = 0;
         for (uint32_t i = 0; i < pipeline->stage_count; i++) {
            nir_shader *st = pipeline->stages[i];
            uint64_t n = 0;
            nir_foreach_function_impl (impl, st)
               nir_foreach_block (block, impl)
                  nir_foreach_instr (instr, block) {
                     n++;
                     if (instr->type == nir_instr_type_intrinsic) {
                        nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
                        if (op == nir_intrinsic_trace_ray)
                           trace_sites++;
                        else if (op == nir_intrinsic_execute_callable)
                           call_sites++;
                     }
                  }
            by_stage[st->info.stage] += n;
            n_by_stage[st->info.stage]++;
         }
         unsigned g_tri = 0, g_proc = 0, g_ahit = 0, g_gen = 0;
         for (uint32_t i = 0; i < pipeline->group_count; i++) {
            const struct tu_rt_group *g = pipeline->groups + i;
            if (g->isec_index != VK_SHADER_UNUSED_KHR)
               g_proc++;
            else if (g->ahit_index != VK_SHADER_UNUSED_KHR ||
                     (g->recursive_index != VK_SHADER_UNUSED_KHR &&
                      pipeline->stages[g->recursive_index]->info.stage == MESA_SHADER_CLOSEST_HIT))
               g_tri++;
            else
               g_gen++;
            if (g->ahit_index != VK_SHADER_UNUSED_KHR)
               g_ahit++;
         }
         mesa_logi("tu-rt: shape groups gen=%u tri=%u (ahit=%u) proc=%u | trace_sites=%u call_sites=%u",
                   g_gen, g_tri, g_ahit, g_proc, trace_sites, call_sites);
         for (unsigned st = 0; st < 32; st++)
            if (n_by_stage[st])
               mesa_logi("tu-rt: shape   %s x%u = %" PRIu64 " instrs",
                         _mesa_shader_stage_to_abbrev((mesa_shader_stage) st),
                         n_by_stage[st], by_stage[st]);
      }
      if (total > budget) {
         c.nohit = true;
         c.has_ahit = false;
         c.has_isec = false;
         mesa_logw("tu-rt: %" PRIu64 " stage instrs over budget %" PRIu64
                   ": building without hit shaders (%u stages, %u groups)",
                   total, budget, pipeline->stage_count, pipeline->group_count);
      }
      if (debug_get_bool_option("TU_RT_LOG", false)) {
         for (uint32_t i = 0; i < create_info->stageCount && i < 4; i++)
            mesa_logi("tu-rt:   stage %u %s %s", i,
                      _mesa_shader_stage_to_abbrev(
                         vk_to_mesa_shader_stage(create_info->pStages[i].stage)),
                      create_info->pStages[i].pName);
      }
   }
   for (uint32_t i = 0; i < pipeline->stage_count; i++)
      if (only_raygen >= 0 && pipeline->stages[i]->info.stage == MESA_SHADER_RAYGEN &&
          i != pipeline->groups[only_raygen].recursive_index)
         continue;
      else if (!c.nohit || (pipeline->stages[i]->info.stage != MESA_SHADER_CLOSEST_HIT &&
                       pipeline->stages[i]->info.stage != MESA_SHADER_ANY_HIT &&
                       pipeline->stages[i]->info.stage != MESA_SHADER_INTERSECTION))
         c.used |= tu_rt_scan_used(pipeline->stages[i]);

   {
      uint32_t ncall = 0;
      for (uint32_t i = 0; i < pipeline->group_count; i++) {
         nir_shader *s = tu_rt_group_stage(&c, pipeline->groups[i].recursive_index);
         if (s && s->info.stage == MESA_SHADER_CALLABLE &&
             !tu_rt_handle_seen(&c, i, pipeline->groups[i].handle, 0))
            ncall++;
      }
      uint64_t bodies = 0, level = 1;
      c.max_call_depth = 0;
      while (c.max_call_depth < TU_RT_MAX_CALL_DEPTH) {
         level *= MAX2(ncall, 1);
         if (c.max_call_depth >= 2 && bodies + level > 64)
            break;
         bodies += level;
         c.max_call_depth++;
      }
   }

   nir_def *launch_id = nir_load_ray_launch_id(b);
   nir_def *launch_size = nir_load_ray_launch_size(b);
   nir_def *oob = nir_ige(b, nir_channel(b, launch_id, 0), nir_channel(b, launch_size, 0));
   oob = nir_ior(b, oob, nir_ige(b, nir_channel(b, launch_id, 1), nir_channel(b, launch_size, 1)));
   oob = nir_ior(b, oob, nir_ige(b, nir_channel(b, launch_id, 2), nir_channel(b, launch_size, 2)));
   nir_push_if(b, oob);
   nir_jump(b, nir_jump_return);
   nir_pop_if(b, NULL);

   if (c.fov_mode & TU_RT_FOV_SHARE) {
      c.fov_sel = nir_variable_create(b->shader, nir_var_shader_temp, glsl_uint_type(), "fov_sel");
      nir_store_var(b, c.fov_sel, tu_rt_fov_select(b, tu_rt_fov_level(b, &c)), 0x1);
   }

   struct tu_rt_sbt_entry raygen = tu_rt_load_sbt_entry(
      b, NULL, offsetof(VkTraceRaysIndirectCommand2KHR, raygenShaderRecordAddress));
   nir_store_var(b, c.state.shader_record_ptr, raygen.shader_record_ptr, 0x1);
   tu_rt_dispatch_recursive(b, &c, raygen.value, MESA_SHADER_RAYGEN,
                            nir_undef(b, 1, 32), 0, 0);

   nir_shader_instructions_pass(b->shader, tu_rt_lower_instr,
                                nir_metadata_none, &c);

   tu_rt_prof_mark("pre-inline", b->shader);
   {
      uint64_t instrs = 0;
      nir_foreach_function_impl (impl, b->shader)
         nir_foreach_block (block, impl)
            nir_foreach_instr (instr, block)
               instrs++;
      static const uint64_t max_instrs =
         debug_get_num_option("TU_RT_MAX_PREINLINE", 400000);
      if (instrs > max_instrs) {
         mesa_loge("tu-rt: pipeline too large to inline (%" PRIu64 " > %" PRIu64
                   " instrs, %u stages, %u groups)", instrs, max_instrs,
                   pipeline->stage_count, pipeline->group_count);
         ralloc_free(c.fn_ctx);
         free(c.fn_cache);
         free(c.isec_cache);
         ralloc_free(b->shader);
         return NULL;
      }
   }
   NIR_PASS(_, b->shader, nir_lower_continue_constructs);
   NIR_PASS(_, b->shader, nir_lower_returns);
   NIR_PASS(_, b->shader, nir_inline_functions);
   tu_rt_prof_mark("post-inline", b->shader);
   nir_remove_non_entrypoints(b->shader);
   NIR_PASS(_, b->shader, nir_opt_deref);
   NIR_PASS(_, b->shader, nir_opt_copy_prop);
   NIR_PASS(_, b->shader, nir_opt_deref);

   NIR_PASS(_, b->shader, nir_split_var_copies);
   NIR_PASS(_, b->shader, nir_lower_var_copies);
   tu_rt_pun_mismatched_vars(b->shader);

   NIR_PASS(_, b->shader, nir_lower_global_vars_to_local);
   NIR_PASS(_, b->shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b->shader, nir_opt_copy_prop);
   NIR_PASS(_, b->shader, nir_opt_dce);

   tu_rt_prof_mark("pre-rq", b->shader);
   NIR_PASS(_, b->shader, nir_opt_ray_queries);
   NIR_PASS(_, b->shader, tu_nir_lower_ray_queries);
   NIR_PASS(_, b->shader, nir_split_var_copies);
   NIR_PASS(_, b->shader, nir_lower_var_copies);
   NIR_PASS(_, b->shader, nir_lower_system_values);

   ralloc_free(c.fn_ctx);
   free(c.fn_cache);
   free(c.isec_cache);

   nir_shader_gather_info(b->shader, nir_shader_get_entrypoint(b->shader));

   tu_rt_prof_mark("pre-optloop", b->shader);
   struct ir3_optimize_options optimize_options = {};
   ir3_optimize_loop(dev->compiler, &optimize_options, b->shader);
   tu_rt_prof_mark("post-optloop", b->shader);

   return b->shader;
}

enum {
   TU_RT_LINK_OFF = 0,
   TU_RT_LINK_WHOLE,
   TU_RT_LINK_FORCE,
   TU_RT_LINK_AUTO,
};

static unsigned
tu_rt_link_mode(void)
{
   static int mode = -1;
   if (mode < 0) {
      const char *v = os_get_option("TU_RT_LINK");
      int m = TU_RT_LINK_OFF;
      if (v && !strcmp(v, "whole"))
         m = TU_RT_LINK_WHOLE;
      else if (v && !strcmp(v, "force"))
         m = TU_RT_LINK_FORCE;
      else if (v && !strcmp(v, "auto"))
         m = TU_RT_LINK_AUTO;
      mode = m;
   }
   return mode;
}

static bool
tu_rt_log_enabled(void)
{
   static int on = -1;
   if (on < 0)
      on = debug_get_bool_option("TU_RT_LOG", false);
   return on;
}

static const char tu_rt_abi_tag[] = "tu-rt-abi-10";

static void
tu_rt_hash_abi(struct mesa_blake3 *h)
{
   _mesa_blake3_update(h, tu_rt_abi_tag, sizeof(tu_rt_abi_tag));
   const uint32_t abi[] = {
      TU_RT_FRAME_PAYLOAD, TU_RT_PAYLOAD_MAX, TU_RT_FRAME_STACK,
      TU_RT_STACK_MAX, TU_RT_FRAME_CALLEE, TU_RT_FRAME_WHOLE,
      (uint32_t) tu_rt_use_case_tree(),
      (uint32_t) debug_get_bool_option("TU_RT_LINK_PERWAVE", false),
   };
   _mesa_blake3_update(h, abi, sizeof(abi));
}

struct tu_rt_part_bin;

struct tu_rt_link_merge {
   const struct tu_rt_part_bin **parts;
   uint32_t nparts;
   const char *fail;
   bool final;
   uint32_t const_offset[IR3_CONST_ALLOC_MAX];
   uint32_t const_size[IR3_CONST_ALLOC_MAX];
};

static bool
tu_rt_merge_const(struct tu_rt_link_merge *m, unsigned k, uint32_t off, uint32_t size)
{
   if (!size)
      return true;
   if (!m->const_size[k] && !m->final) {
      m->const_offset[k] = off;
      m->const_size[k] = size;
      return true;
   }
   if (off != m->const_offset[k])
      return false;
   if (k != IR3_CONST_ALLOC_DRIVER_PARAMS)
      return size == m->const_size[k];
   if (size <= m->const_size[k])
      return true;
   if (m->final)
      return false;
   m->const_size[k] = size;
   return true;
}

static const char *
tu_rt_link_merge_fail(const struct tu_rt_link_merge *m)
{
   return m->fail;
}

struct tu_rt_part_bin {
   uint32_t count;
   int32_t max_reg;
   int32_t max_half_reg;
   uint32_t branchstack;
   uint32_t per_wave;
   uint32_t bindless;
   uint32_t pvtmem_size;
   uint32_t constlen;
   uint32_t num_samp;
   uint32_t num_uavs;
   uint32_t const_offset[IR3_CONST_ALLOC_MAX];
   uint32_t const_size[IR3_CONST_ALLOC_MAX];
   uint32_t instrs_count;
   uint32_t nops_count;
   uint32_t stp_count;
   uint32_t ldp_count;
   uint32_t uses_rt;
   const uint64_t *instrs;
};

#define TU_RT_PART_BIN_HDR offsetof(struct tu_rt_part_bin, instrs)

struct tu_rt_partset {
   struct vk_pipeline_cache_object base;
   uint32_t kind;
   uint32_t nparts;
   uint32_t nclasses;
   uint32_t classes[TU_RT_MAX_CLASSES];
   uint32_t payload_size;
   uint32_t has_payload;
   uint32_t used;
   uint32_t scratch;
   uint32_t active_desc_sets;
   uint32_t nwords;
   struct tu_rt_part_bin *parts;
   uint64_t *words;
};

static bool tu_rt_partset_serialize(struct vk_pipeline_cache_object *object,
                                    struct blob *blob);
static struct vk_pipeline_cache_object *
tu_rt_partset_deserialize(struct vk_pipeline_cache *cache, const void *key_data,
                          size_t key_size, struct blob_reader *blob);

static void
tu_rt_partset_destroy(struct vk_device *vk,
                      struct vk_pipeline_cache_object *object)
{
   struct tu_rt_partset *ps = container_of(object, struct tu_rt_partset, base);
   vk_pipeline_cache_object_finish(&ps->base);
   vk_free(&vk->alloc, ps);
}

static const struct vk_pipeline_cache_object_ops tu_rt_partset_ops = {
   .serialize = tu_rt_partset_serialize,
   .deserialize = tu_rt_partset_deserialize,
   .destroy = tu_rt_partset_destroy,
};

static struct tu_rt_partset *
tu_rt_partset_alloc(struct vk_device *vk, const void *key, size_t key_size,
                    uint32_t nparts, uint32_t nwords)
{
   size_t size = sizeof(struct tu_rt_partset) +
                 MAX2(nparts, 1) * sizeof(struct tu_rt_part_bin) +
                 (size_t) nwords * sizeof(uint64_t) + key_size;
   char *mem = (char *) vk_zalloc(&vk->alloc, size, 8,
                                  VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!mem)
      return NULL;
   struct tu_rt_partset *ps = (struct tu_rt_partset *) mem;
   char *p = mem + sizeof(*ps);
   ps->parts = (struct tu_rt_part_bin *) p;
   p += MAX2(nparts, 1) * sizeof(struct tu_rt_part_bin);
   ps->words = (uint64_t *) p;
   p += (size_t) nwords * sizeof(uint64_t);
   memcpy(p, key, key_size);
   vk_pipeline_cache_object_init(vk, &ps->base, &tu_rt_partset_ops, p, key_size);
   ps->nparts = nparts;
   ps->nwords = nwords;
   return ps;
}

static void
tu_rt_partset_fix_ptrs(struct tu_rt_partset *ps)
{
   uint32_t off = 0;
   for (uint32_t i = 0; i < ps->nparts; i++) {
      ps->parts[i].instrs = ps->words + off;
      off += ps->parts[i].count;
   }
}

static bool
tu_rt_partset_serialize(struct vk_pipeline_cache_object *object,
                        struct blob *blob)
{
   struct tu_rt_partset *ps = container_of(object, struct tu_rt_partset, base);
   blob_write_uint32(blob, ps->kind);
   blob_write_uint32(blob, ps->nparts);
   blob_write_uint32(blob, ps->nwords);
   blob_write_uint32(blob, ps->nclasses);
   blob_write_bytes(blob, ps->classes, sizeof(ps->classes));
   blob_write_uint32(blob, ps->payload_size);
   blob_write_uint32(blob, ps->has_payload);
   blob_write_uint32(blob, ps->used);
   blob_write_uint32(blob, ps->scratch);
   blob_write_uint32(blob, ps->active_desc_sets);
   for (uint32_t i = 0; i < ps->nparts; i++)
      blob_write_bytes(blob, &ps->parts[i], TU_RT_PART_BIN_HDR);
   blob_write_bytes(blob, ps->words, (size_t) ps->nwords * sizeof(uint64_t));
   return true;
}

static struct vk_pipeline_cache_object *
tu_rt_partset_deserialize(struct vk_pipeline_cache *cache, const void *key_data,
                          size_t key_size, struct blob_reader *blob)
{
   uint32_t kind = blob_read_uint32(blob);
   uint32_t nparts = blob_read_uint32(blob);
   uint32_t nwords = blob_read_uint32(blob);
   if (blob->overrun || nparts > 4096 ||
       (uint64_t) nwords * sizeof(uint64_t) > (uint64_t) (blob->end - blob->current))
      return NULL;
   struct tu_rt_partset *ps =
      tu_rt_partset_alloc(cache->base.device, key_data, key_size, nparts, nwords);
   if (!ps)
      return NULL;
   ps->kind = kind;
   ps->nclasses = blob_read_uint32(blob);
   blob_copy_bytes(blob, ps->classes, sizeof(ps->classes));
   ps->payload_size = blob_read_uint32(blob);
   ps->has_payload = blob_read_uint32(blob);
   ps->used = blob_read_uint32(blob);
   ps->scratch = blob_read_uint32(blob);
   ps->active_desc_sets = blob_read_uint32(blob);
   uint64_t total = 0;
   for (uint32_t i = 0; i < nparts; i++) {
      blob_copy_bytes(blob, &ps->parts[i], TU_RT_PART_BIN_HDR);
      total += ps->parts[i].count;
   }
   blob_copy_bytes(blob, ps->words, (size_t) nwords * sizeof(uint64_t));
   if (blob->overrun || total != nwords) {
      tu_rt_partset_destroy(cache->base.device, &ps->base);
      return NULL;
   }
   tu_rt_partset_fix_ptrs(ps);
   return &ps->base;
}

static simple_mtx_t tu_rt_cache_mtx = SIMPLE_MTX_INITIALIZER;

static struct tu_rt_partset *
tu_rt_partset_lookup(struct tu_device *dev, struct vk_pipeline_cache *cache,
                     const unsigned char *key)
{
   simple_mtx_lock(&tu_rt_cache_mtx);
   bool hit = false;
   struct vk_pipeline_cache_object *obj =
      vk_pipeline_cache_lookup_object(cache, key, BLAKE3_KEY_LEN,
                                      &tu_rt_partset_ops, &hit);
   if (!obj && cache != dev->mem_cache)
      obj = vk_pipeline_cache_lookup_object(dev->mem_cache, key, BLAKE3_KEY_LEN,
                                            &tu_rt_partset_ops, &hit);
   if (debug_get_bool_option("TU_RT_LINK_KEYDBG", false)) {
      char hk[17];
      tu_keylog_hex(hk, key, 8);
      mesa_logi("tu-rt-keydbg: lookup %s cache=%p mem=%p -> %p", hk, (void *) cache,
                (void *) dev->mem_cache, (void *) obj);
   }
   simple_mtx_unlock(&tu_rt_cache_mtx);
   return obj ? container_of(obj, struct tu_rt_partset, base) : NULL;
}

static struct tu_rt_partset *
tu_rt_partset_insert(struct tu_device *dev, struct vk_pipeline_cache *cache,
                     struct tu_rt_partset *ps)
{
   simple_mtx_lock(&tu_rt_cache_mtx);
   struct vk_pipeline_cache_object *obj =
      vk_pipeline_cache_add_object(cache, &ps->base);
   if (cache != dev->mem_cache)
      vk_pipeline_cache_object_unref(
         &dev->vk, vk_pipeline_cache_add_object(dev->mem_cache,
                                                vk_pipeline_cache_object_ref(obj)));
   simple_mtx_unlock(&tu_rt_cache_mtx);
   return container_of(obj, struct tu_rt_partset, base);
}

static void
tu_rt_partset_unref(struct tu_device *dev, struct tu_rt_partset *ps)
{
   if (ps)
      vk_pipeline_cache_object_unref(&dev->vk, &ps->base);
}

static bool
tu_rt_lower_part_abi_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *def;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_ray_launch_id:
      def = tu_rt_fload_vec(b, TU_RT_FRAME_LAUNCH_ID, 3);
      break;
   case nir_intrinsic_load_rt_cmd_addr_ir3:
      def = tu_rt_fload_vec(b, TU_RT_FRAME_CMD, 2);
      break;
   default:
      return false;
   }
   nir_def_replace(&intr->def, def);
   return true;
}

static const char *
tu_rt_part_check(const struct tu_shader *sh)
{
   const struct ir3_shader_variant *v = sh->variant;
   const struct ir3_const_state *cs = ir3_const_state(v);
   if (v->inputs_count)
      return "sysval inputs";
   if (v->constant_data_size)
      return "constant data";
   if (cs->consts_ubo.idx >= 0 || cs->driver_params_ubo.idx >= 0 ||
       cs->primitive_map_ubo.idx >= 0 || cs->primitive_param_ubo.idx >= 0)
      return "driver ubo";
   if (cs->ubo_state.num_enabled)
      return "ubo ranges";
   if (v->imm_state.count)
      return "imm consts";
   if (sh->const_state.dynamic_offsets_ubo.idx >= 0 ||
       sh->const_state.inline_uniforms_ubo.idx >= 0 ||
       sh->const_state.bindless_base_addrs_ubo.idx >= 0 ||
       sh->const_state.fdm_ubo.idx >= 0)
      return "tu driver ubo";
   if (v->info.double_threadsize)
      return "wave128";
   if (v->info.preamble_instrs_count || v->early_preamble)
      return "preamble";
   if (v->shared_size || v->has_barrier)
      return "shared";
   return NULL;
}

static void tu_rt_link_fixup(struct ir3_shader_variant *v, void *data);

static const char *
tu_rt_part_compile(struct tu_device *dev, struct tu_pipeline_layout *layout,
                   const struct tu_shader_key *key, nir_shader *nir,
                   unsigned scratch_base, struct tu_rt_part_bin *out,
                   uint64_t **words, uint32_t *desc_sets,
                   const struct ir3_rt_blobs *blobs = NULL,
                   struct tu_rt_link_merge *merge = NULL)
{
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, tu_rt_lower_part_abi_instr,
            nir_metadata_control_flow, NULL);
   if (merge) {
      for (uint32_t i = 0; i < merge->nparts; i++)
         scratch_base = MAX2(scratch_base, merge->parts[i]->pvtmem_size);
   }
   nir->scratch_size = MAX2(nir->scratch_size, scratch_base);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   struct ir3_shader_key ir3_key = {};
   struct tu_shader_info info = {};
   tu_lower_nir(dev, nir, key, &ir3_key, &info);

   static const char part_key[] = "tu-rt-part";
   struct tu_rt_shader_opts opts = {};
   opts.part = true;
   opts.blobs = blobs;
   if (merge) {
      opts.fixup = tu_rt_link_fixup;
      opts.fixup_data = merge;
   }
   struct tu_shader *sh = NULL;
   tu_shader_create(dev, &sh, nir, key, &info, &ir3_key, part_key,
                    sizeof(part_key), layout, false, &opts);
   if (!sh || !sh->variant) {
      if (sh)
         vk_pipeline_cache_object_unref(&dev->vk, &sh->base);
      return "compile";
   }

   const char *reason = tu_rt_part_check(sh);
   if (!reason && merge)
      reason = tu_rt_link_merge_fail(merge);
   if (!reason) {
      const struct ir3_shader_variant *v = sh->variant;
      const uint64_t *bin = (const uint64_t *) v->bin;
      uint32_t n = v->info.size / 8;
      while (n && !bin[n - 1])
         n--;
      *words = (uint64_t *) malloc(MAX2(n, 1) * sizeof(uint64_t));
      memcpy(*words, bin, n * sizeof(uint64_t));
      const struct ir3_const_state *cs = ir3_const_state(v);
      out->count = n;
      out->max_reg = v->info.max_reg;
      out->max_half_reg = v->info.max_half_reg;
      out->branchstack = v->branchstack;
      out->per_wave = v->pvtmem_per_wave;
      out->bindless = (v->bindless_tex ? 1 : 0) | (v->bindless_samp ? 2 : 0) |
                      (v->bindless_ibo ? 4 : 0) | (v->bindless_ubo ? 8 : 0) |
                      (v->has_ssbo ? 16 : 0);
      out->pvtmem_size = v->pvtmem_size;
      out->constlen = v->constlen;
      out->num_samp = MAX2(v->num_samp, 0);
      out->num_uavs = v->num_uavs;
      for (unsigned k = 0; k < IR3_CONST_ALLOC_MAX; k++) {
         out->const_offset[k] = merge ? merge->const_offset[k]
                                      : cs->allocs.consts[k].offset_vec4;
         out->const_size[k] = merge ? merge->const_size[k]
                                    : cs->allocs.consts[k].size_vec4;
      }
      out->instrs_count = v->info.instrs_count;
      out->nops_count = v->info.nops_count;
      out->stp_count = v->info.stp_count;
      out->ldp_count = v->info.ldp_count;
      out->uses_rt = v->info.uses_ray_intersection;
      *desc_sets |= sh->active_desc_sets;
   }
   vk_pipeline_cache_object_unref(&dev->vk, &sh->base);
   return reason;
}

static struct tu_rt_partset *
tu_rt_partset_from_bins(struct tu_device *dev, const unsigned char *pkey,
                        uint32_t kind, uint32_t nparts,
                        struct tu_rt_part_bin *bins, uint64_t **words)
{
   uint32_t nwords = 0;
   for (uint32_t i = 0; i < nparts; i++)
      nwords += bins[i].count;
   struct tu_rt_partset *ps =
      tu_rt_partset_alloc(&dev->vk, pkey, BLAKE3_KEY_LEN, nparts, nwords);
   if (!ps)
      return NULL;
   ps->kind = kind;
   uint32_t off = 0;
   for (uint32_t i = 0; i < nparts; i++) {
      ps->parts[i] = bins[i];
      memcpy(ps->words + off, words[i], bins[i].count * sizeof(uint64_t));
      off += bins[i].count;
   }
   tu_rt_partset_fix_ptrs(ps);
   return ps;
}

struct tu_rt_fake_pipeline {
   struct tu_ray_tracing_pipeline p;
   nir_shader *stages[1];
};

static void
tu_rt_fake_pipeline_init(struct tu_rt_fake_pipeline *f, nir_shader *stage)
{
   memset(f, 0, sizeof(*f));
   f->stages[0] = stage;
   f->p.stages = f->stages;
   f->p.stage_count = 1;
   f->p.group_count = 0;
}

static void
tu_rt_part_compiler_init(struct tu_rt_compiler *c, nir_shader *nir,
                         struct tu_ray_tracing_pipeline *pipeline,
                         VkPipelineCreateFlags2KHR flags,
                         enum tu_rt_part_kind kind)
{
   memset(c, 0, sizeof(*c));
   c->pipeline = pipeline;
   c->flags = flags;
   c->part = kind;
   c->only_raygen = -1;
   c->fn_ctx = _mesa_pointer_hash_table_create(NULL);
   c->fn_cache = (nir_function **)
      calloc(MAX2(pipeline->stage_count, 1) * 2 * (TU_RT_MAX_CALL_DEPTH + 1),
             sizeof(nir_function *));
   c->isec_cache = (nir_function **)
      calloc(MAX2(pipeline->group_count, 1) * (TU_RT_MAX_CALL_DEPTH + 1),
             sizeof(nir_function *));
   tu_rt_state_init(nir, &c->state);
   c->hit_words = tu_rt_hit_words_var(nir);
}

static void
tu_rt_part_compiler_fini(struct tu_rt_compiler *c)
{
   ralloc_free(c->fn_ctx);
   free(c->fn_cache);
   free(c->isec_cache);
}

static nir_builder
tu_rt_part_builder(struct tu_device *dev, const char *name,
                   mesa_shader_stage stage = MESA_SHADER_COMPUTE)
{
   nir_builder b = nir_builder_init_simple_shader(
      stage, ir3_get_compiler_options(dev->compiler), "%s", name);
   b.shader->info.workgroup_size[0] = 8;
   b.shader->info.workgroup_size[1] = 8;
   b.shader->info.workgroup_size[2] = 1;
   return b;
}

static void
tu_rt_part_lower_common(nir_shader *nir, struct tu_rt_compiler *c)
{
   nir_shader_instructions_pass(nir, tu_rt_lower_instr, nir_metadata_none, c);
   NIR_PASS(_, nir, nir_lower_continue_constructs);
   NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_inline_functions);
   nir_remove_non_entrypoints(nir);
   NIR_PASS(_, nir, nir_opt_deref);
   NIR_PASS(_, nir, nir_opt_copy_prop);
   NIR_PASS(_, nir, nir_opt_deref);
   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_lower_var_copies);
   tu_rt_pun_mismatched_vars(nir);
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_opt_copy_prop);
   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_opt_ray_queries);
   NIR_PASS(_, nir, tu_nir_lower_ray_queries);
   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_lower_system_values);
}

static uint64_t
tu_rt_count_instrs(nir_shader *nir)
{
   uint64_t n = 0;
   nir_foreach_function_impl (impl, nir)
      nir_foreach_block (block, impl)
         nir_foreach_instr (instr, block)
            n++;
   return n;
}

static unsigned
tu_rt_scratch_end(nir_shader *nir)
{
   unsigned end = 0;
   nir_foreach_function_impl (impl, nir) {
      nir_foreach_block (block, impl) {
         nir_foreach_instr (instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_rt_trace_ray:
            case nir_intrinsic_rt_execute_callable:
            case nir_intrinsic_rt_resume:
               end = MAX2(end, nir_intrinsic_stack_size(intr));
               break;
            case nir_intrinsic_load_scratch:
               if (nir_src_is_const(intr->src[0]))
                  end = MAX2(end, nir_src_as_uint(intr->src[0]) +
                                     intr->def.num_components * intr->def.bit_size / 8);
               break;
            case nir_intrinsic_store_scratch:
               if (nir_src_is_const(intr->src[1]))
                  end = MAX2(end, nir_src_as_uint(intr->src[1]) +
                                     intr->src[0].ssa->num_components *
                                        intr->src[0].ssa->bit_size / 8);
               break;
            default:
               break;
            }
         }
      }
   }
   return end;
}

static bool
tu_rt_cmd_derived(nir_def *def, unsigned depth)
{
   if (!depth)
      return false;
   nir_instr *instr = nir_def_instr(def);
   switch (instr->type) {
   case nir_instr_type_load_const:
      return true;
   case nir_instr_type_alu: {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
         if (!tu_rt_cmd_derived(alu->src[i].src.ssa, depth - 1))
            return false;
      }
      return true;
   }
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic == nir_intrinsic_load_rt_cmd_addr_ir3)
         return true;
      if (intr->intrinsic == nir_intrinsic_load_global &&
          (nir_intrinsic_access(intr) & ACCESS_NON_WRITEABLE))
         return tu_rt_cmd_derived(intr->src[0].ssa, depth - 1);
      return false;
   }
   default:
      return false;
   }
}

static bool
tu_rt_should_remat(nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_load_rt_cmd_addr_ir3:
      return true;
   case nir_intrinsic_load_global:
   case nir_intrinsic_load_global_constant:
      return (nir_intrinsic_access(intr) & ACCESS_CAN_REORDER) &&
             (nir_intrinsic_access(intr) & ACCESS_NON_WRITEABLE) &&
             tu_rt_cmd_derived(intr->src[0].ssa, 8);
   default:
      return false;
   }
}

static bool
tu_rt_lower_rt_call(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const char **fail = (const char **) data;
   if (intr->intrinsic == nir_intrinsic_rt_resume) {
      nir_instr_remove(&intr->instr);
      return true;
   }
   if (intr->intrinsic == nir_intrinsic_rt_execute_callable) {
      *fail = "callable";
      return false;
   }
   if (intr->intrinsic != nir_intrinsic_rt_trace_ray)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *accel = intr->src[0].ssa;
   if (nir_src_is_intrinsic(intr->src[0]) &&
       nir_def_as_intrinsic(accel)->intrinsic == nir_intrinsic_load_vulkan_descriptor) {
      unsigned off = offsetof(struct tu_accel_struct_header, self_ptr);
      accel = nir_load_uav_ir3(b, 2, 32, accel,
                               nir_vec2(b, nir_imm_int(b, 0), nir_imm_int(b, off / 4)),
                               .access = (gl_access_qualifier) (ACCESS_NON_WRITEABLE |
                                                                ACCESS_CAN_REORDER),
                               .align_mul = AS_RECORD_SIZE, .align_offset = off);
   } else if (accel->bit_size == 64 && accel->num_components == 1) {
      accel = nir_unpack_64_2x32(b, accel);
   } else {
      *fail = "accel struct form";
   }
   if (accel->num_components == 2)
      tu_rt_fstore_vec(b, TU_RT_FRAME_ACCEL, accel);
   tu_rt_fstore(b, TU_RT_FRAME_FLAGS, intr->src[1].ssa);
   tu_rt_fstore(b, TU_RT_FRAME_CULL, intr->src[2].ssa);
   tu_rt_fstore(b, TU_RT_FRAME_SBT_OFFSET, nir_iand_imm(b, intr->src[3].ssa, 0xF));
   tu_rt_fstore(b, TU_RT_FRAME_SBT_STRIDE, nir_iand_imm(b, intr->src[4].ssa, 0xF));
   tu_rt_fstore(b, TU_RT_FRAME_MISS_INDEX, nir_iand_imm(b, intr->src[5].ssa, 0xFFFF));
   tu_rt_fstore_vec(b, TU_RT_FRAME_ORIGIN, intr->src[6].ssa);
   tu_rt_fstore(b, TU_RT_FRAME_TMIN, intr->src[7].ssa);
   tu_rt_fstore_vec(b, TU_RT_FRAME_DIR, intr->src[8].ssa);
   tu_rt_fstore(b, TU_RT_FRAME_TMAX, intr->src[9].ssa);
   tu_rt_fstore(b, TU_RT_FRAME_RET,
                nir_imm_int(b, TU_RT_SEL_RG(nir_intrinsic_call_idx(intr) + 1)));
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
tu_rt_halt_to_return(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_jump)
      return false;
   nir_jump_instr *jump = nir_instr_as_jump(instr);
   if (jump->type != nir_jump_halt)
      return false;
   jump->type = nir_jump_return;
   return true;
}

static const char *
tu_rt_finish_raygen_part(nir_shader *nir)
{
   const char *fail = NULL;
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, tu_rt_lower_rt_call,
            nir_metadata_control_flow, &fail);
   NIR_PASS(_, nir, nir_shader_instructions_pass, tu_rt_halt_to_return,
            nir_metadata_none, NULL);
   NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_remove_dead_derefs);
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_shader_call_data, NULL);
   nir->info.stage = MESA_SHADER_COMPUTE;
   return fail;
}

struct tu_rt_part_env {
   struct tu_device *dev;
   struct tu_pipeline_layout *layout;
   const struct tu_shader_key *key;
   VkPipelineCreateFlags2KHR flags;
};

static const char *
tu_rt_build_raygen_parts(const struct tu_rt_part_env *env, nir_shader *stage,
                         const unsigned char *pkey, struct tu_rt_partset **out,
                         int64_t *probe)
{
   struct tu_device *dev = env->dev;
   struct tu_rt_fake_pipeline fake;
   tu_rt_fake_pipeline_init(&fake, stage);

   nir_builder _b = tu_rt_part_builder(dev, "tu rt raygen part", MESA_SHADER_CALLABLE);
   nir_builder *b = &_b;
   struct tu_rt_compiler c;
   tu_rt_part_compiler_init(&c, b->shader, &fake.p, env->flags, TU_RT_PART_RAYGEN);

   struct tu_rt_sbt_entry raygen = tu_rt_load_sbt_entry(
      b, NULL, offsetof(VkTraceRaysIndirectCommand2KHR, raygenShaderRecordAddress));
   nir_store_var(b, c.state.shader_record_ptr, raygen.shader_record_ptr, 0x1);
   tu_rt_call_stage(b, &c, stage, nir_undef(b, 1, 32), 0, 0);
   tu_rt_part_lower_common(b->shader, &c);
   const char *fail = c.part_fail;
   unsigned nclasses = c.part_nclasses;
   uint32_t classes[TU_RT_MAX_CLASSES];
   memcpy(classes, c.part_classes, sizeof(classes));
   tu_rt_part_compiler_fini(&c);
   if (fail) {
      ralloc_free(b->shader);
      return fail;
   }

   nir_shader *nir = b->shader;
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_opt_deref);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_remove_dead_derefs);
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_function_temp, NULL);
   if (tu_rt_log_enabled() && debug_get_bool_option("TU_RT_PROBE_VARS", false)) {
      nir_foreach_function_temp_variable (var, nir_shader_get_entrypoint(nir)) {
         unsigned size, al;
         glsl_get_natural_size_align_bytes(var->type, &size, &al);
         mesa_logi("tu-rt-probe:   local %s %s %u bytes", var->name ? var->name : "?",
                   glsl_get_type_name(var->type), size);
      }
   }
   nir->scratch_size = TU_RT_FRAME_STACK;
   NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, nir_var_function_temp,
            glsl_get_natural_size_align_bytes);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_function_temp,
            nir_address_format_32bit_offset);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   struct ir3_optimize_options optimize_options = {};
   ir3_optimize_loop(dev->compiler, &optimize_options, nir);
   unsigned locals = nir->scratch_size - TU_RT_FRAME_STACK;
   uint64_t pre = tu_rt_count_instrs(nir);

   static const char *nir_dir = os_get_option("TU_RT_PROBE_NIR_DIR");
   char nir_prefix[17] = "";
   if (nir_dir) {
      tu_keylog_hex(nir_prefix, pkey, 8);
      char path[512];
      snprintf(path, sizeof(path), "%s/rg_%s_pre.nir", nir_dir, nir_prefix);
      FILE *f = fopen(path, "w");
      if (f) {
         nir_print_shader(nir, f);
         fclose(f);
      }
   }

   unsigned scratch = nir->scratch_size;
   nir_lower_shader_calls_options opts = {};
   opts.address_format = nir_address_format_32bit_offset;
   opts.stack_alignment = 4;
   opts.localized_loads = true;
   opts.should_remat_callback = tu_rt_should_remat;
   void *mem_ctx = ralloc_context(NULL);
   nir_shader **resume = NULL;
   uint32_t nresume = 0;
   nir_lower_shader_calls(nir, &opts, &resume, &nresume, mem_ctx);

   uint32_t nparts = 1 + nresume;
   nir_shader **shaders = (nir_shader **) calloc(nparts, sizeof(nir_shader *));
   shaders[0] = nir;
   for (uint32_t i = 0; i < nresume; i++) {
      shaders[1 + i] = resume[i];
      ralloc_steal(NULL, resume[i]);
      resume[i]->info.name = ralloc_strdup(resume[i], resume[i]->info.name);
   }
   ralloc_free(mem_ctx);

   for (uint32_t i = 0; i < nparts; i++) {
      scratch = MAX2(scratch, shaders[i]->scratch_size);
      scratch = MAX2(scratch, tu_rt_scratch_end(shaders[i]));
   }
   scratch = align(scratch, 4);
   uint64_t post = 0;
   for (uint32_t i = 0; i < nparts && !fail; i++) {
      shaders[i]->scratch_size = scratch;
      fail = tu_rt_finish_raygen_part(shaders[i]);
      post += tu_rt_count_instrs(shaders[i]);
      if (nir_dir) {
         char path[512];
         snprintf(path, sizeof(path), "%s/rg_%s_part%u.nir", nir_dir, nir_prefix, i);
         FILE *f = fopen(path, "w");
         if (f) {
            nir_print_shader(shaders[i], f);
            fclose(f);
         }
      }
   }
   if (!fail && scratch > TU_RT_FRAME_CALLEE)
      fail = "stack too large";

   if (probe) {
      probe[0] = nparts;
      probe[1] = scratch > TU_RT_FRAME_STACK ? scratch - TU_RT_FRAME_STACK : 0;
      probe[2] = locals;
      probe[3] = pre;
      probe[4] = post;
      probe[5] = nclasses;
   }

   struct tu_rt_part_bin *bins = (struct tu_rt_part_bin *)
      calloc(nparts, sizeof(struct tu_rt_part_bin));
   uint64_t **words = (uint64_t **) calloc(nparts, sizeof(uint64_t *));
   uint32_t sets = 0;
   uint32_t i = 0;
   for (; i < nparts && !fail; i++) {
      nir_shader *s = shaders[i];
      shaders[i] = NULL;
      fail = tu_rt_part_compile(dev, env->layout, env->key, s, 0, &bins[i],
                                &words[i], &sets);
   }
   for (; i < nparts; i++)
      ralloc_free(shaders[i]);
   free(shaders);

   struct tu_rt_part_bin cbin = {};
   uint64_t *cwords = NULL;
   if (!fail) {
      struct ir3_rt_blob *blobs =
         (struct ir3_rt_blob *) calloc(nparts, sizeof(struct ir3_rt_blob));
      const struct tu_rt_part_bin **pbins =
         (const struct tu_rt_part_bin **) calloc(nparts, sizeof(void *));
      struct tu_rt_case *cases =
         (struct tu_rt_case *) calloc(nparts, sizeof(struct tu_rt_case));
      for (uint32_t j = 0; j < nparts; j++) {
         bins[j].instrs = words[j];
         blobs[j] = (struct ir3_rt_blob) { words[j], bins[j].count };
         pbins[j] = &bins[j];
         cases[j] = (struct tu_rt_case) { TU_RT_SEL_RG(j), j };
      }
      nir_builder _cb = tu_rt_part_builder(dev, "tu rt raygen dispatch");
      nir_builder *cb = &_cb;
      nir_def *sel = tu_rt_fload(cb, TU_RT_FRAME_SEL, 1);
      tu_rt_fstore(cb, TU_RT_FRAME_SEL, nir_imm_int(cb, TU_RT_SEL_DONE));
      tu_rt_emit_cases(cb, sel, cases, nparts, [&](const struct tu_rt_case &cs) {
         tu_rt_emit_blob(cb, cs.group);
      });
      nir_shader_gather_info(cb->shader, nir_shader_get_entrypoint(cb->shader));
      struct ir3_rt_blobs bl = { nparts, blobs };
      struct tu_rt_link_merge merge = { pbins, nparts, NULL };
      fail = tu_rt_part_compile(dev, env->layout, env->key, cb->shader, TU_RT_FRAME_CALLEE,
                                &cbin, &cwords, &sets, &bl, &merge);
      free(blobs);
      free(pbins);
      free(cases);
   }

   if (!fail) {
      *out = tu_rt_partset_from_bins(dev, pkey, TU_RT_PART_RAYGEN, 1, &cbin, &cwords);
      if (!*out) {
         fail = "oom";
      } else {
         (*out)->nclasses = nclasses;
         memcpy((*out)->classes, classes, sizeof(classes));
         (*out)->scratch = scratch;
         (*out)->active_desc_sets = sets;
         (*out)->payload_size = nparts;
      }
   }
   free(cwords);
   for (uint32_t j = 0; j < nparts; j++)
      free(words[j]);
   free(words);
   free(bins);
   return fail;
}

static const char *
tu_rt_build_callee_part(const struct tu_rt_part_env *env, nir_shader *stage,
                        const unsigned char *pkey, struct tu_rt_partset **out)
{
   struct tu_device *dev = env->dev;
   struct tu_rt_fake_pipeline fake;
   tu_rt_fake_pipeline_init(&fake, stage);

   nir_builder _b = tu_rt_part_builder(dev, "tu rt callee part");
   nir_builder *b = &_b;
   struct tu_rt_compiler c;
   tu_rt_part_compiler_init(&c, b->shader, &fake.p, env->flags, TU_RT_PART_CALLEE);
   struct tu_rt_state *s = &c.state;

   bool has = false;
   unsigned size = tu_rt_call_data_size(stage, &has);
   const char *fail = size > TU_RT_PAYLOAD_MAX ? "payload too large" : NULL;
   uint32_t used = tu_rt_scan_used(stage);

   nir_store_var(b, s->flags, tu_rt_fload(b, TU_RT_FRAME_FLAGS, 1), 0x1);
   nir_store_var(b, s->cull_mask, tu_rt_fload(b, TU_RT_FRAME_CULL, 1), 0x1);
   nir_store_var(b, s->origin, tu_rt_fload_vec(b, TU_RT_FRAME_ORIGIN, 3), 0x7);
   nir_store_var(b, s->tmin, tu_rt_fload(b, TU_RT_FRAME_TMIN, 1), 0x1);
   nir_store_var(b, s->dir, tu_rt_fload_vec(b, TU_RT_FRAME_DIR, 3), 0x7);
   nir_store_var(b, s->tmax, tu_rt_fload(b, TU_RT_FRAME_TMAX, 1), 0x1);
   nir_store_var(b, s->shader_record_ptr,
                 nir_pack_64_2x32(b, tu_rt_fload_vec(b, TU_RT_FRAME_RECORD, 2)), 0x1);
   if (stage->info.stage == MESA_SHADER_CLOSEST_HIT) {
      nir_store_var(b, s->hit_kind, tu_rt_fload(b, TU_RT_FRAME_HIT_KIND, 1), 0x1);
      if (used & TU_RT_USE_INSTANCE_ID)
         nir_store_var(b, s->instance_id, tu_rt_fload(b, TU_RT_FRAME_INSTANCE_ID, 1), 0x1);
      if (used & TU_RT_USE_CUSTOM_INDEX)
         nir_store_var(b, s->custom_index, tu_rt_fload(b, TU_RT_FRAME_CUSTOM_INDEX, 1), 0x1);
      if (used & TU_RT_USE_PRIMITIVE_ID)
         nir_store_var(b, s->primitive_id, tu_rt_fload(b, TU_RT_FRAME_PRIMITIVE_ID, 1), 0x1);
      if (used & TU_RT_USE_GEOMETRY_INDEX)
         nir_store_var(b, s->geometry_index, tu_rt_fload(b, TU_RT_FRAME_GEOMETRY_INDEX, 1), 0x1);
      if (used & TU_RT_USE_OBJECT_ORIGIN)
         nir_store_var(b, s->object_origin, tu_rt_fload_vec(b, TU_RT_FRAME_OBJECT_ORIGIN, 3), 0x7);
      if (used & TU_RT_USE_OBJECT_DIR)
         nir_store_var(b, s->object_dir, tu_rt_fload_vec(b, TU_RT_FRAME_OBJECT_DIR, 3), 0x7);
      for (unsigned col = 0; col < 4; col++) {
         if (used & TU_RT_USE_WTO)
            nir_store_var(b, s->wto[col], tu_rt_fload_vec(b, TU_RT_FRAME_WTO + 12 * col, 3), 0x7);
         if (used & TU_RT_USE_OTW)
            nir_store_var(b, s->otw[col], tu_rt_fload_vec(b, TU_RT_FRAME_OTW + 12 * col, 3), 0x7);
      }
      for (unsigned i = 0; i < TU_RT_HIT_WORDS; i++)
         nir_store_deref(b, tu_rt_hw(b, c.hit_words, nir_imm_int(b, i)),
                         tu_rt_fload(b, TU_RT_FRAME_HIT_WORDS + 4 * i, 1), 0x1);
   }

   nir_def *arg = NULL;
   nir_deref_instr *pd = NULL;
   if (has) {
      nir_variable *pvar = NULL;
      nir_foreach_variable_with_modes (var, stage, nir_var_shader_call_data) {
         pvar = nir_local_variable_create(b->impl, var->type, "tu_rt_payload");
         break;
      }
      pd = nir_build_deref_var(b, pvar);
      if (!tu_rt_copy_payload(b, pd, size, false))
         fail = "payload type";
      arg = &pd->def;
   } else {
      arg = nir_undef(b, 1, 32);
   }
   tu_rt_call_stage(b, &c, stage, arg, 1, 0);
   if (pd && !tu_rt_copy_payload(b, nir_build_deref_var(b, pd->var), size, true))
      fail = "payload type";
   tu_rt_fstore(b, TU_RT_FRAME_SEL, tu_rt_fload(b, TU_RT_FRAME_RET, 1));

   tu_rt_part_lower_common(b->shader, &c);
   if (!fail)
      fail = c.part_fail;
   tu_rt_part_compiler_fini(&c);
   if (fail) {
      ralloc_free(b->shader);
      return fail;
   }

   nir_shader *nir = b->shader;
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   struct ir3_optimize_options optimize_options = {};
   ir3_optimize_loop(dev->compiler, &optimize_options, nir);

   struct tu_rt_part_bin bin = {};
   uint64_t *words = NULL;
   uint32_t sets = 0;
   fail = tu_rt_part_compile(dev, env->layout, env->key, nir, TU_RT_FRAME_CALLEE,
                             &bin, &words, &sets);
   if (!fail) {
      *out = tu_rt_partset_from_bins(dev, pkey, TU_RT_PART_CALLEE, 1, &bin, &words);
      if (!*out) {
         fail = "oom";
      } else {
         (*out)->payload_size = size;
         (*out)->has_payload = has;
         (*out)->used = used;
         (*out)->active_desc_sets = sets;
      }
   }
   free(words);
   return fail;
}

struct tu_rt_trav_desc {
   unsigned cls;
   uint32_t used;
   uint32_t used_chit;
   bool has_ahit;
   bool has_isec;
   bool has_chit;
   bool has_miss;
};

static const char *
tu_rt_build_traversal_part(const struct tu_rt_part_env *env,
                           struct tu_ray_tracing_pipeline *pipeline,
                           const struct tu_rt_trav_desc *d,
                           const unsigned char *pkey, struct tu_rt_partset **out)
{
   struct tu_device *dev = env->dev;
   nir_builder _b = tu_rt_part_builder(dev, "tu rt traversal part");
   nir_builder *b = &_b;
   struct tu_rt_compiler c;
   tu_rt_part_compiler_init(&c, b->shader, pipeline, env->flags, TU_RT_PART_TRAVERSAL);
   c.part_payload_size = d->cls;
   c.part_hit_used = d->used_chit;
   c.part_has_chit = d->has_chit;
   c.part_has_miss = d->has_miss;
   c.used = d->used;
   c.has_ahit = d->has_ahit;
   c.has_isec = d->has_isec;

   const unsigned psize = tu_rt_class_size(d->cls);
   unsigned nwords = MAX2(DIV_ROUND_UP(psize, 4), 1);
   nir_variable *w =
      nir_local_variable_create(b->impl, glsl_array_type(glsl_uint_type(), nwords, 4),
                                "tu_rt_trav_payload");
   for (unsigned i = 0; i < DIV_ROUND_UP(psize, 4); i++)
      nir_store_deref(b, tu_rt_hw(b, w, nir_imm_int(b, i)),
                      tu_rt_fload(b, TU_RT_FRAME_PAYLOAD + 4 * i, 1), 0x1);

   struct tu_rt_ray r = {};
   r.accel = nir_pack_64_2x32(b, tu_rt_fload_vec(b, TU_RT_FRAME_ACCEL, 2));
   r.flags = tu_rt_fload(b, TU_RT_FRAME_FLAGS, 1);
   r.cull_mask = tu_rt_fload(b, TU_RT_FRAME_CULL, 1);
   r.sbt_offset = tu_rt_fload(b, TU_RT_FRAME_SBT_OFFSET, 1);
   r.sbt_stride = tu_rt_fload(b, TU_RT_FRAME_SBT_STRIDE, 1);
   r.miss_index = tu_rt_fload(b, TU_RT_FRAME_MISS_INDEX, 1);
   r.origin = tu_rt_fload_vec(b, TU_RT_FRAME_ORIGIN, 3);
   r.tmin = tu_rt_fload(b, TU_RT_FRAME_TMIN, 1);
   r.dir = tu_rt_fload_vec(b, TU_RT_FRAME_DIR, 3);
   r.tmax = tu_rt_fload(b, TU_RT_FRAME_TMAX, 1);
   r.payload = &nir_build_deref_var(b, w)->def;
   nir_push_if(b, nir_ine_imm(b, r.accel, 0));
   tu_rt_trace_ray_emit(b, &c, &r);
   nir_push_else(b, NULL);
   tu_rt_traversal_miss(b, &c, nir_iand_imm(b, r.miss_index, 0xFFFF));
   nir_pop_if(b, NULL);

   for (unsigned i = 0; i < DIV_ROUND_UP(psize, 4); i++)
      tu_rt_fstore(b, TU_RT_FRAME_PAYLOAD + 4 * i,
                   nir_load_deref(b, tu_rt_hw(b, w, nir_imm_int(b, i))));

   tu_rt_part_lower_common(b->shader, &c);
   const char *fail = c.part_fail;
   tu_rt_part_compiler_fini(&c);
   if (fail) {
      ralloc_free(b->shader);
      return fail;
   }

   nir_shader *nir = b->shader;
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   struct ir3_optimize_options optimize_options = {};
   ir3_optimize_loop(dev->compiler, &optimize_options, nir);

   struct tu_rt_part_bin bin = {};
   uint64_t *words = NULL;
   uint32_t sets = 0;
   fail = tu_rt_part_compile(dev, env->layout, env->key, nir, TU_RT_FRAME_CALLEE,
                             &bin, &words, &sets);
   if (!fail) {
      *out = tu_rt_partset_from_bins(dev, pkey, TU_RT_PART_TRAVERSAL, 1, &bin, &words);
      if (!*out) {
         fail = "oom";
      } else {
         (*out)->payload_size = psize;
         (*out)->used = d->used;
         (*out)->active_desc_sets = sets;
      }
   }
   free(words);
   return fail;
}


static void
tu_rt_link_fixup(struct ir3_shader_variant *v, void *data)
{
   struct tu_rt_link_merge *m = (struct tu_rt_link_merge *) data;
   const struct ir3_compiler *compiler = v->compiler;
   const struct ir3_const_state *cs = ir3_const_state(v);

   if (v->rt_blob_live_fail) {
      m->fail = "live across blob";
      return;
   }
   if (v->info.double_threadsize) {
      m->fail = "host wave128";
      return;
   }

   for (unsigned k = 0; k < IR3_CONST_ALLOC_MAX; k++) {
      m->const_offset[k] = cs->allocs.consts[k].offset_vec4;
      m->const_size[k] = cs->allocs.consts[k].size_vec4;
   }

   int max_reg = v->info.max_reg, max_half_reg = v->info.max_half_reg;
   unsigned bs = 0, pvt = v->pvtmem_size, constlen = v->constlen;
   bool per_wave = v->pvtmem_per_wave;
   for (uint32_t i = 0; i < m->nparts; i++) {
      const struct tu_rt_part_bin *p = m->parts[i];
      for (unsigned k = 0; k < IR3_CONST_ALLOC_MAX; k++) {
         if (!tu_rt_merge_const(m, k, p->const_offset[k], p->const_size[k])) {
            m->fail = k == IR3_CONST_ALLOC_PUSH_CONSTS ? "push const layout"
                                                       : "const layout";
            return;
         }
      }
      max_reg = MAX2(max_reg, p->max_reg);
      max_half_reg = MAX2(max_half_reg, p->max_half_reg);
      bs = MAX2(bs, p->branchstack);
      pvt = MAX2(pvt, p->pvtmem_size);
      constlen = MAX2(constlen, p->constlen);
      per_wave = per_wave && p->per_wave;
      v->bindless_tex |= !!(p->bindless & 1);
      v->bindless_samp |= !!(p->bindless & 2);
      v->bindless_ibo |= !!(p->bindless & 4);
      v->bindless_ubo |= !!(p->bindless & 8);
      v->has_ssbo |= !!(p->bindless & 16);
      v->num_samp = MAX2(v->num_samp, (int) p->num_samp);
      v->num_uavs = MAX2(v->num_uavs, p->num_uavs);
      v->info.stp_count += p->stp_count;
      v->info.ldp_count += p->ldp_count;
      v->info.uses_ray_intersection |= !!p->uses_rt;
   }
   v->info.max_reg = max_reg;
   v->info.max_half_reg = max_half_reg;
   v->branchstack += bs;
   v->pvtmem_size = align(pvt, compiler->pvtmem_per_fiber_align);
   v->pvtmem_per_wave = per_wave;
   v->constlen = constlen;
   v->info.constlen = constlen;

   ir3_update_max_waves(v);
}


struct tu_rt_glue_class {
   uint32_t cls;
   uint32_t trav_blob;
   uint32_t disp_blob;
};

static nir_shader *
tu_rt_build_glue(struct tu_device *dev, bool whole, const uint32_t *handles,
                 const uint32_t *first, uint32_t nraygens, uint32_t nrg,
                 struct tu_rt_glue_class *classes, uint32_t nclasses)
{
   nir_builder _b = tu_rt_part_builder(dev, "tu rt link");
   nir_builder *b = &_b;

   nir_def *launch_id = nir_load_global_invocation_id(b, 32);
   nir_def *launch_size =
      tu_rt_load_cmd(b, offsetof(VkTraceRaysIndirectCommand2KHR, width), 3);
   nir_def *oob = nir_ige(b, nir_channel(b, launch_id, 0), nir_channel(b, launch_size, 0));
   oob = nir_ior(b, oob, nir_ige(b, nir_channel(b, launch_id, 1), nir_channel(b, launch_size, 1)));
   oob = nir_ior(b, oob, nir_ige(b, nir_channel(b, launch_id, 2), nir_channel(b, launch_size, 2)));
   nir_push_if(b, nir_inot(b, oob));
   tu_rt_fstore_vec(b, TU_RT_FRAME_LAUNCH_ID, launch_id);
   tu_rt_fstore_vec(b, TU_RT_FRAME_CMD, nir_load_rt_cmd_addr_ir3(b));

   if (whole) {
      tu_rt_emit_blob(b, 0);
   } else {
      struct tu_rt_sbt_entry raygen = tu_rt_load_sbt_entry(
         b, NULL, offsetof(VkTraceRaysIndirectCommand2KHR, raygenShaderRecordAddress));
      tu_rt_fstore(b, TU_RT_FRAME_SEL, nir_imm_int(b, TU_RT_SEL_DONE));
      struct tu_rt_case *hcases =
         (struct tu_rt_case *) calloc(MAX2(nraygens, 1), sizeof(struct tu_rt_case));
      for (uint32_t i = 0; i < nraygens; i++)
         hcases[i] = (struct tu_rt_case) { handles[i], i };
      tu_rt_emit_cases(b, raygen.value, hcases, nraygens, [&](const struct tu_rt_case &cs) {
         tu_rt_fstore(b, TU_RT_FRAME_SEL, nir_imm_int(b, TU_RT_SEL_RG(0)));
         tu_rt_fstore(b, TU_RT_FRAME_RGIDX, nir_imm_int(b, cs.group));
      });
      free(hcases);

      struct tu_rt_case *rcases =
         (struct tu_rt_case *) calloc(MAX2(nraygens, 1), sizeof(struct tu_rt_case));
      for (uint32_t i = 0; i < nraygens; i++)
         rcases[i] = (struct tu_rt_case) { i, first[i] };
      struct tu_rt_case *cases =
         (struct tu_rt_case *) calloc(MAX2(2 * nclasses, 1), sizeof(struct tu_rt_case));
      uint32_t n = 0;
      for (uint32_t i = 0; i < nclasses; i++) {
         cases[n++] = (struct tu_rt_case) { TU_RT_SEL_TRAV(classes[i].cls), classes[i].trav_blob };
         cases[n++] = (struct tu_rt_case) { TU_RT_SEL_CHIT(classes[i].cls), classes[i].disp_blob };
      }

      nir_loop *loop = nir_push_loop(b);
      {
         nir_def *sel = tu_rt_fload(b, TU_RT_FRAME_SEL, 1);
         nir_break_if(b, nir_ieq_imm(b, sel, TU_RT_SEL_DONE));
         nir_def *kind = nir_ushr_imm(b, sel, 24);
         nir_push_if(b, nir_ieq_imm(b, kind, 0));
         {
            if (nraygens == 1) {
               tu_rt_emit_blob(b, first[0]);
            } else {
               nir_def *r = tu_rt_fload(b, TU_RT_FRAME_RGIDX, 1);
               tu_rt_emit_cases(b, r, rcases, nraygens, [&](const struct tu_rt_case &cs) {
                  tu_rt_emit_blob(b, cs.group);
               });
            }
         }
         nir_push_else(b, NULL);
         {
            sel = nir_bcsel(b, nir_ieq_imm(b, kind, 3),
                            nir_iand_imm(b, sel, 0x2ffffff), sel);
            tu_rt_emit_cases_or(b, sel, cases, n, [&](const struct tu_rt_case &cs) {
               tu_rt_emit_blob(b, cs.group);
            }, [&]() {
               tu_rt_fstore(b, TU_RT_FRAME_SEL, nir_imm_int(b, TU_RT_SEL_DONE));
            });
         }
         nir_pop_if(b, NULL);
      }
      nir_pop_loop(b, loop);
      free(cases);
      free(rcases);
   }
   nir_pop_if(b, NULL);

   nir_shader *nir = b->shader;
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   nir->scratch_size = TU_RT_FRAME_CALLEE;
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   struct ir3_optimize_options optimize_options = {};
   ir3_optimize_loop(dev->compiler, &optimize_options, nir);
   return nir;
}

static std::atomic<int64_t> tu_rt_link_lower_ns, tu_rt_link_create_ns;

static nir_shader *
tu_rt_build_dispatch(struct tu_device *dev, uint32_t cls,
                     struct tu_rt_case *chit, uint32_t nchit,
                     struct tu_rt_case *miss, uint32_t nmiss)
{
   nir_builder _b = tu_rt_part_builder(dev, "tu rt callee dispatch");
   nir_builder *b = &_b;
   nir_def *sel = tu_rt_fload(b, TU_RT_FRAME_SEL, 1);
   tu_rt_fstore(b, TU_RT_FRAME_SEL, tu_rt_fload(b, TU_RT_FRAME_RET, 1));
   nir_def *h = tu_rt_fload(b, TU_RT_FRAME_HANDLE, 1);
   nir_push_if(b, nir_ieq_imm(b, sel, TU_RT_SEL_CHIT(cls)));
   tu_rt_emit_cases(b, h, chit, nchit, [&](const struct tu_rt_case &cs) {
      tu_rt_emit_blob(b, cs.group);
   });
   nir_push_else(b, NULL);
   tu_rt_emit_cases(b, h, miss, nmiss, [&](const struct tu_rt_case &cs) {
      tu_rt_emit_blob(b, cs.group);
   });
   nir_pop_if(b, NULL);
   nir_shader *nir = b->shader;
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   struct ir3_optimize_options optimize_options = {};
   ir3_optimize_loop(dev->compiler, &optimize_options, nir);
   return nir;
}

static const char *
tu_rt_link_one(struct tu_device *dev, struct tu_pipeline_layout *layout,
               const struct tu_shader_key *key, nir_shader *glue,
               const struct ir3_rt_blobs *blobs,
               const struct tu_rt_part_bin **parts, uint32_t nparts,
               uint32_t desc_sets, const unsigned char *link_key,
               struct vk_pipeline_cache *cache, struct tu_shader **out)
{
   struct tu_rt_link_merge merge = { parts, nparts, NULL, true };
   struct tu_rt_shader_opts opts = {};
   opts.blobs = blobs;
   opts.fixup = tu_rt_link_fixup;
   opts.fixup_data = &merge;

   for (uint32_t i = 0; i < nparts; i++)
      glue->scratch_size = MAX2(glue->scratch_size, parts[i]->pvtmem_size);

   struct ir3_shader_key ir3_key = {};
   struct tu_shader_info info = {};
   int64_t t0 = os_time_get_nano();
   tu_lower_nir(dev, glue, key, &ir3_key, &info);
   int64_t t1 = os_time_get_nano();
   struct tu_shader *sh = NULL;
   tu_shader_create(dev, &sh, glue, key, &info, &ir3_key, link_key,
                    BLAKE3_KEY_LEN, layout, false, &opts);
   int64_t t2 = os_time_get_nano();
   tu_rt_link_lower_ns.fetch_add(t1 - t0);
   tu_rt_link_create_ns.fetch_add(t2 - t1);
   if (!sh)
      return "host compile";
   static const char *dump_dir = os_get_option("TU_RT_LINK_DISASM");
   if (dump_dir) {
      char path[512], hex[17];
      tu_keylog_hex(hex, link_key, 8);
      snprintf(path, sizeof(path), "%s/link_%s.txt", dump_dir, hex);
      FILE *f = fopen(path, "w");
      if (f) {
         fprintf(f, "; blobs=%u regs=%d half=%d bs=%u pvt=%u perwave=%d constlen=%u instrlen=%u\n",
                 blobs->count, sh->variant->info.max_reg + 1, sh->variant->info.max_half_reg + 1,
                 sh->variant->branchstack, sh->variant->pvtmem_size,
                 sh->variant->pvtmem_per_wave, sh->variant->constlen, sh->variant->instrlen);
         if (merge.fail)
            fprintf(f, "; FAIL %s\n", merge.fail);
         ir3_rt_disasm(sh->variant, f);
         fclose(f);
      }
   }
   if (merge.fail) {
      vk_pipeline_cache_object_unref(&dev->vk, &sh->base);
      return merge.fail;
   }
   sh->active_desc_sets |= desc_sets;
   simple_mtx_lock(&tu_rt_cache_mtx);
   *out = container_of(vk_pipeline_cache_add_object(cache, &sh->base),
                       struct tu_shader, base);
   simple_mtx_unlock(&tu_rt_cache_mtx);
   return NULL;
}

struct tu_rt_par_job {
   std::atomic<uint32_t> next;
   uint32_t count;
   void (*fn)(void *ctx, uint32_t i);
   void *ctx;
};

static void *
tu_rt_par_worker(void *data)
{
   struct tu_rt_par_job *job = (struct tu_rt_par_job *) data;
   for (;;) {
      uint32_t i = job->next.fetch_add(1);
      if (i >= job->count)
         break;
      job->fn(job->ctx, i);
   }
   return NULL;
}

static void
tu_rt_par_run(uint32_t count, void (*fn)(void *ctx, uint32_t i), void *ctx)
{
   if (!count)
      return;
   struct tu_rt_par_job job;
   job.next.store(0);
   job.count = count;
   job.fn = fn;
   job.ctx = ctx;
   static const unsigned max_jobs = debug_get_num_option("TU_RT_JOBS", 4);
   unsigned nthreads = MIN3(MAX2(max_jobs, 1u), count, 16u);
   pthread_t threads[16];
   unsigned started = 0;
   for (unsigned t = 1; t < nthreads; t++) {
      pthread_attr_t attr;
      pthread_attr_init(&attr);
      pthread_attr_setstacksize(&attr, 256u << 20);
      if (pthread_create(&threads[started], &attr, tu_rt_par_worker, &job) == 0)
         started++;
      pthread_attr_destroy(&attr);
   }
   tu_rt_par_worker(&job);
   for (unsigned t = 0; t < started; t++)
      pthread_join(threads[t], NULL);
}

static void
tu_rt_part_key(unsigned char *out, const char *tag, const unsigned char *stage_hash,
               const struct tu_shader_key *key, struct tu_pipeline_layout *layout)
{
   if (debug_get_bool_option("TU_RT_LINK_KEYDBG", false)) {
      char hs[17], hl[17], hk[17];
      blake3_hash kh;
      _mesa_blake3_compute(key, sizeof(*key), kh);
      tu_keylog_hex(hs, stage_hash, 8);
      tu_keylog_hex(hl, layout ? layout->blake3 : (const unsigned char *) "\0\0\0\0\0\0\0\0", 8);
      tu_keylog_hex(hk, kh, 8);
      mesa_logi("tu-rt-keydbg: %s stage=%s layout=%s key=%s", tag, hs, hl, hk);
   }
   struct mesa_blake3 h;
   _mesa_blake3_init(&h);
   _mesa_blake3_update(&h, tag, strlen(tag) + 1);
   tu_rt_hash_abi(&h);
   if (stage_hash)
      _mesa_blake3_update(&h, stage_hash, BLAKE3_KEY_LEN);
   _mesa_blake3_update(&h, key, sizeof(*key));
   enum ir3_shader_debug dbg = ir3_shader_debug_hash_key();
   _mesa_blake3_update(&h, &dbg, sizeof(dbg));
   if (layout)
      _mesa_blake3_update(&h, layout->blake3, sizeof(layout->blake3));
   _mesa_blake3_final(&h, out);
}

struct tu_rt_link_ctx {
   struct tu_rt_part_env env;
   struct vk_pipeline_cache *cache;
   struct tu_ray_tracing_pipeline *pipeline;
   const VkRayTracingPipelineCreateInfoKHR *create_info;
   const uint32_t *groups;
   const unsigned char *link_keys;
   bool whole;
   bool single;
   uint32_t nraygens;

   struct tu_rt_partset **rg_sets;
   uint32_t *rg_stage;

   uint32_t *callee_stages;
   uint32_t ncallee;
   struct tu_rt_partset **callee_sets;
   int32_t *stage_to_callee;

   uint32_t classes[64];
   uint32_t nclasses;
   struct tu_rt_trav_desc trav[64];
   struct tu_rt_partset *trav_sets[64];
   struct tu_rt_part_bin disp_bin[64];
   uint64_t *disp_words[64];
   uint32_t disp_sets[64];

   std::atomic<int> fail_count;
   const char *fail;
   simple_mtx_t lock;

   int64_t t_rg, t_callee, t_trav, t_link, t_link_max;
   std::atomic<int64_t> a_rg, a_callee, a_trav, a_link, a_link_max, a_glue;
   std::atomic<int64_t> a_link_tcpu, a_link_tcpu_max;
   std::atomic<uint32_t> n_rg_built, n_callee_built, n_trav_built;
   int64_t probe[6];
   std::atomic<int64_t> probe_acc[8];
};

static int64_t
tu_rt_thread_cpu_ns(void)
{
   struct timespec ts;
   if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts))
      return 0;
   return (int64_t) ts.tv_sec * 1000000000ll + ts.tv_nsec;
}

static void
tu_rt_note_tcpu(struct tu_rt_link_ctx *lc, int64_t dt)
{
   lc->a_link_tcpu.fetch_add(dt);
   int64_t prev = lc->a_link_tcpu_max.load();
   while (dt > prev && !lc->a_link_tcpu_max.compare_exchange_weak(prev, dt))
      ;
}

static void
tu_rt_link_note_fail(struct tu_rt_link_ctx *lc, const char *why)
{
   simple_mtx_lock(&lc->lock);
   if (!lc->fail)
      lc->fail = why;
   simple_mtx_unlock(&lc->lock);
   lc->fail_count.fetch_add(1);
}

static void
tu_rt_link_rg_job(void *data, uint32_t v)
{
   struct tu_rt_link_ctx *lc = (struct tu_rt_link_ctx *) data;
   if (lc->fail_count.load() || (!lc->single && lc->pipeline->variants[v].shader))
      return;
   uint32_t s = lc->rg_stage[v];
   unsigned char pkey[BLAKE3_KEY_LEN];
   tu_rt_part_key(pkey, "tu-rt-part-rg-1", lc->pipeline->stage_hashes + s * BLAKE3_KEY_LEN,
                  lc->env.key, lc->env.layout);
   struct tu_rt_partset *ps = tu_rt_partset_lookup(lc->env.dev, lc->cache, pkey);
   if (!ps) {
      int64_t t0 = os_time_get_nano();
      int64_t probe[6] = {};
      const char *why = tu_rt_build_raygen_parts(&lc->env, lc->pipeline->stages[s],
                                                 pkey, &ps, probe);
      int64_t dt = os_time_get_nano() - t0;
      lc->a_rg.fetch_add(dt);
      if (why) {
         tu_rt_link_note_fail(lc, why);
         return;
      }
      ps = tu_rt_partset_insert(lc->env.dev, lc->cache, ps);
      lc->n_rg_built.fetch_add(1);
      for (unsigned i = 0; i < 6; i++)
         lc->probe_acc[i].fetch_add(probe[i]);
      if (tu_rt_log_enabled()) {
         uint32_t instrs = 0, regs = 0, stp = 0, ldp = 0;
         for (uint32_t i = 0; i < ps->nparts; i++) {
            instrs += ps->parts[i].instrs_count;
            regs = MAX2(regs, (uint32_t) (ps->parts[i].max_reg + 1));
            stp += ps->parts[i].stp_count;
            ldp += ps->parts[i].ldp_count;
         }
         mesa_logi("tu-rt-probe: raygen handle=0x%x parts=%" PRId64 " stack=%" PRId64
                   " locals=%" PRId64 " nir=%" PRId64 "->%" PRId64 " classes=%" PRId64
                   " ir3=%u regs=%u stp=%u ldp=%u %.1f ms",
                   lc->pipeline->groups[lc->groups[v]].handle, probe[0], probe[1],
                   probe[2], probe[3], probe[4], probe[5], instrs, regs, stp, ldp,
                   dt / 1e6);
      }
   }
   lc->rg_sets[v] = ps;
}

static void
tu_rt_link_callee_job(void *data, uint32_t i)
{
   struct tu_rt_link_ctx *lc = (struct tu_rt_link_ctx *) data;
   if (lc->fail_count.load())
      return;
   if (i >= lc->ncallee) {
      uint32_t k = i - lc->ncallee;
      const struct tu_rt_trav_desc *d = &lc->trav[k];
      unsigned char pkey[BLAKE3_KEY_LEN];
      struct mesa_blake3 h;
      _mesa_blake3_init(&h);
      static const char tag[] = "tu-rt-part-trav-1";
      _mesa_blake3_update(&h, tag, sizeof(tag));
      tu_rt_hash_abi(&h);
      _mesa_blake3_update(&h, lc->env.key, sizeof(*lc->env.key));
      enum ir3_shader_debug dbg = ir3_shader_debug_hash_key();
      _mesa_blake3_update(&h, &dbg, sizeof(dbg));
      if (lc->env.layout)
         _mesa_blake3_update(&h, lc->env.layout->blake3, sizeof(lc->env.layout->blake3));
      VkPipelineCreateFlags2KHR hflags = lc->env.flags &
         (VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_TRIANGLES_BIT_KHR |
          VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_AABBS_BIT_KHR);
      _mesa_blake3_update(&h, &hflags, sizeof(hflags));
      uint32_t desc[7] = { d->cls, d->used, d->used_chit, d->has_ahit, d->has_isec,
                           d->has_chit, d->has_miss };
      _mesa_blake3_update(&h, desc, sizeof(desc));
      struct tu_ray_tracing_pipeline *p = lc->pipeline;
      struct tu_rt_case *cases = (struct tu_rt_case *)
         calloc(MAX2(p->group_count, 1), sizeof(struct tu_rt_case));
      uint32_t n = 0;
      struct tu_rt_compiler tmp = {};
      tmp.pipeline = p;
      tmp.part = TU_RT_PART_TRAVERSAL;
      tmp.part_payload_size = d->cls;
      for (uint32_t g = 0; g < p->group_count; g++) {
         const struct tu_rt_group *grp = p->groups + g;
         if (grp->isec_index != VK_SHADER_UNUSED_KHR) {
            if (!tu_rt_handle_seen(&tmp, g, grp->handle, 2))
               cases[n++] = (struct tu_rt_case) { grp->handle, g };
         } else if (grp->ahit_index != VK_SHADER_UNUSED_KHR) {
            if (!tu_rt_handle_seen(&tmp, g, grp->handle, 1) &&
                tu_rt_compat(&tmp, p->stages[grp->ahit_index], NULL))
               cases[n++] = (struct tu_rt_case) { grp->handle, g };
         }
      }
      qsort(cases, n, sizeof(*cases), tu_rt_case_cmp);
      for (uint32_t j = 0; j < n; j++) {
         const struct tu_rt_group *grp = p->groups + cases[j].group;
         _mesa_blake3_update(&h, &grp->handle, sizeof(grp->handle));
         uint32_t idx[2] = { grp->ahit_index, grp->isec_index };
         for (unsigned k2 = 0; k2 < 2; k2++) {
            if (idx[k2] == VK_SHADER_UNUSED_KHR) {
               uint32_t none = ~0u;
               _mesa_blake3_update(&h, &none, sizeof(none));
            } else {
               _mesa_blake3_update(&h, p->stage_hashes + idx[k2] * BLAKE3_KEY_LEN,
                                   BLAKE3_KEY_LEN);
            }
         }
      }
      free(cases);
      _mesa_blake3_final(&h, pkey);

      struct tu_rt_partset *ps = tu_rt_partset_lookup(lc->env.dev, lc->cache, pkey);
      if (!ps) {
         int64_t t0 = os_time_get_nano();
         const char *why = tu_rt_build_traversal_part(&lc->env, p, d, pkey, &ps);
         int64_t dt = os_time_get_nano() - t0;
         lc->a_trav.fetch_add(dt);
         if (why) {
            tu_rt_link_note_fail(lc, why);
            return;
         }
         ps = tu_rt_partset_insert(lc->env.dev, lc->cache, ps);
         lc->n_trav_built.fetch_add(1);
         if (tu_rt_log_enabled())
            mesa_logi("tu-rt-probe: traversal class=%u ahit=%d isec=%d used=0x%x ir3=%u regs=%u "
                      "stp=%u ldp=%u pvt=%u %.1f ms",
                      d->cls, d->has_ahit, d->has_isec, d->used, ps->parts[0].instrs_count,
                      ps->parts[0].max_reg + 1, ps->parts[0].stp_count,
                      ps->parts[0].ldp_count, ps->parts[0].pvtmem_size, dt / 1e6);
      }
      lc->trav_sets[k] = ps;
      return;
   }

   uint32_t s = lc->callee_stages[i];
   unsigned char pkey[BLAKE3_KEY_LEN];
   tu_rt_part_key(pkey, "tu-rt-part-callee-1", lc->pipeline->stage_hashes + s * BLAKE3_KEY_LEN,
                  lc->env.key, lc->env.layout);
   struct tu_rt_partset *ps = tu_rt_partset_lookup(lc->env.dev, lc->cache, pkey);
   if (!ps) {
      int64_t t0 = os_time_get_nano();
      const char *why = tu_rt_build_callee_part(&lc->env, lc->pipeline->stages[s], pkey, &ps);
      lc->a_callee.fetch_add(os_time_get_nano() - t0);
      if (why) {
         tu_rt_link_note_fail(lc, why);
         return;
      }
      ps = tu_rt_partset_insert(lc->env.dev, lc->cache, ps);
      lc->n_callee_built.fetch_add(1);
   }
   lc->callee_sets[i] = ps;
}

static bool
tu_rt_callee_compat(const struct tu_rt_partset *ps, uint32_t cls)
{
   return !ps->has_payload || ps->payload_size == cls;
}

static void
tu_rt_link_dispatch_job(void *data, uint32_t k)
{
   struct tu_rt_link_ctx *lc = (struct tu_rt_link_ctx *) data;
   struct tu_ray_tracing_pipeline *p = lc->pipeline;
   if (lc->fail_count.load())
      return;
   int64_t t0 = os_time_get_nano();
   uint32_t cls = lc->classes[k];
   struct ir3_rt_blob *blobs =
      (struct ir3_rt_blob *) calloc(MAX2(lc->ncallee, 1), sizeof(struct ir3_rt_blob));
   const struct tu_rt_part_bin **bins =
      (const struct tu_rt_part_bin **) calloc(MAX2(lc->ncallee, 1), sizeof(void *));
   int32_t *callee_blob = (int32_t *) malloc(MAX2(lc->ncallee, 1) * sizeof(int32_t));
   struct tu_rt_case *chit = (struct tu_rt_case *)
      calloc(MAX2(p->group_count, 1), sizeof(struct tu_rt_case));
   struct tu_rt_case *miss = (struct tu_rt_case *)
      calloc(MAX2(p->group_count, 1), sizeof(struct tu_rt_case));
   uint32_t nblobs = 0, nchit = 0, nmiss = 0, sets = 0;
   for (uint32_t i = 0; i < lc->ncallee; i++)
      callee_blob[i] = -1;
   struct tu_rt_compiler tmp = {};
   tmp.pipeline = p;
   for (uint32_t g = 0; g < p->group_count; g++) {
      const struct tu_rt_group *grp = p->groups + g;
      if (grp->recursive_index == VK_SHADER_UNUSED_KHR)
         continue;
      uint8_t st = p->stage_types[grp->recursive_index];
      if (st != MESA_SHADER_CLOSEST_HIT && st != MESA_SHADER_MISS)
         continue;
      if (tu_rt_handle_seen(&tmp, g, grp->handle, 0))
         continue;
      int32_t ce = lc->stage_to_callee[grp->recursive_index];
      const struct tu_rt_partset *ps = lc->callee_sets[ce];
      if (!tu_rt_callee_compat(ps, tu_rt_class_size(cls)))
         continue;
      if (p->stage_types[grp->recursive_index] == MESA_SHADER_CLOSEST_HIT &&
          (cls & TU_RT_CLASS_SKIP_CHIT))
         continue;
      if (callee_blob[ce] < 0) {
         callee_blob[ce] = nblobs;
         blobs[nblobs] = (struct ir3_rt_blob) { ps->parts[0].instrs, ps->parts[0].count };
         bins[nblobs++] = &ps->parts[0];
         sets |= ps->active_desc_sets;
      }
      struct tu_rt_case cs = { grp->handle, (uint32_t) callee_blob[ce] };
      if (st == MESA_SHADER_CLOSEST_HIT)
         chit[nchit++] = cs;
      else
         miss[nmiss++] = cs;
   }
   nir_shader *nir = tu_rt_build_dispatch(lc->env.dev, cls, chit, nchit, miss, nmiss);
   struct ir3_rt_blobs bl = { nblobs, blobs };
   struct tu_rt_link_merge merge = { bins, nblobs, NULL };
   lc->disp_words[k] = NULL;
   const char *why = tu_rt_part_compile(lc->env.dev, lc->env.layout, lc->env.key, nir,
                                        TU_RT_FRAME_CALLEE, &lc->disp_bin[k],
                                        &lc->disp_words[k], &sets, &bl, &merge);
   lc->disp_bin[k].instrs = lc->disp_words[k];
   lc->disp_sets[k] = sets;
   free(blobs);
   free(bins);
   free(callee_blob);
   free(chit);
   free(miss);
   lc->a_trav.fetch_add(os_time_get_nano() - t0);
   if (why)
      tu_rt_link_note_fail(lc, why);
}

static void
tu_rt_link_variant_job(void *data, uint32_t v)
{
   struct tu_rt_link_ctx *lc = (struct tu_rt_link_ctx *) data;
   struct tu_ray_tracing_pipeline *p = lc->pipeline;
   if (lc->fail_count.load() || p->variants[v].shader)
      return;
   int64_t t0 = os_time_get_nano();
   int64_t c0 = tu_rt_thread_cpu_ns();

   const struct tu_rt_partset *rg = lc->rg_sets[v];
   uint32_t nblobs = rg->nparts;
   uint32_t cap = rg->nparts + 2 * TU_RT_MAX_CLASSES;
   struct ir3_rt_blob *blobs =
      (struct ir3_rt_blob *) calloc(cap, sizeof(struct ir3_rt_blob));
   const struct tu_rt_part_bin **bins =
      (const struct tu_rt_part_bin **) calloc(cap, sizeof(void *));
   uint32_t sets = rg->active_desc_sets;
   for (uint32_t i = 0; i < rg->nparts; i++) {
      blobs[i] = (struct ir3_rt_blob) { rg->parts[i].instrs, rg->parts[i].count };
      bins[i] = &rg->parts[i];
   }

   struct tu_rt_glue_class gc[TU_RT_MAX_CLASSES] = {};
   for (uint32_t ci = 0; ci < rg->nclasses; ci++) {
      uint32_t cls = rg->classes[ci];
      uint32_t k = 0;
      while (k < lc->nclasses && lc->classes[k] != cls)
         k++;
      assert(k < lc->nclasses);
      const struct tu_rt_partset *tr = lc->trav_sets[k];
      gc[ci].cls = cls;
      gc[ci].trav_blob = nblobs;
      blobs[nblobs] = (struct ir3_rt_blob) { tr->parts[0].instrs, tr->parts[0].count };
      bins[nblobs++] = &tr->parts[0];
      sets |= tr->active_desc_sets;
      gc[ci].disp_blob = nblobs;
      blobs[nblobs] = (struct ir3_rt_blob) { lc->disp_bin[k].instrs, lc->disp_bin[k].count };
      bins[nblobs++] = &lc->disp_bin[k];
      sets |= lc->disp_sets[k];
   }

   int64_t tg0 = os_time_get_nano();
   uint32_t zero = 0;
   nir_shader *glue = tu_rt_build_glue(lc->env.dev, false, &p->variants[v].handle, &zero, 1,
                                       rg->nparts, gc, rg->nclasses);
   lc->a_glue.fetch_add(os_time_get_nano() - tg0);
   struct ir3_rt_blobs bl = { nblobs, blobs };
   struct tu_shader *sh = NULL;
   const char *why = tu_rt_link_one(lc->env.dev, lc->env.layout, lc->env.key, glue, &bl,
                                    bins, nblobs, sets,
                                    lc->link_keys + v * BLAKE3_KEY_LEN, lc->cache, &sh);
   free(blobs);
   free(bins);
   int64_t dt = os_time_get_nano() - t0;
   lc->a_link.fetch_add(dt);
   int64_t prev = lc->a_link_max.load();
   while (dt > prev && !lc->a_link_max.compare_exchange_weak(prev, dt))
      ;
   tu_rt_note_tcpu(lc, tu_rt_thread_cpu_ns() - c0);
   if (why) {
      tu_rt_link_note_fail(lc, why);
      return;
   }
   p->variants[v].shader = sh;
}

static void
tu_rt_link_all_job(void *data, uint32_t unused)
{
   struct tu_rt_link_ctx *lc = (struct tu_rt_link_ctx *) data;
   struct tu_ray_tracing_pipeline *p = lc->pipeline;
   if (lc->fail_count.load() || p->variants[0].shader)
      return;
   int64_t t0 = os_time_get_nano();
   int64_t c0 = tu_rt_thread_cpu_ns();
   const uint32_t nrg = lc->nraygens;
   uint32_t total = 0;
   for (uint32_t r = 0; r < nrg; r++)
      total += lc->rg_sets[r]->nparts;
   uint32_t cap = total + 2 * lc->nclasses;
   struct ir3_rt_blob *blobs =
      (struct ir3_rt_blob *) calloc(MAX2(cap, 1), sizeof(struct ir3_rt_blob));
   const struct tu_rt_part_bin **bins =
      (const struct tu_rt_part_bin **) calloc(MAX2(cap, 1), sizeof(void *));
   uint32_t *handles = (uint32_t *) calloc(MAX2(nrg, 1), sizeof(uint32_t));
   uint32_t *first = (uint32_t *) calloc(MAX2(nrg, 1), sizeof(uint32_t));
   struct tu_rt_glue_class *gc = (struct tu_rt_glue_class *)
      calloc(MAX2(lc->nclasses, 1), sizeof(struct tu_rt_glue_class));
   uint32_t nblobs = 0, sets = 0;
   for (uint32_t r = 0; r < nrg; r++) {
      const struct tu_rt_partset *rg = lc->rg_sets[r];
      handles[r] = p->groups[lc->groups[r]].handle;
      first[r] = nblobs;
      sets |= rg->active_desc_sets;
      for (uint32_t i = 0; i < rg->nparts; i++) {
         blobs[nblobs] = (struct ir3_rt_blob) { rg->parts[i].instrs, rg->parts[i].count };
         bins[nblobs++] = &rg->parts[i];
      }
   }
   for (uint32_t k = 0; k < lc->nclasses; k++) {
      const struct tu_rt_partset *tr = lc->trav_sets[k];
      gc[k].cls = lc->classes[k];
      gc[k].trav_blob = nblobs;
      blobs[nblobs] = (struct ir3_rt_blob) { tr->parts[0].instrs, tr->parts[0].count };
      bins[nblobs++] = &tr->parts[0];
      sets |= tr->active_desc_sets;
      gc[k].disp_blob = nblobs;
      blobs[nblobs] = (struct ir3_rt_blob) { lc->disp_bin[k].instrs, lc->disp_bin[k].count };
      bins[nblobs++] = &lc->disp_bin[k];
      sets |= lc->disp_sets[k];
   }
   int64_t tg0 = os_time_get_nano();
   nir_shader *glue = tu_rt_build_glue(lc->env.dev, false, handles, first, nrg, total,
                                       gc, lc->nclasses);
   lc->a_glue.fetch_add(os_time_get_nano() - tg0);
   struct ir3_rt_blobs bl = { nblobs, blobs };
   struct tu_shader *sh = NULL;
   const char *why = tu_rt_link_one(lc->env.dev, lc->env.layout, lc->env.key, glue, &bl,
                                    bins, nblobs, sets, lc->link_keys, lc->cache, &sh);
   free(blobs);
   free(bins);
   free(handles);
   free(first);
   free(gc);
   int64_t dt = os_time_get_nano() - t0;
   lc->a_link.fetch_add(dt);
   lc->a_link_max.store(dt);
   tu_rt_note_tcpu(lc, tu_rt_thread_cpu_ns() - c0);
   if (why) {
      tu_rt_link_note_fail(lc, why);
      return;
   }
   p->variants[0].shader = sh;
}

static void
tu_rt_link_whole_job(void *data, uint32_t v)
{
   struct tu_rt_link_ctx *lc = (struct tu_rt_link_ctx *) data;
   struct tu_ray_tracing_pipeline *p = lc->pipeline;
   if (lc->fail_count.load() || p->variants[v].shader)
      return;
   int64_t t0 = os_time_get_nano();
   nir_shader *nir = tu_rt_build_shader(lc->env.dev, p, lc->create_info, lc->env.flags,
                                        (int) lc->groups[v], TU_RT_PART_WHOLE);
   if (!nir) {
      tu_rt_link_note_fail(lc, "build");
      return;
   }
   struct tu_rt_part_bin bin = {};
   uint64_t *words = NULL;
   uint32_t sets = 0;
   const char *why = tu_rt_part_compile(lc->env.dev, lc->env.layout, lc->env.key, nir,
                                        TU_RT_FRAME_WHOLE, &bin, &words, &sets);
   int64_t t1 = os_time_get_nano();
   lc->a_rg.fetch_add(t1 - t0);
   if (why) {
      tu_rt_link_note_fail(lc, why);
      return;
   }
   bin.instrs = words;
   struct ir3_rt_blob blob = { words, bin.count };
   struct ir3_rt_blobs bl = { 1, &blob };
   const struct tu_rt_part_bin *bins[1] = { &bin };
   nir_shader *glue = tu_rt_build_glue(lc->env.dev, true, NULL, NULL, 0, 0, NULL, 0);
   struct tu_shader *sh = NULL;
   why = tu_rt_link_one(lc->env.dev, lc->env.layout, lc->env.key, glue, &bl, bins, 1,
                        sets, lc->link_keys + v * BLAKE3_KEY_LEN, lc->cache, &sh);
   free(words);
   int64_t dt = os_time_get_nano() - t1;
   lc->a_link.fetch_add(dt);
   int64_t prev = lc->a_link_max.load();
   while (dt > prev && !lc->a_link_max.compare_exchange_weak(prev, dt))
      ;
   if (why) {
      tu_rt_link_note_fail(lc, why);
      return;
   }
   p->variants[v].shader = sh;
}

static VkResult
tu_rt_link_variants(struct tu_device *dev, struct vk_pipeline_cache *cache,
                    struct tu_ray_tracing_pipeline *pipeline,
                    const VkRayTracingPipelineCreateInfoKHR *create_info,
                    VkPipelineCreateFlags2KHR flags,
                    struct tu_pipeline_layout *layout,
                    const struct tu_shader_key *key,
                    const unsigned char *base_blake3,
                    const uint32_t *groups, uint32_t count, bool whole)
{
   int64_t t_start = os_time_get_nano();
   cache = cache ? cache : dev->mem_cache;
   const bool single = !whole && !debug_get_bool_option("TU_RT_LINK_PER_RAYGEN", false);
   const uint32_t nvar = single ? 1 : count;
   pipeline->variant_count = nvar;
   pipeline->variants = rzalloc_array(pipeline->mem_ctx, struct tu_rt_variant, nvar);
   unsigned char *keys = rzalloc_array(pipeline->mem_ctx, unsigned char,
                                       nvar * BLAKE3_KEY_LEN);
   uint32_t missing = 0;
   for (uint32_t v = 0; v < nvar; v++) {
      const struct tu_rt_group *g = pipeline->groups + groups[v];
      pipeline->variants[v].handle = g->handle;
      struct mesa_blake3 h;
      _mesa_blake3_init(&h);
      _mesa_blake3_update(&h, base_blake3, BLAKE3_KEY_LEN);
      static const char tag[] = "tu-rt-link-1";
      static const char tag_all[] = "tu-rt-link-all-1";
      if (single)
         _mesa_blake3_update(&h, tag_all, sizeof(tag_all));
      else
         _mesa_blake3_update(&h, tag, sizeof(tag));
      uint64_t mode[3] = { whole ? 1u : 2u,
                           debug_get_num_option("TU_RT_HIT_BUDGET", 250000),
                           debug_get_num_option("TU_RT_MAX_PREINLINE", 400000) };
      _mesa_blake3_update(&h, mode, sizeof(mode));
      tu_rt_hash_abi(&h);
      _mesa_blake3_update(&h, &g->handle, sizeof(g->handle));
      _mesa_blake3_final(&h, keys + v * BLAKE3_KEY_LEN);
      pipeline->variants[v].shader =
         tu_compute_pipeline_cache_lookup(dev, cache, keys + v * BLAKE3_KEY_LEN);
      if (!pipeline->variants[v].shader)
         missing++;
   }

   if (!missing)
      goto done;
   if (flags & VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT_KHR)
      return VK_PIPELINE_COMPILE_REQUIRED;

   {
      void *nir_ctx = ralloc_context(NULL);
      VkResult result = tu_rt_compile_stages(dev, cache, pipeline, create_info,
                                             flags, key, nir_ctx);
      if (result != VK_SUCCESS) {
         ralloc_free(nir_ctx);
         return result;
      }

      for (uint32_t s = 0; s < pipeline->stage_count; s++) {
         if (pipeline->stage_types[s] == MESA_SHADER_CALLABLE) {
            ralloc_free(nir_ctx);
            memset(pipeline->stages, 0, pipeline->stage_count * sizeof(pipeline->stages[0]));
            mesa_logw("tu-rt: link: callable stages, using inline path");
            return VK_ERROR_FEATURE_NOT_PRESENT;
         }
      }

      struct tu_rt_link_ctx *lc = new tu_rt_link_ctx();
      lc->env.dev = dev;
      lc->env.layout = layout;
      lc->env.key = key;
      lc->env.flags = flags;
      lc->cache = cache;
      lc->pipeline = pipeline;
      lc->create_info = create_info;
      lc->groups = groups;
      lc->link_keys = keys;
      lc->whole = whole;
      lc->single = single;
      lc->nraygens = count;
      lc->fail_count.store(0);
      lc->fail = NULL;
      simple_mtx_init(&lc->lock, mtx_plain);
      for (unsigned i = 0; i < 8; i++)
         lc->probe_acc[i].store(0);
      lc->a_rg.store(0);
      lc->a_callee.store(0);
      lc->a_trav.store(0);
      lc->a_link.store(0);
      lc->a_link_max.store(0);
      lc->a_link_tcpu.store(0);
      lc->a_link_tcpu_max.store(0);
      lc->a_glue.store(0);
      lc->n_rg_built.store(0);
      lc->n_callee_built.store(0);
      lc->n_trav_built.store(0);

      int64_t t0 = os_time_get_nano(), t1 = t0, t2 = t0;
      if (whole) {
         tu_rt_par_run(count, tu_rt_link_whole_job, lc);
      } else {
         lc->rg_sets = (struct tu_rt_partset **) calloc(count, sizeof(void *));
         lc->rg_stage = (uint32_t *) calloc(count, sizeof(uint32_t));
         for (uint32_t v = 0; v < count; v++)
            lc->rg_stage[v] = pipeline->groups[groups[v]].recursive_index;
         tu_rt_par_run(count, tu_rt_link_rg_job, lc);
         t1 = os_time_get_nano();

         lc->stage_to_callee = (int32_t *) malloc(MAX2(pipeline->stage_count, 1) * sizeof(int32_t));
         lc->callee_stages = (uint32_t *) malloc(MAX2(pipeline->stage_count, 1) * sizeof(uint32_t));
         lc->ncallee = 0;
         for (uint32_t s = 0; s < pipeline->stage_count; s++) {
            lc->stage_to_callee[s] = -1;
            if (pipeline->stage_types[s] == MESA_SHADER_CLOSEST_HIT ||
                pipeline->stage_types[s] == MESA_SHADER_MISS) {
               lc->stage_to_callee[s] = lc->ncallee;
               lc->callee_stages[lc->ncallee++] = s;
            }
         }
         lc->callee_sets = (struct tu_rt_partset **) calloc(MAX2(lc->ncallee, 1), sizeof(void *));

         lc->nclasses = 0;
         if (!lc->fail_count.load()) {
            for (uint32_t v = 0; v < count; v++) {
               const struct tu_rt_partset *rg = lc->rg_sets[v];
               if (!rg)
                  continue;
               for (uint32_t ci = 0; ci < rg->nclasses; ci++) {
                  uint32_t k = 0;
                  while (k < lc->nclasses && lc->classes[k] != rg->classes[ci])
                     k++;
                  if (k == lc->nclasses) {
                     if (lc->nclasses == ARRAY_SIZE(lc->classes)) {
                        tu_rt_link_note_fail(lc, "too many classes");
                        break;
                     }
                     lc->classes[lc->nclasses++] = rg->classes[ci];
                  }
               }
            }
         }

         bool has_ahit = false, has_isec = false;
         for (uint32_t g = 0; g < pipeline->group_count; g++) {
            if (pipeline->groups[g].ahit_index != VK_SHADER_UNUSED_KHR)
               has_ahit = true;
            if (pipeline->groups[g].isec_index != VK_SHADER_UNUSED_KHR)
               has_isec = true;
         }
         for (uint32_t k = 0; k < lc->nclasses; k++) {
            struct tu_rt_trav_desc *d = &lc->trav[k];
            d->cls = lc->classes[k];
            d->has_ahit = has_ahit;
            d->has_isec = has_isec;
            d->used = 0;
            d->used_chit = 0;
            d->has_chit = false;
            d->has_miss = false;
            struct tu_rt_compiler tmp = {};
            tmp.pipeline = pipeline;
            tmp.part = TU_RT_PART_TRAVERSAL;
            tmp.part_payload_size = d->cls;
            for (uint32_t g = 0; g < pipeline->group_count; g++) {
               const struct tu_rt_group *grp = pipeline->groups + g;
               if (grp->ahit_index != VK_SHADER_UNUSED_KHR)
                  d->used |= tu_rt_scan_used(pipeline->stages[grp->ahit_index]);
               if (grp->isec_index != VK_SHADER_UNUSED_KHR)
                  d->used |= tu_rt_scan_used(pipeline->stages[grp->isec_index]);
               if (grp->recursive_index != VK_SHADER_UNUSED_KHR &&
                   pipeline->stage_types[grp->recursive_index] == MESA_SHADER_CLOSEST_HIT &&
                   tu_rt_compat(&tmp, pipeline->stages[grp->recursive_index], NULL) &&
                   !(d->cls & TU_RT_CLASS_SKIP_CHIT)) {
                  d->used_chit |= tu_rt_scan_used(pipeline->stages[grp->recursive_index]);
                  d->has_chit = true;
               }
               if (grp->recursive_index != VK_SHADER_UNUSED_KHR &&
                   pipeline->stage_types[grp->recursive_index] == MESA_SHADER_MISS &&
                   tu_rt_compat(&tmp, pipeline->stages[grp->recursive_index], NULL))
                  d->has_miss = true;
            }
            d->used |= d->used_chit;
         }

         if (!lc->fail_count.load())
            tu_rt_par_run(lc->ncallee + lc->nclasses, tu_rt_link_callee_job, lc);
         if (!lc->fail_count.load())
            tu_rt_par_run(lc->nclasses, tu_rt_link_dispatch_job, lc);
         t2 = os_time_get_nano();
         if (!lc->fail_count.load()) {
            if (single)
               tu_rt_par_run(1, tu_rt_link_all_job, lc);
            else
               tu_rt_par_run(count, tu_rt_link_variant_job, lc);
         }
      }
      int64_t t3 = os_time_get_nano();

      ralloc_free(nir_ctx);
      memset(pipeline->stages, 0, pipeline->stage_count * sizeof(pipeline->stages[0]));

      const char *fail = lc->fail;
      if (tu_rt_log_enabled()) {
         mesa_logi("tu-rt-link: %s %u variants (%u to link) stages=%u groups=%u classes=%u "
                   "callees=%u | rg parts built=%u %.1f ms cpu, callee built=%u %.1f ms cpu, "
                   "trav built=%u %.1f ms cpu, link %.1f ms cpu (glue nir %.1f, max %.2f ms) | wall rg=%.1f "
                   "callee+trav=%.1f link=%.1f ms | link thread cpu %.1f ms (max %.2f ms)%s%s",
                   whole ? "whole" : "force", count, missing, pipeline->stage_count,
                   pipeline->group_count, lc->nclasses, lc->ncallee,
                   lc->n_rg_built.load(), lc->a_rg.load() / 1e6,
                   lc->n_callee_built.load(), lc->a_callee.load() / 1e6,
                   lc->n_trav_built.load(), lc->a_trav.load() / 1e6,
                   lc->a_link.load() / 1e6, lc->a_glue.load() / 1e6, lc->a_link_max.load() / 1e6,
                   (t1 - t0) / 1e6, (t2 - t1) / 1e6, (t3 - t2) / 1e6,
                   lc->a_link_tcpu.load() / 1e6, lc->a_link_tcpu_max.load() / 1e6,
                   fail ? " FAIL: " : "", fail ? fail : "");
         mesa_logi("tu-rt-link: cumulative host lower %.1f ms, host create %.1f ms",
                   tu_rt_link_lower_ns.load() / 1e6, tu_rt_link_create_ns.load() / 1e6);
         if (!whole && lc->n_rg_built.load()) {
            mesa_logi("tu-rt-probe: raygens built=%u parts=%" PRId64 " stack_bytes=%" PRId64
                      " locals=%" PRId64 " nir_pre=%" PRId64 " nir_post=%" PRId64,
                      lc->n_rg_built.load(), lc->probe_acc[0].load(), lc->probe_acc[1].load(),
                      lc->probe_acc[2].load(), lc->probe_acc[3].load(),
                      lc->probe_acc[4].load());
         }
      }

      if (!whole) {
         for (uint32_t v = 0; v < count; v++)
            tu_rt_partset_unref(dev, lc->rg_sets[v]);
         for (uint32_t i = 0; i < lc->ncallee; i++)
            tu_rt_partset_unref(dev, lc->callee_sets[i]);
         for (uint32_t k = 0; k < lc->nclasses; k++)
            tu_rt_partset_unref(dev, lc->trav_sets[k]);
         for (uint32_t k = 0; k < lc->nclasses; k++)
            free(lc->disp_words[k]);
         free(lc->rg_sets);
         free(lc->rg_stage);
         free(lc->stage_to_callee);
         free(lc->callee_stages);
         free(lc->callee_sets);
      }
      simple_mtx_destroy(&lc->lock);
      delete lc;

      if (fail) {
         mesa_logw("tu-rt: link failed (%s), using inline path", fail);
         for (uint32_t v = 0; v < nvar; v++) {
            if (pipeline->variants[v].shader)
               vk_pipeline_cache_object_unref(&dev->vk, &pipeline->variants[v].shader->base);
            pipeline->variants[v].shader = NULL;
         }
         return VK_ERROR_FEATURE_NOT_PRESENT;
      }
   }

done:
   if (tu_rt_log_enabled()) {
      uint64_t instrs = 0;
      unsigned max_regs = 0;
      for (uint32_t v = 0; v < nvar; v++) {
         instrs += pipeline->variants[v].shader->variant->info.instrs_count;
         max_regs = MAX2(max_regs, pipeline->variants[v].shader->variant->info.max_reg + 1u);
      }
      mesa_logi("tu-rt: %u raygen variants (%u linked) total instrs=%" PRIu64
                " max regs=%u %.1f ms (%u raygens)",
                nvar, missing, instrs, max_regs, (os_time_get_nano() - t_start) / 1e6, count);
   }
   return VK_SUCCESS;
}

static std::atomic<int64_t> tu_rt_prebuild_ns;
static std::atomic<uint32_t> tu_rt_prebuild_count;

static void
tu_rt_prebuild_parts(struct tu_device *dev, struct vk_pipeline_cache *cache,
                     const VkRayTracingPipelineCreateInfoKHR *create_info,
                     VkPipelineCreateFlags2KHR flags,
                     const struct tu_shader_key *key,
                     const unsigned char *stage_hash, nir_shader *nir)
{
   const unsigned mode = tu_rt_link_mode();
   if (mode != TU_RT_LINK_FORCE && mode != TU_RT_LINK_AUTO)
      return;
   if (!debug_get_bool_option("TU_RT_LINK_PREBUILD", true))
      return;
   if (dev->physical_device->info->chip < 7 ||
       (flags & VK_PIPELINE_CREATE_2_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR))
      return;
   mesa_shader_stage st = nir->info.stage;
   if (st != MESA_SHADER_RAYGEN && st != MESA_SHADER_CLOSEST_HIT &&
       st != MESA_SHADER_MISS)
      return;

   VK_FROM_HANDLE(tu_pipeline_layout, layout, create_info->layout);
   if (!layout)
      return;
   struct tu_rt_part_env env = { dev, layout, key, flags };
   unsigned char pkey[BLAKE3_KEY_LEN];
   tu_rt_part_key(pkey, st == MESA_SHADER_RAYGEN ? "tu-rt-part-rg-1" : "tu-rt-part-callee-1",
                  stage_hash, key, layout);
   struct tu_rt_partset *ps = tu_rt_partset_lookup(dev, cache, pkey);
   if (ps) {
      tu_rt_partset_unref(dev, ps);
      return;
   }

   int64_t t0 = os_time_get_nano();
   const char *why;
   int64_t probe[6] = {};
   if (st == MESA_SHADER_RAYGEN)
      why = tu_rt_build_raygen_parts(&env, nir, pkey, &ps, probe);
   else
      why = tu_rt_build_callee_part(&env, nir, pkey, &ps);
   int64_t dt = os_time_get_nano() - t0;
   uint32_t n = tu_rt_prebuild_count.fetch_add(1) + 1;
   int64_t total = tu_rt_prebuild_ns.fetch_add(dt) + dt;
   if (tu_rt_log_enabled()) {
      uint32_t instrs = 0, regs = 0, stp = 0, ldp = 0, pvt = 0;
      char sizes[256] = "";
      unsigned len = 0;
      for (uint32_t i = 0; ps && !why && i < ps->nparts; i++) {
         instrs += ps->parts[i].instrs_count;
         regs = MAX2(regs, (uint32_t) (ps->parts[i].max_reg + 1));
         stp += ps->parts[i].stp_count;
         ldp += ps->parts[i].ldp_count;
         pvt = MAX2(pvt, ps->parts[i].pvtmem_size);
         if (len + 8 < sizeof(sizes))
            len += snprintf(sizes + len, sizeof(sizes) - len, "%u,", ps->parts[i].instrs_count);
      }
      mesa_logi("tu-rt-prebuild: %s %.1f ms%s%s parts=%" PRId64 " stack=%" PRId64
                " nir=%" PRId64 "->%" PRId64 " ir3=%u [%s] regs=%u stp=%u ldp=%u pvt=%u"
                " (total %u parts %.1f ms)",
                _mesa_shader_stage_to_abbrev(st), dt / 1e6, why ? " FAIL: " : "",
                why ? why : "", probe[0], probe[1], probe[3], probe[4], instrs, sizes,
                regs, stp, ldp, pvt, n, total / 1e6);
   }
   if (!why) {
      struct tu_rt_partset *ins = tu_rt_partset_insert(dev, cache, ps);
      if (debug_get_bool_option("TU_RT_LINK_KEYDBG", false)) {
         char hk[17];
         tu_keylog_hex(hk, pkey, 8);
         mesa_logi("tu-rt-keydbg: insert %s cache=%p obj=%p ref=%d", hk, (void *) cache,
                   (void *) ins, (int) ins->base.ref_cnt);
      }
      tu_rt_partset_unref(dev, ins);
   }
}

struct tu_rt_variant_job {
   struct tu_device *dev;
   struct tu_ray_tracing_pipeline *pipeline;
   const VkRayTracingPipelineCreateInfoKHR *create_info;
   VkPipelineCreateFlags2KHR flags;
   struct tu_pipeline_layout *layout;
   struct vk_pipeline_cache *cache;
   const struct tu_shader_key *key;
   const uint32_t *groups;
   const unsigned char *hashes;
   uint32_t count;
   std::atomic<uint32_t> next;
   std::atomic<int> result;
};

static void *
tu_rt_variant_worker(void *data)
{
   struct tu_rt_variant_job *job = (struct tu_rt_variant_job *) data;
   for (;;) {
      uint32_t v = job->next.fetch_add(1);
      if (v >= job->count)
         break;
      struct tu_rt_variant *var = job->pipeline->variants + v;
      if (var->shader)
         continue;
      nir_shader *nir = tu_rt_build_shader(job->dev, job->pipeline, job->create_info,
                                           job->flags, (int) job->groups[v]);
      if (!nir) {
         job->result.store(VK_ERROR_OUT_OF_HOST_MEMORY);
         continue;
      }
      struct tu_shader *shader = NULL;
      VkResult r = tu_compute_shader_compile(job->dev, job->layout, job->cache, nir,
                                             job->key, job->hashes + v * BLAKE3_KEY_LEN,
                                             &shader);
      if (r != VK_SUCCESS || !shader) {
         job->result.store(r != VK_SUCCESS ? r : VK_ERROR_OUT_OF_HOST_MEMORY);
         continue;
      }
      var->shader = shader;
   }
   return NULL;
}

static VkResult
tu_rt_build_variants(struct tu_device *dev, struct vk_pipeline_cache *cache,
                     struct tu_ray_tracing_pipeline *pipeline,
                     const VkRayTracingPipelineCreateInfoKHR *create_info,
                     VkPipelineCreateFlags2KHR flags,
                     struct tu_pipeline_layout *layout,
                     const struct tu_shader_key *key,
                     const unsigned char *base_blake3,
                     const uint32_t *groups, uint32_t count)
{
   pipeline->variant_count = count;
   pipeline->variants = rzalloc_array(pipeline->mem_ctx, struct tu_rt_variant, count);
   unsigned char *hashes = rzalloc_array(pipeline->mem_ctx, unsigned char,
                                         count * BLAKE3_KEY_LEN);
   uint32_t missing = 0;
   for (uint32_t v = 0; v < count; v++) {
      const struct tu_rt_group *g = pipeline->groups + groups[v];
      pipeline->variants[v].handle = g->handle;
      struct mesa_blake3 ctx;
      _mesa_blake3_init(&ctx);
      _mesa_blake3_update(&ctx, base_blake3, BLAKE3_KEY_LEN);
      static const char tag[] = "tu-rt-raygen-1";
      _mesa_blake3_update(&ctx, tag, sizeof(tag));
      _mesa_blake3_update(&ctx, &g->handle, sizeof(g->handle));
      _mesa_blake3_final(&ctx, hashes + v * BLAKE3_KEY_LEN);
      if (!(flags & VK_PIPELINE_CREATE_2_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR))
         pipeline->variants[v].shader =
            tu_compute_pipeline_cache_lookup(dev, cache, hashes + v * BLAKE3_KEY_LEN);
      if (!pipeline->variants[v].shader)
         missing++;
   }

   if (missing) {
      if (flags & VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT_KHR)
         return VK_PIPELINE_COMPILE_REQUIRED;
      void *nir_ctx = ralloc_context(NULL);
      VkResult result = tu_rt_compile_stages(dev, cache, pipeline, create_info,
                                             flags, key, nir_ctx);
      if (result != VK_SUCCESS) {
         ralloc_free(nir_ctx);
         return result;
      }

      struct tu_rt_variant_job job;
      job.dev = dev;
      job.pipeline = pipeline;
      job.create_info = create_info;
      job.flags = flags;
      job.layout = layout;
      job.cache = cache;
      job.key = key;
      job.groups = groups;
      job.hashes = hashes;
      job.count = count;
      job.next.store(0);
      job.result.store(VK_SUCCESS);

      static const unsigned max_jobs = debug_get_num_option("TU_RT_JOBS", 4);
      unsigned nthreads = MIN2(MAX2(max_jobs, 1u), missing);
      pthread_t threads[16];
      unsigned started = 0;
      for (unsigned t = 1; t < MIN2(nthreads, 16u); t++) {
         pthread_attr_t attr;
         pthread_attr_init(&attr);
         pthread_attr_setstacksize(&attr, 256u << 20);
         if (pthread_create(&threads[started], &attr, tu_rt_variant_worker, &job) == 0)
            started++;
         pthread_attr_destroy(&attr);
      }
      tu_rt_variant_worker(&job);
      for (unsigned t = 0; t < started; t++)
         pthread_join(threads[t], NULL);

      ralloc_free(nir_ctx);
      memset(pipeline->stages, 0, pipeline->stage_count * sizeof(pipeline->stages[0]));
      if (job.result.load() != VK_SUCCESS)
         return (VkResult) job.result.load();
   }

   if (debug_get_bool_option("TU_RT_LOG", false)) {
      uint64_t instrs = 0;
      unsigned max_regs = 0;
      for (uint32_t v = 0; v < count; v++) {
         instrs += pipeline->variants[v].shader->variant->info.instrs_count;
         max_regs = MAX2(max_regs, pipeline->variants[v].shader->variant->info.max_reg + 1u);
      }
      mesa_logi("tu-rt: %u raygen variants (%u compiled) total instrs=%" PRIu64 " max regs=%u",
                count, missing, instrs, max_regs);
   }
   return VK_SUCCESS;
}

static VkResult
tu_rt_pipeline_create(struct tu_device *dev,
                      struct vk_pipeline_cache *cache,
                      const VkRayTracingPipelineCreateInfoKHR *create_info,
                      VkPipelineCreateFlags2KHR flags,
                      const VkAllocationCallbacks *alloc,
                      VkPipeline *out_pipeline)
{
   VK_FROM_HANDLE(tu_pipeline_layout, layout, create_info->layout);
   VkResult result;

   int64_t t_start = os_time_get_nano();
   struct tu_ray_tracing_pipeline *pipeline =
      (struct tu_ray_tracing_pipeline *) vk_object_zalloc(
         &dev->vk, alloc, sizeof(*pipeline), VK_OBJECT_TYPE_PIPELINE);
   if (!pipeline)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   pipeline->base.base.type = TU_PIPELINE_RAY_TRACING;
   pipeline->base.base.executables_mem_ctx = ralloc_context(NULL);
   util_dynarray_init(&pipeline->base.base.executables,
                      pipeline->base.base.executables_mem_ctx);
   pipeline->base.base.active_stages = TU_RT_STAGES;
   pipeline->mem_ctx = ralloc_context(NULL);

   pipeline->stage_count = create_info->stageCount;
   pipeline->group_count = create_info->groupCount;
   if (create_info->pLibraryInfo) {
      for (uint32_t l = 0; l < create_info->pLibraryInfo->libraryCount; l++) {
         VK_FROM_HANDLE(tu_pipeline, lib_base,
                        create_info->pLibraryInfo->pLibraries[l]);
         struct tu_ray_tracing_pipeline *lib =
            tu_pipeline_to_ray_tracing(lib_base);
         pipeline->stage_count += lib->stage_count;
         pipeline->group_count += lib->group_count;
      }
   }

   pipeline->stages = rzalloc_array(pipeline->mem_ctx, nir_shader *,
                                    MAX2(pipeline->stage_count, 1));
   pipeline->stage_hashes = rzalloc_array(pipeline->mem_ctx, uint8_t,
                                          MAX2(pipeline->stage_count, 1) * BLAKE3_KEY_LEN);
   pipeline->stage_types = rzalloc_array(pipeline->mem_ctx, uint8_t,
                                         MAX2(pipeline->stage_count, 1));
   pipeline->groups = rzalloc_array(pipeline->mem_ctx, struct tu_rt_group,
                                    MAX2(pipeline->group_count, 1));

   struct tu_shader_key key = {};
   tu_shader_key_subgroup_size(&key, false, false, NULL, dev);
   struct vk_pipeline_robustness_state rs;
   vk_pipeline_robustness_state_fill(&dev->vk.robustness_state, &rs,
                                     create_info->pNext, NULL);
   tu_shader_key_robustness(&key, &rs);
   key.fast_math = debug_get_bool_option("TU_RT_FAST_MATH", false);
   key.ray_tracing = true;

   {
      uint32_t s = 0;
      for (; s < create_info->stageCount; s++) {
         vk_pipeline_hash_shader_stage(flags, create_info->pStages + s, &rs,
                                       pipeline->stage_hashes + s * BLAKE3_KEY_LEN);
         pipeline->stage_types[s] = vk_to_mesa_shader_stage(create_info->pStages[s].stage);
      }
      if (create_info->pLibraryInfo) {
         for (uint32_t l = 0; l < create_info->pLibraryInfo->libraryCount; l++) {
            VK_FROM_HANDLE(tu_pipeline, lib_base,
                           create_info->pLibraryInfo->pLibraries[l]);
            struct tu_ray_tracing_pipeline *lib = tu_pipeline_to_ray_tracing(lib_base);
            memcpy(pipeline->stage_hashes + s * BLAKE3_KEY_LEN, lib->stage_hashes,
                   lib->stage_count * BLAKE3_KEY_LEN);
            memcpy(pipeline->stage_types + s, lib->stage_types, lib->stage_count);
            s += lib->stage_count;
         }
      }
   }

   tu_rt_init_groups(dev, pipeline, create_info);

   if (tu_keylog_enabled()) {
      mesa_logi("tu-key: rt %s stages=%u groups=%u own_stages=%u libs=%d",
                (flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR) ? "library" : "final",
                pipeline->stage_count, pipeline->group_count, create_info->stageCount,
                create_info->pLibraryInfo ? (int) create_info->pLibraryInfo->libraryCount : 0);
      for (uint32_t s = 0; s < create_info->stageCount; s++) {
         const VkPipelineShaderStageCreateInfo *st = create_info->pStages + s;
         VK_FROM_HANDLE(vk_shader_module, module, st->module);
         const VkShaderModuleCreateInfo *minfo =
            (const VkShaderModuleCreateInfo *) vk_find_struct_const(st->pNext, SHADER_MODULE_CREATE_INFO);
         blake3_hash mh = {};
         if (module)
            memcpy(mh, module->hash, sizeof(mh));
         else if (minfo)
            vk_shader_module_hash(minfo, mh);
         blake3_hash sp = {};
         uint32_t sp_n = 0, sp_sz = 0;
         if (st->pSpecializationInfo) {
            struct mesa_blake3 sc;
            _mesa_blake3_init(&sc);
            _mesa_blake3_update(&sc, st->pSpecializationInfo->pMapEntries,
                                st->pSpecializationInfo->mapEntryCount *
                                sizeof(*st->pSpecializationInfo->pMapEntries));
            _mesa_blake3_update(&sc, st->pSpecializationInfo->pData,
                                st->pSpecializationInfo->dataSize);
            _mesa_blake3_final(&sc, sp);
            sp_n = st->pSpecializationInfo->mapEntryCount;
            sp_sz = (uint32_t) st->pSpecializationInfo->dataSize;
         }
         char hs[17], hm[17], hp[17];
         tu_keylog_hex(hs, pipeline->stage_hashes + s * BLAKE3_KEY_LEN, 8);
         tu_keylog_hex(hm, mh, 8);
         tu_keylog_hex(hp, sp, 8);
         mesa_logi("tu-key:  s%u %s name=%s sh=%s mod=%s%s sflags=0x%x spec=%u/%u/%s",
                   s, _mesa_shader_stage_to_abbrev(vk_to_mesa_shader_stage(st->stage)),
                   st->pName ? st->pName : "?", hs, hm,
                   module ? "" : (minfo ? "(code)" : "(ident)"),
                   (unsigned) st->flags, sp_n, sp_sz, hp);
      }
   }

   if (flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR) {
      pipeline->stage_set = tu_rt_stage_set_create(pipeline->mem_ctx, pipeline->stage_count);
      if (!pipeline->stage_set) {
         result = VK_ERROR_OUT_OF_HOST_MEMORY;
         goto fail;
      }
      result = tu_rt_compile_stages(dev, cache, pipeline, create_info, flags, &key, NULL);
      if (result != VK_SUCCESS)
         goto fail;
   } else {
      static const char version[] = "tu-rt-3";
      unsigned char blake3[BLAKE3_KEY_LEN];
      struct mesa_blake3 ctx;
      _mesa_blake3_init(&ctx);
      _mesa_blake3_update(&ctx, version, sizeof(version));
      if (layout)
         _mesa_blake3_update(&ctx, layout->blake3, sizeof(layout->blake3));
      _mesa_blake3_update(&ctx, &key, sizeof(key));
      enum ir3_shader_debug ir3_debug_key = ir3_shader_debug_hash_key();
      _mesa_blake3_update(&ctx, &ir3_debug_key, sizeof(ir3_debug_key));
      VkPipelineCreateFlags2KHR hflags = flags &
         (VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_TRIANGLES_BIT_KHR |
          VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_AABBS_BIT_KHR);
      _mesa_blake3_update(&ctx, &hflags, sizeof(hflags));
      uint32_t depth = MIN2(create_info->maxPipelineRayRecursionDepth, 1);
      _mesa_blake3_update(&ctx, &depth, sizeof(depth));
      const uint32_t fov_mode = tu_rt_fov_mode(dev);
      if (fov_mode) {
         static const char fov_tag[] = "tu-rt-fov-1";
         _mesa_blake3_update(&ctx, fov_tag, sizeof(fov_tag));
         _mesa_blake3_update(&ctx, &fov_mode, sizeof(fov_mode));
      }
      static const bool key_sort = debug_get_bool_option("TU_RT_KEY_SORT", false) && tu_rt_use_case_tree();
      uint32_t *key_order = (uint32_t *) malloc(MAX2(pipeline->group_count, 1) * sizeof(uint32_t));
      if (!key_order) {
         result = VK_ERROR_OUT_OF_HOST_MEMORY;
         goto fail;
      }
      for (uint32_t g = 0; g < pipeline->group_count; g++)
         key_order[g] = g;
      if (key_sort)
         qsort_r(key_order, pipeline->group_count, sizeof(uint32_t), tu_rt_group_key_cmp, pipeline);
      if (tu_keylog_enabled()) {
         uint32_t hsum = 0;
         for (uint32_t g = 0; g < pipeline->group_count; g++)
            hsum = hsum * 31u + pipeline->groups[g].handle;
         char line[1024];
         unsigned len = 0;
         for (uint32_t g = 0; g < pipeline->group_count && len + 12 < sizeof(line); g++)
            len += snprintf(line + len, sizeof(line) - len, "%08x ", pipeline->groups[g].handle);
         mesa_logi("tu-key:  groups api-order-sum=%08x sorted=%d first: %s", hsum, (int) key_sort, line);
      }
      for (uint32_t gi = 0; gi < pipeline->group_count; gi++) {
         const uint32_t g = key_order[gi];
         const struct tu_rt_group *grp = pipeline->groups + g;
         _mesa_blake3_update(&ctx, &grp->handle, sizeof(grp->handle));
         uint32_t idx[3] = { grp->recursive_index, grp->ahit_index, grp->isec_index };
         for (unsigned k = 0; k < 3; k++) {
            if (idx[k] == VK_SHADER_UNUSED_KHR) {
               uint32_t none = ~0u;
               _mesa_blake3_update(&ctx, &none, sizeof(none));
            } else {
               _mesa_blake3_update(&ctx, pipeline->stage_hashes + idx[k] * BLAKE3_KEY_LEN,
                                   BLAKE3_KEY_LEN);
            }
         }
      }
      free(key_order);
      _mesa_blake3_final(&ctx, blake3);

      if (tu_keylog_enabled()) {
         blake3_hash kh;
         _mesa_blake3_compute(&key, sizeof(key), kh);
         char hb[17], hl[17], hk[17], kb[2 * sizeof(key) + 1];
         tu_keylog_hex(hb, blake3, 8);
         if (layout)
            tu_keylog_hex(hl, layout->blake3, 8);
         else
            strcpy(hl, "none");
         tu_keylog_hex(hk, kh, 8);
         tu_keylog_hex(kb, (const unsigned char *) &key, sizeof(key));
         mesa_logi("tu-key:  final b3=%s layout=%s keyh=%s dbg=0x%x hflags=0x%llx depth=%u key=%s",
                   hb, hl, hk, (unsigned) ir3_debug_key, (unsigned long long) hflags, depth, kb);
      }

      uint32_t *rg_groups = rzalloc_array(pipeline->mem_ctx, uint32_t,
                                          MAX2(pipeline->group_count, 1));
      uint32_t rg_count = 0;
      for (uint32_t g = 0; g < pipeline->group_count; g++) {
         const struct tu_rt_group *grp = pipeline->groups + g;
         if (grp->recursive_index == VK_SHADER_UNUSED_KHR ||
             pipeline->stage_types[grp->recursive_index] != MESA_SHADER_RAYGEN)
            continue;
         bool dup = false;
         for (uint32_t k = 0; k < rg_count; k++)
            dup |= pipeline->groups[rg_groups[k]].handle == grp->handle;
         if (!dup)
            rg_groups[rg_count++] = g;
      }

      struct tu_shader *shader = NULL;
      nir_shader *nir = NULL;
      const unsigned link_mode = tu_rt_link_mode();
      bool linked = false;
      if (link_mode != TU_RT_LINK_OFF && dev->physical_device->info->chip >= 7 &&
          rg_count >= 1 &&
          !(flags & VK_PIPELINE_CREATE_2_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR) &&
          (link_mode != TU_RT_LINK_AUTO || pipeline->stage_count >= 50)) {
         result = tu_rt_link_variants(dev, cache, pipeline, create_info, flags,
                                      layout, &key, blake3, rg_groups, rg_count,
                                      link_mode == TU_RT_LINK_WHOLE);
         if (result == VK_SUCCESS) {
            linked = true;
            shader = pipeline->variants[0].shader;
            vk_pipeline_cache_object_ref(&shader->base);
         } else if (result != VK_ERROR_FEATURE_NOT_PRESENT) {
            goto fail;
         } else {
            pipeline->variant_count = 0;
            pipeline->variants = NULL;
         }
      }
      if (!linked && rg_count > 1 && !debug_get_bool_option("TU_RT_NO_VARIANTS", false)) {
         result = tu_rt_build_variants(dev, cache, pipeline, create_info, flags,
                                       layout, &key, blake3, rg_groups, rg_count);
         if (result != VK_SUCCESS)
            goto fail;
         shader = pipeline->variants[0].shader;
         vk_pipeline_cache_object_ref(&shader->base);
      } else if (!linked &&
                 !(flags & VK_PIPELINE_CREATE_2_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR)) {
         shader = tu_compute_pipeline_cache_lookup(dev, cache, blake3);
      }
      if (!shader) {
         if (flags & VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT_KHR) {
            result = VK_PIPELINE_COMPILE_REQUIRED;
            goto fail;
         }
         tu_rt_prof_begin();
         void *nir_ctx = ralloc_context(NULL);
         result = tu_rt_compile_stages(dev, cache, pipeline, create_info, flags, &key, nir_ctx);
         if (result != VK_SUCCESS) {
            ralloc_free(nir_ctx);
            goto fail;
         }
         nir = tu_rt_build_shader(dev, pipeline, create_info, flags, -1);
         if (!nir) {
            ralloc_free(nir_ctx);
            tu_rt_prof_end("final-abort");
            result = VK_ERROR_OUT_OF_HOST_MEMORY;
            goto fail;
         }
         ralloc_steal(pipeline->mem_ctx, nir);
         ralloc_free(nir_ctx);
         memset(pipeline->stages, 0, pipeline->stage_count * sizeof(pipeline->stages[0]));
      }

      if (nir)
         tu_rt_prof_mark("init_nir", nir);
      result = tu_compute_pipeline_init_nir(dev, &pipeline->base, layout, cache,
                                            shader, nir, &key, blake3);
      tu_rt_prof_end("final");
      if (result != VK_SUCCESS)
         goto fail;
      for (uint32_t v = 1; v < pipeline->variant_count; v++)
         pipeline->base.base.active_desc_sets |=
            pipeline->variants[v].shader->active_desc_sets;
   }

   if (debug_get_bool_option("TU_RT_LOG", false)) {
      const struct tu_shader *sh = pipeline->base.base.shaders[MESA_SHADER_COMPUTE];
      struct mallinfo2 mi = mallinfo2();
      mesa_logi("tu-rt: pipeline %s stages=%u groups=%u lib=%d %.1f ms instrs=%u regs=%u heap=%zuM",
                (flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR) ? "library" : "final",
                pipeline->stage_count, pipeline->group_count,
                create_info->pLibraryInfo ? (int) create_info->pLibraryInfo->libraryCount : 0,
                (os_time_get_nano() - t_start) / 1e6,
                sh ? sh->variant->info.instrs_count : 0,
                sh ? sh->variant->info.max_reg + 1 : 0,
                (mi.uordblks + mi.hblkhd) >> 20);
      uint32_t logged = pipeline->variants ? pipeline->variant_count : (sh ? 1 : 0);
      for (uint32_t v = 0; v < logged; v++) {
         const struct ir3_shader_variant *iv =
            pipeline->variants ? pipeline->variants[v].shader->variant : sh->variant;
         mesa_logi("tu-rt: variant %u/%u handle=0x%x blake3=%s instrs=%u nops=%u regs=%u half=%d "
                   "stp=%u ldp=%u loops=%u waves=%d wave%u pvt=%u pw=%d bs=%u size=%u",
                   v, logged, pipeline->variants ? pipeline->variants[v].handle : 0, iv->blake3_str,
                   iv->info.instrs_count, iv->info.nops_count, iv->info.max_reg + 1,
                   iv->info.max_half_reg + 1, iv->info.stp_count, iv->info.ldp_count,
                   iv->info.loops, iv->info.max_waves, iv->info.double_threadsize ? 128 : 64,
                   iv->pvtmem_size, iv->pvtmem_per_wave, iv->branchstack, iv->info.size / 8);
      }
   }

   *out_pipeline = tu_pipeline_to_handle(&pipeline->base.base);
   return VK_SUCCESS;

fail:
   tu_pipeline_finish(&pipeline->base.base, dev, alloc);
   vk_object_free(&dev->vk, alloc, pipeline);
   return result;
}

struct tu_rt_create_job {
   struct tu_device *dev;
   struct vk_pipeline_cache *cache;
   const VkRayTracingPipelineCreateInfoKHR *info;
   VkPipelineCreateFlags2KHR flags;
   const VkAllocationCallbacks *alloc;
   VkPipeline *out;
   VkResult result;
};

static void *
tu_rt_create_job_run(void *data)
{
   struct tu_rt_create_job *job = (struct tu_rt_create_job *) data;
   job->result = tu_rt_pipeline_create(job->dev, job->cache, job->info,
                                       job->flags, job->alloc, job->out);
   if (debug_get_bool_option("TU_RT_TRIM", true)) {
      if (debug_get_bool_option("TU_RT_LOG", false)) {
         struct mallinfo2 a = mallinfo2();
         malloc_trim(0);
         struct mallinfo2 b = mallinfo2();
         mesa_logi("tu-rt: trim used=%zuM free=%zuM -> free=%zuM arena=%zuM",
                   (a.uordblks + a.hblkhd) >> 20, a.fordblks >> 20, b.fordblks >> 20, b.arena >> 20);
      } else {
         malloc_trim(0);
      }
   }
   return NULL;
}

static void
tu_rt_run_on_big_stack(struct tu_rt_create_job *job)
{
   pthread_attr_t attr;
   pthread_t thread;
   if (pthread_attr_init(&attr) == 0) {
      bool ok = pthread_attr_setstacksize(&attr, 256u << 20) == 0 &&
                pthread_create(&thread, &attr, tu_rt_create_job_run, job) == 0;
      pthread_attr_destroy(&attr);
      if (ok) {
         pthread_join(thread, NULL);
         return;
      }
   }
   tu_rt_create_job_run(job);
}

VKAPI_ATTR VkResult VKAPI_CALL
tu_CreateRayTracingPipelinesKHR(VkDevice _device,
                                VkDeferredOperationKHR deferredOperation,
                                VkPipelineCache pipelineCache,
                                uint32_t createInfoCount,
                                const VkRayTracingPipelineCreateInfoKHR *pCreateInfos,
                                const VkAllocationCallbacks *pAllocator,
                                VkPipeline *pPipelines)
{
   VK_FROM_HANDLE(tu_device, dev, _device);
   VK_FROM_HANDLE(vk_pipeline_cache, cache, pipelineCache);
   VkResult result = VK_SUCCESS;

   uint32_t i = 0;
   for (; i < createInfoCount; i++) {
      VkPipelineCreateFlags2KHR flags =
         vk_rt_pipeline_create_flags(&pCreateInfos[i]);
      struct tu_rt_create_job job = { dev, cache, &pCreateInfos[i], flags,
                                      pAllocator, &pPipelines[i],
                                      VK_ERROR_OUT_OF_HOST_MEMORY };
      tu_rt_run_on_big_stack(&job);
      VkResult r = job.result;
      if (r != VK_SUCCESS) {
         result = r;
         pPipelines[i] = VK_NULL_HANDLE;
         if (flags & VK_PIPELINE_CREATE_2_EARLY_RETURN_ON_FAILURE_BIT_KHR)
            break;
      }
   }

   for (; i < createInfoCount; i++)
      pPipelines[i] = VK_NULL_HANDLE;

   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
tu_GetRayTracingShaderGroupHandlesKHR(VkDevice device,
                                      VkPipeline _pipeline,
                                      uint32_t firstGroup,
                                      uint32_t groupCount,
                                      size_t dataSize,
                                      void *pData)
{
   VK_FROM_HANDLE(tu_pipeline, base, _pipeline);
   struct tu_ray_tracing_pipeline *pipeline = tu_pipeline_to_ray_tracing(base);

   uint8_t *data = (uint8_t *) pData;
   memset(data, 0, dataSize);
   for (uint32_t i = 0; i < groupCount; i++) {
      memcpy(data + i * TU_RT_GROUP_HANDLE_SIZE,
             &pipeline->groups[firstGroup + i].handle, sizeof(uint32_t));
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
tu_GetRayTracingCaptureReplayShaderGroupHandlesKHR(VkDevice device,
                                                   VkPipeline pipeline,
                                                   uint32_t firstGroup,
                                                   uint32_t groupCount,
                                                   size_t dataSize,
                                                   void *pData)
{
   memset(pData, 0, dataSize);
   return VK_SUCCESS;
}

VKAPI_ATTR VkDeviceSize VKAPI_CALL
tu_GetRayTracingShaderGroupStackSizeKHR(VkDevice device,
                                        VkPipeline pipeline,
                                        uint32_t group,
                                        VkShaderGroupShaderKHR groupShader)
{
   return 4;
}

VKAPI_ATTR void VKAPI_CALL
tu_CmdSetRayTracingPipelineStackSizeKHR(VkCommandBuffer commandBuffer,
                                        uint32_t pipelineStackSize)
{
}

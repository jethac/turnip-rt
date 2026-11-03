rd_entrypoints = custom_target(
  'rd_entrypoints',
  input : [vk_entrypoints_gen, vk_api_xml],
  output : ['rd_entrypoints.h', 'rd_entrypoints.c'],
  command : [prog_python, '@INPUT0@', '--xml', '@INPUT1@', '--proto', '--weak',
             '--out-h', '@OUTPUT0@', '--out-c', '@OUTPUT1@', '--prefix', 'rd',
             '--beta', with_vulkan_beta.to_string()],
  depend_files : vk_entrypoints_gen_depend_files,
)

libvulkan_radish = shared_library(
  'vulkan_radish',
  ['rd_instance.c', 'rd_device.c', 'rd_memory.c', 'rd_descriptor.c', 'rd_pipeline.c', 'rd_pipeline_cache.cpp',
   'rd_command.c', 'rd_sampling.c', 'rd_nir_sampling.c', 'rd_nir_resources.c', 'rd_nir_robust.c', 'rd_resources.c', 'rd_transfer.c', 'rd_image.c',
   'rd_event.c', 'rd_query.c', 'rd_queue.c', 'rd_trace.c', rd_entrypoints, freedreno_xml_header_files],
  include_directories : [inc_freedreno, inc_include, inc_src],
  link_with : [libfreedreno_drm, libfreedreno_ir3, libfreedreno_layout],
  dependencies : [idep_libfreedreno_common, idep_vulkan_util,
                  idep_vulkan_runtime, idep_mesautil, dep_libdrm, dep_thread,
                  dep_dl, dep_m],
  c_args : [no_override_init_args],
  gnu_symbol_visibility : 'hidden',
  link_args : [vulkan_icd_link_args, ld_args_bsymbolic, ld_args_gc_sections],
  link_depends : vulkan_icd_link_depends,
  install : true,
)

custom_target(
  'radish_icd',
  input : [vk_icd_gen, vk_api_xml],
  output : 'radish_icd.' + vulkan_manifest_suffix,
  command : [prog_python, '@INPUT0@', '--api-version', '1.1', '--xml', '@INPUT1@',
             '--sizeof-pointer', sizeof_pointer,
             '--icd-lib-path', vulkan_icd_lib_path,
             '--icd-filename', 'libvulkan_radish.so', '--out', '@OUTPUT@'],
  build_by_default : true,
  install : true,
  install_dir : with_vulkan_icd_dir,
)

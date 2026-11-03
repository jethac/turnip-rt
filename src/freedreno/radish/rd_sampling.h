#ifndef RD_SAMPLING_H
#define RD_SAMPLING_H
#include "rd_image.h"
#include "util/format/u_format.h"
struct nir_shader;
struct rd_device;
struct rd_dispatch;
struct rd_pipeline;
struct rd_sampling_level { uint32_t offset, row_pitch, array_pitch, width, height; };
struct rd_sampling_view {
   VkFormat format;
   VkComponentMapping swizzle;
   uint32_t levels, layers, cpp, samples;
   struct rd_sampling_level level[RD_IMAGE_MAX_LEVELS];
};
struct rd_sampling_key {
   uint32_t mask;
   uint32_t samplers[16][4];
   struct rd_sampling_view views[16];
};
struct rd_sampling_entry {
   struct rd_sampling_entry *next;
   struct rd_sampling_key key;
   struct rd_pipeline *pipeline;
};
bool rd_nir_lower_array_fetch(struct nir_shader *nir, const struct rd_sampling_key *key,
                              uint32_t metadata_buffer);
VkResult rd_sampling_compile(struct rd_device *device, struct rd_pipeline *source,
                             const struct rd_sampling_key *key, struct rd_pipeline **out);
VkResult rd_prepare_sampling(struct rd_device *device, struct rd_dispatch *dispatch);
#endif

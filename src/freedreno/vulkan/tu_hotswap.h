/*
 * Copyright © 2026 Local verification
 * SPDX-License-Identifier: MIT
 */

#ifndef TU_HOTSWAP_H
#define TU_HOTSWAP_H

#include "tu_common.h"

#define TU_HOTSWAP_MAX_VARIANTS 32

/* Optional per-alternative dispatch-grid rule (TU_HOTSWAP_GRID "<target>.<alt>=xhalf[:GXxGY]").
 * xdiv == 0: no rule.  Applied by vkCmdDispatch(Base) to direct dispatches with a
 * zero base group while the alternative is bound: X groups := ceil(X / xdiv),
 * only if gx == 0 or the recorded dims equal gx x gy. */
struct tu_hotswap_grid {
   uint16_t xdiv;
   int16_t target;
   uint32_t gx, gy;
};

struct tu_hotswap_pipe {
   int target;
   uint64_t dxil;
   VkPipeline variant[TU_HOTSWAP_MAX_VARIANTS];
   uint32_t alias;
};

bool tu_hotswap_enabled(void);

int tu_hotswap_match(const VkPipelineShaderStageCreateInfo *stage,
                     mesa_shader_stage st);

typedef VkResult (*tu_hotswap_create_cb)(void *data, VkShaderModule module,
                                         VkPipeline *out);

void tu_hotswap_build(struct tu_device *dev, struct tu_pipeline *base,
                      int target, const VkPipelineShaderStageCreateInfo *stage,
                      mesa_shader_stage st,
                      const struct tu_pipeline_layout *layout,
                      const void *feedback_pnext,
                      const VkAllocationCallbacks *alloc,
                      tu_hotswap_create_cb create, void *data);

typedef VkResult (*tu_hotswap_link_cb)(void *data, unsigned lib_index,
                                       VkPipeline lib_variant, VkPipeline *out);

void tu_hotswap_build_linked(struct tu_device *dev, struct tu_pipeline *base,
                             const struct tu_pipeline *lib, unsigned lib_index,
                             const VkAllocationCallbacks *alloc,
                             tu_hotswap_link_cb create, void *data);

struct tu_pipeline *tu_hotswap_bind(struct tu_device *dev,
                                    struct tu_pipeline *base,
                                    struct tu_hotswap_grid *grid);

void tu_hotswap_grid_applied(const struct tu_hotswap_grid *grid);

void tu_hotswap_destroy(struct tu_device *dev, struct tu_pipeline *base,
                        const VkAllocationCallbacks *alloc);

#endif /* TU_HOTSWAP_H */

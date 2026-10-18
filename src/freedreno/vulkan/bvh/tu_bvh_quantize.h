/*
 * Copyright © 2022 Friedrich Vock
 * SPDX-License-Identifier: MIT
 */

#ifndef TU_BVH_QUANTIZE_H
#define TU_BVH_QUANTIZE_H

/* Truncate to bfloat16 while rounding down. bfloat16 is used to store the bases.
 */

u16vec3 to_bfloat_round_down(vec3 coord)
{
   u32vec3 icoord = floatBitsToUint(coord);
   return u16vec3(mix(icoord >> 16, (icoord + 0xffff) >> 16, notEqual(icoord & u32vec3(0x80000000), u32vec3(0))));
}

/* Approximate subtraction while rounding up. Return a result greater than or
 * equal to the infinitely-precise result. This just uses the native
 * subtraction and then shifts one ULP towards infinity. Because the result is
 * further rounded, it should usually be good enough while being faster than
 * emulated floating-point math.
 *
 * We assume here that the result is always nonnegative, because it's only used
 * to subtract away the base.
 */

vec3 subtract_round_up_approx(vec3 a, vec3 b)
{
   vec3 f = a - b;
   u32vec3 i = floatBitsToUint(f);

   i++;

   /* Handle infinity/zero special cases */
   i = mix(i, floatBitsToUint(f), isinf(f));
   i = mix(i, floatBitsToUint(f), equal(f, vec3(0)));

   return uintBitsToFloat(i);
}

vec3 subtract_round_down_approx(vec3 a, vec3 b)
{
   vec3 f = a - b;
   u32vec3 i = floatBitsToUint(f);

   i--;

   /* Handle infinity/zero special cases */
   i = mix(i, floatBitsToUint(f), isinf(f));
   i = mix(i, floatBitsToUint(f), equal(f, vec3(0)));

   return uintBitsToFloat(i);
}

u32vec3 extract_mantissa(vec3 f)
{
   return mix((floatBitsToUint(f) & 0x7fffff) | 0x800000, u32vec3(0), equal(f, vec3(0)));
}

void
encode_node_base(REF(tu_internal_node) dst_node, vec3 min_offset, vec3 max_offset,
                 out vec3 base_float, out u16vec3 exponents)
{
   u16vec3 base_bfloat = to_bfloat_round_down(min_offset);
   base_float = uintBitsToFloat(u32vec3(base_bfloat) << 16);
   DEREF(dst_node).bases[0] = base_bfloat.x;
   DEREF(dst_node).bases[1] = base_bfloat.y;
   DEREF(dst_node).bases[2] = base_bfloat.z;

   vec3 children_max = subtract_round_up_approx(max_offset, base_float);

   /* The largest child offset will be encoded in 8 bits, including the
    * explicit leading 1. We need to downcast to this precision while rounding
    * up to catch cases where the exponent is increased by rounding up, then
    * extract the exponent. Because children_max is always nonnegative, we can
    * do the downcast with "(floatBitsToUint(children_max) + 0xffff) >> 16",
    * and then we further shift to get the rounded exponent.
    */
   exponents = u16vec3((floatBitsToUint(children_max) + 0xffff) >> 23);
   u8vec3 exponents_u8 = u8vec3(exponents);
   DEREF(dst_node).exponents[0] = exponents_u8.x;
   DEREF(dst_node).exponents[1] = exponents_u8.y;
   DEREF(dst_node).exponents[2] = exponents_u8.z;
}

void
encode_child_box(REF(tu_internal_node) dst_node, uint32_t i, vec3 base_float, u16vec3 exponents,
                 vk_aabb child_aabb)
{
   /* Note: because we subtract from the minimum, we should never have a
    * negative value here.
    */
   vec3 child_min = subtract_round_down_approx(child_aabb.min, base_float);
   vec3 child_max = subtract_round_up_approx(child_aabb.max, base_float);

   u16vec3 child_min_exponents = u16vec3(floatBitsToUint(child_min) >> 23);
   u16vec3 child_max_exponents = u16vec3(floatBitsToUint(child_max) >> 23);

   u16vec3 child_min_shift = u16vec3(16) + exponents - child_min_exponents;
   /* Divide the mantissa by 2**child_min_shift, rounding down */
   u8vec3 child_min_mantissas =
      mix(u8vec3(extract_mantissa(child_min) >> child_min_shift), u8vec3(0),
          greaterThanEqual(child_min_shift, u16vec3(32)));
   u16vec3 child_max_shift = u16vec3(16) + exponents - child_max_exponents;
   /* Divide the mantissa by 2**child_max_shift, rounding up */
   u8vec3 child_max_mantissas =
      mix(u8vec3((extract_mantissa(child_max) + ((u32vec3(1u) << u32vec3(child_max_shift)) - 1)) >> child_max_shift),
          u8vec3(notEqual(extract_mantissa(child_max), u32vec3(0))),
          greaterThanEqual(child_max_shift, u16vec3(32)));

   DEREF(dst_node).mantissas[i][0][0] = child_min_mantissas.x;
   DEREF(dst_node).mantissas[i][0][1] = child_min_mantissas.y;
   DEREF(dst_node).mantissas[i][0][2] = child_min_mantissas.z;
   DEREF(dst_node).mantissas[i][1][0] = child_max_mantissas.x;
   DEREF(dst_node).mantissas[i][1][1] = child_max_mantissas.y;
   DEREF(dst_node).mantissas[i][1][2] = child_max_mantissas.z;
}

#endif

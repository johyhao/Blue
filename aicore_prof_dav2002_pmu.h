/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aicore_prof_dav2002_pmu.h
 * \brief 310P3 (DAV_2002) PMU register offsets.
 *
 * Register layout source: inc/soc/cloud_v1/aic_sc_reg_reg_offset.h + device .ko disassembly +
 *                        board verification (310p_hardware_address.md §6.4 / §11.4).
 *
 * Key differences from DAV_2201:
 *   - PMU_CNTx_IDX offsets: 0x260~0x298 (stride 8) vs DAV_2201's 0x1280~0x129C (stride 4)
 *   - No PMU_CNT_TOTAL1 (0x254 absent; 0x258 is PMU_MIN_OV_CNT)
 *   - No PMU_START_CNT_CYC_1 (0x2A4) / PMU_STOP_CNT_CYC_1 (0x2AC)
 *   - No PMU_CTRL_1
 */

#ifndef AICORE_PROF_DAV2002_PMU_H
#define AICORE_PROF_DAV2002_PMU_H

#include <cstdint>

namespace npu::tile_fwk::dynamic {

namespace DAV_2002 {
const uint32_t PMU_CTRL_0 = 0x200;
const uint32_t PMU_CNT0 = 0x210;
const uint32_t PMU_CNT1 = 0x218;
const uint32_t PMU_CNT2 = 0x220;
const uint32_t PMU_CNT3 = 0x228;
const uint32_t PMU_CNT4 = 0x230;
const uint32_t PMU_CNT5 = 0x238;
const uint32_t PMU_CNT6 = 0x240;
const uint32_t PMU_CNT7 = 0x248;
const uint32_t PMU_CNT_TOTAL0 = 0x250;
const uint32_t PMU_CNT0_IDX = 0x260;
const uint32_t PMU_CNT1_IDX = 0x268;
const uint32_t PMU_CNT2_IDX = 0x270;
const uint32_t PMU_CNT3_IDX = 0x278;
const uint32_t PMU_CNT4_IDX = 0x280;
const uint32_t PMU_CNT5_IDX = 0x288;
const uint32_t PMU_CNT6_IDX = 0x290;
const uint32_t PMU_CNT7_IDX = 0x298;
const uint32_t PMU_START_CNT_CYC_0 = 0x2A0;
const uint32_t PMU_STOP_CNT_CYC_0 = 0x2A8;
}; // namespace DAV_2002

} // namespace npu::tile_fwk::dynamic

#endif // AICORE_PROF_DAV2002_PMU_H

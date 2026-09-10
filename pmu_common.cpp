/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pmu_common.cpp
 * \brief
 */

#include "machine/runtime/runner/pmu_common.h"
#include <string>
#include "tilefwk/pypto_fwk_log.h"
#include "interface/utils/common.h"

namespace npu::tile_fwk {
namespace {
// pmu event type
constexpr int32_t ARITHMETIC_UTILIZATION = 1;
constexpr int32_t PIPE_UTILIZATION = 2;
constexpr int32_t MEMORY = 4;
constexpr int32_t MEMORY_L0 = 5;
constexpr int32_t RESOURCE_CONFLICT_RATION = 6;
constexpr int32_t MEMORY_UB = 7;
constexpr int32_t L2_CACHE = 8;

constexpr int PMU_EVENT_TYPE_MAX_DAV2201 = 8;
constexpr int PMU_EVENT_TYPE_MAX_DAV3510 = 10;

void SetPmuEventTypeDAV2201(int32_t profPmuType, std::vector<int64_t>& pmuEvtType)
{
    // 按照环境变量设置的数值，获取pmu事件类型
    switch (profPmuType) {
        case ARITHMETIC_UTILIZATION:
            pmuEvtType = {0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f, 0x0};
            break;
        case PIPE_UTILIZATION:
            pmuEvtType = {0x08, 0x0a, 0x09, 0x0b, 0x0c, 0x0d, 0x55, 0x54};
            break;
        case MEMORY:
            pmuEvtType = {0x15, 0x16, 0x31, 0x32, 0x0f, 0x10, 0x12, 0x13};
            break;
        case MEMORY_L0:
            pmuEvtType = {0x1b, 0x1c, 0x21, 0x22, 0x27, 0x28, 0x0, 0x0};
            break;
        case RESOURCE_CONFLICT_RATION:
            pmuEvtType = {0x64, 0x65, 0x66, 0x0, 0x0, 0x0, 0x0, 0x0};
            break;
        case MEMORY_UB:
            pmuEvtType = {0x3d, 0x10, 0x13, 0x3e, 0x43, 0x44, 0x37, 0x38};
            break;
        case L2_CACHE:
            pmuEvtType = {0x500, 0x502, 0x504, 0x506, 0x508, 0x50a, 0x0, 0x0};
            break;
        default:
            MACHINE_LOGW("Invalid profPmuType %d, only support [1,2,4,5,6,7,8].\n", profPmuType);
    }
}

void SetPmuEventTypeDAV3510(int32_t profPmuType, std::vector<int64_t>& pmuEvtType)
{
    // 按照环境变量设置的数值，获取pmu事件类型
    switch (profPmuType) {
        case ARITHMETIC_UTILIZATION:
            pmuEvtType = {0x323, 0x324, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
            break;
        case PIPE_UTILIZATION:
            pmuEvtType = {0x501, 0x301, 0x1, 0x701, 0x202, 0x203, 0x34, 0x35, 0x714, 0x0};
            break;
        case MEMORY:
            pmuEvtType = {0x0, 0x0, 0x400, 0x401, 0x56f, 0x571, 0x570, 0x572, 0x707, 0x709};
            break;
        case MEMORY_L0:
            pmuEvtType = {0x304, 0x703, 0x306, 0x705, 0x712, 0x30a, 0x308, 0x0, 0x0, 0x0};
            break;
        case RESOURCE_CONFLICT_RATION:
            pmuEvtType = {0x3556, 0x3540, 0x3502, 0x3528, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
            break;
        case MEMORY_UB:
            pmuEvtType = {0x3, 0x5, 0x70c, 0x206, 0x204, 0x571, 0x572, 0x0, 0x0, 0x0};
            break;
        case L2_CACHE:
            pmuEvtType = {0x424, 0x425, 0x426, 0x42a, 0x42b, 0x42c, 0x0, 0x0, 0x0, 0x0};
            break;
        default:
            MACHINE_LOGW("Invalid profPmuType %d, only support [1,2,4,5,6,7,8].\n", profPmuType);
    }
}
} // namespace

void PmuCommon::InitPmuEventType(const ArchInfo& archInfo, std::vector<int64_t>& pmuEvtType)
{
    size_t pmuEvtTypeSize = (archInfo == ArchInfo::DAV_3510) ? PMU_EVENT_TYPE_MAX_DAV3510 : PMU_EVENT_TYPE_MAX_DAV2201;
    pmuEvtType.resize(pmuEvtTypeSize, 0x0);
    // 获取pmu事件类型环境变量获取方式
    std::string eventTypeStr = GetEnvVar("PROF_PMU_EVENT_TYPE");
    if (eventTypeStr.empty()) {
        MACHINE_LOGW("Dont support PROF_PMU_EVENT_TYPE env, use default pmu event type PIPE_UTILIZATION.\n");
        eventTypeStr = "2";
    }
    int32_t profPmuType = PIPE_UTILIZATION;
    try {
        profPmuType = std::stoi(eventTypeStr);
    } catch (const std::exception& e) {
        MACHINE_LOGW(
            "Invalid PROF_PMU_EVENT_TYPE value [%s], use default PIPE_UTILIZATION. error: %s", eventTypeStr.c_str(),
            e.what());
    }

    if (archInfo == ArchInfo::DAV_2201 || archInfo == ArchInfo::DAV_2002) {
        SetPmuEventTypeDAV2201(profPmuType, pmuEvtType);
    } else if (archInfo == ArchInfo::DAV_3510) {
        SetPmuEventTypeDAV3510(profPmuType, pmuEvtType);
    }
}
} // namespace npu::tile_fwk

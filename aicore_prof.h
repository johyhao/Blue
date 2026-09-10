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
 * \file aicore_prof.h
 * \brief
 */

#ifndef AICORE_PROF_H
#define AICORE_PROF_H

#include <vector>
#include <map>
#include <sstream>
#include <string>
#include <sys/syscall.h>
#include "tilefwk/aicpu_common.h"
#include "machine/device/dynamic/aicore_prof_dav3510_pmu.h"
#include "machine/device/dynamic/aicore_prof_dav2002_pmu.h"

typedef void* VOID_PTR;
typedef int32_t (*ProfCommandHandle)(uint32_t type, void *data, uint32_t len);
using PyptoProfGetFuncPtr = void(*)(void*);
extern "C" {
__attribute__((weak)) int32_t AdprofReportAdditionalInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length);
#ifndef MSVP_PROF_API
__attribute__((weak)) int32_t MsprofReportAdditionalInfo(uint32_t nonPersistantFlag, const VOID_PTR data, uint32_t length);
__attribute__((weak)) int32_t MsprofRegisterCallback(uint32_t moduleId, ProfCommandHandle handle);
#endif
__attribute__((weak)) int32_t AdprofCheckFeatureIsOn(uint64_t feature);
};

typedef int32_t (*ProfReportAdditionalInfoFunc)(uint32_t agingFlag, const VOID_PTR data, uint32_t length);

namespace npu::tile_fwk::dynamic {
class AiCoreManager;
constexpr uint32_t PATH_SIZE = 1024;
constexpr uint32_t PROF_DATA_SIZE = 4096;
constexpr uint32_t DEVICE_ID_LIST_SIZE = 64;
struct PyPtoMsprofAdditionalInfo { // for MsprofReportAdditionalInfo buffer data
    uint16_t magicNumber = 0x5A5AU;
    uint16_t level;
    uint32_t type;
    uint32_t threadId;
    uint32_t dataLen;
    uint64_t timeStamp;
    uint8_t data[232];
};

struct PyptoProfDataparam {
    int32_t coreIdx;
    uint32_t subGraphId;
    uint32_t taskId;
    const struct TaskStat *taskStat;
};

struct PyPtoMsprofCommandHandleParams {
    uint32_t pathLen;
    uint32_t storageLimit;  // MB
    uint32_t profDataLen;
    char path[PATH_SIZE];
    char profData[PROF_DATA_SIZE];
};

struct PyPtoMsprofCommandHandle {
    uint64_t profSwitch;
    uint64_t profSwitchHi;
    uint32_t devNums;
    uint32_t devIdList[DEVICE_ID_LIST_SIZE];
    uint32_t modelId;
    uint32_t type;
    uint32_t cacheFlag;
    struct PyPtoMsprofCommandHandleParams params;
};

constexpr uint32_t PYPTO_MSPROF_REPORT_AICPU_LEVEL = 6000U;
constexpr uint32_t PYPTO_MSPROF_REPORT_AICPU_NODE_TYPE = 10U; /* type info: DATA_PREPROCESS.AICPU */

constexpr uint64_t PROF_TASK_TIME_L0 = 0x00000008ULL;
constexpr uint64_t PROF_TASK_TIME_L1 = 0x00000010ULL;
constexpr uint64_t PROF_TASK_TIME_L2 = 0x00000020ULL;
constexpr uint64_t PROF_TASK_TIME_L3 = 0x00000040ULL;

constexpr bool GLB_PMU_EN = true;
constexpr bool USER_PMU_MODE_EN = (GLB_PMU_EN && true);
constexpr bool SAMPLE_PMU_MODE_EN = (GLB_PMU_EN && USER_PMU_MODE_EN);
constexpr bool DUAL_PAGE_EN = false; // 双页表是否使能，如何感知？？？通过runtime？
constexpr uint32_t MAX_PMU_CNT = 8;

constexpr int32_t PMU_CYCLE = 80; // 记录按照了20MHZ的时钟周期，单位归一按照1600MHZ的时钟周期进行统一，所以80
constexpr int64_t NUM_TWO_PMU = 2;

typedef enum AiCoreProfLevel {
    PROF_LEVEL_OFF = 0,
    PROF_LEVEL_FUNC = 1,

    PROF_LEVEL_FUNC_LOG = 2,
    PROF_LEVEL_FUNC_LOG_PMU = 3,
    PROF_LEVEL_MAX = 3
} AiCoreProfLevel;

typedef enum AiCoreProfDataType {
    PROF_DATATYPE_UNKNOWN = 0,
    PROF_DATATYPE_FUNC = 1,
    PROF_DATATYPE_LOG = 2,
    PROF_DATATYPE_PMU = 3,
    PROF_DATATYPE_HAND_SHAKE = 4,
    PROF_DATATYPE_EXE = 5
} AiCoreProfDataType;

typedef enum AiCoreRegister {
    PMU_CTRL_0 = 0x200,
    PMU_CNT0 = 0x210,
    PMU_CNT1 = 0x218,
    PMU_CNT2 = 0x220,
    PMU_CNT3 = 0x228,
    PMU_CNT4 = 0x230,
    PMU_CNT5 = 0x238,
    PMU_CNT6 = 0x240,
    PMU_CNT7 = 0x248,
    PMU_CNT_TOTAL0 = 0x250,
    PMU_CNT_TOTAL1 = 0x254,
    PMU_CNT0_IDX = 0x1280,
    PMU_CNT1_IDX = 0x1284,
    PMU_CNT2_IDX = 0x1288,
    PMU_CNT3_IDX = 0x128C,
    PMU_CNT4_IDX = 0x1290,
    PMU_CNT5_IDX = 0x1294,
    PMU_CNT6_IDX = 0x1298,
    PMU_CNT7_IDX = 0x129C,
    PMU_START_CNT_CYC_0 = 0x2A0,
    PMU_START_CNT_CYC_1 = 0x2A4,
    PMU_STOP_CNT_CYC_0 = 0x2A8,
    PMU_STOP_CNT_CYC_1 = 0x2AC,
} AiCoreRegister;

struct ArchPmuConfig {
    std::vector<uint32_t> pmuCntIdxOffsets;
    std::vector<uint32_t> pmuCntOffsets;
    uint32_t pmuCntTotal0Offset;
    uint32_t pmuCntTotal1Offset;
    uint32_t ctrl0Offset;
    uint32_t ctrl1Offset;
    uint32_t startCntCyc0Offset;
    uint32_t startCntCyc1Offset;
    uint32_t stopCntCyc0Offset;
    uint32_t stopCntCyc1Offset;
    uint32_t ctrl0Val;
    uint32_t ctrl1Val;
};

inline const std::map<ArchInfo, ArchPmuConfig> kArchPmuConfigs = {
    {ArchInfo::DAV_2201,
     {{PMU_CNT0_IDX, PMU_CNT1_IDX, PMU_CNT2_IDX, PMU_CNT3_IDX, PMU_CNT4_IDX, PMU_CNT5_IDX, PMU_CNT6_IDX, PMU_CNT7_IDX},
      {PMU_CNT0, PMU_CNT1, PMU_CNT2, PMU_CNT3, PMU_CNT4, PMU_CNT5, PMU_CNT6, PMU_CNT7},
      PMU_CNT_TOTAL0,
      PMU_CNT_TOTAL1,
      PMU_CTRL_0,
      0,
      PMU_START_CNT_CYC_0,
      PMU_START_CNT_CYC_1,
      PMU_STOP_CNT_CYC_0,
      PMU_STOP_CNT_CYC_1,
      GLB_PMU_EN + (USER_PMU_MODE_EN << 1) + (SAMPLE_PMU_MODE_EN << NUM_TWO_PMU),
      0}},
    {ArchInfo::DAV_2002,
     {{DAV_2002::PMU_CNT0_IDX, DAV_2002::PMU_CNT1_IDX, DAV_2002::PMU_CNT2_IDX, DAV_2002::PMU_CNT3_IDX,
       DAV_2002::PMU_CNT4_IDX, DAV_2002::PMU_CNT5_IDX, DAV_2002::PMU_CNT6_IDX, DAV_2002::PMU_CNT7_IDX},
      {DAV_2002::PMU_CNT0, DAV_2002::PMU_CNT1, DAV_2002::PMU_CNT2, DAV_2002::PMU_CNT3,
       DAV_2002::PMU_CNT4, DAV_2002::PMU_CNT5, DAV_2002::PMU_CNT6, DAV_2002::PMU_CNT7},
      DAV_2002::PMU_CNT_TOTAL0,
      DAV_2002::PMU_CNT_TOTAL1,
      DAV_2002::PMU_CTRL_0,
      0,
      DAV_2002::PMU_START_CNT_CYC_0,
      DAV_2002::PMU_START_CNT_CYC_1,
      DAV_2002::PMU_STOP_CNT_CYC_0,
      DAV_2002::PMU_STOP_CNT_CYC_1,
      GLB_PMU_EN + (USER_PMU_MODE_EN << 1) + (SAMPLE_PMU_MODE_EN << NUM_TWO_PMU),
      0}},
    {ArchInfo::DAV_3510,
     {{DAV_3510::PMU_CNT0_IDX, DAV_3510::PMU_CNT1_IDX, DAV_3510::PMU_CNT2_IDX, DAV_3510::PMU_CNT3_IDX,
       DAV_3510::PMU_CNT4_IDX, DAV_3510::PMU_CNT5_IDX, DAV_3510::PMU_CNT6_IDX, DAV_3510::PMU_CNT7_IDX,
       DAV_3510::PMU_CNT8_IDX, DAV_3510::PMU_CNT9_IDX},
      {DAV_3510::PMU_CNT0, DAV_3510::PMU_CNT1, DAV_3510::PMU_CNT2, DAV_3510::PMU_CNT3, DAV_3510::PMU_CNT4,
       DAV_3510::PMU_CNT5, DAV_3510::PMU_CNT6, DAV_3510::PMU_CNT7, DAV_3510::PMU_CNT8, DAV_3510::PMU_CNT9},
      DAV_3510::PMU_CNT_TOTAL0,
      DAV_3510::PMU_CNT_TOTAL1,
      DAV_3510::PMU_CTRL_0,
      DAV_3510::PMU_CTRL_1,
      DAV_3510::PMU_START_CNT_CYC_0,
      DAV_3510::PMU_START_CNT_CYC_1,
      DAV_3510::PMU_STOP_CNT_CYC_0,
      DAV_3510::PMU_STOP_CNT_CYC_1,
      USER_PMU_MODE_EN + (SAMPLE_PMU_MODE_EN << 1),
      GLB_PMU_EN}}};

typedef enum AiCorePmuEvent {
    VEC_BUSY_CYCLE = 0x8,
    SU_BUSY_CYCLE = 0x9,
    CUBE_BUSY_CYCLE = 0xA,
    MTE2_BUSY_CYCLE = 0xC,
    MTE3_BUSY_CYCLE = 0xD,

    ICACHE_REQ_CNT = 0x54,
    ICACHE_MISS_CNT = 0x55,
    DCACHE_HIT_CNT = 0xA6,
    DCACHE_MISS_CNT = 0xA7,

    FIXP_L1_WR_CYCLE = 0x206,
    FIXP_L1_RD_CYCLE = 0x209,
    FIXP_L0C_RD_CYCLE = 0x20C,
    FIXP_BUSY_CYCLE = 0x20D,
    FIXP_ACTIVE_CYCLE = 0x303, // FIXP busy minus hset/hwait time

    L2_WR_HIT_CNT = 0x500,
    L2_WR_MISS_CNT = 0x501, // L2BUF_HIT, ineffective
    L2_WR_MISS_ALLOC_CNT = 0x502,
    L2_WR_MISS_NON_ALLOC_CNT = 0x503,
    L2_R0_HIT_CNT = 0x504,
    L2_R0_MISS_CNT = 0x505,
    L2_R0_MISS_ALLOC_CNT = 0x506,
    L2_R0_MISS_NON_ALLOC_CNT = 0x507,
    L2_R1_HIT_CNT = 0x508,
    L2_R1_MISS_CNT = 0x509,
    L2_R1_MISS_ALLOC_CNT = 0x50A,
    L2_R1_MISS_NON_ALLOC_CNT = 0x50B,
} AiCorePmuEvent;

const uint16_t AIV_EVENT_LIST[MAX_PMU_CNT] = {
    MTE3_BUSY_CYCLE, L2_WR_HIT_CNT,        L2_WR_MISS_ALLOC_CNT, MTE2_BUSY_CYCLE,
    L2_R0_HIT_CNT,   L2_R0_MISS_ALLOC_CNT, L2_R1_HIT_CNT,        L2_R0_MISS_ALLOC_CNT,
};

const uint16_t AIC_EVENT_LIST[MAX_PMU_CNT] = {
    FIXP_BUSY_CYCLE, L2_WR_HIT_CNT,        L2_WR_MISS_ALLOC_CNT, MTE2_BUSY_CYCLE,
    L2_R0_HIT_CNT,   L2_R0_MISS_ALLOC_CNT, L2_R1_HIT_CNT,        L2_R0_MISS_ALLOC_CNT,
};

struct MsprofAicpuPyPtoPmuData {
    uint32_t seqNo{0};
    uint32_t taskId{0};
    uint64_t totalCyc{0};
    uint32_t pmuCnt0{0}; // 单个task不能超过3s, 按50MHZ计算
    uint32_t pmuCnt1{0};
    uint32_t pmuCnt2{0};
    uint32_t pmuCnt3{0};
    uint32_t pmuCnt4{0};
    uint32_t pmuCnt5{0};
    uint32_t pmuCnt6{0};
    uint32_t pmuCnt7{0};
    uint32_t pmuCnt8{0};
    uint32_t pmuCnt9{0};
};

// !!注意和 TaskStat 前面的数据区保持一致
struct MsprofAicpuPyPtoLogData {
    int16_t seqNo;
    int16_t subGraphId;
    int32_t taskId{0};
    int64_t execStart{0};
    int64_t execEnd{0};
};

struct MsprofAicpuPyPtoHead {
    uint16_t magicNumber = 0x6BD3U;
    uint16_t coreId : 7;
    uint16_t coreType : 3;
    uint16_t dataType : 6;
    uint32_t taskId;
    uint16_t streamId;
    uint8_t cnt;
    uint8_t rsv[29];
};

using MsprofAicpuPyPtoPmuHead = struct MsprofAicpuPyPtoHead;
using MsprofAicpuPyPtoLogHead = struct MsprofAicpuPyPtoHead;

bool ProfCheckLevel(uint64_t feature);
AiCoreProfLevel CreateProfLevel(ProfConfig profConfig);

class AiCoreProf {
public:
    explicit AiCoreProf(AiCoreManager& aicoreMng) : hostAicoreMng_(aicoreMng) {}
    ~AiCoreProf() {}
#ifdef __DEVICE__
    static void RegDevProf();
    static int DevProfInit(uint32_t type, void *data, uint32_t len);
    void GetIsOpenDevProf();
    static uint64_t devProfSwitch_;
    static uint32_t devProfType_ ;
#endif
    void ProfInit(DeviceArgs *deviceArgs);
    void ProfStart();
    void ProfGetSwitch(int64_t& flag) const;
    void ProfStop();
    void AsmCntvc(uint64_t& cntvct) const;
    bool ProfIsEnable() { return profLevel_ != PROF_LEVEL_OFF; }

    void ProfInitPmu(int64_t* regAddrs, int64_t* pmuEventAddrs);
    void ProfStartPmu();
    void ProfStopPmu();
    void ProfGetPmu(int32_t coreIdx, uint32_t subGraphId, uint32_t taskId, uint64_t taskCtrlTaskId);
    void ProfGetLog(int32_t coreIdx, const struct TaskStat* taskStat);

private:
    struct PmuCtrlAddrs {
        uint32_t* ctrl0Addr{nullptr};
        uint32_t* ctrl1Addr{nullptr};
        uint32_t* startCntCyc0Addr{nullptr};
        uint32_t* startCntCyc1Addr{nullptr};
        uint32_t* stopCntCyc0Addr{nullptr};
        uint32_t* stopCntCyc1Addr{nullptr};
    };
    inline void ProfInitLog();
    inline void ProfStopLog();
    void ReadPmuCounters(const int32_t coreIdx) const;
    void SetPmuEvents(void* mapBase, const int32_t coreIdx) const;
    PmuCtrlAddrs InitPmuRegAddrsForCore(void* addr, void* mapBase, int coreIdx);
    void ProgramPmuStartForCore(void* mapBase, int coreIdx, const PmuCtrlAddrs& addrs);
    void FillPmuData(
        MsprofAicpuPyPtoPmuData& data, int32_t& coreIdx, uint32_t& subGraphId, uint32_t& taskId,
        uint64_t taskCtrlTaskId) const;
    void DebugPmuData(int32_t coreIdx, const MsprofAicpuPyPtoPmuData& data) const;
    uint64_t ProfGetCurCpuTimestamp();

private:
    uint32_t ctrl0Val_;
    uint32_t ctrl1Val_;
    PmuCtrlAddrs addrs_;
    int32_t coreNum_ = 0;
    AiCoreProfLevel profLevel_ = PROF_LEVEL_OFF;
    uint64_t taskCnt_ = 0;
    int64_t* regAddrs_{nullptr};
    int64_t* pmuEventAddrs_{nullptr};
    ProfReportAdditionalInfoFunc profReportAdditionalInfoFunc_{nullptr};
    ArchInfo archInfo_{ArchInfo::DAV_2201};

    // PMU_CNT0 ~ PMU_CNT7 共计8个cnt寄存器,32位寄存器,用来获取对应读数,单位为cycle
    std::vector<volatile uint32_t*> pmuCnt0Plain_;
    std::vector<volatile uint32_t*> pmuCnt1Plain_;
    std::vector<volatile uint32_t*> pmuCnt2Plain_;
    std::vector<volatile uint32_t*> pmuCnt3Plain_;
    std::vector<volatile uint32_t*> pmuCnt4Plain_;
    std::vector<volatile uint32_t*> pmuCnt5Plain_;
    std::vector<volatile uint32_t*> pmuCnt6Plain_;
    std::vector<volatile uint32_t*> pmuCnt7Plain_;
    std::vector<volatile uint32_t*> pmuCntTotal0Plain_;
    std::vector<volatile uint32_t*> pmuCntTotal1Plain_;

    std::vector<volatile uint32_t*> pmuCnt8Plain_;
    std::vector<volatile uint32_t*> pmuCnt9Plain_;
    // pmu data
    uint32_t pmuDataMaxNum_ = 3;
    uint32_t pmuMsgSize_ = 0;
    uint32_t pmuDataSize_ = 0;
    uint32_t pmuHeadSize_ = 0;
    std::vector<PyPtoMsprofAdditionalInfo> pmuMsg_;
    std::vector<MsprofAicpuPyPtoPmuHead*> pmuHead_;
    std::vector<MsprofAicpuPyPtoPmuData*> pmuData_;

    // log data
    uint32_t logDataMaxNum_ = 8;
    uint32_t logMsgSize_ = 0;
    uint32_t logDataSize_ = 0;
    uint32_t logHeadSize_ = 0;
    std::vector<PyPtoMsprofAdditionalInfo> logMsg_;
    std::vector<MsprofAicpuPyPtoLogHead*> logHead_;
    std::vector<MsprofAicpuPyPtoLogData*> logData_;

    // handshake data
    const uint32_t handkShakeMaxNum_ = 8;
    uint32_t handkShakeMsgSize_{0};
    uint32_t handShakeDataSize_{0};
    uint32_t handkShakeHeadSize_{0};
    std::vector<PyPtoMsprofAdditionalInfo> HandShakeMsg_;
    // AiCputStat data
    const uint32_t aiCpuStatMaxNum_ = 6;
    uint32_t aiCpuStatMsgSize_{0};
    uint32_t aiCpuStatDataSize_{0};
    uint32_t aiCpuStatHeadSize_{0};
    std::vector<PyPtoMsprofAdditionalInfo> aiCpuStatMsg_;
    AiCoreManager& hostAicoreMng_;
};
} // namespace npu::tile_fwk::dynamic
#endif

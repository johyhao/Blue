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
 * \file aicore_prof.cpp
 * \brief
 */

#include "aicore_prof.h"
#include "aicore_manager.h"
#include "machine/device/tilefwk/aicpu_common.h"
namespace {
constexpr int AICPUNUM = 6;
constexpr int64_t HIG_32BIT = 32;
constexpr uint32_t PYPTO_PROF_COMMANDHANDLE_TYPE_START = 1;
} // namespace

namespace npu::tile_fwk::dynamic {
AiCoreProfLevel CreateProfLevel(ProfConfig profConfig)
{
    if (profConfig.Contains(ProfConfig::AICORE_PMU)) {
        return PROF_LEVEL_FUNC_LOG_PMU;
    } else if (profConfig.Contains(ProfConfig::AICORE_TIME)) {
        return PROF_LEVEL_FUNC_LOG;
    } else if (profConfig.Contains(ProfConfig::AICPU_FUNC)) {
        return PROF_LEVEL_FUNC;
    }
    return PROF_LEVEL_OFF;
}

bool ProfCheckLevel(uint64_t feature)
{
    if (AdprofCheckFeatureIsOn == nullptr) {
        return false;
    }
    return AdprofCheckFeatureIsOn(feature) > 0;
}

#ifdef __DEVICE__
uint64_t AiCoreProf::devProfSwitch_ = 0;
uint32_t AiCoreProf::devProfType_ = 0;
int32_t AiCoreProf::DevProfInit(uint32_t type, void *data, uint32_t len) {
    if (data == nullptr || len == 0) {
        DEV_WARN("Para is invalid");
        return -1;
    }
    if (type != 1) {
        DEV_WARN("Prof type [%u] is invalid", type);
        return -1;
    }
    if (len < sizeof(PyPtoMsprofCommandHandle)) {
        DEV_WARN("Prof CommandHandle len [%u] is invalid", len);
        return -1;
    }
    PyPtoMsprofCommandHandle *hostProfHandleConfig = reinterpret_cast<PyPtoMsprofCommandHandle *>(data);
    devProfSwitch_ = hostProfHandleConfig->profSwitch;
    devProfType_ = hostProfHandleConfig->type;
    DEV_DEBUG("Dev prof profSwitch is %lu profType is %u", devProfSwitch_, devProfType_);
    return 0;
}

void AiCoreProf::RegDevProf() {
    if (MsprofRegisterCallback == nullptr) {
        DEV_DEBUG("MsprofRegister is not supproted");
        return;
    }
    int ret = MsprofRegisterCallback(AICPU, DevProfInit);
    if (ret != 0) {
        DEV_WARN("Pypto Msporf reg not success");
    }
}

void AiCoreProf::GetIsOpenDevProf() {
    if (ProfCheckLevel(PROF_TASK_TIME_L3)) {
        profLevel_ = PROF_LEVEL_FUNC_LOG_PMU;
        return;
    }
    if (ProfCheckLevel(PROF_TASK_TIME_L2)) {
        profLevel_ = PROF_LEVEL_FUNC_LOG;
        return;
    }
    if (devProfType_ != PYPTO_PROF_COMMANDHANDLE_TYPE_START) {
        DEV_DEBUG("Dev prof not open");
        return;
    }
    if ((PROF_TASK_TIME_L3 & devProfSwitch_) != 0) {
        profLevel_ = PROF_LEVEL_FUNC_LOG_PMU;
        return;
    }
    if ((PROF_TASK_TIME_L2 & devProfSwitch_) != 0) {
        profLevel_ = PROF_LEVEL_FUNC_LOG;
    }
}
#endif

void AiCoreProf::ProfInit(DeviceArgs *deviceArgs) {
    DEV_DEBUG("Begin Prof init");
    profLevel_ = CreateProfLevel(deviceArgs->toSubMachineConfig.profConfig);
#ifdef __DEVICE__
    GetIsOpenDevProf();
#endif
    coreNum_ = hostAicoreMng_.GetAllAiCoreNum();
    if (AdprofReportAdditionalInfo != nullptr) {
        DEV_DEBUG("Pypto config prof level is %d, current env support api is AdprofReportAdditionalInfo", profLevel_);
        profReportAdditionalInfoFunc_ = AdprofReportAdditionalInfo;
    } else {
        profReportAdditionalInfoFunc_ = MsprofReportAdditionalInfo;
    }
    DEV_DEBUG("Pypto config prof level is %d, profFuncPtr: %p", profLevel_, profReportAdditionalInfoFunc_);
    archInfo_ = deviceArgs->archInfo;
    if ((profLevel_ == PROF_LEVEL_FUNC_LOG) || (profLevel_ == PROF_LEVEL_FUNC_LOG_PMU)) {
        profLevel_ = PROF_LEVEL_FUNC_LOG;
        ProfInitLog();
#if PMU_COLLECT
        ProfInitPmu(
            reinterpret_cast<int64_t*>(deviceArgs->corePmuRegAddr),
            reinterpret_cast<int64_t*>(deviceArgs->pmuEventAddr));
        profLevel_ = PROF_LEVEL_FUNC_LOG_PMU;
#endif
    } else {
        profLevel_ = PROF_LEVEL_OFF;
        DEV_INFO("aicore profiling is closed..");
        return;
    }
    hostAicoreMng_.SetDotStatus(static_cast<int64_t>(profLevel_));
    DEV_INFO("aicore profiling is opened, level is %d.", profLevel_);
}

void AiCoreProf::ProfStart()
{
    if (profLevel_ == PROF_LEVEL_OFF) {
        return;
    }

    DEV_INFO("aicore profiling start.");

    if (profLevel_ == PROF_LEVEL_FUNC_LOG_PMU) {
        ProfStartPmu();
    }
}

void AiCoreProf::ProfGetSwitch(int64_t& flag) const
{
    if (profLevel_ == PROF_LEVEL_FUNC_LOG) {
        flag |= 0x1;
    } else if (profLevel_ == PROF_LEVEL_FUNC_LOG_PMU) {
        flag |= 0x3;
    }
}

void AiCoreProf::ProfStop()
{
    if (profLevel_ == PROF_LEVEL_OFF) {
        return;
    } else if (profLevel_ == PROF_LEVEL_FUNC_LOG) {
        ProfStopLog();
    } else if (profLevel_ == PROF_LEVEL_FUNC_LOG_PMU) {
        ProfStopPmu();
        ProfStopLog();
    }
    DEV_INFO("aicore profiling stop, total run task num: %lu.", taskCnt_);
}

inline void AiCoreProf::ProfInitLog()
{
    logMsgSize_ = sizeof(PyPtoMsprofAdditionalInfo);
    logHeadSize_ = sizeof(MsprofAicpuPyPtoLogHead);
    logDataSize_ = sizeof(MsprofAicpuPyPtoLogData);
    logMsg_.resize(coreNum_);
    logHead_.resize(coreNum_, nullptr);
    logData_.resize(coreNum_, nullptr);
    for (int32_t i = 0; i < coreNum_; i++) {
        logHead_[i] = reinterpret_cast<MsprofAicpuPyPtoLogHead*>(&logMsg_[i].data);
        logData_[i] =
            reinterpret_cast<MsprofAicpuPyPtoLogData*>(reinterpret_cast<uintptr_t>(logHead_[i]) + logHeadSize_);
        logHead_[i]->cnt = 0;

        logMsg_[i].magicNumber = 0x5A5AU;
        logMsg_[i].level = PYPTO_MSPROF_REPORT_AICPU_LEVEL;
        logMsg_[i].type = PYPTO_MSPROF_REPORT_AICPU_NODE_TYPE;
        logMsg_[i].threadId = hostAicoreMng_.aicpuIdx_;
        logMsg_[i].dataLen = logHeadSize_;
        logHead_[i]->magicNumber = 0x6BD3U;
        logHead_[i]->coreId = i;
        logHead_[i]->coreType = static_cast<uint16_t>(hostAicoreMng_.AicoreType(i));
        logHead_[i]->dataType = PROF_DATATYPE_LOG;
        logHead_[i]->taskId = 0;
        logHead_[i]->streamId = 0;
    }
    DEV_INFO("ProfInitLog finish.");
}

inline void AiCoreProf::ProfStopLog()
{
    if (!ProfCheckLevel(PROF_TASK_TIME_L2)) {
        return;
    }
    hostAicoreMng_.ForEachManageAicore([&](int coreIdx) {
        if (logHead_[coreIdx]->cnt != 0) {
            int32_t ret = profReportAdditionalInfoFunc_(1, &logMsg_[coreIdx], sizeof(PyPtoMsprofAdditionalInfo));
            DEV_DEBUG(
                "aicore profiling send log mesg, core id: %d, task num: %d, ret: %d.", coreIdx, logHead_[coreIdx]->cnt,
                ret);
            (void)(ret);
            memset_s(&logMsg_[coreIdx], logMsgSize_, 0, logMsgSize_);
        }
    });
}

void AiCoreProf::ProfGetLog(int32_t coreIdx, const struct TaskStat* taskStat)
{
    if (!ProfCheckLevel(PROF_TASK_TIME_L2)) {
        return;
    }
    MsprofAicpuPyPtoLogHead* logHead = logHead_[coreIdx];
    PyPtoMsprofAdditionalInfo& logMsg = logMsg_[coreIdx];
    if (logHead->cnt < logDataMaxNum_ - 1) {
        memcpy_s(
            reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(logData_[coreIdx]) + logDataSize_ * logHead->cnt),
            logDataSize_, taskStat, logDataSize_);
        logMsg.dataLen += logDataSize_;
        logHead->cnt++;
        DEV_DEBUG(
            "aicore profiling gen log mesg, taskid: %d core id: %d, task start: %ld, end: %ld.", taskStat->taskId,
            coreIdx, taskStat->execStart, taskStat->execEnd);
    } else if (logHead->cnt == logDataMaxNum_ - 1) {
        memcpy_s(
            reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(logData_[coreIdx]) + logDataSize_ * logHead->cnt),
            logDataSize_, taskStat, logDataSize_);
        logHead->cnt++;
        logMsg.dataLen += logDataSize_;
        int32_t ret = profReportAdditionalInfoFunc_(1, &logMsg, sizeof(PyPtoMsprofAdditionalInfo));
        DEV_DEBUG("aicore profiling send log mesg, core id: %d, task num: %d, ret: %d.", coreIdx, logHead->cnt, ret);
        // reset
        (void)(ret);
        logHead->cnt = 0;
        logMsg.dataLen = logHeadSize_;
    }
    taskCnt_++;
}

void AiCoreProf::ProfInitPmu(int64_t* regAddrs, int64_t* pmuEventAddrs)
{
    pmuMsgSize_ = sizeof(PyPtoMsprofAdditionalInfo);
    pmuHeadSize_ = sizeof(MsprofAicpuPyPtoPmuHead);
    pmuDataSize_ = sizeof(MsprofAicpuPyPtoPmuData);
    pmuMsg_.resize(coreNum_);
    pmuHead_.resize(coreNum_, nullptr);
    pmuData_.resize(coreNum_, nullptr);
    for (int32_t i = 0; i < coreNum_; i++) {
        pmuHead_[i] = reinterpret_cast<MsprofAicpuPyPtoPmuHead*>(&pmuMsg_[i].data);
        pmuData_[i] =
            reinterpret_cast<MsprofAicpuPyPtoPmuData*>(reinterpret_cast<uintptr_t>(pmuHead_[i]) + pmuHeadSize_);
        pmuHead_[i]->cnt = 0;
    }

    pmuCnt0Plain_.resize(coreNum_, nullptr);
    pmuCnt1Plain_.resize(coreNum_, nullptr);
    pmuCnt2Plain_.resize(coreNum_, nullptr);
    pmuCnt3Plain_.resize(coreNum_, nullptr);
    pmuCnt4Plain_.resize(coreNum_, nullptr);
    pmuCnt5Plain_.resize(coreNum_, nullptr);
    pmuCnt6Plain_.resize(coreNum_, nullptr);
    pmuCnt7Plain_.resize(coreNum_, nullptr);
    pmuCntTotal0Plain_.resize(coreNum_, nullptr);
    pmuCntTotal1Plain_.resize(coreNum_, nullptr);

    pmuCnt8Plain_.resize(coreNum_, nullptr);
    pmuCnt9Plain_.resize(coreNum_, nullptr);
    regAddrs_ = regAddrs;
    pmuEventAddrs_ = pmuEventAddrs;

    auto it = kArchPmuConfigs.find(archInfo_);
    if (it != kArchPmuConfigs.end()) {
        size_t pmuCntSize = it->second.pmuCntIdxOffsets.size();
        if (pmuCntSize == MAX_PMU_CNT) {
            DEV_INFO(
                "0: %x, 1: %x, 2: %x, 3: %x, 4: %x, 5: %x, 6: %x, 7: %x.", (uint32_t)pmuEventAddrs_[0],
                (uint32_t)pmuEventAddrs_[1], (uint32_t)pmuEventAddrs_[2], (uint32_t)pmuEventAddrs_[3],
                (uint32_t)pmuEventAddrs_[4], (uint32_t)pmuEventAddrs_[5], (uint32_t)pmuEventAddrs_[6],
                (uint32_t)pmuEventAddrs_[7]);
        } else if (pmuCntSize == MAX_PMU_CNT_3510) {
            DEV_INFO(
                "0: %x, 1: %x, 2: %x, 3: %x, 4: %x, 5: %x, 6: %x, 7: %x, 8: %x, 9: %x.", (uint32_t)pmuEventAddrs_[0],
                (uint32_t)pmuEventAddrs_[1], (uint32_t)pmuEventAddrs_[2], (uint32_t)pmuEventAddrs_[3],
                (uint32_t)pmuEventAddrs_[4], (uint32_t)pmuEventAddrs_[5], (uint32_t)pmuEventAddrs_[6],
                (uint32_t)pmuEventAddrs_[7], (uint32_t)pmuEventAddrs_[8], (uint32_t)pmuEventAddrs_[9]);
        }
    }
}

void AiCoreProf::ReadPmuCounters(const int32_t coreIdx) const
{
    volatile uint32_t dummy_read = 0;
    auto read_reg = [&dummy_read](volatile uint32_t* reg) {
        dummy_read = *reg; // 通过volatile访问确保实际读取操作
    };

    auto it = kArchPmuConfigs.find(archInfo_);
    if (it == kArchPmuConfigs.end()) {
        return;
    }
    const auto& cfg = it->second;

    std::vector<const std::vector<volatile uint32_t*>*> pmuCntPlains = {
        &pmuCnt0Plain_, &pmuCnt1Plain_, &pmuCnt2Plain_, &pmuCnt3Plain_, &pmuCnt4Plain_,
        &pmuCnt5Plain_, &pmuCnt6Plain_, &pmuCnt7Plain_, &pmuCnt8Plain_, &pmuCnt9Plain_};

    for (size_t i = 0; i < cfg.pmuCntOffsets.size(); ++i) {
        read_reg((*pmuCntPlains[i])[coreIdx]);
    }

    read_reg(pmuCntTotal0Plain_[coreIdx]);
    if (pmuCntTotal1Plain_[coreIdx] != nullptr) {
        read_reg(pmuCntTotal1Plain_[coreIdx]);
    }

    (void)dummy_read; // 抑制未使用变量警告
}

void AiCoreProf::SetPmuEvents(void* mapBase, const int32_t coreIdx) const
{
    auto it = kArchPmuConfigs.find(archInfo_);
    if (it == kArchPmuConfigs.end()) {
        return;
    }
    const auto& cfg = it->second;
    for (size_t i = 0; i < cfg.pmuCntIdxOffsets.size(); ++i) {
        uint32_t* cntIdxAddr =
            reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(mapBase) + cfg.pmuCntIdxOffsets[i]);
        *cntIdxAddr = pmuEventAddrs_[i];
    }
    (void)coreIdx;
}

AiCoreProf::PmuCtrlAddrs AiCoreProf::InitPmuRegAddrsForCore(void* addr, void* mapBase, int coreIdx)
{
    PmuCtrlAddrs addrs;
    auto it = kArchPmuConfigs.find(archInfo_);
    if (it == kArchPmuConfigs.end()) {
        return addrs;
    }
    const auto& cfg = it->second;

    std::vector<std::vector<volatile uint32_t*>*> pmuCntPlains = {
        &pmuCnt0Plain_, &pmuCnt1Plain_, &pmuCnt2Plain_, &pmuCnt3Plain_, &pmuCnt4Plain_,
        &pmuCnt5Plain_, &pmuCnt6Plain_, &pmuCnt7Plain_, &pmuCnt8Plain_, &pmuCnt9Plain_};

    for (size_t i = 0; i < cfg.pmuCntOffsets.size(); ++i) {
        (*pmuCntPlains[i])[coreIdx] =
            reinterpret_cast<volatile uint32_t*>(reinterpret_cast<uint8_t*>(addr) + cfg.pmuCntOffsets[i]);
    }

    pmuCntTotal0Plain_[coreIdx] =
        reinterpret_cast<volatile uint32_t*>(reinterpret_cast<uint8_t*>(addr) + cfg.pmuCntTotal0Offset);
    if (cfg.pmuCntTotal1Offset != 0) {
        pmuCntTotal1Plain_[coreIdx] =
            reinterpret_cast<volatile uint32_t*>(reinterpret_cast<uint8_t*>(addr) + cfg.pmuCntTotal1Offset);
    } else {
        pmuCntTotal1Plain_[coreIdx] = nullptr;
    }

    addrs.ctrl0Addr = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(mapBase) + cfg.ctrl0Offset);
    if (cfg.ctrl1Offset != 0) {
        addrs.ctrl1Addr = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(mapBase) + cfg.ctrl1Offset);
    }
    addrs.startCntCyc0Addr = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(mapBase) + cfg.startCntCyc0Offset);
    if (cfg.startCntCyc1Offset != 0) {
        addrs.startCntCyc1Addr = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(mapBase) + cfg.startCntCyc1Offset);
    }
    addrs.stopCntCyc0Addr = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(mapBase) + cfg.stopCntCyc0Offset);
    if (cfg.stopCntCyc1Offset != 0) {
        addrs.stopCntCyc1Addr = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(mapBase) + cfg.stopCntCyc1Offset);
    }

    return addrs;
}

void AiCoreProf::ProgramPmuStartForCore(void* mapBase, int coreIdx, const PmuCtrlAddrs& addrs)
{
    // 在enable前先读取一次寄存器,将cnt清0
    ReadPmuCounters(coreIdx);

    // 设置PMU寄存器记录类型事件
    SetPmuEvents(mapBase, coreIdx);

    auto it = kArchPmuConfigs.find(archInfo_);
    if (it == kArchPmuConfigs.end()) {
        return;
    }
    const auto& cfg = it->second;

    *addrs.startCntCyc0Addr = 0x0;
    if (addrs.startCntCyc1Addr != nullptr) {
        *addrs.startCntCyc1Addr = 0x0;
    }
    *addrs.stopCntCyc0Addr = 0xFFFFFFFF;
    if (addrs.stopCntCyc1Addr != nullptr) {
        *addrs.stopCntCyc1Addr = 0xFFFFFFFF;
    }
    ctrl0Val_ = *addrs.ctrl0Addr;
    *addrs.ctrl0Addr = cfg.ctrl0Val;
    if (cfg.ctrl1Offset != 0 && addrs.ctrl1Addr != nullptr) {
        ctrl1Val_ = *addrs.ctrl1Addr;
        *addrs.ctrl1Addr = cfg.ctrl1Val;
    }
}

void AiCoreProf::ProfStartPmu()
{
    hostAicoreMng_.ForEachManageAicore([&](int coreIdx) {
        void* addr = reinterpret_cast<void*>(regAddrs_[hostAicoreMng_.GetPhyIdByBlockId(coreIdx)]);
        uint32_t pageSize = static_cast<uint32_t>(sysconf(_SC_PAGESIZE));
        void* mapBase =
            reinterpret_cast<void*>(reinterpret_cast<uint64_t>(addr) & ~(static_cast<uint64_t>(pageSize) - 1));
        addrs_ = InitPmuRegAddrsForCore(addr, mapBase, coreIdx);
        ProgramPmuStartForCore(mapBase, coreIdx, addrs_);
    });
}

void AiCoreProf::ProfStopPmu()
{
    *addrs_.ctrl0Addr = ctrl0Val_;
    if (archInfo_ == ArchInfo::DAV_3510) {
        *addrs_.ctrl1Addr = ctrl1Val_;
    }
    if (!ProfCheckLevel(PROF_TASK_TIME_L2)) {
        return;
    }
    hostAicoreMng_.ForEachManageAicore([&](int coreIdx) {
        if (pmuHead_[coreIdx]->cnt != 0) {
            int32_t ret = profReportAdditionalInfoFunc_(1, &pmuMsg_[coreIdx], sizeof(PyPtoMsprofAdditionalInfo));
            DEV_DEBUG(
                "aicore profiling send pmu mesg, core id: %d, task num: %d, ret: %d.", coreIdx, pmuHead_[coreIdx]->cnt,
                ret);
            (void)(ret);
            memset_s(&pmuMsg_[coreIdx], pmuMsgSize_, 0, pmuMsgSize_);
        }
    });
}

void AiCoreProf::FillPmuData(
    MsprofAicpuPyPtoPmuData& data, int32_t& coreIdx, uint32_t& subGraphId, uint32_t& taskId,
    uint64_t taskCtrlTaskId) const
{
    data.seqNo = static_cast<uint32_t>(taskCtrlTaskId);
    data.taskId = taskId;
    if (pmuCntTotal1Plain_[coreIdx] != nullptr) {
        data.totalCyc =
            *(pmuCntTotal0Plain_[coreIdx]) + (static_cast<uint64_t>(*(pmuCntTotal1Plain_[coreIdx])) << HIG_32BIT);
    } else {
        data.totalCyc = *(pmuCntTotal0Plain_[coreIdx]);
    }
    data.pmuCnt0 = *(pmuCnt0Plain_[coreIdx]);
    data.pmuCnt1 = *(pmuCnt1Plain_[coreIdx]);
    data.pmuCnt2 = *(pmuCnt2Plain_[coreIdx]);
    data.pmuCnt3 = *(pmuCnt3Plain_[coreIdx]);
    data.pmuCnt4 = *(pmuCnt4Plain_[coreIdx]);
    data.pmuCnt5 = *(pmuCnt5Plain_[coreIdx]);
    data.pmuCnt6 = *(pmuCnt6Plain_[coreIdx]);
    data.pmuCnt7 = *(pmuCnt7Plain_[coreIdx]);
    if (archInfo_ == ArchInfo::DAV_3510) {
        data.pmuCnt8 = *(pmuCnt8Plain_[coreIdx]);
        data.pmuCnt9 = *(pmuCnt9Plain_[coreIdx]);
    }
    (void)subGraphId;
}

void AiCoreProf::DebugPmuData(int32_t coreIdx, const MsprofAicpuPyPtoPmuData& data) const
{
    DEV_DEBUG(
        "aicore profiling pmu info, core id: %d: (%u, %u | %lu | %p=%u, %p=%u, %p=%u, %p=%u, "
        "%p=%u, %p=%u, %p=%u, %p=%u, %p=%u, %p=%u).",
        coreIdx, data.seqNo, data.taskId, data.totalCyc, pmuCnt0Plain_[coreIdx], data.pmuCnt0, pmuCnt1Plain_[coreIdx],
        data.pmuCnt1, pmuCnt2Plain_[coreIdx], data.pmuCnt2, pmuCnt3Plain_[coreIdx], data.pmuCnt3,
        pmuCnt4Plain_[coreIdx], data.pmuCnt4, pmuCnt5Plain_[coreIdx], data.pmuCnt5, pmuCnt6Plain_[coreIdx],
        data.pmuCnt6, pmuCnt7Plain_[coreIdx], data.pmuCnt7, pmuCnt8Plain_[coreIdx], data.pmuCnt8,
        pmuCnt9Plain_[coreIdx], data.pmuCnt9);
}

void AiCoreProf::ProfGetPmu(int32_t coreIdx, uint32_t subGraphId, uint32_t taskId, uint64_t taskCtrlTaskId)
{
    if (!ProfCheckLevel(PROF_TASK_TIME_L2)) {
        return;
    }

    MsprofAicpuPyPtoPmuData data = {0};
    FillPmuData(data, coreIdx, subGraphId, taskId, taskCtrlTaskId);
    DebugPmuData(coreIdx, data);

    if (pmuHead_[coreIdx]->cnt == 0) {
        pmuMsg_[coreIdx].magicNumber = 0x5A5AU;
        pmuMsg_[coreIdx].level = PYPTO_MSPROF_REPORT_AICPU_LEVEL;
        pmuMsg_[coreIdx].type = PYPTO_MSPROF_REPORT_AICPU_NODE_TYPE;
        pmuMsg_[coreIdx].threadId = syscall(SYS_gettid);
        pmuHead_[coreIdx]->magicNumber = 0x6BD3U;
        pmuHead_[coreIdx]->coreId = coreIdx;
        pmuHead_[coreIdx]->coreType = static_cast<uint16_t>(hostAicoreMng_.AicoreType(coreIdx));
        pmuHead_[coreIdx]->dataType = PROF_DATATYPE_PMU;
        pmuHead_[coreIdx]->taskId = 0;
        pmuHead_[coreIdx]->streamId = 0;
        memcpy_s(pmuData_[coreIdx], pmuDataSize_, &data, pmuDataSize_);
        pmuMsg_[coreIdx].dataLen = pmuHeadSize_ + pmuDataSize_;
        pmuHead_[coreIdx]->cnt++;
    } else if (pmuHead_[coreIdx]->cnt == pmuDataMaxNum_ - 1) {
        pmuMsg_[coreIdx].timeStamp = ProfGetCurCpuTimestamp();
        memcpy_s(
            reinterpret_cast<void*>(
                (reinterpret_cast<uintptr_t>(pmuData_[coreIdx]) + pmuDataSize_ * pmuHead_[coreIdx]->cnt)),
            pmuDataSize_, &data, pmuDataSize_);
        pmuMsg_[coreIdx].dataLen += pmuDataSize_;
        pmuHead_[coreIdx]->cnt++;
        int32_t ret = profReportAdditionalInfoFunc_(1, &pmuMsg_[coreIdx], sizeof(PyPtoMsprofAdditionalInfo));
        DEV_DEBUG(
            "aicore profiling send pmu mesg, core id: %d, task num: %d, ret: %d.", coreIdx, pmuHead_[coreIdx]->cnt,
            ret);
        (void)(ret);
        memset_s(&pmuMsg_[coreIdx], pmuMsgSize_, 0, pmuMsgSize_);
    } else {
        memcpy_s(
            reinterpret_cast<void*>(
                (reinterpret_cast<uintptr_t>(pmuData_[coreIdx]) + pmuDataSize_ * pmuHead_[coreIdx]->cnt)),
            pmuDataSize_, &data, pmuDataSize_);
        pmuMsg_[coreIdx].dataLen += pmuDataSize_;
        pmuHead_[coreIdx]->cnt++;
    }
}

void AiCoreProf::AsmCntvc(uint64_t& cntvct) const
{
#if defined __aarch64__
    asm volatile("mrs %0, cntvct_el0" : "=r"(cntvct));
#else
    cntvct = 0;
#endif
}

uint64_t AiCoreProf::ProfGetCurCpuTimestamp()
{
    uint64_t cntvct;
    AsmCntvc(cntvct);
    return cntvct;
}

} // namespace npu::tile_fwk::dynamic

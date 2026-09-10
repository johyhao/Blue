/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aicore_entry.h
 * \brief
 */

#ifndef AICORE_ENTRY_H
#define AICORE_ENTRY_H

#include <stdint.h>
#include <cstdint>
#include "tilefwk/aicpu_common.h"
#include "tilefwk/aicore_runtime.h"
#include "tilefwk/aicore_print.h"
#include "tilefwk/core_func_data.h"

// device switch head file begin
namespace npu::tile_fwk {

#define DEBUG_SWITCH 1

} // namespace npu::tile_fwk
// device switch head file end

#define TO_STRING_IMPL(str) #str
#define TO_STRING(str) TO_STRING_IMPL(str)

#ifdef __HAS_SUB_FUNC__
#if defined(__MIX__) && defined(__AIV__)
#include TO_STRING(__HEAD_FILE__)
#else
#include TO_STRING(__HEAD_FILE__)
#endif
#endif

INLINE void SetLastWordStatus(__gm__ KernelArgs* args, int64_t val);
INLINE uint32_t WaitWaveSignal(__gm__ KernelArgs* args);
INLINE void Trap()
{
#if IS_AICORE
    trap();
#endif
}

#define AICORE_DEVICE_TASK_WAIT_TIME_OUT 250000000 // unit is cycle. => 250 ms?
#define AICORE_LEAF_TASK_RUN_TIMEOUT 3000000000 // => 3s
#define AICORE_LEAF_TASK_WAIT_TIMEOUT 300000000 // => 300ms
#define AICORE_GM_DCCI_TIMEOUT 50000000 // => 50ms
#define AICORE_WAVEFLAG_WAIT_TIMEOUT 350000000 // => 350ms
#define AICORE_GET_LEAF_HIGHREG_TIMEOUT 400000000 // => 400ms

#if defined(__DAV_V200) && IS_AICORE
#define GET_SYS_CNT() ((uint64_t)0) // @NOTE: AICore不支持该接口
#define AICORE_TIMEOUT_CHECK_IMPL(t0, loop_count, timelen, lastStatus, action) \
    (void)(t0); \
    ++loop_count; \
    if ((loop_count % (timelen / 1000)) == 0) { \
        SetLastWordStatus(args, lastStatus); \
        Trap(); \
        action; \
    }
#else
#define GET_SYS_CNT() get_sys_cnt()
#define AICORE_TIMEOUT_CHECK_IMPL(t0, loop_count, timelen, lastStatus, action) \
    ++loop_count; \
    if (((loop_count % 1000) == 0) && (GET_SYS_CNT() - t0 > timelen)) { \
        SetLastWordStatus(args, lastStatus); \
        Trap(); \
        action; \
    }
#endif

#define AICORE_TIMEOUT_CHECK_BEGIN(t0, loop_count) \
    uint64_t t0 = GET_SYS_CNT(); \
    uint64_t loop_count = 0;

#define AICORE_TIMEOUT_CHECK_RETURN(t0, loop_count, timelen, lastStatus, retval) \
    AICORE_TIMEOUT_CHECK_IMPL(t0, loop_count, timelen, lastStatus, return retval)

#define AICORE_TIMEOUT_CHECK_RETURN_VOID(t0, loop_count, timelen, lastStatus) \
    AICORE_TIMEOUT_CHECK_IMPL(t0, loop_count, timelen, lastStatus, return)

using npu::tile_fwk::CoreFunctionData;
using npu::tile_fwk::DevRawTensorDesc;
using npu::tile_fwk::DynFuncBin;
using npu::tile_fwk::DynFuncData;
using npu::tile_fwk::DynFuncHeader;

enum DFX_STAGE_STATUS {
    STAGE_HANDSHAKE_START = 1,
    STAGE_HANDSHAKE_END = 2,
    STAGE_CORE_EXIT = 3,
    STAGE_GET_NEXT_TASK_STOP = 4,
    STAGE_PRE_EXEC_COREFUNC_KERNEL = 5,
    STAGE_FINISH_EXEC_COREFUNC_KERNEL = 6,
    STAGE_FINISH_PIPE_SYNC = 7,
    STAGE_FINISH_CUR_TASK = 8,
    STAGE_GET_PARALLEL_DEVTASK_TIMEOUT = 9,
    STAGE_GET_NEXT_TASK_TIMEOUT = 10,
    STAGE_WAVE_TIMEOUT = 11,
    STAGE_GET_HIGH_REG_TIMEOUT = 12,
    STAGE_UPDATE_PARALLEL_DEVTASK_TIMEOUT = 13,
    STAGE_UPDATE_PARALLEL_DEVTASK_ID_TIMEOUT = 14,
    STAGE_RUN_LEAFTASK_TIMEOUT = 15,
    STAGE_GET_FUNCDATA_STOP = 16,
    STAGE_KERNEL_ENTRY_0,
    STAGE_KERNEL_ENTRY_0_1,
    STAGE_KERNEL_ENTRY_1,
    STAGE_KERNEL_ENTRY_2 = 20,
    STAGE_KERNEL_ENTRY_3,
    STAGE_KERNEL_ENTRY_4,
    STAGE_KERNEL_ENTRY_5,
    STAGE_KERNEL_ENTRY_6,
    STAGE_KERNEL_ENTRY_7,
    STAGE_KERNEL_ENTRY_8,
    STAGE_KERNEL_ENTRY_9,
    STAGE_KERNEL_ENTRY_10,
    STAGE_KERNEL_ENTRY_11,
    STAGE_KERNEL_ENTRY_12,
    STAGE_EXEC_DYN_0,
    STAGE_EXEC_DYN_1,
    STAGE_EXEC_DYN_2 = 30,

    KERNEL_FUNC_1 = 31
};

struct ExecuteContext {
    __gm__ KernelArgs* args;
    int32_t blockIdx;
    volatile __gm__ ParallelDevTask *parallelDevTask{nullptr};
    uint32_t curLeafTaskParallelIdx{0};
    uint32_t seqNo{0};
    struct CachedDevTask {
        uint32_t seqNo{0};
        __gm__ DynFuncHeader *header{nullptr};
        __gm__ DynFuncData *funcDataList{nullptr};
        __gm__ DynFuncBin* cceBinary{nullptr};
    } cachedDevTasks[npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM];
    uint64_t lastTaskFinishCycle{0};
    bool openDumpPerfTrace{false};
    bool commByReg{true};  // false = GM fallback via shakeBuffer (read from args->commByReg at startup)
#if ENABLE_AICORE_PRINT
    AicoreLogger logger;
#endif
    uint32_t SeqNo() { return cachedDevTasks[curLeafTaskParallelIdx].seqNo; }
};

#if IS_AICORE
INLINE uint64_t GetDataMainBase()
{
    uint64_t coreStatus = 0;
    __asm__ volatile("MOV %0, DATA_MAIN_BASE\n" : "+l"(coreStatus));
    return coreStatus;
}
#endif

INLINE void Barrier()
{
#if defined(__CCE_KT_TEST__) && __CCE_KT_TEST__ == 1
    __asm__ __volatile__("" ::: "memory");
#else
    __asm__ __volatile__("");
#endif
}

// Read the 64-bit task register: DATA_MAIN_BASE (register) or shakeBufferCpuToCore[1] (GM fallback)
INLINE uint64_t GetTaskReg(__gm__ KernelArgs* args, bool commByReg)
{
    if (commByReg) {
        return GetDataMainBase();
    }
    // GM fallback: DCCI invalidate cache line before reading task ID from AICPU
    DCCI_STUB(&args->shakeBufferCpuToCore[SHAK_BUF_GM_DATA_MAIN_BASE_INDEX], SINGLE_CACHE_LINE, CACHELINE_OUT);
    Barrier();
    return static_cast<uint64_t>(args->shakeBufferCpuToCore[SHAK_BUF_GM_DATA_MAIN_BASE_INDEX]);
}

// Write the 64-bit cond value: COND register or shakeBuffer[4] (GM fallback)
INLINE void SendCond(__gm__ KernelArgs* args, bool commByReg, uint64_t value)
{
    if (commByReg) {
        set_cond(value);
        return;
    }
    // GM fallback: write to shakeBuffer[4] and flush cache line to GM
    args->shakeBuffer[SHAK_BUF_GM_COND_INDEX] = static_cast<int64_t>(value);
    Barrier();
    DCCI_STUB(&args->shakeBuffer[SHAK_BUF_GM_COND_INDEX], SINGLE_CACHE_LINE, CACHELINE_OUT);
    Barrier();
}

INLINE uint32_t GetLeafTaskId(__gm__ KernelArgs* args, bool commByReg) {
    return ((GetTaskReg(args, commByReg) & 0xFFFFFFFF) - 1);
}

INLINE uint32_t GetNextLeafTask(__gm__ KernelArgs* args, bool commByReg, uint32_t lastTaskIdx)
{
    uint32_t nextLowIdx = 0;
    uint64_t t0 = GET_SYS_CNT();
    uint64_t loop_count = 0;
    do {
        nextLowIdx = GetLeafTaskId(args, commByReg);

#if defined(__DAV_V200)
        if (!commByReg) {
            // GM fallback mode: re-enable timeout to avoid infinite polling
            ++loop_count;
            if ((loop_count % 1000 == 0) && (GET_SYS_CNT() - t0 > AICORE_LEAF_TASK_WAIT_TIMEOUT)) {
                Trap();
                return AICORE_TASK_ABNORMAL_STOP;
            }
        }
#else
        ++loop_count;
        if ((loop_count % 1000 == 0) && (GET_SYS_CNT() - t0 > AICORE_LEAF_TASK_WAIT_TIMEOUT)) {
            Trap();
            return AICORE_TASK_ABNORMAL_STOP;
        }
#endif
    } while (nextLowIdx == lastTaskIdx);

    return nextLowIdx;
}

INLINE uint32_t GetRegHighValue(__gm__ KernelArgs* args, bool commByReg, uint32_t lastHighRegValue)
{
    uint32_t highRegVal = 0;
    uint64_t regValue = 0;
    AICORE_TIMEOUT_CHECK_BEGIN(t0, loop_count);
    do {
        regValue = GetTaskReg(args, commByReg);
        highRegVal = (uint32_t)(regValue >> REG_HIGH_DTASKID_SHIFT);
        AICORE_TIMEOUT_CHECK_RETURN(t0, loop_count, AICORE_GET_LEAF_HIGHREG_TIMEOUT, STAGE_GET_HIGH_REG_TIMEOUT, AICORE_TASK_ABNORMAL_STOP);
    } while (highRegVal == 0 || lastHighRegValue == highRegVal);
    return highRegVal;
}

INLINE void PipeSync()
{
#if defined(__AIV__) || defined(__DAV_V200)
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
#else
    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
#endif
}

INLINE void HandshakeClient(__gm__ KernelArgs* args, bool commByReg)
{
    SendCond(args, commByReg, AICORE_TASK_INIT);
    volatile __gm__ int64_t* hello = args->shakeBuffer;

    int64_t coreId = get_coreid(); // phyId
    *hello = ((int64_t)coreId << 32) | AICORE_SAY_HELLO;

    Barrier();
    DCCI_STUB(hello, SINGLE_CACHE_LINE, CACHELINE_OUT);
    Barrier();
}

// val from DFX_STAGE_STATUS
INLINE void SetStatus(__gm__ KernelArgs* args, int64_t val)
{
    if (!IS_AICORE || DEBUG_SWITCH) {
        Barrier();
        args->shakeBuffer[2] = val;
        DCCI_STUB(args->shakeBuffer, SINGLE_CACHE_LINE, CACHELINE_OUT);
    }
}

INLINE void SetLastWordStatus(__gm__ KernelArgs* args, int64_t val)
{
    Barrier();
    args->shakeBuffer[3] = val;
    DCCI_STUB(args->shakeBuffer, SINGLE_CACHE_LINE, CACHELINE_OUT);
}

INLINE void SendRegFinish(__gm__ KernelArgs* args, bool commByReg, uint32_t curTaskIdx) {
    SendCond(args, commByReg, curTaskIdx | AICORE_FIN_MASK);
}

INLINE void SendRegAck(__gm__ KernelArgs* args, bool commByReg, uint32_t taskIdx) {
    SendCond(args, commByReg, taskIdx);
}

INLINE void PerfTraceRecord(
    uint32_t devTaskId, __gm__ Metrics* metric, AicorePerfTrace type, __gm__ KernelArgs* args,
    bool openDumpPerfTrace, uint64_t cycle = 0)
{
    if (unlikely(openDumpPerfTrace) && metric->turnNum < MAX_ROUND_NUM) {
        uint32_t turn = metric->turnNum;
        uint32_t cnt = metric->perfTraceCnt[turn][type];
        if (cnt < PERF_TRACE_INST_MAX_NUM_EVERY_TYPE) {
            metric->perfTrace[turn][type][cnt] = cycle == 0 ? GET_SYS_CNT() : cycle;
            metric->perfTraceDevTaskId[turn][type][cnt] = devTaskId;
            metric->perfTraceCnt[turn][type]++;
        }
    }
    (void)args;
}

INLINE void AddMetricStatistic(ExecuteContext* ctx, uint32_t seqNo, uint32_t taskId, int32_t subGraphId, int64_t t1)
{
    auto m = (__gm__ Metrics*)(ctx->args->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
    if (m && m->taskCount < MAX_DFX_TASK_NUM_PER_CORE) {
#if !defined(__DAV_V200)
        m->tasks[m->taskCount].subGraphId = subGraphId;
        m->tasks[m->taskCount].seqNo = seqNo;
        m->tasks[m->taskCount].taskId = taskId & TASKID_FROM_CTRL_TOPO_MASK;
        m->tasks[m->taskCount].execStart = t1;
        ctx->lastTaskFinishCycle = GET_SYS_CNT();
        m->tasks[m->taskCount].execEnd = ctx->lastTaskFinishCycle;
        m->taskCount++;
#endif
    }
}

INLINE void FlushMetricStatistic(__gm__ volatile KernelArgs* args)
{
    __gm__ volatile Metrics* m = (__gm__ volatile Metrics*)(args->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
    if (m == nullptr) {
        return;
    }

#ifndef __DAV_V200
    for (uint32_t i = 0; i < m->taskCount; i++) {
        DCCI_STUB(&m->tasks[i], SINGLE_CACHE_LINE, CACHELINE_OUT);
    }
#endif

    m->isMetricStop = 1;
    DCCI_STUB(m, SINGLE_CACHE_LINE, CACHELINE_OUT);
}

INLINE void DfxProcWhenCoreExit(ExecuteContext* ctx, __gm__ KernelArgs* args, __gm__ Metrics* metric)
{
    PerfTraceRecord(INVALID_DEV_TASK_ID, metric, PERF_TRACE_CORE_WAIT_EXIT_NOTIFY, args, ctx->openDumpPerfTrace);
    if (unlikely(
            args->taskEntry.reserved[0] == PRO_LEVEL2 || args->taskEntry.reserved[0] == PRO_LEVEL1 ||
            ctx->openDumpPerfTrace)) {
        metric->turnNum++;
        FlushMetricStatistic(args);
    }
}

INLINE void DfxProcWhenDevTaskStop(ExecuteContext *ctx, __gm__ KernelArgs *args, __gm__ Metrics* metric)
{
    if (ctx->lastTaskFinishCycle > 0) {
        PerfTraceRecord(
            ctx->SeqNo(), metric, PERF_TRACE_CORE_DEV_TASK_LEAF_TASK_EXEC, args, ctx->openDumpPerfTrace,
            ctx->lastTaskFinishCycle);
    }
}

INLINE void UpdateCacheDevTask(ExecuteContext *ctx, uint32_t parallelIdx, int64_t devTaskPtr)
{
        __gm__ DynFuncHeader *header = (__gm__ DynFuncHeader *)devTaskPtr;
        ctx->cachedDevTasks[parallelIdx].header = header;
        ctx->cachedDevTasks[parallelIdx].seqNo = header->seqNo;
        ctx->cachedDevTasks[parallelIdx].funcDataList = (__gm__ DynFuncData*)(header + 1);
        ctx->cachedDevTasks[parallelIdx].cceBinary = (__gm__ npu::tile_fwk::DynFuncBin*)(header->cceBinary);
}

INLINE volatile __gm__ ParallelDevTask* GetCoreFunctionData(ExecuteContext *ctx, __gm__ KernelArgs *args)
{
    AICORE_TIMEOUT_CHECK_BEGIN(t0, loop_count);
    // kernel start , init prallelDevtask
    volatile __gm__ ParallelDevTask* parallelDevTask = &args->parallelDevTask;
    while (true) {
        if (parallelDevTask->rear - parallelDevTask->front == 0) {
            DCCI_STUB(parallelDevTask, SINGLE_CACHE_LINE, CACHELINE_OUT);

#if !defined(__DAV_V200)
            AICORE_TIMEOUT_CHECK_RETURN(t0, loop_count, AICORE_DEVICE_TASK_WAIT_TIME_OUT, STAGE_GET_PARALLEL_DEVTASK_TIMEOUT, nullptr);
#endif

            // The aicore received the task stop notification before receiving the leaftask.
            if (GetLeafTaskId(args, ctx->commByReg) == AICORE_TASK_STOP) {
                SetStatus(args, STAGE_CORE_EXIT);
                WaitWaveSignal(args);
                return nullptr;
            }
            continue;
        }

        // make sure all devtask dcci successfully
        for (uint32_t i = parallelDevTask->front; i < parallelDevTask->rear; ++i) {
            uint32_t idx = i % npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM;
            int64_t elemPtr = 0;
            do {
                AICORE_TIMEOUT_CHECK_RETURN(t0, loop_count, AICORE_DEVICE_TASK_WAIT_TIME_OUT, STAGE_GET_PARALLEL_DEVTASK_TIMEOUT, nullptr);

                // The aicore received the task stop notification before receiving the leaftask.
                if (GetLeafTaskId(args, ctx->commByReg) == AICORE_TASK_STOP) {
                    SetStatus(args, STAGE_CORE_EXIT);
                    WaitWaveSignal(args);
                    return nullptr;
                }

                DCCI_STUB(&parallelDevTask->ptrElements[idx], SINGLE_CACHE_LINE, CACHELINE_OUT);
                elemPtr = parallelDevTask->ptrElements[idx];
            } while (elemPtr == 0);

            DCCI_STUB((__gm__ void *)elemPtr, SINGLE_CACHE_LINE, CACHELINE_OUT);
            UpdateCacheDevTask(ctx, idx, elemPtr);
        }

        return parallelDevTask;
    }

    return nullptr;
}

INLINE void PmuTestBegin(__gm__ KernelArgs* args)
{
    UNUSED(args);
#if IS_AICORE
    // set_ctrl/get_ctrl are CANN CCE intrinsics; unavailable on host AICore model simulation.
    if (args->taskEntry.reserved[0] == PRO_LEVEL2) {
        set_ctrl((uint64_t)get_ctrl() | 0x1);
    }
#endif
}

INLINE void PmuTestEnd(__gm__ KernelArgs* args)
{
    UNUSED(args);
#if IS_AICORE
    if (args->taskEntry.reserved[0] == PRO_LEVEL2) {
        set_ctrl((uint64_t)get_ctrl() - 1);
    }
#endif
}

INLINE __gm__ TaskStat* InitTaskStat(ExecuteContext* ctx)
{
    __gm__ TaskStat* taskStat = nullptr;
    (void)ctx;
#ifdef __DAV_V310
    if (unlikely(ctx->args->taskEntry.reserved[0] >= PRO_LEVEL1)) {
        auto m = (__gm__ Metrics*)(ctx->args->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
        taskStat = &m->tasks[m->taskCount];
        taskStat->perfDataBaseAddr = reinterpret_cast<uint64_t>(m);
        if (m->taskCount == 0) {
            taskStat->setEventAddr = PERF_DATA_BASE_SIZE;
            taskStat->waitEventAddr = PERF_DATA_BASE_SIZE + SET_WAIT_EVENT_DATA_SIZE;
        } else {
            __gm__ TaskStat* prevTask = &m->tasks[m->taskCount - 1];
            taskStat->setEventAddr = prevTask->setEventAddr + prevTask->setEventNum * sizeof(uint64_t);
            taskStat->waitEventAddr = prevTask->waitEventAddr + prevTask->waitEventNum * sizeof(uint64_t);
        }
        taskStat->setEventNum = 0;
        taskStat->waitEventNum = 0;
    }
#endif
    return taskStat;
}

#define FuncNum(id) TaskID(id)

#ifdef __HAS_SUB_FUNC__
INLINE void ExecDynCoreFunctionKernel(ExecuteContext* ctx, uint32_t taskId)
{
#if !defined(__DAV_V200)
    uint64_t t1 = GET_SYS_CNT();
#endif
    SetStatus(ctx->args, ((uint64_t)taskId << 32) | STAGE_PRE_EXEC_COREFUNC_KERNEL); // high 32 bits used for taskId

    auto funcData = &ctx->cachedDevTasks[ctx->curLeafTaskParallelIdx].funcDataList[npu::tile_fwk::FuncID(taskId)];
    auto opAttrs = &funcData->opAttrs[funcData->opAtrrOffsets[npu::tile_fwk::TaskID(taskId)]];
#if ENABLE_AICORE_PRINT
    CoreFuncParam param = {funcData, opAttrs, funcData->exprTbl, taskId, ctx->logger.Context()};
#else
    CoreFuncParam param = {funcData, opAttrs, funcData->exprTbl, taskId, nullptr};
#endif

#ifdef __ENABLE_MAIN_BLOCK
#define __MAIN_BLOCK 1
#else
#define __MAIN_BLOCK 0
#endif

#ifdef __DAV_V310
   // for mix coretasks, use cube's stackworkspace
    int index = __MAIN_BLOCK ? (opAttrs[0] + 1) / 2 : opAttrs[0];
    int64_t blockIndex = (ctx->cachedDevTasks[ctx->curLeafTaskParallelIdx].cceBinary[index].mixResourceType != 0) ?
        get_block_idx() : ctx->blockIdx;
    int64_t gmStackAddr = funcData->stackWorkSpaceAddr + blockIndex * funcData->stackWorkSpaceSize;
#else
    int64_t gmStackAddr = funcData->stackWorkSpaceAddr + ctx->blockIdx * funcData->stackWorkSpaceSize;
#endif

    SetStatus(ctx->args, ((uint64_t)taskId << 32) | STAGE_EXEC_DYN_0); // high 32 bits used for taskId

    __gm__ TaskStat* taskStat = InitTaskStat(ctx);

    SetStatus(ctx->args, ((uint64_t)taskId << 32) | STAGE_EXEC_DYN_1); // high 32 bits used for taskId

    CallSubFuncTask(
        opAttrs[0] + funcData->exprTbl[0], &param,
        gmStackAddr,
        (__gm__ int64_t*)funcData->startArgs->commContexts, taskStat, ctx->args);

    SetStatus(ctx->args, STAGE_FINISH_EXEC_COREFUNC_KERNEL);
    PipeSync();
    SetStatus(ctx->args, STAGE_FINISH_PIPE_SYNC);

    // A1芯片在AICPU侧进行采集
#if !defined(__DAV_V200)
    if (unlikely(ctx->args->taskEntry.reserved[0] == PRO_LEVEL2 || ctx->args->taskEntry.reserved[0] == PRO_LEVEL1)) {
        AddMetricStatistic(ctx, ctx->SeqNo(), taskId, opAttrs[0], t1);
    }
#endif

    if (unlikely(ctx->openDumpPerfTrace)) {
        ctx->lastTaskFinishCycle = GET_SYS_CNT();
    }
}
#endif

INLINE void InitLogger(ExecuteContext *ctx)
{
#if ENABLE_AICORE_PRINT
    auto buffer = reinterpret_cast<__gm__ uint8_t *>(ctx->args->shakeBuffer[SHAK_BUF_PRINT_BUFFER_INDEX]);
    if (buffer != 0 && ctx->logger.GetBuffer() != buffer) {
        ctx->logger.Init(buffer, PRINT_BUFFER_SIZE);
        (void)GetStaticLogContext(ctx->logger.Context());
    }

    DCCI_STUB(ctx->args->shakeBuffer, SINGLE_CACHE_LINE, CACHELINE_OUT);
#else
    (void)ctx;
#endif

    return;
}

INLINE void InitCtx(ExecuteContext *ctx, __gm__ Metrics* metric, volatile __gm__ ParallelDevTask* prallelDevTask)
{
    ctx->curLeafTaskParallelIdx = 0; // default init first devtask 
    PerfTraceRecord(
        ctx->SeqNo(), metric, PERF_TRACE_CORE_DEV_TASK_RCV_MODEL, ctx->args, ctx->openDumpPerfTrace);
    ctx->lastTaskFinishCycle = 0;
    ctx->parallelDevTask = prallelDevTask;

    DCCI_STUB(ctx->parallelDevTask, SINGLE_CACHE_LINE, CACHELINE_OUT);
    return;
}

INLINE void ExecCoreFunctionKernel(ExecuteContext* ctx, uint32_t curTaskIdx)
{
    UNUSED(ctx);
    UNUSED(curTaskIdx);
#ifdef __HAS_SUB_FUNC__
    ExecDynCoreFunctionKernel(ctx, curTaskIdx);
    return;
#endif
}

INLINE uint32_t WaitWaveSignal(__gm__ KernelArgs* args)
{
    AICORE_TIMEOUT_CHECK_BEGIN(t0, loop_count);
    volatile __gm__ int64_t* waveBuffer = args->waveBufferCpuToCore;
    while (true) {
        DCCI_STUB(waveBuffer, SINGLE_CACHE_LINE, CACHELINE_OUT);
        if (*waveBuffer == AICORE_SAY_GOODBYE) {
            return 0;
        }
        AICORE_TIMEOUT_CHECK_RETURN(t0, loop_count, AICORE_WAVEFLAG_WAIT_TIMEOUT, STAGE_WAVE_TIMEOUT, AICORE_TASK_ABNORMAL_STOP);
    }
}

INLINE uint32_t RefreshParallelDevTaskByModifyFlag(__gm__ KernelArgs *args, ExecuteContext *ctx, uint32_t highRegValue)
{
    uint32_t curLeafDevTaskId = npu::tile_fwk::DevTaskId(highRegValue);
    uint32_t mask = npu::tile_fwk::ParallelDevTaskModifyFlag(highRegValue);
    int32_t modifyCnt = __builtin_popcount(mask);
    while (mask) {
        int idx = __builtin_ffs(mask) - 1;

        // dcci read new devtask id
        uint32_t newDevTaskId;
        if (modifyCnt == 1) { // un-parallel devtask scene
            newDevTaskId = curLeafDevTaskId;
        } else {
            uint32_t oldDevTaskId = ctx->cachedDevTasks[idx].seqNo;
            AICORE_TIMEOUT_CHECK_BEGIN(t0, loop_count);
            do {
                AICORE_TIMEOUT_CHECK_RETURN(t0, loop_count, AICORE_GM_DCCI_TIMEOUT, STAGE_UPDATE_PARALLEL_DEVTASK_ID_TIMEOUT, AICORE_TASK_ABNORMAL_STOP);
                DCCI_STUB(&ctx->parallelDevTask->idElements[idx], SINGLE_CACHE_LINE, CACHELINE_OUT);
                newDevTaskId = ctx->parallelDevTask->idElements[idx];
            } while (newDevTaskId == oldDevTaskId);
        }

        // dcci read new devtask ptr
        int64_t newElemPtr;
        __gm__ DynFuncHeader *oldHeader = ctx->cachedDevTasks[idx].header;
        AICORE_TIMEOUT_CHECK_BEGIN(t0, loop_count);
        do {
            AICORE_TIMEOUT_CHECK_RETURN(t0, loop_count, AICORE_GM_DCCI_TIMEOUT, STAGE_UPDATE_PARALLEL_DEVTASK_TIMEOUT, AICORE_TASK_ABNORMAL_STOP);
            DCCI_STUB(&ctx->parallelDevTask->ptrElements[idx], SINGLE_CACHE_LINE, CACHELINE_OUT);
            newElemPtr = ctx->parallelDevTask->ptrElements[idx];
            if (newElemPtr != (int64_t)oldHeader) {
                DCCI_STUB((__gm__ void *)newElemPtr, SINGLE_CACHE_LINE, CACHELINE_OUT);
                break;
            }

            if (newElemPtr) {
                DCCI_STUB((__gm__ void *)newElemPtr, SINGLE_CACHE_LINE, CACHELINE_OUT);
            } 
        } while (newElemPtr == 0 || (((__gm__ DynFuncHeader *)newElemPtr)->seqNo != newDevTaskId));
        UpdateCacheDevTask(ctx, idx, newElemPtr);
        mask &= (mask - 1);
    }

    return 0;
}

INLINE uint32_t RefreshParallelDevTask(__gm__ KernelArgs *args, ExecuteContext *ctx, __gm__ Metrics* metric, uint32_t &lastRegHighVal)
{
    uint32_t newRegHighVal = GetRegHighValue(args, ctx->commByReg, lastRegHighVal);
    if (newRegHighVal == AICORE_TASK_ABNORMAL_STOP) {
        return AICORE_TASK_ABNORMAL_STOP;
    }

    uint32_t ret = RefreshParallelDevTaskByModifyFlag(args, ctx, newRegHighVal);
    if (ret == AICORE_TASK_ABNORMAL_STOP) {
        return AICORE_TASK_ABNORMAL_STOP;
    }

    lastRegHighVal = newRegHighVal;
    DCCI_STUB(ctx->parallelDevTask, SINGLE_CACHE_LINE, CACHELINE_OUT);

    // start new devtask
    PerfTraceRecord(
        ctx->SeqNo(), metric, PERF_TRACE_CORE_DEV_TASK_RCV_MODEL, ctx->args, ctx->openDumpPerfTrace);
    ctx->lastTaskFinishCycle = 0;
    return 0;
}

INLINE void KernelEntry(
    int64_t ffts_addr, int64_t inputs, int64_t outputs, int64_t workspace, int64_t tilingdata, int64_t cfgdata)
{
    uint64_t start = GET_SYS_CNT();
    UNUSED(ffts_addr);
    UNUSED(inputs);
    UNUSED(outputs);
    UNUSED(workspace);
    UNUSED(tilingdata);
#if defined(__AIV__) && defined(__MIX__) && !defined(__DAV_V200)
    int32_t blockIdx = get_block_idx() * get_subblockdim() + get_subblockid() + get_block_num();
#else
    int32_t blockIdx = get_block_idx();
#endif
#if defined(__DAV_V200)
    __gm__ DeviceArgs* devArgs = (__gm__ DeviceArgs*)cfgdata;
#else
    auto devArgs = (DeviceArgs*)cfgdata;
#endif
    __gm__ KernelArgs* args = (__gm__ KernelArgs*)(devArgs->sharedBuffer + blockIdx * SHARED_BUFFER_SIZE);
    __gm__ Metrics* metric = (__gm__ Metrics*)(args->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);

    ExecuteContext ctx = {};
    ctx.args = args;
    ctx.blockIdx = blockIdx;
    ctx.openDumpPerfTrace = (((__gm__ DevDfxArgs*)devArgs->devDfxArgAddr)->isOpenPerfTrace != 0);
    ctx.commByReg = devArgs->commByReg;  // read from DeviceArgs (host set before launch, no race with AICPU)
    PerfTraceRecord(INVALID_DEV_TASK_ID, metric, PERF_TRACE_CORE_BEGIN, args, ctx.openDumpPerfTrace, start);
    bool isFirstTask = true;

    SetStatus(args, STAGE_HANDSHAKE_START);
    HandshakeClient(args, ctx.commByReg);
    SetStatus(args, STAGE_HANDSHAKE_END); /// 

    set_mask_norm();

    uint32_t curTaskIdx;
    uint32_t lastTaskIdx = AICORE_TASK_INIT;
    uint32_t lastRegHighVal = 0;

    PerfTraceRecord(INVALID_DEV_TASK_ID, metric, PERF_TRACE_CORE_INIT, args, ctx.openDumpPerfTrace);

    volatile __gm__ ParallelDevTask* parallelDevTask = GetCoreFunctionData(&ctx, args);
    if (parallelDevTask == nullptr) {
        DfxProcWhenCoreExit(&ctx, args, metric);
        return; // no data exit
    }

    // init logger first, DO NOT init much more earlier!!
    InitLogger(&ctx);

    InitCtx(&ctx, metric, parallelDevTask);

    AiCoreLogF(0, "===> start sched core id: %lld, blockIdx: %d", (int64_t)get_coreid(), blockIdx);

    AICORE_TIMEOUT_CHECK_BEGIN(t0, loop_count);
    while (true) {
        SetStatus(args, STAGE_KERNEL_ENTRY_0);
        AICORE_TIMEOUT_CHECK_RETURN_VOID(t0, loop_count, AICORE_LEAF_TASK_RUN_TIMEOUT, STAGE_RUN_LEAFTASK_TIMEOUT);

        curTaskIdx = GetNextLeafTask(args, ctx.commByReg, lastTaskIdx);
        if (curTaskIdx == AICORE_TASK_STOP) {
            DfxProcWhenDevTaskStop(&ctx, args, metric);
            SetStatus(args, STAGE_CORE_EXIT);

            (void)WaitWaveSignal(args); // no data exit
            DfxProcWhenCoreExit(&ctx, args, metric);
            return;
        }
        if (curTaskIdx == AICORE_TASK_ABNORMAL_STOP) {
            SetLastWordStatus(args, STAGE_GET_NEXT_TASK_TIMEOUT);
            return;
        }

        SetStatus(args, STAGE_KERNEL_ENTRY_1);

        if (npu::tile_fwk::DevTaskDcciFlag(curTaskIdx) == 1) {
            SetStatus(args, STAGE_KERNEL_ENTRY_2);

            DfxProcWhenDevTaskStop(&ctx, args, metric); // perf trace stop last devtask

            ctx.curLeafTaskParallelIdx = npu::tile_fwk::ParallelIndex(curTaskIdx);

            SetStatus(args, STAGE_KERNEL_ENTRY_3);

            // need dcci new prallel devtask
            uint32_t ret = RefreshParallelDevTask(args, &ctx, metric, lastRegHighVal);
            if (ret == AICORE_TASK_ABNORMAL_STOP) {
                return;
            }

            SetStatus(args, STAGE_KERNEL_ENTRY_4);

            isFirstTask = true;
            t0 = GET_SYS_CNT(); // reset time out
        }

        // update cur leaftask parallelindex
        ctx.curLeafTaskParallelIdx = npu::tile_fwk::ParallelIndex(curTaskIdx);

        SetStatus(args, STAGE_KERNEL_ENTRY_5);

        if (isFirstTask) {
            PerfTraceRecord(ctx.SeqNo(), metric, PERF_TRACE_CORE_DEV_TASK_WAIT_RCV_FIRST_LEAF_TASK, args,
                ctx.openDumpPerfTrace);
            isFirstTask = false;
        }

        SetStatus(args, STAGE_KERNEL_ENTRY_6);

        SendRegAck(args, ctx.commByReg, curTaskIdx);

        SetStatus(args, STAGE_KERNEL_ENTRY_7);

        PmuTestBegin(args);
        ExecCoreFunctionKernel(&ctx, curTaskIdx);
        PmuTestEnd(args);

        SetStatus(args, STAGE_KERNEL_ENTRY_8);

        SendRegFinish(args, ctx.commByReg, curTaskIdx);
        lastTaskIdx = curTaskIdx;
        SetStatus(args, STAGE_FINISH_CUR_TASK);
    }
}

#endif

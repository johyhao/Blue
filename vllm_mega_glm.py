"""GLM-5.2 MegaKernel integration for vLLM-Ascend.

When ``ENABLE_MEGAKERNEL=1`` and the model is ``glm_moe_dsa``
(``GlmMoeDsaForCausalLM``), the forward is routed through the MegaKernel
(``blockrt.models.glm_5_2.Glm52MegaKernel``) instead of the vLLM/Ascend
op-by-op path.

This mirrors the sglang_npu ``qwen3_moe.py`` integration:

* weights are loaded once into the MegaKernel model from the tensors
  vLLM-Ascend already loaded (``Glm52MegaKernel(vllm_ascend_weights=True)``
  direct path, no ``load_glm5_2_weights`` re-read);
* the MegaKernel reuses the paged KV / DSA indexer caches already allocated by
  vLLM-Ascend (fp8 packed SFA C8 layout), so prefill writes and decode reads
  the same physical cache; vLLM drives scheduling and provides metadata
  (slot mapping, block tables, seq lens);
* persistent per-batch-size device buffers keep tensor addresses stable so
  the MegaKernel DAG graph-cache key does not change between decode steps
  (same trick as the sglang adapter's ``kv_seq_len_cache``/``q_seq_len_cache``).

Usage:

    ENABLE_MEGAKERNEL=1 vllm serve /path/to/GLM-5.2-10L-W4A8C8-A5 \
        --tensor-parallel-size 1 ...

Notes:
* TP must be 1 in this milestone (``Glm52MegaKernel(tp_size=1)``).
* ``MEGA_KERNEL_HOME`` / megakernel python path must be on ``PYTHONPATH``
  (``blockrt`` import) and the megakernel CANN registration must be done
  (``megakernel/install.sh``).
"""

from __future__ import annotations  # 启用 PEP 563 延迟注解求值，便于前置类型引用

import os  # 导入操作系统接口，用于读取环境变量
from typing import Any, Dict, List, Optional  # 导入类型注解工具类型

import torch  # 导入 PyTorch 核心张量与算子库
import torch.nn as nn  # 导入神经网络模块基类，供自定义 Module 继承

from vllm.forward_context import get_forward_context  # 获取当前前向上下文（含注意力元数据）
from vllm.config import VllmConfig  # vLLM 全局配置类型，承载模型/缓存/调度等子配置
from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP  # 复用 vLLM 的 DeepSeek MTP 草稿基类
from vllm.model_executor.models.deepseek_v2 import GlmMoeDsaForCausalLM  # 上游 GLM MoE DSA 因果语言模型基类
from vllm.sequence import IntermediateTensors  # 流水线并行/PP 中间张量类型

from vllm.logger import logger  # vLLM 统一日志器

_ENABLED = os.environ.get("ENABLE_MEGAKERNEL", "0") in ("1", "true", "True")  # 读取开关：是否启用 MegaKernel 路径

# Packed-KV layout constants are resolved from vLLM config at state init
# (cache_config.block_size / scheduler_config.max_num_batched_tokens /
# model_config.max_model_len) — see _MegaKernelGLM52State.__init__.


def _mega_enabled() -> bool:  # 定义查询函数：返回当前是否启用 MegaKernel
    return _ENABLED  # 直接返回模块级开关常量


def _model_args_from_hf_config(config: Any):  # 根据 vLLM 的 HF 配置构造 blockrt ModelArgsGLM5
    """Build a blockrt ModelArgsGLM5 from the vLLM HF config."""
    from blockrt.models.glm_5_2.model_args import ModelArgsGLM5  # 延迟导入 blockrt 的 GLM5 模型参数类

    args = ModelArgsGLM5()  # 实例化默认模型参数对象
    for key in (  # 遍历需要从 HF 配置拷贝的字段名列表
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_head_dim",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "intermediate_size",
        "moe_intermediate_size",
        "n_routed_experts",
        "n_shared_experts",
        "num_experts_per_tok",
        "first_k_dense_replace",
        "index_topk",
        "index_n_heads",
        "index_head_dim",
        "max_position_embeddings",
        "vocab_size",
        "rms_norm_eps",
    ):
        val = getattr(config, key, None)  # 从 HF 配置中按字段名取值，缺失则为 None
        if val is not None:  # 仅在配置中确实存在该字段时写入参数对象
            setattr(args, key, int(val) if isinstance(val, (int, float)) else val)  # 数值转 int，其余原样赋值
    args.quantized = True  # 标记模型为量化模式（W4A8C8 等量化权重）
    # Keep the MTP module enabled when the checkpoint provides MTP layers so
    # the MegaKernel-backed draft (DeepSeekMTPModel) can share this runtime.
    # The target forward never calls the MTP module.
    args.num_nextn_predict_layers = int(  # 设置 MTP（下一 token 预测）层数量
        getattr(config, "num_nextn_predict_layers", 0) or 0  # 缺省时取 0 表示无 MTP
    )
    configured_start = getattr(config, "mtp_start_layer_idx", None)  # 读取配置中显式指定的 MTP 起始层
    args.mtp_start_layer_idx = (  # 设置 MTP 起始层索引
        int(configured_start)  # 若配置显式指定则使用该绝对层索引
        if configured_start is not None
        else args.num_hidden_layers  # 否则默认接在主网络最后一层之后
    )
    return args  # 返回构造完成的 blockrt 模型参数对象


# Registry so the MegaKernel-backed MTP draft (a separate vLLM model instance)
# can share the target model's weights/runtime instead of loading them twice.
_MEGA_STATE_REGISTRY: Dict[str, "_MegaKernelGLM52State"] = {}  # 模型路径 -> 运行时状态的注册表，供 MTP 草稿复用


def _extract_dsa_metadata(attn_metadata: Any) -> AscendDSAMetadata:  # 将 v1 ForwardContext 的注意力元数据归一化为单个 AscendDSAMetadata
    """Normalize v1 ForwardContext attn_metadata to one AscendDSAMetadata."""
    if isinstance(attn_metadata, dict):  # v1 上下文中元数据为 {层名: AttentionMetadata} 字典
        # v1: {layer_name: AttentionMetadata}; all layers share the batch shape
        for meta in attn_metadata.values():  # 遍历各层元数据，取第一个非空项
            if meta is not None:
                return meta  # 所有层共享同一批次形状，返回任一非空元数据即可
        raise RuntimeError("MegaKernel: no attention metadata in forward context")  # 字典内全为空则报错
    if isinstance(attn_metadata, (list, tuple)):  # 若为列表/元组形式
        return _extract_dsa_metadata(attn_metadata[0])  # 递归取第一个元素进行归一化
    return attn_metadata  # 本身即为元数据对象，直接返回


def _has_usable_metadata(attn_meta: Any) -> bool:  # 判断注意力元数据是否具备 MegaKernel 桥接所需的字段
    """True if attn_meta exposes the fields the mega kernel bridge needs."""
    if attn_meta is None:  # 元数据为空则不可用
        return False
    if getattr(attn_meta, "num_actual_tokens", 0) <= 0:  # 实际 token 数小于等于 0 则不可用
        return False
    req = getattr(attn_meta, "req_metadata", None) or attn_meta  # 取请求级元数据子对象，缺失则用元数据本身
    return (  # 三者均非空才判定为可用
        getattr(req, "block_table", None) is not None  # 需要块表（KV 缓存页映射）
        and getattr(req, "seq_lens", None) is not None  # 需要序列长度
        and getattr(req, "slot_mapping", None) is not None  # 需要槽位映射
    )


def _cache_tensors(cache: Any) -> tuple[torch.Tensor, ...]:  # 将 vLLM 绑定的 KV 缓存归一化为张量元组
    """Normalize a vLLM bound KV cache into a tuple of tensors."""
    if cache is None:  # 缓存为空则返回空元组
        return ()
    if isinstance(cache, (tuple, list)):  # 多张量容器则过滤空项后转为元组
        return tuple(t for t in cache if t is not None)
    return (cache,)  # 单张量则包成单元素元组


class _MegaKernelGLM52State:  # 单个 GLM-5.2 模型的 MegaKernel 运行时状态（懒初始化）
    """Lazily-initialized MegaKernel runtime for one GLM-5.2 model."""

    def __init__(self, vllm_config: VllmConfig, vllm_model: Any):  # 构造运行时，需 vLLM 配置与已加载权重的模型实例
        self.vllm_config = vllm_config  # 保存 vLLM 全局配置
        hf_config = vllm_config.model_config.hf_config  # 取 HF 模型配置对象
        cache_config = vllm_config.cache_config  # 取缓存配置对象

        # Prefer vLLM/vLLM-Ascend-owned configuration over MegaKernel-specific
        # magic numbers. ``packed_kv_dim`` is the same value AscendSFA uses for
        # its packed C8 KV layout.
        self.block_size = int(getattr(cache_config, "block_size", 128))  # KV 缓存块大小，缺省 128
        self.rope_head_dim = int(getattr(hf_config, "qk_rope_head_dim", 64))  # MLA RoPE 头维度，缺省 64
        self.index_head_dim = int(getattr(hf_config, "index_head_dim", 128))  # DSA 索引器头维度，缺省 128
        self.packed_kv_dim = self._resolve_packed_kv_dim(hf_config)  # 解析打包 KV 头维度（C8 布局）
        self.max_tokens_default = self._resolve_max_tokens_default(  # 解析默认最大 token 数（用于预分配缓冲区）
            vllm_config
        )

        # The MegaKernel can selectively load layers 0..num_hidden_layers-1
        # from the FULL checkpoint even when vLLM serves a reduced one.
        self.model_dir = os.environ.get(  # 模型权重目录：优先环境变量，否则用 vLLM 配置中的模型路径
            "MEGA_WEIGHTS_DIR", vllm_config.model_config.model
        )
        self.model_args = _model_args_from_hf_config(  # 由 HF 配置构造 blockrt 模型参数
            vllm_config.model_config.hf_config
        )
        # Optional reduced-main-layer mode: vLLM may serve the FULL checkpoint
        # while the MegaKernel selectively loads a reduced layer prefix from
        # MEGA_WEIGHTS_DIR (e.g. the 10-layer + MTP index).  The MTP start
        # index stays at the checkpoint's absolute layer index.
        env_layers = os.environ.get("MEGA_MAIN_LAYERS")  # 读取可选的裁剪主网络层数环境变量
        if env_layers:  # 若指定了裁剪层数
            self.model_args.num_hidden_layers = max(int(env_layers), 1)  # 覆盖主网络层数，至少为 1
            if (  # 当 MTP 起始层不大于裁剪后的层数（会与主网络层重叠）时需修正
                self.model_args.mtp_start_layer_idx is not None
                and self.model_args.mtp_start_layer_idx
                <= self.model_args.num_hidden_layers
            ):
                self.model_args.mtp_start_layer_idx = int(  # 将 MTP 起始层重设为 HF 配置中的原始绝对层索引
                    getattr(
                        vllm_config.model_config.hf_config,
                        "num_hidden_layers",
                        self.model_args.num_hidden_layers,
                    )
                )

        num_blocks = self._num_blocks()  # 计算 KV 缓存块总数
        self.num_blocks = num_blocks  # 保存块总数供后续缓冲区分配使用
        self.k_cache, self.index_k_buffer, self.index_k_scale_buffer = (  # 借用 vLLM 已分配的主 KV/索引器缓存
            self._borrow_vllm_kv_cache(vllm_model)
        )
        # indexer_rope indexes this table with absolute token positions.  Reuse
        # vLLM's complete RoPE table just like the SGLang handoff does; a
        # per-step [num_tokens, 64] buffer is invalid once decode positions are
        # non-zero.
        self.index_cos_sin_cache = self._borrow_vllm_index_rope_cache(  # 借用 vLLM 完整索引器 RoPE 表（绝对位置）
            vllm_model
        )
        # Target and MTP use the same indexer RoPE configuration.  Start with
        # the target's complete table; fill_rope_caches may replace this with
        # the draft module's own complete table if it exposes one.
        self.mtp_index_cos_sin_cache = self.index_cos_sin_cache  # MTP 索引器 RoPE 表初始复用目标模型表
        self.mla_cos_cache = torch.empty(  # 预分配 MLA cos 缓存（per-token，BF16）
            (self.max_tokens_default, self.rope_head_dim),
            dtype=torch.bfloat16,
            device="npu",
        )
        self.mla_sin_cache = torch.empty(  # 预分配 MLA sin 缓存（per-token，BF16）
            (self.max_tokens_default, self.rope_head_dim),
            dtype=torch.bfloat16,
            device="npu",
        )
        # Per-batch-size persistent metadata buffers (stable addresses).
        self.meta_buffers: Dict[int, Dict[str, torch.Tensor]] = {}  # 批次大小 -> 持久化元数据缓冲区字典

        # TP rank/world are set once on the model in
        # AscendGlm52MegaForCausalLM.__init__ (SHMEM is initialized there too,
        # exactly once per process - never call init_shemm again here).
        self.tp_size = int(getattr(vllm_model, "tp_size", 1) or 1)  # 张量并行规模，缺省 1
        self.tp_rank = int(getattr(vllm_model, "tp_rank", 0) or 0)  # 当前张量并行秩，缺省 0

        # Weights: reuse the vLLM-Ascend nn.Module state_dict (the tensors
        # vLLM-Ascend already loaded and post-processed), same idea as
        # sglang_npu ``glm4_moe.py`` ``_collect_megakernel_weights``, and hand
        # them straight to the MegaKernel via the ``vllm_ascend_weights``
        # direct path instead of re-reading the checkpoint with
        # ``load_glm5_2_weights``.
        from blockrt.models.glm_5_2.glm5_2 import Glm52MegaKernel  # 延迟导入 blockrt 的 GLM-5.2 MegaKernel 类

        weight_dict = {  # 从 vLLM 模型 state_dict 构建 detach 的权重字典（避免 autograd 跟踪）
            name: tensor.detach()
            for name, tensor in vllm_model.state_dict().items()
        }


        # SFA disposes kv_b_proj after deriving W_UK_T / W_UV, leaving an empty
        # state_dict entry.  The vLLM-specific load_weights hook snapshots the
        # original tensors before attention post-processing so MegaKernel can
        # derive its own w_kc / w_vc without changing the generic SFA path.
        _saved_kv_b = dict(getattr(vllm_model, "_mega_kv_b_weights", {}))  # 取 load_weights 阶段快照保存的 kv_b_proj 原始权重
        for _k, _v in _saved_kv_b.items():  # 遍历每个被 SFA 清空的 kv_b_proj 权重
            _existing = weight_dict.get(_k)  # 查看当前权重字典中对应键的值
            if _existing is not None and _existing.numel() == 0:  # 若该键已被 SFA 清空（元素数为 0）
                weight_dict[_k] = _v  # 用快照的原始权重还原，供 MegaKernel 派生 w_kc/w_vc
        self.mega = Glm52MegaKernel(  # 实例化 MegaKernel，通过 vllm_ascend_weights 直传路径注入权重
            self.model_args, weight_dict,
            tp_size=self.tp_size, tp_rank=self.tp_rank,
            vllm_ascend_weights=True,
        )
        # MegaKernel now owns Parameter wrappers for the borrowed tensors; the
        # vLLM wrapper no longer needs to keep a second set of Python refs.
        getattr(vllm_model, "_mega_kv_b_weights", {}).clear()  # 清空 vLLM 侧的 kv_b_proj 快照引用

        # Dedicated KV/index caches for the MTP layer(s).  vLLM's slot mapping
        # and block tables index the shared-pool layout; a pool with the same
        # block_size / block count is address-compatible with those indices.
        mtp_start = self.model_args.mtp_start_layer_idx  # MTP 起始层索引
        mtp_end = mtp_start + self.model_args.num_nextn_predict_layers  # MTP 结束层索引（不含）
        self.mtp_k_cache: List[Optional[torch.Tensor]] = [None] * max(mtp_end, 1)  # 按层索引初始化 MTP KV 缓存列表
        self.mtp_index_k_buffer: List[Optional[torch.Tensor]] = [None] * max(  # 按层索引初始化 MTP 索引器 K 缓存列表
            mtp_end, 1
        )
        self.mtp_index_k_scale_buffer: List[Optional[torch.Tensor]] = [None] * max(  # 按层索引初始化 MTP 索引器 K 缩放缓存列表
            mtp_end, 1
        )
        for layer_id in range(mtp_start, mtp_end):  # 为每个 MTP 层分配独立的 KV/索引器缓存
            self.mtp_k_cache[layer_id] = torch.zeros(  # 分配 MTP 主 KV 缓存（FP8 打包 C8 布局）
                (num_blocks, self.block_size, 1, self.packed_kv_dim),
                dtype=torch.float8_e4m3fn,
                device="npu",
            )
            self.mtp_index_k_buffer[layer_id] = torch.zeros(  # 分配 MTP 索引器 K 缓存（展平为 [块数*块大小, 头维度]）
                (num_blocks * self.block_size, self.index_head_dim),
                dtype=torch.float8_e4m3fn,
                device="npu",
            )
            self.mtp_index_k_scale_buffer[layer_id] = torch.zeros(  # 分配 MTP 索引器 K 缩放缓存（每 token 一个 FP32 scale）
                (num_blocks * self.block_size, 1),
                dtype=torch.float32,
                device="npu",
            )

        # Share this runtime with the MegaKernel MTP draft (same checkpoint),
        # so the draft does not reload weights or build a second runtime.
        _MEGA_STATE_REGISTRY[str(vllm_config.model_config.model)] = self  # 将本运行时注册到全局表，供 MTP 草稿共享

    @staticmethod
    def _validate_index_rope_cache(cache: torch.Tensor, owner: str) -> torch.Tensor:  # 校验索引器 RoPE 缓存满足 blockrt 要求
        """Validate the full absolute-position cache required by BlockRT."""
        if (  # 校验维度/数据类型/设备/连续性是否满足 blockRT 要求
            cache.ndim != 2  # 必须为 2 维
            or cache.shape[1] != 64  # 第二维必须为 64
            or cache.dtype != torch.bfloat16  # 必须为 BF16
            or not cache.is_npu  # 必须位于 NPU 设备
            or not cache.is_contiguous()  # 必须连续
        ):
            raise RuntimeError(  # 校验失败则抛出详细错误
                "MegaKernel indexer RoPE cache must be contiguous NPU BF16 "
                f"[max_positions, 64], got owner={owner}, "
                f"shape={tuple(cache.shape)}, dtype={cache.dtype}, "
                f"device={cache.device}, contiguous={cache.is_contiguous()}"
            )
        return cache  # 校验通过返回原缓存

    @classmethod
    def _borrow_vllm_index_rope_cache(  # 借用 vLLM 已构建的完整索引器 RoPE 表（零拷贝）
        cls, vllm_model: Any
    ) -> torch.Tensor:
        """Borrow vLLM's full indexer RoPE table without copying storage."""
        layers = getattr(getattr(vllm_model, "model", None), "layers", None)  # 取 vLLM 解码器层列表
        if layers is None:  # 找不到解码器层则报错
            raise RuntimeError(
                "MegaKernel: cannot locate vLLM decoder layers for indexer "
                "RoPE cache"
            )
        for layer_id, layer in enumerate(layers):  # 遍历各层寻找索引器 RoPE 的 cos_sin_cache
            self_attn = getattr(layer, "self_attn", None)  # 取该层自注意力模块
            rope = getattr(self_attn, "indexer_rope_emb", None)  # 取索引器 RoPE 模块
            cache = getattr(rope, "cos_sin_cache", None)  # 取 cos/sin 缓存张量
            if isinstance(cache, torch.Tensor):  # 找到张量型缓存即返回（经校验）
                return cls._validate_index_rope_cache(
                    cache, f"model.layers.{layer_id}.self_attn.indexer_rope_emb"
                )
        raise RuntimeError(  # 所有层都没有索引器 RoPE 缓存则报错
            "MegaKernel: vLLM model exposes no indexer RoPE cos_sin_cache"
        )

    def _resolve_packed_kv_dim(self, hf_config: Any) -> int:  # 解析打包 MLA KV 头维度
        """Resolve the packed MLA KV head dimension from vLLM-Ascend.

        Uses the vLLM cache block size (``cache_config.block_size``, stored on
        ``self.block_size`` at init) as the packed-KV tile width, matching the
        layout AscendSFA uses for the C8 packed cache.
        """
        try:
            from vllm_ascend.attention.utils import (  # 尝试导入 Ascend SFA 打包头维度计算工具
                get_sfa_qsfa_packed_head_dim,
            )

            return int(  # 调用工具计算打包头维度
                get_sfa_qsfa_packed_head_dim(
                    int(getattr(hf_config, "kv_lora_rank", 512)),  # MLA KV lora 秩，缺省 512
                    int(getattr(hf_config, "qk_rope_head_dim", 64)),  # RoPE 头维度，缺省 64
                    self.block_size,  # 打包瓦片宽度（块大小）
                )
            )
        except Exception as exc:  # noqa: BLE001  # 工具不可用时降级为模型配置推算
            logger.warning(
                "[MegaKernel] falling back to model-config packed cache "
                f"dimension: {exc}",
            )
        kv_lora_rank = int(getattr(hf_config, "kv_lora_rank", 512))  # 取 KV lora 秩
        rope_head_dim = int(getattr(hf_config, "qk_rope_head_dim", 64))  # 取 RoPE 头维度
        scale_dim = (  # 计算缩放维度（按块大小对齐的 4 字节倍数）
            kv_lora_rank // self.block_size
        ) * 4
        return kv_lora_rank + rope_head_dim * 2 + scale_dim  # 返回降级估算的打包头维度

    @staticmethod
    def _resolve_max_tokens_default(vllm_config: VllmConfig) -> int:  # 解析预分配缓冲区的默认最大 token 数
        env_value = os.environ.get("MEGA_MAX_TOKENS")  # 优先读取环境变量显式指定值
        if env_value is not None:
            return max(int(env_value), 1)  # 至少为 1
        scheduler = getattr(vllm_config, "scheduler_config", None)  # 取调度器配置
        max_num_batched_tokens = getattr(  # 取单批次最大 token 数
            scheduler, "max_num_batched_tokens", None
        )
        if max_num_batched_tokens is not None:
            return max(int(max_num_batched_tokens), 1)  # 用调度器配置，至少为 1
        model_cfg = getattr(vllm_config, "model_config", None)  # 取模型配置
        max_model_len = getattr(model_cfg, "max_model_len", None)  # 取最大模型长度
        if max_model_len is not None:
            return max(int(max_model_len), 1)  # 用模型长度，至少为 1
        return 8192  # 全部缺失时使用 8192 作为兜底默认值

    def _borrow_vllm_kv_cache(  # 借用 vLLM 已绑定的 KV/索引器缓存视图
        self, vllm_model: Any
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Read the KV cache views already bound by vLLM.

        AscendConfig now forces ``enable_sparse_sfa_c8`` /
        ``enable_sparse_li_c8`` in MegaKernel mode, so every MLA layer's bound
        main cache has the same ``[num_blocks, block_size, 1, 656]`` packed
        FP8 layout as blockrt's ``rms_norm_rope_scatter`` and
        ``kv_quant_sparse_flash_attention`` kernels. Indexer layers likewise
        expose ``(k, scale)`` caches that can be flattened into blockrt's
        ``[num_blocks * block_size, head_dim]`` / ``[..., 1]`` views.
        """
        layers = getattr(getattr(vllm_model, "model", None), "layers", None)  # 取 vLLM 解码器层列表
        if layers is None or len(layers) < self.model_args.num_hidden_layers:  # 层数不足则报错
            raise RuntimeError(
                "MegaKernel: cannot locate vLLM decoder layers for shared KV cache"
            )

        main_caches: list[torch.Tensor] = []  # 主 KV 缓存列表（每层一个）
        indexer_caches: list[torch.Tensor] = []  # 索引器 K 缓存列表（每层一个）
        indexer_scale_caches: list[torch.Tensor] = []  # 索引器 K 缩放缓存列表（每层一个）

        for layer_id in range(self.model_args.num_hidden_layers):  # 遍历每个主网络层
            layer = layers[layer_id]  # 取当前层模块
            self_attn = getattr(layer, "self_attn", None)  # 取自注意力模块
            if self_attn is None:  # 缺失自注意力模块则报错
                raise RuntimeError(
                    f"MegaKernel: layer {layer_id} has no self_attn"
                )

            # vLLM 0.27.1 DeepSeek/GLM MLA wrapper keeps the bound cache on
            # ``self_attn.mla_attn``. Fall back to the older ``attn`` name.
            mla_attn = getattr(self_attn, "mla_attn", None) or getattr(  # 取 MLA 注意力包装器，兼容新/旧命名
                self_attn, "attn", None
            )
            if mla_attn is None:  # 缺失 MLA 包装器则报错
                raise RuntimeError(
                    f"MegaKernel: layer {layer_id} has no MLA attention wrapper"
                )
            mla_attn = getattr(mla_attn, "mla_attn", None) or mla_attn  # 再解一层嵌套的 mla_attn（兼容多层包装）

            main = _cache_tensors(getattr(mla_attn, "kv_cache", None))  # 归一化主 KV 缓存为张量元组
            if len(main) != 1:  # 主缓存必须为单个打包张量，否则报错
                raise RuntimeError(
                    "MegaKernel shared KV cache requires packed SFA C8 main cache; "
                    f"layer {layer_id} got {len(main)} cache tensor(s). "
                    "Check that AscendConfig.enable_sparse_sfa_c8 is enabled."
                )
            packed = main[0]  # 取主缓存张量
            if packed.ndim != 4 or packed.shape[-1] != self.packed_kv_dim:  # 校验主缓存形状维度
                raise RuntimeError(
                    "MegaKernel shared KV cache has unexpected shape "
                    f"{tuple(packed.shape)} for layer {layer_id}; expected "
                    f"[num_blocks, {self.block_size}, 1, "
                    f"{self.packed_kv_dim}]."
                )
            main_caches.append(packed)  # 加入主缓存列表

            indexer = getattr(self_attn, "indexer", None)  # 取该层的 DSA 索引器模块
            indexer_cache_module = getattr(indexer, "k_cache", None)  # 取索引器的 k_cache 子模块
            indexer_cache = _cache_tensors(  # 归一化索引器缓存为张量元组
                getattr(indexer_cache_module, "kv_cache", None)
            )

            if indexer is None or len(indexer_cache) == 0:  # 无索引器层（共享 DSA 层复用前层 top-k）
                # Shared DSA layers reuse the previous layer's top-k indices
                # and never touch an indexer cache. Keep empty placeholders so
                # ForwardBatchInfo still has one list entry per layer.
                indexer_caches.append(  # 添加空占位张量以保持列表长度一致
                    torch.empty(
                        (0,), dtype=torch.float8_e4m3fn, device=packed.device
                    )
                )
                indexer_scale_caches.append(  # 添加空缩放占位张量
                    torch.zeros((0, 1), dtype=torch.float32, device=packed.device)
                )
                continue  # 跳过后续索引器缓存处理

            indexer_k = indexer_cache[0]  # 取索引器 K 张量
            indexer_caches.append(  # 展平为 [块数*块大小, 头维度] 视图后加入列表
                indexer_k.reshape(-1, self.index_head_dim)
            )
            if len(indexer_cache) == 2:  # 若索引器缓存含 scale 张量
                indexer_scale_caches.append(indexer_cache[1].reshape(-1, 1))  # 展平 scale 为 [N,1] 加入列表
            else:
                indexer_scale_caches.append(  # 无 scale 则用全零张量占位
                    torch.zeros((packed.shape[0] * packed.shape[1], 1), dtype=torch.float32, device=packed.device)
                )

        return main_caches, indexer_caches, indexer_scale_caches  # 返回主/索引器/索引器scale三类缓存列表

    def _num_blocks(self) -> int:  # 计算 KV 缓存块总数
        cfg = self.vllm_config.cache_config  # 取缓存配置
        nb = getattr(cfg, "num_gpu_blocks", None)  # 优先取 GPU/NPU 块数
        if nb is None:
            nb = getattr(cfg, "num_blocks", None)  # 兼容旧字段名
        if nb is None:
            nb = int(os.environ.get("MEGA_KV_BLOCKS", "2048"))  # 兜底取环境变量或默认 2048
        return int(nb)  # 返回块数

    def get_meta_buffers(self, batch_size: int) -> Dict[str, torch.Tensor]:  # 获取/创建指定批大小的持久化元数据缓冲区
        if batch_size not in self.meta_buffers:  # 该批大小尚未分配缓冲区则新建
            max_tokens = max(self.max_tokens_default, batch_size * 4)  # 缓冲区 token 容量取默认值与批*4 的较大者
            self.meta_buffers[batch_size] = {  # 为该批大小分配一组稳定地址的缓冲区
                "input_ids": torch.empty(  # 输入 token id 缓冲区（int32）
                    (max_tokens,), dtype=torch.int32, device="npu"
                ),
                "positions": torch.empty(  # 位置 id 缓冲区（int64）
                    (max_tokens,), dtype=torch.int64, device="npu"
                ),
                "slot_mapping": torch.empty(  # 槽位映射缓冲区（int64，匹配 scatter_update ABI）
                    # BlockRT scatter_update reads this input as int64_t.
                    # Keep the persistent graph buffer on that exact ABI;
                    # using vLLM's int32 metadata directly makes the kernel
                    # combine two adjacent entries into one out-of-range
                    # cache index (especially visible for batch size 1).
                    (max_tokens,), dtype=torch.int64, device="npu"
                ),
                "block_tables": torch.empty(  # 块表缓冲区（int32），初始填充 -1 表示未占用
                    (batch_size, self.num_blocks),
                    dtype=torch.int32,
                    device="npu",
                ).fill_(-1),
                "q_seq_len": torch.empty(  # 每 sequence 的 query 长度缓冲区（int32）
                    (batch_size,), dtype=torch.int32, device="npu"
                ),
                "kv_seq_len": torch.empty(  # 每 sequence 的 KV 长度缓冲区（int32）
                    (batch_size,), dtype=torch.int32, device="npu"
                ),
            }
        return self.meta_buffers[batch_size]  # 返回该批大小的缓冲区字典

    def build_forward_batch_info(  # 构造目标模型的 ForwardBatchInfo（blockrt 前向所需的批次元信息）
        self,
        attn_meta: AscendDSAMetadata,
        meta_bufs: Dict[str, torch.Tensor],
        num_tokens: int,
    ):
        from blockrt.models.glm_5_2.model_args import ForwardBatchInfo  # 延迟导入 blockrt 前向批次信息类

        # Fields for ForwardBatchInfo come from the persistent meta_bufs and
        # the caches of this state; per-request fields (block_table, seq_lens,
        # slot_mapping) are read from the metadata object itself in
        # _forward_mega (AscendSFAMetadata has no req_metadata sub-object).
        req = getattr(attn_meta, "req_metadata", None) or attn_meta  # 取请求级元数据子对象
        del req  # 此处仅做兼容取值，实际 per-request 字段在 _forward_mega 中直接读取
        # Per-token caches are persistent full-width buffers; blockrt consumes
        # them as [num_tokens, ...] views (view shares storage, so the DAG
        # captured data pointers stay stable across decode steps).
        return ForwardBatchInfo(  # 组装并返回 ForwardBatchInfo
            sin_cos_cache=self.mla_cos_cache[:num_tokens],  # MLA sin/cos 缓存切片视图
            k_cache=self.k_cache,  # 主 KV 缓存列表（每层一个）
            v_cache=self.k_cache,  # v_cache 复用 k_cache（打包布局中 K/V 同一缓冲）
            num_blocks=self.num_blocks,  # 块总数
            block_tables=meta_bufs["block_tables"],  # 块表
            slot_mapping=meta_bufs["slot_mapping"][:num_tokens].reshape(  # 槽位映射（reshape 为列向量）
                num_tokens, 1
            ),
            kv_seq_len=meta_bufs["kv_seq_len"],  # KV 序列长度
            q_seq_len=meta_bufs["q_seq_len"],  # query 序列长度
            num_tokens=num_tokens,  # 实际 token 数
            mask=meta_bufs["mask"] if "mask" in meta_bufs else None,  # 注意力掩码（可选）
            mask_type=1,  # 掩码类型标识（1 表示全掩码模式）
            index_cos_sin_cache=self.index_cos_sin_cache,  # 索引器 RoPE 完整表
            index_rope_positions=meta_bufs["positions"][:num_tokens],  # 索引器 RoPE 位置
            index_k_buffer=self.index_k_buffer,  # 索引器 K 缓存列表
            index_k_scale_buffer=self.index_k_scale_buffer,  # 索引器 K 缩放缓存列表
            mla_cos_cache=self.mla_cos_cache[:num_tokens],  # MLA cos 缓存切片
            mla_sin_cache=self.mla_sin_cache[:num_tokens],  # MLA sin 缓存切片
        )

    def build_mtp_forward_batch_info(  # 构造 MTP 草稿层的 ForwardBatchInfo
        self,
        meta_bufs: Dict[str, torch.Tensor],
        num_tokens: int,
        batch_size: int,
    ) -> Any:
        """Build a ForwardBatchInfo for one MegaKernel MTP step.

        Uses the dedicated MTP cache pool (same block layout as the vLLM
        shared pool) and the persistent per-step metadata buffers.  The
        per-token RoPE caches are filled by the draft's ``_fill_rope_caches``
        before this call.
        """
        from blockrt.models.glm_5_2.model_args import ForwardBatchInfo  # 延迟导入 ForwardBatchInfo 类

        return ForwardBatchInfo(  # 组装 MTP 专用 ForwardBatchInfo 并返回
            sin_cos_cache=self.mla_cos_cache[:num_tokens],  # MLA sin/cos 缓存切片
            k_cache=self.mtp_k_cache,  # MTP 专用 KV 缓存列表
            v_cache=self.mtp_k_cache,  # v_cache 复用 mtp_k_cache
            num_blocks=self.num_blocks,  # 块总数（与目标模型一致以保证地址兼容）
            block_tables=meta_bufs["block_tables"],  # 块表（复用目标模型的块表）
            slot_mapping=meta_bufs["slot_mapping"][:num_tokens].reshape(  # 槽位映射列向量
                num_tokens, 1
            ),
            kv_seq_len=meta_bufs["kv_seq_len"][:batch_size],  # KV 序列长度（按批大小切片）
            q_seq_len=meta_bufs["q_seq_len"][:batch_size],  # query 序列长度（按批大小切片）
            num_tokens=num_tokens,  # 实际 token 数
            mask=meta_bufs["mask"] if "mask" in meta_bufs else None,  # 注意力掩码（可选）
            mask_type=1,  # 掩码类型标识
            index_cos_sin_cache=self.mtp_index_cos_sin_cache,  # MTP 索引器 RoPE 完整表
            index_rope_positions=meta_bufs["positions"][:num_tokens],  # 索引器 RoPE 位置
            index_k_buffer=self.mtp_index_k_buffer,  # MTP 索引器 K 缓存列表
            index_k_scale_buffer=self.mtp_index_k_scale_buffer,  # MTP 索引器 K 缩放缓存列表
            mla_cos_cache=self.mla_cos_cache[:num_tokens],  # MLA cos 缓存切片
            mla_sin_cache=self.mla_sin_cache[:num_tokens],  # MLA sin 缓存切片
        )

    def fill_rope_caches(  # 填充持久化 MLA/索引器 RoPE 缓存（每步调用）
        self,
        positions: torch.Tensor,
        num_tokens: int,
        self_attn: Any,
    ) -> None:
        """Populate the persistent MLA/indexer RoPE caches for one step.

        ``self_attn`` may come from the target model (``layers[0].self_attn``)
        or the MTP draft layer; both expose ``rotary_emb`` and the optional
        ``indexer_rope_emb`` with identical conventions.
        """
        if num_tokens <= 0:  # 无 token 则跳过填充
            return

        def _cos_sin(rope, positions_cpu):  # 在 CPU 上复现 vLLM 的 cos/sin 计算
            inv_freq = 1.0 / (  # 计算逆频率向量
                rope.base
                ** (
                    torch.arange(
                        0, rope.rotary_dim, 2, dtype=torch.float32
                    )
                    / rope.rotary_dim
                )
            )
            freqs = torch.einsum("i,j->ij", positions_cpu.float(), inv_freq)  # 位置与逆频率外积得相位矩阵
            return freqs.cos(), freqs.sin()  # 返回 cos、sin

        pos_cpu = positions[:num_tokens].cpu()  # 取位置到 CPU（避免 NPU index_select 内核问题）
        cos32, sin32 = _cos_sin(self_attn.rotary_emb, pos_cpu)  # 计算 MLA 的 cos/sin
        self.mla_cos_cache[:num_tokens].copy_(  # 写入 MLA cos 缓存（repeat_interleave 成对系数后转 BF16）
            cos32.repeat_interleave(2, dim=-1).to(torch.bfloat16).npu()
        )
        self.mla_sin_cache[:num_tokens].copy_(  # 写入 MLA sin 缓存
            sin32.repeat_interleave(2, dim=-1).to(torch.bfloat16).npu()
        )

        irope = getattr(self_attn, "indexer_rope_emb", None)  # 取索引器 RoPE 模块
        index_cache = getattr(irope, "cos_sin_cache", None)  # 取索引器 cos/sin 缓存
        if isinstance(index_cache, torch.Tensor):  # 若存在张量型索引器缓存
            # MTP also uses absolute positions, so retain the complete table
            # instead of filling a per-token cache that would be indexed a
            # second time by indexer_rope.
            self.mtp_index_cos_sin_cache = self._validate_index_rope_cache(  # 用完整表替换 MTP 索引器 RoPE 缓存
                index_cache, "MTP self_attn.indexer_rope_emb"
            )


class AscendGlm52MegaForCausalLM(GlmMoeDsaForCausalLM):  # 启用 MegaKernel 时的 GLM-5.2 因果语言模型包装类
    """GLM-5.2 model that routes through MegaKernel when enabled."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):  # 构造函数
        # One-time SHMEM init BEFORE the base vLLM-Ascend model is built,
        # while the device is still free (mirrors run_tp2/test_forward.py).
        # aclshmem must be initialized exactly once per process; failure here
        # is fatal (fail fast at startup).
        super().__init__(vllm_config=vllm_config, prefix=prefix)  # 调用基类构造完成 vLLM-Ascend 模型构建

        pc = vllm_config.parallel_config  # 取并行配置
        self.tp_size = int(getattr(pc, "tensor_parallel_size", 1) or 1)  # 取张量并行规模
        if _mega_enabled() and self.tp_size > 1:  # 启用 MegaKernel 且 TP>1 时初始化共享内存通信
            from vllm.distributed import get_tensor_model_parallel_rank  # 导入获取 TP rank 的函数
            self.tp_rank = int(get_tensor_model_parallel_rank())  # 获取当前 TP rank
            import torch_npu  # noqa: F401  # 导入 torch_npu 以启用 NPU 扩展

            if torch.npu.is_available():  # NPU 可用时设置当前设备
                torch.npu.set_device(torch.npu.current_device())
            from blockrt.dist.utils import init_shemm  # 导入 blockrt 共享内存初始化函数
            # Match the initialization path verified by MegaKernel's TP
            # test_forward: a 1-GiB symmetric heap and one free bootstrap port
            # selected/broadcast through torch.distributed.  A fixed port can
            # still be requested explicitly for deployments that need it.
            _ms = int(os.environ.get("MEGA_SHMEM_SIZE", str(1 << 30)))  # 共享内存大小，默认 1GiB
            _shmem_ip_port = os.environ.get("MEGA_SHMEM_IP_PORT")  # 可选的固定 bootstrap 端口
            logger.debug(
                f"[MegaKernel] init_shemm rank={self.tp_rank} "
                f"world={self.tp_size} mem_size={_ms}",
            )
            try:
                init_shemm(  # 初始化进程间共享内存堆
                    rank=self.tp_rank,
                    world_size=self.tp_size,
                    mem_size=_ms,
                    ip_port=_shmem_ip_port,
                )
            except Exception:  # 初始化失败时输出诊断信息并重新抛出
                _lm = [
                    l for l in open("/proc/self/maps") if "libshmem.so" in l
                ]
                logger.error(
                    "[MegaKernel] init_shemm FAILED; libshmem maps:\n"
                    + "\n".join(_lm),
                )
                raise
        else:
            self.tp_rank = 0  # 非 TP 模式 rank 固定为 0

        self.vllm_config = vllm_config  # 保存 vLLM 配置
        self._mega_state: Optional[_MegaKernelGLM52State] = None  # MegaKernel 运行时状态，懒初始化
        self._mega_raw_weights: Optional[Dict[str, torch.Tensor]] = None  # 保留原始权重引用（可选）
        logger.info(
            "[MegaKernel] AscendGlm52MegaForCausalLM initialized, "
            f"ENABLE_MEGAKERNEL={_mega_enabled()}",
        )

    def _ensure_mega(self, vllm_config: VllmConfig) -> _MegaKernelGLM52State:  # 懒初始化并返回 MegaKernel 运行时状态
        if self._mega_state is None:  # 尚未初始化则在新线程中完成构建
            # vLLM runs the model forward under torch.inference_mode(), which
            # is thread-local. Tensors created there are "inference tensors"
            # and cannot be mutated in worker threads (the blockrt weight
            # loader parallelizes expert fusion with a thread pool). Run the
            # whole initialization in a fresh thread so every tensor is a
            # regular tensor.
            import threading  # 导入线程模块

            logger.info("[MegaKernel] initializing GLM-5.2 mega kernel")
            result: Dict[str, Any] = {}  # 用于跨线程收集构建结果
            errors: list[BaseException] = []  # 用于跨线程收集异常
            # NPU device selection is thread-local.  Capture the worker's
            # device here; a fresh thread otherwise reports device 0 on every
            # TP rank and allocates rank-local MegaKernel weights on npu:0.
            worker_device = (  # 捕获当前 worker 的 NPU 设备号，供新线程使用
                torch.npu.current_device() if torch.npu.is_available() else None
            )

            def _init() -> None:  # 新线程执行体：设置设备并构建运行时状态
                try:
                    if worker_device is not None:
                        torch.npu.set_device(worker_device)  # 在新线程中恢复正确的 NPU 设备
                    result["state"] = _MegaKernelGLM52State(vllm_config, self)  # 构建运行时状态
                except BaseException as exc:  # noqa: BLE001  # 捕获所有异常以便主线程重抛
                    import traceback

                    traceback.print_exc()
                    errors.append(exc)

            t = threading.Thread(target=_init)  # 创建初始化线程
            t.start()  # 启动线程
            t.join()  # 等待线程结束
            if errors:
                raise errors[0]  # 若有异常则重抛
            self._mega_state = result["state"]  # 保存构建好的运行时状态
            # Drain the NPU stream: the weight loader enqueues a large number
            # of expert npu_format_cast kernels asynchronously.  The first
            # decode step syncs the stream inside compute_graph_cache_key
            # (int(q_seq_len)); if those kernels are still in flight, the
            # sync hits the AICPU timeout.  Force completion here instead.
            if torch.npu.is_available():
                torch.npu.synchronize()  # 强制等待所有异步算子完成，避免首步解码超时
            logger.info("[MegaKernel] GLM-5.2 mega kernel ready")
        return self._mega_state  # 返回运行时状态

    def forward(  # 模型前向入口：解码步路由到 MegaKernel，否则走基类 op-by-op 路径
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if _mega_enabled() and input_ids is not None:  # 启用 MegaKernel 且有输入 token 时尝试解码路径
            attn_meta = None  # 注意力元数据，待从 forward context 提取
            try:
                fc = get_forward_context()  # 获取当前前向上下文
                if fc is not None:
                    attn_meta = _extract_dsa_metadata(fc.attn_metadata)  # 归一化注意力元数据
            except Exception:
                attn_meta = None  # 提取失败则回退到基类路径
            is_decode = (  # 判定是否为纯解码步
                attn_meta is not None
                and self._is_decode_step(attn_meta)
                and _has_usable_metadata(attn_meta)
            )
            if is_decode:  # 纯解码步则路由到 MegaKernel 前向
                try:
                    return self._forward_mega(input_ids, positions, attn_meta)
                except Exception:
                    import traceback

                    traceback.print_exc()
                    raise
        return super().forward(  # 非解码步或未启用时走基类前向
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )

    def compute_logits(  # 计算 logits：mega 模式下前向已返回 logits，直接透传
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        # In mega mode the model forward already returned logits; keep them
        # as-is (skip the vLLM LM-head pass).
        if _mega_enabled() and self._mega_state is not None:  # mega 模式且状态已初始化
            return hidden_states  # 直接返回前向产出的 logits
        return super().compute_logits(hidden_states)  # 否则走基类 LM-head 计算

    def load_weights(self, weights):  # 权重加载钩子：过滤 MTP 头并快照 kv_b_proj 原始权重
        # The reduced checkpoint keeps the MTP head under layer index
        # `num_hidden_layers` (renamed from the full model's MTP layer), which
        # the base (non-spec) vLLM model does not define (e.g. `rot`). The
        # MegaKernel path does not use the MTP head either, so drop it here.
        if _mega_enabled():  # 启用 mega 模式时过滤掉 MTP 头权重
            weights = (
                (n, t)
                for n, t in weights
                if not n.startswith("rot.")  # 过滤 rot.* MTP 头权重
                and not n.startswith(f"model.layers.{self.config.num_hidden_layers}.")  # 过滤越界 MTP 层权重
            )
        loaded_weights = super().load_weights(weights)  # 调用基类加载过滤后的权重

        # Ascend's attention post-load hook turns kv_b_proj into W_UK_T/W_UV
        # and disposes the original Parameter. Snapshot zero-copy aliases here,
        # after checkpoint loading but before that hook runs. Keeping this on
        # the MegaKernel-specific wrapper leaves the generic SFA path intact.
        if _mega_enabled():  # 启用 mega 模式时快照 kv_b_proj 原始权重
            self._mega_kv_b_weights = {}  # 初始化快照字典
            for module_name, module in self.named_modules():  # 遍历所有子模块
                if not module_name.endswith(".self_attn.kv_b_proj"):  # 仅处理 kv_b_proj 模块
                    continue
                weight = getattr(module, "weight", None)  # 取权重张量
                if weight is not None and weight.numel() > 0:  # 权重存在且非空则保存 detach 别名
                    self._mega_kv_b_weights[f"{module_name}.weight"] = (
                        weight.detach()
                    )

        return loaded_weights  # 返回基类加载的权重统计

    @staticmethod
    def _is_decode_step(attn_meta: Any) -> bool:  # 判定当前批次是否为纯解码步
        """True for a pure decode batch (AscendAttentionState.DecodeOnly).

        Prefill and mixed chunked-prefill batches keep using the native
        vLLM-Ascend path; only decode steps are routed through MegaKernel.
        """
        from vllm_ascend.attention.attention_v1 import AscendAttentionState  # 导入 Ascend 注意力状态枚举

        state = getattr(attn_meta, "attn_state", None)  # 取注意力状态
        if state is not None:
            return state == AscendAttentionState.DecodeOnly  # 严格匹配 DecodeOnly 状态
        # Fallback: no prefill tokens but some decode tokens.
        return (  # 回退判定：无 prefill 且有 decode token
            getattr(attn_meta, "num_prefills", 0) == 0
            and getattr(attn_meta, "num_decode_tokens", 0) > 0
        )

    def _fill_rope_caches(  # 填充本步的持久化 MLA/索引器 RoPE 缓存
        self, state: _MegaKernelGLM52State, positions: torch.Tensor, num_tokens: int
    ) -> None:
        """Populate the persistent MLA/indexer RoPE caches for this step.

        vLLM stores per-position cos/sin as [B, rotary_dim//2] pair
        coefficients (interleaved rope, is_neox_style=False).  MegaKernel's
        MLA kernels consume per-token [B, 64] cos/sin tensors with the pair
        coefficients repeat-interleaved (c0,c0,c1,c1,...).  The DSA indexer
        separately borrows vLLM's full [max_positions, 64] compact table.
        """
        if num_tokens <= 0:  # 无 token 则跳过
            return

        def _cos_sin(rope, positions_cpu):  # 在 CPU 上复现 vLLM cos/sin 计算
            # Reproduce vLLM cos_sin_cache on the host: cache = cat(cos, sin)
            # with rotary_dim//2 interleaved pair coefficients per position.
            # Computed on CPU (and copied in) to avoid NPU index_select, whose
            # Index kernel fails to launch on this stack (driver halMemAlloc).
            inv_freq = 1.0 / (
                rope.base
                ** (torch.arange(0, rope.rotary_dim, 2, dtype=torch.float32) / rope.rotary_dim)
            )
            freqs = torch.einsum("i,j->ij", positions_cpu.float(), inv_freq)  # 外积得相位矩阵
            return freqs.cos(), freqs.sin()  # 返回 cos、sin

        pos_cpu = positions[:num_tokens].cpu()  # 位置搬到 CPU
        attn0 = self.model.layers[0].self_attn  # 取第 0 层自注意力以获取 RoPE 配置

        cos32, sin32 = _cos_sin(attn0.rotary_emb, pos_cpu)  # 计算 cos/sin
        state.mla_cos_cache[:num_tokens].copy_(  # 写入状态对象的 MLA cos 缓存
            cos32.repeat_interleave(2, dim=-1).to(torch.bfloat16).npu()
        )
        state.mla_sin_cache[:num_tokens].copy_(  # 写入状态对象的 MLA sin 缓存
            sin32.repeat_interleave(2, dim=-1).to(torch.bfloat16).npu()
        )

        # The indexer uses absolute positions to read vLLM's full
        # ``state.index_cos_sin_cache``.  Do not build a per-token table here:
        # passing that table together with absolute positions double-indexes
        # it and reads out of bounds after prefill.

    def _forward_mega(  # MegaKernel 解码前向：填充缓冲区并调用 mega.forward
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_meta: AscendDSAMetadata,
    ) -> torch.Tensor:
        # The metadata may carry per-request fields directly or under a
        # `req_metadata` sub-object (AscendDSAMetadata vs. MLA/SFA metadata).
        req = getattr(attn_meta, "req_metadata", None) or attn_meta  # 取请求级元数据子对象
        state = self._ensure_mega(self.vllm_config)  # type: ignore[attr-defined]  # 懒初始化运行时状态

        num_tokens = int(  # 计算实际 token 数（优先元数据字段，回退输入形状）
            getattr(attn_meta, "num_actual_tokens", None)
            or getattr(attn_meta, "num_tokens", None)
            or input_ids.shape[0]
        )
        seq_lens = getattr(req, "seq_lens", None)  # 取序列长度张量
        block_table = getattr(req, "block_table", None)  # 取块表张量
        slot_mapping = getattr(req, "slot_mapping", None)  # 取槽位映射张量
        query_start_loc = getattr(req, "query_start_loc", None)  # 取 query 起始偏移张量
        if seq_lens is None or block_table is None:  # 关键字段缺失则报错
            raise RuntimeError(
                "MegaKernel: attention metadata lacks required fields "
                f"({type(attn_meta).__name__}): "
                f"seq_lens={seq_lens is not None}, "
                f"block_table={block_table is not None}, "
                f"slot_mapping={slot_mapping is not None}, "
                f"query_start_loc={query_start_loc is not None}"
            )
        batch_size = int(seq_lens.shape[0])  # 批大小 = 序列数
        bufs = state.get_meta_buffers(batch_size)  # 获取该批大小的持久化缓冲区

        # Fill the persistent MLA/indexer RoPE caches for this step.
        self._fill_rope_caches(state, positions, num_tokens)  # 填充 RoPE 缓存

        # ── Copy per-step inputs into persistent buffers (stable addresses) ──
        ids = input_ids.to(torch.int32).contiguous()  # token id 转 int32 并连续化
        pos = positions.to(torch.int64).contiguous()  # 位置转 int64 并连续化
        bufs["input_ids"][:num_tokens].copy_(ids)  # 写入持久 input_ids 缓冲区
        bufs["positions"][:num_tokens].copy_(pos)  # 写入持久 positions 缓冲区

        # MegaKernel's scatter_update ABI requires int64 indices.  vLLM's
        # attention metadata may expose int32 slot mappings, so normalize
        # before copying into the stable graph buffer.
        slot = slot_mapping.to(torch.int64).reshape(-1).contiguous()  # 槽位映射转 int64 展平连续化
        bufs["slot_mapping"][:num_tokens].copy_(slot[:num_tokens])  # 写入持久 slot_mapping 缓冲区

        block_table = block_table.to(torch.int32).contiguous()  # 块表转 int32 连续化
        max_blocks = min(block_table.shape[1], self._mega_state.num_blocks)  # 实际可用块数取较小者
        bufs["block_tables"][:, :max_blocks].copy_(  # 写入持久 block_tables 缓冲区
            block_table[:, :max_blocks]
        )

        seq_lens = seq_lens.to(torch.int32).contiguous()  # 序列长度转 int32 连续化
        bufs["kv_seq_len"][:batch_size].copy_(seq_lens[:batch_size])  # 写入持久 kv_seq_len 缓冲区
        # q_seq_len per sequence derived from query_start_loc (works for both
        # prefill and decode batches).
        qsl_t = query_start_loc  # query 起始偏移张量
        if qsl_t is None:
            qsl_t = getattr(attn_meta, "cum_query_lens", None)  # 回退到累计 query 长度字段
        if qsl_t is None:
            # Fall back to per-token q_seq_len = 1 (decode-only assumption).
            bufs["q_seq_len"][:batch_size].fill_(1)  # 无偏移信息则假设每序列 query=1（纯解码）
        else:
            qsl = qsl_t.to(torch.int64).contiguous()  # 偏移张量转 int64 连续化
            if qsl.shape[0] >= batch_size + 1:  # 偏移张量足够长则差分得每序列 query 长度
                q_seq = qsl[1 : batch_size + 1] - qsl[:batch_size]
            else:
                q_seq = torch.ones(  # 否则回退为全 1
                    (batch_size,), dtype=torch.int64, device=qsl.device
                )
            bufs["q_seq_len"][:batch_size].copy_(q_seq.to(torch.int32))  # 写入持久 q_seq_len 缓冲区

        fbi = state.build_forward_batch_info(attn_meta, bufs, num_tokens)  # 构建 ForwardBatchInfo
        logits = state.mega.forward(  # 调用 MegaKernel 前向得到 logits
            fbi,
            bufs["input_ids"][:num_tokens],
            bufs["positions"][:num_tokens],
            None,  # intermediate_tensors 占位
            None,  # inputs_embeds 占位
        )
        # vLLM sampling expects float32 logits [num_tokens, vocab_size].
        return logits.to(torch.float32)  # 转为 float32 返回供采样器使用


class _MegaMTPStepGraph(nn.Module):  # 可被图捕获的 embed + 单步 MTP 解码层计算图
    """Capture-safe graph of embed + one MTP decoder-layer step.

    Returns the same tuple as the native DeepSeekMTP draft contract:
    ``(pre_final_norm_hidden, post_final_norm_recycle_hidden)``.  The
    vocabulary head is applied later by ``compute_logits``.
    """

    def __init__(self, mtp_model: Any, embed_weight: torch.Tensor):  # 构造计算图模块
        super().__init__()
        self.mtp_model = mtp_model  # 保存 MTP 模型引用
        self.embed_weight = embed_weight  # 保存 embedding 权重引用

    def forward(  # 前向：embedding 查表 + MTP 模型单步
        self,
        forward_batch_info: Any,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        prev_hidden: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        token_hidden = torch.ops.mk_ascendc.gather_v2(  # 用自定义算子按 input_ids 查 embedding 权重
            self.embed_weight, input_ids
        )
        # mtp_model.forward -> (pre_final_norm, recycle_post_norm, topk)
        return self.mtp_model(  # 调用 MTP 模型前向并返回三元组
            forward_batch_info,
            positions,
            token_hidden,
            prev_hidden,
            spec_step_idx=spec_step_idx,
        )


def _shape_hash(value: Any) -> int:  # 仅基于形状的哈希，用作图缓存键（值存于缓冲区）
    """Small shape-only hash for graph-cache keys (values live in buffers)."""
    h = 0  # 初始哈希值

    def _mix(v: int, x: int) -> int:  # 哈希混合函数
        return (v * 1000003) ^ (x & 0xFFFFFFFF)  # 乘法 + 异或混合

    def _visit(v: Any) -> None:  # 递归访问值并累计哈希
        nonlocal h
        if isinstance(v, torch.Tensor):  # 张量：混入维度与各维大小
            h = _mix(h, v.ndim)
            for s in v.shape:
                h = _mix(h, int(s))
        elif isinstance(v, (list, tuple)):  # 列表/元组：递归访问每个元素
            for item in v:
                _visit(item)
        elif v is not None:  # 其他非空值：混入其哈希
            h = _mix(h, hash(v))

    _visit(value)  # 访问根值
    return h  # 返回最终哈希


class _MegaMTPStepRunner(nn.Module):  # MTP 草稿的逐步图捕获/缓存/执行器
    """Per-step graph capture/cache/run for the MegaKernel MTP draft.

    One 1-token (or packed first-pass) graph per shape; persistent input
    buffers keep tensor addresses stable so the captured graph is replayed
    instead of re-captured on every decode step (SGLang per-step design).
    """

    def __init__(  # 构造执行器
        self,
        step_graph: _MegaMTPStepGraph,
        *,
        device_id: int | None = None,
        num_experts: int = 256,
        skip_steps: int = 2,
        cache_capacity: int = 16,
    ) -> None:
        super().__init__()
        self.step_graph = step_graph  # 保存计算图模块
        self.device_id = (  # 保存 NPU 设备号，缺省取当前设备
            torch.npu.current_device() if device_id is None else device_id
        )
        self.num_experts = num_experts  # 专家数（用于图构建的专家跳过配置）
        self.skip_steps = skip_steps if num_experts else 0  # 跳过步数（无专家则置 0）
        from blockrt.models.model_cache import ModelCache  # 导入 blockrt 图缓存容器

        self.graph_cache = ModelCache(capacity=cache_capacity)  # 创建图缓存（容量可配置）
        self._outputs: Dict[Any, Any] = {}  # 缓存每个 key 对应的输出张量
        self._address_signatures: Dict[Any, tuple[int, ...]] = {}  # 缓存每个 key 对应的输入地址签名

    @staticmethod
    def _collect_ptrs(  # 收集输入的数据指针签名（用于判断地址是否变化）
        forward_batch_info: Any,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        prev_hidden: torch.Tensor,
    ) -> tuple[int, ...]:
        ptrs: list[int] = []  # 指针列表

        def _add(v: Any) -> None:  # 递归收集张量数据指针
            if isinstance(v, torch.Tensor):
                ptrs.append(v.data_ptr())  # 张量：记录数据指针
            elif isinstance(v, (list, tuple)):  # 列表/元组：递归收集
                for item in v:
                    _add(item)
            elif v is not None:  # 其他对象：按字段名收集其中的张量指针
                for field in (
                    "k_cache",
                    "v_cache",
                    "block_tables",
                    "slot_mapping",
                    "kv_seq_len",
                    "q_seq_len",
                    "index_cos_sin_cache",
                    "index_rope_positions",
                    "index_k_buffer",
                    "index_k_scale_buffer",
                    "mla_cos_cache",
                    "mla_sin_cache",
                ):
                    _add(getattr(v, field, None))  # 递归收集该字段的张量指针

        _add(forward_batch_info)  # 收集 ForwardBatchInfo 中的指针
        _add(positions)  # 收集 positions 指针
        _add(input_ids)  # 收集 input_ids 指针
        _add(prev_hidden)  # 收集 prev_hidden 指针
        return tuple(ptrs)  # 返回指针元组签名

    def _capture(  # 在 DAGCapture 上下文中执行一次计算以捕获计算图
        self,
        forward_batch_info: Any,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        prev_hidden: torch.Tensor,
        spec_step_idx: int,
    ) -> tuple[Any, Any]:
        from blockrt.runtime.tracer import DAGCapture  # 导入 blockrt DAG 捕获器

        with DAGCapture() as capture:  # 进入 DAG 捕获上下文
            output = self.step_graph(  # 执行一次计算图以记录算子 DAG
                forward_batch_info,
                positions,
                input_ids,
                prev_hidden,
                spec_step_idx,
            )
        graph = capture.build(  # 将捕获的 DAG 构建为可重放的图
            device_id=self.device_id,
            callback="libskip_inactive_callback.so",  # 专家跳过回调库
            num_experts=self.num_experts,
            skip_steps=self.skip_steps,
        )
        return graph, output  # 返回构建好的图与输出

    def forward(  # 执行 MTP 单步：命中缓存则重放，否则捕获新图
        self,
        forward_batch_info: Any,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        prev_hidden: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        key = (  # 构造图缓存键（token 数/num_tokens/步索引/形状哈希）
            int(input_ids.numel()),
            int(forward_batch_info.num_tokens),
            int(spec_step_idx),
            _shape_hash(
                (
                    forward_batch_info.block_tables,
                    forward_batch_info.slot_mapping,
                    forward_batch_info.kv_seq_len,
                    forward_batch_info.q_seq_len,
                    forward_batch_info.index_cos_sin_cache,
                    forward_batch_info.mla_cos_cache,
                    forward_batch_info.mla_sin_cache,
                )
            ),
        )
        graph = self.graph_cache.get_graph(key)  # 查询缓存中是否已有该形状的图
        signature = self._collect_ptrs(  # 收集当前输入的地址签名
            forward_batch_info, positions, input_ids, prev_hidden
        )
        if graph is None or self._address_signatures.get(key) != signature:  # 无图或地址变化则重新捕获
            graph, output = self._capture(
                forward_batch_info,
                positions,
                input_ids,
                prev_hidden,
                spec_step_idx,
            )
            self.graph_cache.insert_graph(key, graph)  # 存入图缓存
            self._outputs[key] = output  # 缓存输出
            self._address_signatures[key] = signature  # 更新地址签名
            torch.npu.synchronize()  # 捕获后同步确保图就绪
        graph.run()  # 重放捕获的图
        return self._outputs[key]  # 返回缓存的输出


class AscendGlm52MegaMTP(DeepSeekMTP):  # MegaKernel 支撑的 GLM-5.2 MTP 草稿模型
    """MegaKernel-backed GLM-5.2 MTP draft (DeepSeekMTPModel registry entry).

    Keeps the native DeepSeekMTP module tree so vLLM's model loader and the
    speculative proposer machinery work unchanged, but routes every MTP step
    through the MegaKernel MTP model as an SGLang-style per-step graph.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):  # 构造 MTP 草稿模型
        super().__init__(vllm_config=vllm_config, prefix=prefix)  # 调用基类构造保留 DeepSeekMTP 模块树
        self.vllm_config = vllm_config  # 保存 vLLM 配置
        self._mtp_state: Optional[_MegaKernelGLM52State] = None  # 共享的 MegaKernel 运行时状态，懒初始化
        self._step_runner: Optional[_MegaMTPStepRunner] = None  # 逐步图执行器，懒初始化

    def _ensure_mtp_state(self) -> _MegaKernelGLM52State:  # 懒初始化并返回共享的 MegaKernel 状态
        if self._mtp_state is None:  # 尚未初始化则从全局注册表获取目标模型运行时
            model_path = str(self.vllm_config.model_config.model)  # 取模型路径作为注册表键
            state = _MEGA_STATE_REGISTRY.get(model_path)  # 查找目标模型注册的运行时
            if state is None:  # 目标模型未先运行则报错
                raise RuntimeError(
                    "MegaKernel MTP draft: no shared MegaKernel state for "
                    f"{model_path}. The target model must run with "
                    "ENABLE_MEGAKERNEL=1 first (it registers the runtime)."
                )
            mtp_model = getattr(state.mega.model, "mtp_model", None)  # 取 MegaKernel 中的 MTP 子模型
            if mtp_model is None:  # 检查点无 MTP 层则报错
                raise RuntimeError(
                    "MegaKernel MTP draft: checkpoint has no MTP layers "
                    "(num_nextn_predict_layers=0)."
                )
            embed_weight = state.mega.model.model.embed_tokens.weight  # 取共享的 embedding 权重
            self._step_runner = _MegaMTPStepRunner(  # 创建逐步图执行器
                _MegaMTPStepGraph(mtp_model, embed_weight),
                device_id=torch.npu.current_device(),
                num_experts=int(
                    getattr(state.model_args, "n_routed_experts", 256)  # 专家数取模型参数，缺省 256
                ),
            )
            self._mtp_state = state  # 保存共享状态
        return self._mtp_state  # 返回运行时状态

    def forward(  # MTP 草稿前向：填充缓冲区并通过 step_runner 执行单步图
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if input_ids is None:  # MTP 草稿必须有 input_ids
            raise RuntimeError("MegaKernel MTP draft requires input_ids")
        state = self._ensure_mtp_state()  # 确保运行时状态就绪
        runner = self._step_runner
        assert runner is not None  # 确保执行器已创建

        fc = get_forward_context()  # 获取前向上下文
        attn_meta = _extract_dsa_metadata(fc.attn_metadata) if fc is not None else None  # 归一化注意力元数据
        if attn_meta is None or not _has_usable_metadata(attn_meta):  # 元数据不可用则报错
            raise RuntimeError(
                "MegaKernel MTP draft: no usable attention metadata in forward "
                "context"
            )
        req = getattr(attn_meta, "req_metadata", None) or attn_meta  # 取请求级元数据子对象
        num_tokens = int(  # 计算实际 token 数
            getattr(attn_meta, "num_actual_tokens", None)
            or getattr(attn_meta, "num_tokens", None)
            or input_ids.shape[0]
        )
        seq_lens = getattr(req, "seq_lens", None)  # 取序列长度
        block_table = getattr(req, "block_table", None)  # 取块表
        slot_mapping = getattr(req, "slot_mapping", None)  # 取槽位映射
        query_start_loc = getattr(req, "query_start_loc", None)  # 取 query 起始偏移
        if seq_lens is None or block_table is None or slot_mapping is None:  # 关键字段缺失则报错
            raise RuntimeError(
                "MegaKernel MTP draft: metadata lacks "
                f"seq_lens={seq_lens is not None}, "
                f"block_table={block_table is not None}, "
                f"slot_mapping={slot_mapping is not None}"
            )
        batch_size = int(seq_lens.shape[0])  # 批大小 = 序列数
        bufs = state.get_meta_buffers(batch_size)  # 获取该批大小的持久化缓冲区
        if "hidden_states" not in bufs:  # 首次需要时分配 hidden_states 缓冲区
            bufs["hidden_states"] = torch.empty(
                (self._max_buf_tokens(batch_size), state.model_args.hidden_size),
                dtype=torch.bfloat16,
                device="npu",
            )

        # Stable-address persistent buffers for graph replay.
        ids = input_ids.to(torch.int32).contiguous()  # token id 转 int32 连续化
        pos = positions.to(torch.int64).contiguous()  # 位置转 int64 连续化
        bufs["input_ids"][:num_tokens].copy_(ids[:num_tokens])  # 写入持久 input_ids 缓冲区
        bufs["positions"][:num_tokens].copy_(pos[:num_tokens])  # 写入持久 positions 缓冲区
        # Keep the MTP path on the same BlockRT scatter_update int64 ABI as
        # the target decode path.
        slot = slot_mapping.to(torch.int64).reshape(-1).contiguous()  # 槽位映射转 int64 展平连续化
        bufs["slot_mapping"][:num_tokens].copy_(slot[:num_tokens])  # 写入持久 slot_mapping 缓冲区
        block_table = block_table.to(torch.int32).contiguous()  # 块表转 int32 连续化
        max_blocks = min(block_table.shape[1], state.num_blocks)  # 实际可用块数取较小者
        bufs["block_tables"][:, :max_blocks].copy_(  # 写入持久 block_tables 缓冲区
            block_table[:, :max_blocks]
        )
        seq_lens_t = seq_lens.to(torch.int32).contiguous()  # 序列长度转 int32 连续化
        bufs["kv_seq_len"][:batch_size].copy_(seq_lens_t[:batch_size])  # 写入持久 kv_seq_len 缓冲区
        if query_start_loc is not None:  # 有偏移信息则差分得每序列 query 长度
            qsl = query_start_loc.to(torch.int64).contiguous()
            if qsl.shape[0] >= batch_size + 1:
                q_seq = qsl[1 : batch_size + 1] - qsl[:batch_size]
            else:
                q_seq = torch.ones(  # 不足则回退为全 1
                    batch_size, dtype=torch.int64, device=qsl.device
                )
            bufs["q_seq_len"][:batch_size].copy_(q_seq.to(torch.int32))  # 写入持久 q_seq_len 缓冲区
        else:
            bufs["q_seq_len"][:batch_size].fill_(1)  # 无偏移则假设每序列 query=1

        # Draft layer attention module (rotary_emb / indexer_rope_emb).
        mtp_start = state.model_args.mtp_start_layer_idx  # 取 MTP 起始层索引
        draft_layer = self.model.layers[str(mtp_start)]  # 取 MTP 草稿层（按字符串索引）
        draft_self_attn = draft_layer.mtp_block.self_attn  # 取草稿层自注意力模块
        state.fill_rope_caches(pos, num_tokens, draft_self_attn)  # 用草稿层 RoPE 配置填充缓存

        bufs["hidden_states"][:num_tokens].copy_(  # 写入持久 hidden_states 缓冲区（BF16）
            hidden_states[:num_tokens].to(torch.bfloat16).contiguous()
        )
        fbi = state.build_mtp_forward_batch_info(bufs, num_tokens, batch_size)  # 构建 MTP ForwardBatchInfo
        pre_norm, recycle, _topk = runner(  # 通过图执行器运行 MTP 单步
            fbi,
            bufs["positions"][:num_tokens],
            bufs["input_ids"][:num_tokens],
            bufs["hidden_states"][:num_tokens],
            spec_step_idx,
        )
        return pre_norm, recycle  # 返回前范数隐藏态与回收隐藏态

    def _max_buf_tokens(self, batch_size: int) -> int:  # 计算缓冲区所需的最大 token 容量
        state = self._mtp_state
        return max(
            getattr(state, "max_tokens_default", 8192) if state is not None else 8192,  # 状态默认值，缺省 8192
            batch_size * 4,  # 批大小 *4 作为下限
        )

    def compute_logits(  # MTP 草稿的 logits 计算：调用 MegaKernel MTP 模型的词表头
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        state = self._ensure_mtp_state()  # 确保运行时状态就绪
        logits = state.mega.model.mtp_model.compute_logits(  # 调用 MegaKernel MTP 子模型计算 logits
            hidden_states, spec_step_idx
        )
        return logits.to(torch.float32)  # 转 float32 返回


def register_megakernel_model() -> None:  # 将 GLM-5.2 注册表项指向 MegaKernel 包装类
    """Point the GLM-5.2 registry entry at the MegaKernel wrapper."""
    from vllm import ModelRegistry  # 导入 vLLM 模型注册表

    ModelRegistry.register_model(  # 注册目标模型类
        "GlmMoeDsaForCausalLM",
        "vllm_ascend.models.glm_5_2_mega:AscendGlm52MegaForCausalLM",
    )


def register_megakernel_mtp_model() -> None:  # 将 DeepSeekMTPModel 注册表项指向 MegaKernel MTP 草稿
    """Point the DeepSeekMTPModel registry entry at the MegaKernel MTP draft."""
    from vllm import ModelRegistry  # 导入 vLLM 模型注册表

    ModelRegistry.register_model(  # 注册 MTP 草稿模型类
        "DeepSeekMTPModel",
        "vllm_ascend.models.glm_5_2_mega:AscendGlm52MegaMTP",
    )

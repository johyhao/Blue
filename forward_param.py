    def _get_attn_layer(self, layer_idx: int):
        self_attn = self.layers[layer_idx].self_attn
        if hasattr(self_attn, "mla_attn"):
            inner = self_attn.mla_attn.mla_attn
            return inner.layer_name, inner.kv_cache
        if hasattr(self_attn, "dsa_attn"):
            wrapper = self_attn.dsa_attn
            return wrapper.prefix, wrapper.kv_cache
        raise RuntimeError(
            f"Layer {layer_idx}: unsupported attention type "
            f"{type(self_attn).__name__}"
        )

    def _get_kv_cache_params(
        self,
        kv_cache: torch.Tensor | tuple,
    ) -> dict[str, torch.Tensor | int | None]:
        if kv_cache is None:
            return self._empty_kv_cache_params()
        if isinstance(kv_cache, torch.Tensor):
            if kv_cache.numel() == 0:
                return self._empty_kv_cache_params()
            return {
                "k_cache": kv_cache,
                "v_cache": kv_cache,
                "num_blocks": kv_cache.shape[0],
                "index_k_buffer": None,
                "index_k_scale_buffer": None,
            }
        return {
            "k_cache": kv_cache[0],
            "v_cache": kv_cache[0],
            "num_blocks": kv_cache[0].shape[0] if kv_cache[0] is not None else 0,
            "index_k_buffer": kv_cache[4] if len(kv_cache) > 4 else None,
            "index_k_scale_buffer": kv_cache[5] if len(kv_cache) > 5 else None,
        }

    @staticmethod
    def _empty_kv_cache_params() -> dict[str, None]:
        return {
            "k_cache": None,
            "v_cache": None,
            "num_blocks": 0,
            "index_k_buffer": None,
            "index_k_scale_buffer": None,
        }

    def _get_attn_meta(self, layer_name: str, forward_context):
        meta = forward_context.attn_metadata.get(layer_name)
        if meta is None:
            return None
        block_tables = getattr(
            meta, "block_table",
            getattr(meta, "block_table_tensor", None),
        )
        q_seq_len = getattr(
            meta, "cum_query_lens",
            getattr(meta, "query_start_loc", None),
        )
        return {
            "block_tables": block_tables,
            "slot_mapping": getattr(meta, "slot_mapping", None),
            "kv_seq_len": getattr(meta, "seq_lens", None),
            "q_seq_len": q_seq_len,
            "num_tokens": getattr(meta, "num_input_tokens", 0),
            "mask": getattr(meta, "attn_mask", None),
            "mask_type": getattr(meta, "attn_state", None),
        }

    def _build_rotary_cache(self, positions: torch.Tensor):
        from vllm_ascend.ops.rotary_embedding import (
            _cos_cache,
            _sin_cache,
            _cos_mla,
            _sin_mla,
        )
        cos_sin = (_cos_cache[positions], _sin_cache[positions])
        return {
            "sin_cos_cache": cos_sin,
            "index_cos_sin_cache": cos_sin,
            "mla_cos_cache": _cos_mla,
            "mla_sin_cache": _sin_mla,
        }

    def extract_glm52_params(
        self,
        positions: torch.Tensor,
    ) -> dict:
        from vllm.forward_context import get_forward_context
        forward_context = get_forward_context()
        per_layer = []
        for idx in range(self.start_layer, self.end_layer):
            layer_name, kv_cache = self._get_attn_layer(idx)
            cache_params = self._get_kv_cache_params(kv_cache)
            meta_params = self._get_attn_meta(
                layer_name, forward_context
            )
            if meta_params is None:
                per_layer.append(None)
                continue
            per_layer.append({
                "layer_name": layer_name,
                **cache_params,
                **meta_params,
            })
        return {
            **self._build_rotary_cache(positions),
            "per_layer": per_layer,
        }




def get_indexer_kv_buffers_from_layers(
    layers,
    layer_idx: int,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    从 model.layers 中提取指定层的 index_k_buffer 和 index_k_scale_buffer。

    Args:
        layers: model.layers (DeepseekV2Model.layers)
        layer_idx: 层索引

    Returns:
        (index_k_buffer, index_k_scale_buffer)
        如果该层没有 indexer，返回 (None, None)
        如果未启用 LI C8，index_k_scale_buffer 为 None
    """
    layer = layers[layer_idx]
    mla_attn_wrapper = layer.self_attn.mla_attn
    mla_attn = mla_attn_wrapper.mla_attn
    impl = mla_attn.impl
    kv_cache = mla_attn.kv_cache

    if kv_cache is None or not getattr(impl, "has_indexer", False):
        return None, None

    enable_sparse_sfa_c8 = getattr(impl, "enable_sparse_sfa_c8", False)
    enable_sparse_li_c8 = getattr(impl, "enable_sparse_li_c8", False)

    if enable_sparse_sfa_c8:
        k_idx = 1
        scale_idx = 2 if enable_sparse_li_c8 else None
    else:
        k_idx = 2
        scale_idx = 3 if enable_sparse_li_c8 else None

    index_k_buffer = kv_cache[k_idx]
    index_k_scale_buffer = kv_cache[scale_idx] if scale_idx is not None else None

    return index_k_buffer, index_k_scale_buffer


def get_all_indexer_kv_buffers_from_layers(
    layers,
    num_hidden_layers: int,
) -> dict[int, tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
    """
    从 model.layers 中提取所有层的 index_k_buffer 和 index_k_scale_buffer。

    Args:
        layers: model.layers
        num_hidden_layers: 模型总层数

    Returns:
        {layer_idx: (index_k_buffer, index_k_scale_buffer)}
    """
    result = {}
    for i in range(num_hidden_layers):
        result[i] = get_indexer_kv_buffers_from_layers(layers, i)
    return result



# deepseek_v2.py:1282 之后添加

def _should_use_external_decode(
    self, intermediate_tensors: IntermediateTensors | None
) -> bool:
    """判断是否使用外部 decode 接口"""
    # 1. 必须是 GLM-5.2 模型
    if getattr(self.config, "model_type", "") != "glm_moe_dsa":
        return False

    # 2. 必须是首个 PP rank (有 input_ids)
    if not get_pp_group().is_first_rank:
        return False

    # 3. 检查是否为 decode-only batch
    from vllm.forward_context import get_forward_context
    fc = get_forward_context()
    attn_metadata = fc.attn_metadata

    # 从 attn_metadata 判断 decode-only
    if attn_metadata is None:
        return False

    # 检查 attn_state (如果是 AscendAttentionState)
    if hasattr(attn_metadata, "attn_state"):
        from vllm_ascend.attention.attention_v1 import AscendAttentionState
        return attn_metadata.attn_state in (
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.SpecDecoding,
        )

    # 回退：检查 num_actual_tokens == num_reqs (每请求1 token)
    if hasattr(attn_metadata, "num_actual_tokens") and hasattr(attn_metadata, "num_reqs"):
        return attn_metadata.num_actual_tokens == attn_metadata.num_reqs

    return False


def _external_glm52_decode(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None,
    inputs_embeds: torch.Tensor | None,
) -> torch.Tensor:
    """调用外部 GLM-5.2 decode 接口"""
    from vllm.forward_context import get_forward_context

    fc = get_forward_context()
    attn_metadata = fc.attn_metadata

    # ─── 1. 收集模型级参数 ───
    if inputs_embeds is not None:
        hidden_states_input = inputs_embeds
    else:
        hidden_states_input = self.embed_input_ids(input_ids)

    # ─── 2. 收集 RoPE cache ───
    # 从第一层获取 rotary_emb
    first_layer = self.layers[self.start_layer]
    rotary_emb = first_layer.self_attn.rotary_emb
    cos_sin_cache = rotary_emb.cos_sin_cache

    # 如果有 indexer_rope_emb (GLM-5.2 的 Indexer RoPE)
    index_cos_sin_cache = None
    if hasattr(first_layer.self_attn, "indexer_rope_emb") and first_layer.self_attn.indexer_rope_emb is not None:
        index_cos_sin_cache = first_layer.self_attn.indexer_rope_emb.cos_sin_cache

    # ─── 3. 收集每层的 KV cache ───
    layer_kv_caches = {}
    for idx in range(self.start_layer, self.end_layer):
        layer = self.layers[idx]
        attn = layer.self_attn
        # MLA attention 的 kv_cache 在 mla_attn.mla_attn.kv_cache
        if hasattr(attn, "mla_attn"):
            mla_wrapper = attn.mla_attn  # MultiHeadLatentAttentionWrapper
            if hasattr(mla_wrapper, "mla_attn"):
                mla_attn = mla_wrapper.mla_attn  # MLAAttention
                if hasattr(mla_attn, "kv_cache"):
                    layer_kv_caches[idx] = mla_attn.kv_cache

    # ─── 4. 收集 attention metadata ───
    # attn_metadata 可能是 dict (每层独立) 或单一对象
    if isinstance(attn_metadata, dict):
        # 取第一层的 metadata 作为参考
        first_layer_name = f"model.layers.{self.start_layer}.self_attn.attn"
        attn_meta = attn_metadata.get(first_layer_name, next(iter(attn_metadata.values())))
    else:
        attn_meta = attn_metadata

    block_tables = attn_meta.block_table if hasattr(attn_meta, "block_table") else None
    slot_mapping = fc.slot_mapping if hasattr(fc, "slot_mapping") else None
    seq_lens = attn_meta.seq_lens if hasattr(attn_meta, "seq_lens") else None
    cum_query_lens = attn_meta.cum_query_lens if hasattr(attn_meta, "cum_query_lens") else None
    num_actual_tokens = attn_meta.num_actual_tokens if hasattr(attn_meta, "num_actual_tokens") else positions.shape[0]

    # ─── 5. 计算 num_blocks ───
    num_blocks = 0
    if layer_kv_caches:
        first_kv = next(iter(layer_kv_caches.values()))
        if isinstance(first_kv, tuple) and len(first_kv) > 0:
            num_blocks = first_kv[0].shape[0]

    # ─── 6. 获取 MLA cos/sin cache (昇腾全局变量) ───
    mla_cos_cache = None
    mla_sin_cache = None
    try:
        from vllm_ascend.ops.rotary_embedding import _cos_cache, _sin_cache
        mla_cos_cache = _cos_cache
        mla_sin_cache = _sin_cache
    except ImportError:
        pass

    # ─── 7. 收集 indexer KV buffer ───
    # GLM-5.2 的 indexer k_cache 在 kv_cache tuple 的 index 2 (Native) 或 1 (C8)
    layer_index_k_buffers = {}
    layer_index_k_scale_buffers = {}
    for idx, kv_cache in layer_kv_caches.items():
        if isinstance(kv_cache, tuple):
            # 根据 tuple 长度判断 C8 配置
            if len(kv_cache) >= 3:
                layer_index_k_buffers[idx] = kv_cache[2] if len(kv_cache) == 3 else kv_cache[1]
            if len(kv_cache) >= 4:
                layer_index_k_scale_buffers[idx] = kv_cache[3]
            elif len(kv_cache) == 3:
                layer_index_k_scale_buffers[idx] = kv_cache[2]

    # ─── 8. 调用外部 decode 接口 ───
    result = self._call_external_decode_api(
        input_ids=input_ids,
        positions=positions,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds,
        hidden_states=hidden_states_input,
        cos_sin_cache=cos_sin_cache,
        index_cos_sin_cache=index_cos_sin_cache,
        layer_kv_caches=layer_kv_caches,
        layer_index_k_buffers=layer_index_k_buffers,
        layer_index_k_scale_buffers=layer_index_k_scale_buffers,
        num_blocks=num_blocks,
        block_tables=block_tables,
        slot_mapping=slot_mapping,
        kv_seq_len=seq_lens,
        q_seq_len=cum_query_lens,
        num_tokens=num_actual_tokens,
        mask=attn_meta.attn_mask if hasattr(attn_meta, "attn_mask") else None,
        mask_type=None,
        mla_cos_cache=mla_cos_cache,
        mla_sin_cache=mla_sin_cache,
    )

    return result


def _call_external_decode_api(
    self,
    input_ids,
    positions,
    intermediate_tensors,
    inputs_embeds,
    hidden_states,
    cos_sin_cache,
    index_cos_sin_cache,
    layer_kv_caches,
    layer_index_k_buffers,
    layer_index_k_scale_buffers,
    num_blocks,
    block_tables,
    slot_mapping,
    kv_seq_len,
    q_seq_len,
    num_tokens,
    mask,
    mask_type,
    mla_cos_cache,
    mla_sin_cache,
) -> torch.Tensor:
    """
    调用外部 GLM-5.2 decode 接口。
    这里需要根据实际的外部接口签名进行适配。
    """
    # TODO: 替换为实际的外部 decode 接口调用
    # 示例：
    # from external_glm52_api import glm52_decode
    # return glm52_decode(
    #     input_ids=input_ids,
    #     positions=positions,
    #     sin_cos_cache=cos_sin_cache,
    #     k_cache=layer_kv_caches,  # 每层的 k_cache
    #     v_cache=layer_kv_caches,  # 每层的 v_cache (实际是 k_rope)
    #     num_blocks=num_blocks,
    #     block_tables=block_tables,
    #     slot_mapping=slot_mapping,
    #     kv_seq_len=kv_seq_len,
    #     q_seq_len=q_seq_len,
    #     num_tokens=num_tokens,
    #     mask=mask,
    #     mask_type=mask_type,
    #     index_cos_sin_cache=index_cos_sin_cache,
    #     index_k_buffer=layer_index_k_buffers,
    #     index_k_scale_buffer=layer_index_k_scale_buffers,
    #     mla_cos_cache=mla_cos_cache,
    #     mla_sin_cache=mla_sin_cache,
    #     hidden_states=hidden_states,
    # )

    raise NotImplementedError("External GLM-5.2 decode API not implemented")

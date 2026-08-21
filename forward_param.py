def _get_forward_metadata(first_self_attn) -> dict:
    """Extract forward-level metadata once per step, before the layer loop.

    Dynamic metadata (block_tables, slot_mapping, etc.) is identical across
    all layers within a single forward step. Static metadata (sin_cos_cache,
    k_cache, etc.) is extracted from the first layer as a representative
    sample — all layers share the same rotary_emb and kv_cache shape.

    Called once per forward step (not per layer), producing a single graph
    break that preserves torch.compile effectiveness for the layer loop.
    """
    from vllm.forward_context import get_forward_context
    _fc = get_forward_context()

    # ── Static: from first layer's self_attn (representative for all layers) ──
    sin_cos_cache = first_self_attn.rotary_emb.cos_sin_cache
    mla_cos_cache = sin_cos_cache
    mla_sin_cache = sin_cos_cache

    _mla_kv = first_self_attn.mla_attn.mla_attn.kv_cache
    k_cache = _mla_kv
    v_cache = _mla_kv
    num_blocks = _mla_kv.shape[0] if _mla_kv is not None else 0

    _idx_rope = getattr(first_self_attn, "indexer_rope_emb", None)
    index_cos_sin_cache = _idx_rope.cos_sin_cache if _idx_rope is not None else None

    _indexer = getattr(first_self_attn, "indexer", None)
    _idx_k_cache = getattr(_indexer, "k_cache", None) if _indexer is not None else None
    index_k_buffer = getattr(_idx_k_cache, "kv_cache", None) if _idx_k_cache is not None else None
    index_k_scale_buffer = index_k_buffer

    # ── Dynamic: from forward_context (same for all layers in this step) ──
    # Use first layer's name to get representative metadata
    _attn_name = "model.layers.0.self_attn.attn"
    _swa_name = "model.layers.0.self_attn.swa_cache"
    _attn_meta = _fc.attn_metadata.get(_attn_name)
    if _attn_meta is None:
        _attn_meta = _fc.attn_metadata.get(_swa_name)
    _req_meta = getattr(_attn_meta, "req_metadata", None) if _attn_meta is not None else None

    block_tables = getattr(_req_meta, "block_table", None) if _req_meta is not None else None
    slot_mapping = _fc.slot_mapping.get(_attn_name)
    kv_seq_len = getattr(_req_meta, "seq_lens", None) if _req_meta is not None else None
    _qsl = getattr(_req_meta, "query_start_loc", None) if _req_meta is not None else None
    q_seq_len = (_qsl[1:] - _qsl[:-1]) if _qsl is not None else None
    num_tokens = getattr(_attn_meta, "num_actual_tokens", 0) if _attn_meta is not None else 0
    mask = getattr(_req_meta, "attn_mask", None) if _req_meta is not None else None
    mask_type = getattr(_attn_meta, "attn_state", None) if _attn_meta is not None else None

    return {
        "sin_cos_cache": sin_cos_cache,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "num_blocks": num_blocks,
        "block_tables": block_tables,
        "slot_mapping": slot_mapping,
        "kv_seq_len": kv_seq_len,
        "q_seq_len": q_seq_len,
        "num_tokens": num_tokens,
        "mask": mask,
        "mask_type": mask_type,
        "index_cos_sin_cache": index_cos_sin_cache,
        "index_k_buffer": index_k_buffer,
        "index_k_scale_buffer": index_k_scale_buffer,
        "mla_cos_cache": mla_cos_cache,
        "mla_sin_cache": mla_sin_cache,
    }

_sa = layer.self_attn

            # sin_cos_cache: 主注意力 RoPE 预计算表 [max_pos, rotary_dim]
            sin_cos_cache = _sa.rotary_emb.cos_sin_cache

            # mla_cos_cache / mla_sin_cache: MLA prolog 使用同一个 RoPE 表
            mla_cos_cache = sin_cos_cache
            mla_sin_cache = sin_cos_cache

            # k_cache / v_cache: MLA 绑定的 KV cache (由 model runner 通过 bind_kv_cache 设置)
            # MLAAttention.kv_cache 是一个 Tensor (压缩 latent + rope)，不是传统 K+V 分离
            _mla_kv = _sa.mla_attn.mla_attn.kv_cache
            k_cache = _mla_kv
            v_cache = _mla_kv
            num_blocks = _mla_kv.shape[0] if _mla_kv is not None else 0

            # index_cos_sin_cache: Indexer RoPE 预计算表
            _idx_rope = getattr(_sa, "indexer_rope_emb", None)
            index_cos_sin_cache = _idx_rope.cos_sin_cache if _idx_rope is not None else None

            # index_k_buffer: Indexer K cache (由 model runner 绑定到 Indexer.k_cache.kv_cache)
            _indexer = getattr(_sa, "indexer", None)
            _idx_k_cache = getattr(_indexer, "k_cache", None) if _indexer is not None else None
            index_k_buffer = getattr(_idx_k_cache, "kv_cache", None) if _idx_k_cache is not None else None

            # index_k_scale_buffer: Indexer K scale (与 k_buffer 打包在同一个 tensor 中)
            # k_cache head_dim = index_head_dim + index_head_dim // quant_block_size * 4
            # 前 index_head_dim 是 K 数据，后面是 scale
            index_k_scale_buffer = index_k_buffer

            # ── forward_context 动态元数据 (每步变化) ──
            _attn_name = f"model.layers.{idx}.self_attn.attn"
            _swa_name = f"model.layers.{idx}.self_attn.swa_cache"
            _attn_meta = _fc.attn_metadata.get(_attn_name)
            if _attn_meta is None:
                _attn_meta = _fc.attn_metadata.get(_swa_name)
            _req_meta = getattr(_attn_meta, "req_metadata", None) if _attn_meta is not None else None

            # block_tables: 分页 block 表 [num_reqs, max_blocks_per_req]
            block_tables = getattr(_req_meta, "block_table", None) if _req_meta is not None else None

            # slot_mapping: KV 写入槽位 [num_tokens]
            slot_mapping = _fc.slot_mapping.get(_attn_name)

            # kv_seq_len: 每请求 KV 序列长度 [num_reqs]
            kv_seq_len = getattr(_req_meta, "seq_lens", None) if _req_meta is not None else None

            # q_seq_len: 每请求 query 长度 [num_reqs]
            _qsl = getattr(_req_meta, "query_start_loc", None) if _req_meta is not None else None
            q_seq_len = (_qsl[1:] - _qsl[:-1]) if _qsl is not None else None

            # num_tokens: 实际 token 数
            num_tokens = getattr(_attn_meta, "num_actual_tokens", 0) if _attn_meta is not None else 0

            # mask: 显式注意力掩码 (DSA 通常为 None，掩码编码在 sas_metadata 中)
            mask = getattr(_req_meta, "attn_mask", None) if _req_meta is not None else None

            # mask_type: 注意力状态 (DecodeOnly / ChunkedPrefill / SpecDecoding)
            mask_type = getattr(_attn_meta, "attn_state", 

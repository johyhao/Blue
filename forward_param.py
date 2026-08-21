# ── Extract per-layer attention metadata ──
            attn_name = f"model.layers.{idx}.self_attn.attn"
            swa_name = f"model.layers.{idx}.self_attn.swa_cache"
            idx_k_name = f"model.layers.{idx}.self_attn.indexer.k_cache"

            attn_layer = forward_context.no_compile_layers.get(attn_name)
            attn_metadata = forward_context.attn_metadata.get(attn_name)
            if attn_metadata is None:
                attn_metadata = forward_context.attn_metadata.get(swa_name)
            req_metadata = getattr(attn_metadata, "req_metadata", None)

            # ── kv_cache tuple: [compress_kv, swa_kv, state, idx_state, idx_k, idx_scale] ──
            kv_cache_tuple = attn_layer.kv_cache if attn_layer is not None else None

            k_cache = kv_cache_tuple[0] if kv_cache_tuple is not None else None
            v_cache = kv_cache_tuple[1] if kv_cache_tuple is not None else None
            num_blocks = k_cache.shape[0] if k_cache is not None else 0

            # ── block_tables / slot_mapping ──
            block_tables = req_metadata.block_table if req_metadata is not None else None
            slot_mapping = forward_context.slot_mapping.get(attn_name)

            # ── kv_seq_len / q_seq_len ──
            kv_seq_len = req_metadata.seq_lens if req_metadata is not None else None
            query_start_loc = req_metadata.query_start_loc if req_metadata is not None else None
            q_seq_len = (query_start_loc[1:] - query_start_loc[:-1]) if query_start_loc is not None else None

            # ── num_tokens ──
            num_tokens = attn_metadata.num_actual_tokens if attn_metadata is not None else 0

            # ── mask / mask_type (DSA uses SAS metadata, not explicit mask) ──
            mask = req_metadata.attn_mask if req_metadata is not None else None
            mask_type = attn_metadata.attn_state if attn_metadata is not None else None

            # ── sin_cos_cache (RoPE cos/sin for main attention) ──
            sin_cos_cache = (
                (req_metadata.cos.get(attn_name), req_metadata.sin.get(attn_name))
                if req_metadata is not None and hasattr(req_metadata, "cos")
                else (None, None)
            )

            # ── mla_cos_cache / mla_sin_cache (same as main RoPE for MLA prolog) ──
            mla_cos_cache = sin_cos_cache[0]
            mla_sin_cache = sin_cos_cache[1]

            # ── index_cos_sin_cache (Indexer RoPE cos/sin) ──
            index_cos_sin_cache = (
                (req_metadata.cos.get(idx_k_name), req_metadata.sin.get(idx_k_name))
                if req_metadata is not None and hasattr(req_metadata, "cos")
                else (None, None)
            )

            # ── index_k_buffer / index_k_scale_buffer (Indexer K cache) ──
            index_k_buffer = kv_cache_tuple[4] if kv_cache_tuple is not None and len(kv_cache_tuple) > 4 else None
            index_k_scale_buffer = kv_cache_tuple[5] if kv_cache_tuple is not None and len(kv_cache_tuple) > 5 else None

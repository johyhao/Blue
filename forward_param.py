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

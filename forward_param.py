from vllm.forward_context import get_forward_context
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.device.device_op import DeviceOperator

def _patched_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds=None):
    ctx = get_forward_context()

    # ─── 全局信息 ───
    num_tokens = positions.shape[0]
    num_tokens_padded = _EXTRA_CTX.num_tokens

    # ─── 逐层提取 ───
    for idx in range(self.start_layer, self.end_layer):
        attn_name  = f"model.layers.{idx}.self_attn.attn"
        swa_name   = f"model.layers.{idx}.self_attn.swa_cache"
        idx_k_name = f"model.layers.{idx}.self_attn.indexer.k_cache"

        # 1. Attention metadata
        attn_meta = ctx.attn_metadata.get(attn_name) or ctx.attn_metadata.get(swa_name)
        req_meta = attn_meta.req_metadata

        # 2. KV Cache
        attn_layer = ctx.no_compile_layers[attn_name]
        kv_cache_tuple = attn_layer.kv_cache
        (compress_kv, swa_kv, state_cache,
         idx_k_cache, idx_scale_cache, _) = DeviceOperator.unpack_dsa_forward_kv_cache(
            kv_cache_tuple, compress_ratio=4)

        # 3. Block table & slot mapping
        block_table   = req_meta.block_table       # [num_reqs, max_blocks]
        slot_mapping  = req_meta.slot_mapping       # [num_tokens, 2]
        num_blocks    = compress_kv.shape[0]

        # 4. Sequence lengths
        kv_seq_len    = req_meta.seq_lens           # [num_reqs]
        query_start   = req_meta.query_start_loc    # [num_reqs+1]
        q_seq_len     = query_start[1:] - query_start[:-1]

        # 5. RoPE cos/sin
        mla_cos       = req_meta.cos[attn_name]     # [num_tokens, 1, 1, rope_dim]
        mla_sin       = req_meta.sin[attn_name]
        index_cos     = req_meta.cos.get(idx_k_name)
        index_sin     = req_meta.sin.get(idx_k_name)

        # 6. Mask info (DSA uses SAS metadata, not explicit mask)
        sas_metadata  = req_meta.sas_metadata        # [1024] int32
        ori_win_left  = req_meta.ori_win_left        # window_size - 1
        attn_state    = attn_meta.attn_state          # DecodeOnly / ChunkedPrefill

        # 7. Indexer K buffer
        index_k_buffer = idx_k_cache                 # [num_blocks, bs, 1, head_dim]
        index_k_scale  = idx_scale_cache             # [num_blocks, bs, 1, scale_dim]

    # ... 原有 forward 逻辑 ...

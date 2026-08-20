# vllm_ascend/patch/worker/patch_decoder_layer.py

from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer

_original_forward = DeepseekV2DecoderLayer.forward

def _external_decoder_layer_forward(
    self,
    positions,
    hidden_states,
    residual,
    llama_4_scaling=None,
):
    # ===== 自定义逻辑 =====
    # 方式 A: 完全替换 — 调用外部接口
    result_hidden, result_residual = external_service.compute(
        layer_idx=self.layer_idx,
        positions=positions,
        hidden_states=hidden_states,
        residual=residual,
    )
    return result_hidden, result_residual

    # 方式 B: 部分拦截 — 外部处理 attention，本地跑 MLP
    # if residual is None:
    #     residual = hidden_states.clone()
    #     hidden_states = self.input_layernorm(hidden_states)
    # else:
    #     hidden_states, residual = self.input_layernorm(hidden_states, residual)
    #
    # hidden_states = external_attn_service.call(
    #     layer_idx=self.layer_idx,
    #     positions=positions,
    #     hidden_states=hidden_states,
    # )
    #
    # hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    # hidden_states = self.mlp(hidden_states)
    # return hidden_states, residual

    # 方式 C: 条件分支 — 某些层走外部，其余走原始
    # if self.layer_idx in EXTERNAL_LAYERS:
    #     return external_service.compute(...)
    # return _original_forward(self, positions, hidden_states, residual, llama_4_scaling)

DeepseekV2DecoderLayer.forward = _external_decoder_layer_forward

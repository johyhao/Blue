#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Validate GLM-5.2 generation with MTP speculative decoding on 1 card.

Run pytest tests/e2e/pull_request/one_card/test_glm5_2.py.
"""

import os
from unittest.mock import patch

import pytest
from vllm.config import CompilationConfig
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner, cleanup_dist_env_and_memory

MAIN_MODEL = "Eco-Tech/GLM-5.2-w4a8"
MTP_NUM_SPECULATIVE_TOKENS = 3

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def _run_mtp_speculative_decoding(
    speculative_config: dict[str, object],
    num_speculative_tokens: int,
    compilation_config: CompilationConfig,
) -> list[float]:
    example_prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]

    with VllmRunner(
        MAIN_MODEL,
        quantization="ascend",
        tensor_parallel_size=1,
        max_model_len=2048,
        max_num_seqs=2,
        disable_log_stats=False,
        enforce_eager=True,
        speculative_config=speculative_config,
        compilation_config=compilation_config,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(example_prompts, max_tokens=64)
        metrics = vllm_model.model.get_metrics()

    assert len(outputs) == len(example_prompts)
    assert all(output_ids and output_text for output_ids, output_text in outputs)

    num_drafts = 0
    num_accepted_tokens_per_pos = [0] * num_speculative_tokens
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            assert len(metric.values) == num_speculative_tokens
            for pos, value in enumerate(metric.values):
                num_accepted_tokens_per_pos[pos] += value

    assert num_drafts > 0, "Speculative decoding did not generate any draft tokens"
    acceptance_per_pos = [accepted / num_drafts for accepted in num_accepted_tokens_per_pos]
    assert any(acceptance_per_pos), "Speculative decoding did not accept any draft tokens"
    assert all(0 <= acceptance <= 1 for acceptance in acceptance_per_pos)

    cleanup_dist_env_and_memory()
    return acceptance_per_pos


@pytest.mark.e2e_model(MAIN_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp",
    parallel="TP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W4A8",
    graph_mode="eager",
)
@patch.dict(
    os.environ,
    {
        "ASCEND_RT_VISIBLE_DEVICES": "0",
        "HCCL_BUFFSIZE": "512",
        "HCCL_OP_EXPANSION_MODE": "AIV",
    },
)
def test_glm_5_2_mtp_acceptance_tp1() -> None:
    _run_mtp_speculative_decoding(
        speculative_config={
            "method": "deepseek_mtp",
            "num_speculative_tokens": MTP_NUM_SPECULATIVE_TOKENS,
            "enforce_eager": True,
        },
        num_speculative_tokens=MTP_NUM_SPECULATIVE_TOKENS,
        compilation_config=CompilationConfig(
            cudagraph_mode="NONE",
        ),
    )

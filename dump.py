import os
import time

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

DUMP_DIR = os.getenv("VLLM_ASCEND_DUMP_MODEL_DIR", "")


def dump_model_weights(model: torch.nn.Module) -> None:
    if not DUMP_DIR:
        return

    from vllm.distributed import get_tensor_model_parallel_rank

    tp_rank = get_tensor_model_parallel_rank()
    rank_dir = os.path.join(DUMP_DIR, f"tp_rank_{tp_rank}")
    os.makedirs(rank_dir, exist_ok=True)

    logger.info("Dumping model weights to %s ...", rank_dir)
    t0 = time.perf_counter()

    try:
        from safetensors.torch import save_file

        state_dict = {}
        for name, param in model.named_parameters():
            state_dict[name] = param.data.detach().cpu()
        for name, buf in model.named_buffers():
            state_dict[name] = buf.data.detach().cpu()

        save_file(state_dict, os.path.join(rank_dir, "model.safetensors"))

        with open(os.path.join(rank_dir, "meta.txt"), "w") as f:
            for name, param in model.named_parameters():
                f.write(
                    f"{name}\tshape={list(param.shape)}\t"
                    f"dtype={param.dtype}\t"
                    f"device={param.device}\n"
                )
    except ImportError:
        torch.save(
            {n: p.data.detach().cpu() for n, p in model.named_parameters()},
            os.path.join(rank_dir, "model.pt"),
        )

    elapsed = time.perf_counter() - t0
    logger.info("Model dump done in %.2fs, saved to %s", elapsed, rank_dir)

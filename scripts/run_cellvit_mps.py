"""Run CellViT++ (cellvit==1.0.9) on Apple Silicon (MPS) without editing the installed package.

Same CLI as `cellvit-inference`. Runtime patches (checked against the cellvit 1.0.9 source):
  1. utils/ressource_manager.get_gpu_resources  – detects CUDA only -> report one MPS device.
     The reported memory (CELLVIT_MPS_MEMORY_GB, default 20) is what _check_devices uses to cap
     the batch size; < 22 GB keeps HIPT at ≤ 8.
  2. inference.py:188 / :241  self.device = "cuda:{gpu}"  -> "mps" before the model is moved.
  3. inference.py:778  autocast(device_type="cuda", fp16)  -> mixed precision off (fp32 on MPS).
  4. predictions are moved to CPU before they are sent to the ray post-processing actors, so the
     actors never need the GPU (post-processing uses the numpy backend; cupy is CUDA-only).
  5. inference/cli.py:187  asserts 0 <= gpu < torch.cuda.device_count()  -> accept gpu 0 on MPS.
A full grep of cellvit 1.0.9 finds no other CUDA references outside the optional cupy modules.
Ops not implemented on MPS fall back to CPU (PYTORCH_ENABLE_MPS_FALLBACK=1).

Must run in the `cellvit` conda env (cellvit pins numpy<2 etc., kept apart from `hemorph`).
"""
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402
from cellvit import detect_cells  # noqa: E402
from cellvit.inference import cli as CLI  # noqa: E402
from cellvit.inference import inference as INF  # noqa: E402
from cellvit.utils import ressource_manager as RM  # noqa: E402

MEMORY_GB = float(os.environ.get("CELLVIT_MPS_MEMORY_GB", "20"))
torch.set_num_threads(int(os.environ.get("CELLVIT_TORCH_THREADS", "4")))


def _gpu_resources(logger=None):
    if not torch.backends.mps.is_available():
        raise SystemError("MPS not available")
    return {"has_gpu": True, "gpu_count": 1, "details": {"backend": "mps"},
            "devices": {0: {"name": "Apple MPS", "total_memory_gb": MEMORY_GB}}}


RM.get_gpu_resources = _gpu_resources

_load_model = INF.CellViTInference._load_model
_setup_amp = INF.CellViTInference._setup_amp
_reorder = INF.CellViTInference.apply_softmax_reorder


def _load_model_mps(self):
    self.device = torch.device("mps")
    _load_model(self)


def _setup_amp_off(self, enforce_amp: bool = False):
    _setup_amp(self, enforce_amp=False)
    self.mixed_precision = False


def _reorder_to_cpu(self, predictions):
    out = _reorder(self, predictions)
    return {k: (v.detach().to("cpu") if torch.is_tensor(v) else v) for k, v in out.items()}


def _set_gpu_mps(self, config: dict) -> None:
    gpu = (config.get("inference") or {}).get("gpu")
    if gpu is not None:
        assert gpu == 0, "only one MPS device (gpu 0) on Apple Silicon"
        self.gpu = gpu


CLI.InferenceConfiguration._InferenceConfiguration__set_gpu = _set_gpu_mps
INF.CellViTInference._load_model = _load_model_mps
INF.CellViTInference._setup_amp = _setup_amp_off
INF.CellViTInference.apply_softmax_reorder = _reorder_to_cpu

if __name__ == "__main__":
    sys.exit(detect_cells.main())

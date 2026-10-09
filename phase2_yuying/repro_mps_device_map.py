"""Minimal repro: loading with device_map={"": "mps"} gives wrong gradients.

Computes the same masked-CE loss and backward pass several times on one fixed
batch and compares parameter gradients against a CPU reference. Run each mode in
a fresh process (the bug depends on the MPS allocator's prior state).

Usage:  python phase2_yuying/repro_mps_device_map.py [to_mps | device_map]
Expect: to_mps     -> 3.4e-04 on every call
        device_map -> 1.2e-02 on call 0 (every run); later calls vary run to run,
                      from recovering to 3.4e-04 up to 1e+05 and beyond
"""
import sys

import torch
import torch.nn.functional as F

import dllm

mode = sys.argv[1] if len(sys.argv) > 1 else "device_map"
BASE = ".models/a2d/Qwen3-0.6B"
Model = dllm.pipelines.a2d.A2DQwen3LMHeadModel

torch.manual_seed(0)
x = torch.randint(0, 150000, (12, 64))
masked = torch.rand(12, 64) < 0.5
inp = torch.where(masked, 151669, x)  # 151669 = <|mask|>


def layer_grads(model, device):
    model.zero_grad(set_to_none=True)
    logits = model(input_ids=inp.to(device), attention_mask=torch.ones_like(inp).to(device)).logits
    nll = F.cross_entropy(logits.transpose(1, 2), x.to(device), reduction="none")
    ((nll * masked.to(device)).sum() / x.numel()).backward()
    return torch.cat([
        p.grad.flatten() for n, p in model.named_parameters()
        if "layers.0." in n or "layers.27." in n
    ]).cpu()


ref = layer_grads(Model.from_pretrained(BASE, dtype=torch.float32).eval(), "cpu")

if mode == "device_map":
    model = Model.from_pretrained(BASE, dtype=torch.float32, device_map={"": "mps"})
else:
    model = Model.from_pretrained(BASE, dtype=torch.float32).to("mps")
model.eval()

errs = [((layer_grads(model, "mps") - ref).norm() / ref.norm()).item() for _ in range(5)]
print(f"{mode}: torch {torch.__version__}, grad rel err vs CPU per call:",
      " ".join(f"{e:.1e}" for e in errs))

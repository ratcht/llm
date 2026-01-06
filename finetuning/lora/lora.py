from torch.nn.utils import parametrize
import torch as t
import torch.nn as nn

class LoRAParameterization(nn.Module):
    def __init__(self, in_features, out_features, rank=1, alpha=1., device='cuda'):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.lora_A = nn.Parameter(t.randn((rank, self.in_features), dtype=t.float32).to(device))
        self.lora_B = nn.Parameter(t.zeros((self.out_features, rank), dtype=t.float32).to(device))

        self.scale = alpha/rank
        self.enabled = False
    
    def forward(self, w: t.Tensor):
        if self.enabled:
            assert w.shape == (self.out_features, self.in_features)
            delta = (self.lora_B @ self.lora_A) * self.scale
            return w + delta.to(dtype=w.dtype)
        return w

def apply_lora(model: nn.Module, target_modules=("q_proj"), rank=8, alpha=16):    
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        
        if not any(m in name for m in target_modules):
            continue
        
        parametrize.register_parametrization(
            module,
            "weight",
            LoRAParameterization(
                in_features=module.in_features,
                out_features=module.out_features,
                rank=rank,
                alpha=alpha,
            )
        )

def enable_lora(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    
    for m in model.modules():
        if not parametrize.is_parametrized(m, "weight"):
            continue
                
        m.parametrizations.weight[0].enabled = True
        for p in m.parametrizations.weight[0].parameters():
            p.requires_grad = True

    added_params = model.num_parameters(True)
    
    return added_params

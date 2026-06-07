import torch
from torch import nn


class ExpertPISM(nn.Module):
    # PISM/DeepSets 元网络：根据每个 client-expert 的状态特征生成聚合权重。
    def __init__(self, input_dim: int = 5, hidden_size: int = 64, dropout: float = 0.0, eps: float = 1e-12):
        super(ExpertPISM, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.eps = eps
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x, tau=1.0, mask=None, return_logits=False):
        if tau <= 0:
            raise ValueError("tau must be positive")

        squeeze_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True
        elif x.dim() != 3:
            raise ValueError("ExpertPISM input must have shape [N, D] or [B, N, D]")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"input feature dim must be {self.input_dim}, got {x.size(-1)}")

        batch_size, num_clients, _ = x.shape
        if mask is not None:
            mask = mask.to(device=x.device, dtype=torch.bool)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            if mask.shape != (batch_size, num_clients):
                raise ValueError("mask must have shape [N] or [B, N]")

        h = self.encoder(x)
        if mask is None:
            context = h.mean(dim=1, keepdim=True)
        else:
            mask_float = mask.unsqueeze(-1).to(dtype=h.dtype)
            valid_count = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
            context = (h * mask_float).sum(dim=1, keepdim=True) / valid_count

        context_expand = context.expand(-1, num_clients, -1)
        relation_input = torch.cat([h, context_expand], dim=-1)
        logits = self.decoder(relation_input).squeeze(-1)

        scaled_logits = logits / max(float(tau), self.eps)
        if mask is not None:
            # 若某组 mask 全 False，退化为全体 uniform；aggregator 后续应避免传入这种情况。
            all_invalid = ~mask.any(dim=1, keepdim=True)
            effective_mask = torch.where(all_invalid, torch.ones_like(mask), mask)
            scaled_logits = scaled_logits.masked_fill(~effective_mask, -1e9)
            scaled_logits = torch.where(
                all_invalid,
                torch.zeros_like(scaled_logits),
                scaled_logits,
            )

        weights = torch.softmax(scaled_logits, dim=1)
        if squeeze_batch:
            weights = weights.squeeze(0)
            logits = logits.squeeze(0)

        if return_logits:
            return {"weights": weights, "logits": logits}
        return weights


def normalize_pism_inputs(x, eps=1e-6):
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)
    if x.dim() == 2:
        client_dim = 0
    elif x.dim() == 3:
        client_dim = 1
    else:
        raise ValueError("PISM input must have shape [N, D] or [B, N, D]")

    mean = x.mean(dim=client_dim, keepdim=True)
    std = x.std(dim=client_dim, keepdim=True, unbiased=False)
    return (x - mean) / (std + eps)


def build_pism_feature_tensor(
    client_loss,
    expert_usage,
    delta_norm,
    total_layer_usage=None,
    device=None,
    dtype=None,
):
    # s_i,l,e / FOGA score 仍然只用于 meta loss 监督，不作为 PISM 输入特征。
    if dtype is None:
        dtype = torch.float32

    client_loss = torch.as_tensor(client_loss, device=device, dtype=dtype).reshape(-1)
    expert_usage = torch.as_tensor(expert_usage, device=device, dtype=dtype).reshape(-1)
    delta_norm = torch.as_tensor(delta_norm, device=device, dtype=dtype).reshape(-1)
    if total_layer_usage is None:
        total_layer_usage = expert_usage.clone()
    else:
        total_layer_usage = torch.as_tensor(
            total_layer_usage,
            device=device,
            dtype=dtype,
        ).reshape(-1)
    if not (
        client_loss.numel()
        == expert_usage.numel()
        == delta_norm.numel()
        == total_layer_usage.numel()
    ):
        raise ValueError(
            "client_loss, expert_usage, delta_norm, and total_layer_usage must have the same length"
        )

    eps = 1e-12
    expert_usage = expert_usage.clamp_min(0)
    delta_norm = delta_norm.clamp_min(0)
    total_layer_usage = total_layer_usage.clamp_min(0)
    total_for_ratio = torch.maximum(total_layer_usage, expert_usage).clamp_min(eps)
    expert_usage_ratio = expert_usage / total_for_ratio
    delta_norm_per_sqrt_usage = delta_norm / torch.sqrt(expert_usage + 1.0)

    return torch.stack(
        [
            client_loss,
            torch.log1p(expert_usage),
            expert_usage_ratio,
            torch.log1p(delta_norm),
            torch.log1p(delta_norm_per_sqrt_usage.clamp_min(0)),
        ],
        dim=-1,
    )


def build_grad_dot_pism_feature_tensor(
    expert_client_loss,
    client_grad_sample_count,
    device=None,
    dtype=None,
):
    """
    构造 grad_dot 版本的 PISM 输入：
    [
      expert_client_loss_i,l,e,
      log1p(client_grad_sample_count_i,l,e)
    ]

    注意：
    - FOGA score 不作为 PISM 输入。
    - score 只作为 meta loss 的监督信号。
    """
    if dtype is None:
        dtype = torch.float32

    expert_client_loss = torch.as_tensor(
        expert_client_loss,
        device=device,
        dtype=dtype,
    ).reshape(-1)
    client_grad_sample_count = torch.as_tensor(
        client_grad_sample_count,
        device=device,
        dtype=dtype,
    ).reshape(-1)
    if expert_client_loss.numel() != client_grad_sample_count.numel():
        raise ValueError(
            "expert_client_loss and client_grad_sample_count must have the same length"
        )

    client_grad_sample_count = client_grad_sample_count.clamp_min(0)
    return torch.stack(
        [
            expert_client_loss,
            torch.log1p(client_grad_sample_count),
        ],
        dim=-1,
    )

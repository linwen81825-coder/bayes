import math

import torch
import torch.nn.functional as F
from torch import nn


def make_group_norm(num_channels, max_groups=8):
    # GroupNorm 不依赖 batch statistics，更适合客户端数据分布不同的 FL 场景。
    for num_groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % num_groups == 0:
            return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    return nn.GroupNorm(num_groups=1, num_channels=num_channels)


class DenseFFN(nn.Module):
    # Transformer block 中的普通 dense FFN：D -> hidden -> D。
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout_rate=0.1):
        super(DenseFFN, self).__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class SwitchFFNExpert(nn.Module):
    # Token-level Switch FFN 中的单个专家，结构和 Transformer MLP 一致。
    def __init__(self, embed_dim, hidden_dim, dropout_rate=0.1):
        super(SwitchFFNExpert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class TokenSwitchFFN(nn.Module):
    # Token-level top-1 Switch FFN。
    # 输入/输出 shape 都是 [B, N, D]，router 会给每个 token 单独选择 1 个专家。
    def __init__(
        self,
        embed_dim,
        num_experts,
        mlp_ratio=4.0,
        dropout_rate=0.1,
        router_jitter_noise=0.0,
        capacity_factor=1.25,
        min_capacity=4,
        drop_tokens=True,
        top_k=1,
    ):
        super(TokenSwitchFFN, self).__init__()
        if top_k != 1:
            raise ValueError("TokenSwitchFFN currently supports top_k=1 only")

        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.router_jitter_noise = router_jitter_noise
        self.capacity_factor = capacity_factor
        self.min_capacity = min_capacity
        self.drop_tokens = drop_tokens
        hidden_dim = int(embed_dim * mlp_ratio)
        self.router = nn.Linear(embed_dim, num_experts)
        self.experts = nn.ModuleList([
            SwitchFFNExpert(embed_dim, hidden_dim, dropout_rate)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape
        router_input = x
        if self.training and self.router_jitter_noise > 0:
            noise = torch.empty_like(router_input).uniform_(
                1.0 - self.router_jitter_noise,
                1.0 + self.router_jitter_noise,
            )
            router_input = router_input * noise

        router_logits = self.router(router_input)
        router_probs = F.softmax(router_logits.float(), dim=-1).to(x.dtype)
        top1_probs, top1_indices = torch.max(router_probs, dim=-1)

        flat_x = x.reshape(batch_size * num_tokens, embed_dim)
        flat_output = torch.zeros_like(flat_x)
        flat_indices = top1_indices.reshape(-1)
        flat_top1_probs = top1_probs.reshape(-1)
        total_tokens = max(batch_size * num_tokens, 1)
        capacity = max(
            self.min_capacity,
            math.ceil(self.capacity_factor * total_tokens / self.num_experts),
        )
        selected_counts = torch.bincount(
            flat_indices,
            minlength=self.num_experts,
        ).to(x.device)
        expert_activations = torch.zeros(self.num_experts, device=x.device, dtype=torch.long)
        overflow_counts = torch.zeros(self.num_experts, device=x.device, dtype=torch.long)

        for expert_id, expert in enumerate(self.experts):
            token_positions = torch.nonzero(flat_indices == expert_id, as_tuple=False).flatten()
            if token_positions.numel() == 0:
                continue

            overflow_count = max(token_positions.numel() - capacity, 0)
            overflow_counts[expert_id] = overflow_count
            if self.drop_tokens:
                accepted_positions = token_positions[:capacity]
            else:
                accepted_positions = token_positions

            expert_activations[expert_id] = accepted_positions.numel()
            if accepted_positions.numel() > 0:
                expert_output = expert(flat_x[accepted_positions])
                flat_output[accepted_positions] = expert_output * flat_top1_probs[accepted_positions].unsqueeze(-1)

        # overflow token 的 FFN 增量保持为 0，外层 residual 会让这些 token 走 identity bypass。
        output = flat_output.reshape(batch_size, num_tokens, embed_dim)
        usage_fraction = selected_counts.float() / float(total_tokens)
        avg_router_probs = router_probs.float().mean(dim=(0, 1))

        # Switch Transformer load-balancing auxiliary loss:
        # aux = E * sum_e(f_e * p_e)
        # f_e 是 top-1 路由到 expert e 的 token 比例，p_e 是 router 对 expert e 的平均概率。
        # f_e 是离散选择，不参与梯度；p_e 保留梯度，推动 router 更均衡地使用专家。
        router_aux_loss = self.num_experts * torch.sum(usage_fraction.detach() * avg_router_probs)

        # router z-loss: mean(logsumexp(router_logits)^2)，抑制 router logits 过大。
        router_z_loss = torch.mean(torch.logsumexp(router_logits.float(), dim=-1) ** 2)

        return {
            "hidden": output,
            "router_aux_loss": router_aux_loss,
            "router_z_loss": router_z_loss,
            "expert_activations": expert_activations,
            "selected_counts": selected_counts,
            "overflow_counts": overflow_counts,
            "capacity": capacity,
            "avg_router_probs": avg_router_probs,
        }


class HybridTransformerBlock(nn.Module):
    # 标准 pre-norm Transformer block；FFN 可选择 dense 或 token-level Switch FFN。
    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_ratio=4.0,
        dropout_rate=0.1,
        use_switch_ffn=False,
        num_experts=8,
        router_jitter_noise=0.0,
        capacity_factor=1.25,
        min_capacity=4,
        drop_tokens=True,
        top_k=1,
        layer_id=0,
    ):
        super(HybridTransformerBlock, self).__init__()
        self.layer_id = layer_id
        self.use_switch_ffn = use_switch_ffn
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        if use_switch_ffn:
            self.ffn = TokenSwitchFFN(
                embed_dim=embed_dim,
                num_experts=num_experts,
                mlp_ratio=mlp_ratio,
                dropout_rate=dropout_rate,
                router_jitter_noise=router_jitter_noise,
                capacity_factor=capacity_factor,
                min_capacity=min_capacity,
                drop_tokens=drop_tokens,
                top_k=top_k,
            )
        else:
            self.ffn = DenseFFN(
                embed_dim=embed_dim,
                mlp_ratio=mlp_ratio,
                dropout_rate=dropout_rate,
            )

    def forward(self, x):
        norm_x = self.norm1(x)
        attention_out, _ = self.attention(norm_x, norm_x, norm_x, need_weights=False)
        x = x + self.dropout(attention_out)

        ffn_input = self.norm2(x)
        if self.use_switch_ffn:
            switch_result = self.ffn(ffn_input)
            x = x + self.dropout(switch_result["hidden"])
            return x, {
                "layer_id": self.layer_id,
                "router_aux_loss": switch_result["router_aux_loss"],
                "router_z_loss": switch_result["router_z_loss"],
                "expert_activations": switch_result["expert_activations"],
                "selected_counts": switch_result["selected_counts"],
                "overflow_counts": switch_result["overflow_counts"],
                "capacity": switch_result["capacity"],
                "avg_router_probs": switch_result["avg_router_probs"],
            }

        x = x + self.dropout(self.ffn(ffn_input))
        return x, None


class HybridSwitchTransformer(nn.Module):
    # Hybrid CNN Stem + Switch Transformer。
    # 这里先把 CNN stem 的 feature map token 化，再进入 Transformer blocks；
    # 只有 moe_layers 指定的部分 block 使用 token-level Switch FFN，其余 block 使用 dense FFN。
    def __init__(
        self,
        num_classes=100,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        num_experts=8,
        moe_layers=None,
        dropout_rate=0.1,
        router_jitter_noise=0.0,
        capacity_factor=1.25,
        min_capacity=4,
        drop_tokens=True,
        top_k=1,
        stem_channels=64,
        token_grid_size=8,
        use_cls_token=False,
        router_aux_loss_coef=0.01,
        router_z_loss_coef=0.001,
    ):
        super(HybridSwitchTransformer, self).__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.depth = depth
        self.moe_layers = set(moe_layers or [])
        self.token_grid_size = token_grid_size
        self.use_cls_token = use_cls_token
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_z_loss_coef = router_z_loss_coef

        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=3, stride=1, padding=1),
            make_group_norm(stem_channels),
            nn.GELU(),
            nn.Conv2d(stem_channels, stem_channels, kernel_size=3, stride=2, padding=1),
            make_group_norm(stem_channels),
            nn.GELU(),
        )
        self.token_pool = nn.AdaptiveAvgPool2d((token_grid_size, token_grid_size))
        self.token_projection = nn.Conv2d(stem_channels, embed_dim, kernel_size=1)
        num_position_tokens = token_grid_size * token_grid_size + (1 if use_cls_token else 0)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_position_tokens, embed_dim)
        )
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None
        self.position_dropout = nn.Dropout(dropout_rate)

        self.blocks = nn.ModuleList([
            HybridTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout_rate=dropout_rate,
                use_switch_ffn=layer_id in self.moe_layers,
                num_experts=num_experts,
                router_jitter_noise=router_jitter_noise,
                capacity_factor=capacity_factor,
                min_capacity=min_capacity,
                drop_tokens=drop_tokens,
                top_k=top_k,
                layer_id=layer_id,
            )
            for layer_id in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def get_expert_state_dict_by_layer(self):
        expert_states = {}
        for layer_id, block in enumerate(self.blocks):
            if not block.use_switch_ffn:
                continue
            expert_states[str(layer_id)] = {
                str(expert_id): expert.state_dict()
                for expert_id, expert in enumerate(block.ffn.experts)
            }
        return expert_states

    def get_router_state_dict_by_layer(self):
        router_states = {}
        for layer_id, block in enumerate(self.blocks):
            if block.use_switch_ffn:
                router_states[str(layer_id)] = block.ffn.router.state_dict()
        return router_states

    def get_moe_parameter_groups(self):
        parameter_groups = []
        for layer_id, block in enumerate(self.blocks):
            if not block.use_switch_ffn:
                continue
            parameter_groups.append({
                "type": "router",
                "layer_id": str(layer_id),
                "params": block.ffn.router.parameters(),
            })
            for expert_id, expert in enumerate(block.ffn.experts):
                parameter_groups.append({
                    "type": "expert",
                    "layer_id": str(layer_id),
                    "expert_id": str(expert_id),
                    "params": expert.parameters(),
                })
        return parameter_groups

    def forward(self, x):
        feature_map = self.stem(x)
        feature_map = self.token_pool(feature_map)
        tokens = self.token_projection(feature_map)
        tokens = tokens.flatten(2).transpose(1, 2)
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(tokens.size(0), -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = self.position_dropout(tokens + self.position_embedding)

        router_aux_loss = tokens.new_tensor(0.0)
        router_z_loss = tokens.new_tensor(0.0)
        expert_activations = torch.zeros(self.num_experts, device=tokens.device)
        selected_counts = torch.zeros(self.num_experts, device=tokens.device)
        overflow_counts = torch.zeros(self.num_experts, device=tokens.device)
        avg_router_probs = torch.zeros(self.num_experts, device=tokens.device)
        expert_stats_by_layer = {}
        expert_activations_by_layer = {}
        selected_counts_by_layer = {}
        overflow_counts_by_layer = {}
        avg_router_probs_by_layer = {}
        capacity_by_layer = {}
        switch_layer_count = 0

        for block in self.blocks:
            tokens, switch_stats = block(tokens)
            if switch_stats is None:
                continue

            layer_key = str(switch_stats["layer_id"])
            router_aux_loss = router_aux_loss + switch_stats["router_aux_loss"]
            router_z_loss = router_z_loss + switch_stats["router_z_loss"]
            expert_activations = expert_activations + switch_stats["expert_activations"]
            selected_counts = selected_counts + switch_stats["selected_counts"]
            overflow_counts = overflow_counts + switch_stats["overflow_counts"]
            avg_router_probs = avg_router_probs + switch_stats["avg_router_probs"]
            layer_stats = {
                "expert_activations": switch_stats["expert_activations"],
                "selected_counts": switch_stats["selected_counts"],
                "overflow_counts": switch_stats["overflow_counts"],
                "capacity": switch_stats["capacity"],
                "avg_router_probs": switch_stats["avg_router_probs"],
            }
            expert_stats_by_layer[layer_key] = layer_stats
            expert_activations_by_layer[layer_key] = layer_stats["expert_activations"]
            selected_counts_by_layer[layer_key] = layer_stats["selected_counts"]
            overflow_counts_by_layer[layer_key] = layer_stats["overflow_counts"]
            avg_router_probs_by_layer[layer_key] = layer_stats["avg_router_probs"]
            capacity_by_layer[layer_key] = layer_stats["capacity"]
            switch_layer_count += 1

        if switch_layer_count > 0:
            router_aux_loss = router_aux_loss / switch_layer_count
            router_z_loss = router_z_loss / switch_layer_count
            avg_router_probs = avg_router_probs / switch_layer_count

        total_router_loss = (
            self.router_aux_loss_coef * router_aux_loss
            + self.router_z_loss_coef * router_z_loss
        )
        tokens = self.norm(tokens)
        pooled = tokens[:, 0] if self.cls_token is not None else tokens.mean(dim=1)
        logits = self.classifier(pooled)

        return {
            "logits": logits,
            "feature": pooled,
            "aux_loss": router_aux_loss,
            "router_aux_loss": router_aux_loss,
            "router_z_loss": router_z_loss,
            "total_router_loss": total_router_loss,
            # Summary 仅用于日志展示；FL expert 聚合应优先使用 expert_stats_by_layer。
            "expert_activations": expert_activations,
            "expert_activations_summary": expert_activations,
            "selected_counts_summary": selected_counts,
            "overflow_counts_summary": overflow_counts,
            "avg_router_probs": avg_router_probs,
            "expert_stats_by_layer": expert_stats_by_layer,
            "expert_activations_by_layer": expert_activations_by_layer,
            "selected_counts_by_layer": selected_counts_by_layer,
            "overflow_counts_by_layer": overflow_counts_by_layer,
            "avg_router_probs_by_layer": avg_router_probs_by_layer,
            "capacity_by_layer": capacity_by_layer,
        }

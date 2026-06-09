from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50


@dataclass
class SwitchRouterStats:
    """
    单层 Switch FFN 的路由统计。

    注意：
    - router_probs / router_logits 是 flatten 后的 token 级张量，shape=[B*T, E]
    - selected_experts 是 token 级 top-1 expert id，shape=[B, T]
    - expert_counts 是本层所有 token 对每个 expert 的使用次数，shape=[E]
    - sample_expert_counts 是每个样本内部每个 expert 的 token 数，shape=[B, E]
    """

    router_probs: torch.Tensor
    expert_counts: torch.Tensor
    sample_expert_counts: torch.Tensor
    selected_experts: torch.Tensor
    aux_loss: torch.Tensor


class SwitchFeedForward(nn.Module):
    """
    VSMC 风格的 Top-1 Switch FFN。

    正常前向：
        router(x) -> softmax -> top1 expert -> expert_output * top1_prob

    expert 结构：
        Linear(hidden_dim, ffn_dim)
        GELU
        Dropout
        Linear(ffn_dim, hidden_dim)
        Dropout
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_experts: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_experts = int(num_experts)
        self.router = nn.Linear(hidden_dim, num_experts)

        # 保留 VSMC 原项目里的 server-side router adaptation 参数。
        # 只跑普通 FedAvg / 你的算法时，默认不会训练这些参数。
        self.router.server_temperature = nn.Parameter(
            torch.ones(()),
            requires_grad=False,
        )
        self.router.server_bias = nn.Parameter(
            torch.zeros(num_experts),
            requires_grad=False,
        )
        self.router.server_temperature_min = 0.5
        self.router.server_temperature_max = 2.0

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, ffn_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, hidden_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_experts)
            ]
        )

    def _apply_server_router_adapt(
        self,
        router_logits: torch.Tensor,
        server_adapt_config: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        兼容 VSMC 原项目的 server_temperature / server_bias。
        普通 FedAvg 和当前 MPSL 算法一般不会显式使用。
        """
        if hasattr(self.router, "server_temperature") and self.router.server_temperature is not None:
            temp_min = float(
                getattr(
                    server_adapt_config,
                    "server_adapt_router_temp_min",
                    getattr(self.router, "server_temperature_min", 0.5),
                )
            )
            temp_max = float(
                getattr(
                    server_adapt_config,
                    "server_adapt_router_temp_max",
                    getattr(self.router, "server_temperature_max", 2.0),
                )
            )
            router_logits = router_logits / self.router.server_temperature.clamp(
                temp_min,
                temp_max,
            )

        if hasattr(self.router, "server_bias") and self.router.server_bias is not None:
            router_logits = router_logits + self.router.server_bias

        return router_logits

    def forward(
        self,
        x: torch.Tensor,
        server_adapt_config: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, SwitchRouterStats]:
        """
        参数：
            x: [B, T, D]

        返回：
            output: [B, T, D]
            stats:  路由统计
        """
        batch_size, seq_len, hidden_dim = x.shape
        flat_x = x.reshape(batch_size * seq_len, hidden_dim)

        router_logits = self.router(flat_x)
        router_logits = self._apply_server_router_adapt(
            router_logits,
            server_adapt_config=server_adapt_config,
        )

        router_probs = F.softmax(router_logits, dim=-1)
        top1_probs, top1_experts = torch.max(router_probs, dim=-1)

        flat_output = torch.zeros_like(flat_x)

        for expert_id, expert in enumerate(self.experts):
            token_mask = top1_experts == expert_id
            if token_mask.any():
                flat_output[token_mask] = (
                    expert(flat_x[token_mask])
                    * top1_probs[token_mask].unsqueeze(-1)
                )

        expert_mask = F.one_hot(
            top1_experts,
            num_classes=self.num_experts,
        ).float()

        sample_expert_counts = expert_mask.view(
            batch_size,
            seq_len,
            self.num_experts,
        ).sum(dim=1)

        # Switch Transformer 常用 load-balancing auxiliary loss。
        density = expert_mask.mean(dim=0)
        density_proxy = router_probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(density * density_proxy)

        stats = SwitchRouterStats(
            router_probs=router_probs.detach(),
            expert_counts=expert_mask.sum(dim=0).detach(),
            sample_expert_counts=sample_expert_counts.detach(),
            selected_experts=top1_experts.view(batch_size, seq_len).detach(),
            aux_loss=aux_loss,
        )

        # 这些 raw 张量用于模型 forward 汇总分层统计，也用于 collect_uoc_evidence。
        # router_probs_raw 不 detach，方便将来如需 router adaptation 时保留梯度。
        stats.router_probs_raw = router_probs
        stats.router_logits_raw = router_logits
        stats.router_probs_tokens_raw = router_probs.view(
            batch_size,
            seq_len,
            self.num_experts,
        )
        stats.router_logits_tokens_raw = router_logits.view(
            batch_size,
            seq_len,
            self.num_experts,
        )

        return flat_output.reshape(batch_size, seq_len, hidden_dim), stats

    def forward_force_expert(
        self,
        x: torch.Tensor,
        expert_id: int,
        gate_mode: str = "one",
    ) -> torch.Tensor:
        """
        强制所有 token 经过指定 expert。

        这个接口只用于 UOC-FOGA / PISM 的 query 梯度计算，
        不影响正常 forward。

        参数：
            x: [B, T, D]，某层 FFN 输入 hidden
            expert_id: 强制使用的 expert 编号
            gate_mode:
                - "one": 不乘 router gate，直接使用 expert 输出
                - "top1_gate" / "router_prob": 乘当前 router 对该 expert 的概率
        """
        expert_id = int(expert_id)
        if expert_id < 0 or expert_id >= self.num_experts:
            raise ValueError(
                f"expert_id must be in [0, {self.num_experts}), got {expert_id}"
            )

        hidden_dim = x.size(-1)
        flat_x = x.reshape(-1, hidden_dim)
        forced_output = self.experts[expert_id](flat_x)

        if gate_mode in {"one", "force_one", None}:
            pass
        elif gate_mode in {"top1_gate", "router_prob", "prob"}:
            router_logits = self.router(flat_x)
            router_logits = self._apply_server_router_adapt(
                router_logits,
                server_adapt_config=None,
            )
            router_probs = F.softmax(router_logits, dim=-1)
            gate = router_probs[:, expert_id].unsqueeze(-1)
            forced_output = forced_output * gate
        else:
            raise ValueError(
                f"Unsupported gate_mode={gate_mode!r}. "
                f"Supported: 'one', 'top1_gate', 'router_prob'."
            )

        return forced_output.reshape_as(x)


class SwitchTransformerBlock(nn.Module):
    """
    VSMC 风格 Transformer block：

        x -> LN -> MHA -> residual
          -> LN -> Switch FFN -> residual
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_experts: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.switch_ffn = SwitchFeedForward(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        server_adapt_config: Optional[Any] = None,
        return_moe_inputs: bool = False,
    ) -> Tuple[torch.Tensor, SwitchRouterStats]:
        attn_input = self.attn_norm(x)
        attn_output, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            need_weights=False,
        )
        x = x + self.attn_dropout(attn_output)

        ffn_residual_input = x
        ffn_input = self.ffn_norm(x)
        ffn_output, stats = self.switch_ffn(
            ffn_input,
            server_adapt_config=server_adapt_config,
        )
        x = x + ffn_output

        if return_moe_inputs:
            # hidden 是 FFN 输入，即 ffn_norm 后的 token 表示。
            stats.moe_input_raw = ffn_input

            # residual 是 FFN 残差相加前的 block 状态。
            stats.block_residual_input_raw = ffn_residual_input

            # block_output_raw 是当前 block 完成后的输出。
            stats.block_output_raw = x

        return x, stats

    def forward_from_ffn_input(
        self,
        hidden: torch.Tensor,
        force_expert_id: int,
        gate_mode: str = "one",
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        从采集到的 FFN 输入 hidden 开始，强制经过指定 expert。

        参数：
            hidden:
                当前 MoE 层 FFN 输入，通常是 ffn_norm 后的 hidden，shape=[B,T,D]
            force_expert_id:
                强制使用的 expert id
            gate_mode:
                强制 expert 输出是否乘 gate
            residual:
                FFN 残差相加前的 block 状态，shape=[B,T,D]

        返回：
            当前 block 的输出，shape=[B,T,D]
        """
        forced_ffn_output = self.switch_ffn.forward_force_expert(
            hidden,
            expert_id=force_expert_id,
            gate_mode=gate_mode,
        )

        if residual is None:
            # 兼容没有 residual 的旧 evidence；不推荐但能避免直接崩。
            residual = hidden

        return residual + forced_ffn_output


class SwitchTransformerClassifier(nn.Module):
    """
    迁移到 MPSL 的 VSMC Switch Transformer 分类模型。

    模型结构：
        image
        -> tokenizer，可选 patch_embed / resnet18 / resnet34 / resnet50
        -> flatten tokens
        -> prepend CLS token
        -> position embedding
        -> SwitchTransformerBlock * num_layers
        -> LayerNorm
        -> classifier(CLS)

    注意：
    - forward 返回 MPSL client.py 需要的 dict。
    - 参数名保留 blocks.*.switch_ffn.experts.*，方便 MPSL 聚合器识别 expert 参数。
    - 额外补了 collect_uoc_evidence / forward_uoc_from_hidden，用于 UOC-FOGA / PISM。
    """

    def __init__(
        self,
        num_classes: int,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        backbone_name: str = "patch_embed",
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 256,
        num_experts: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.backbone_name = str(backbone_name)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.ffn_dim = int(ffn_dim)
        self.num_experts = int(num_experts)
        self.dropout_rate = float(dropout)

        self.tokenizer = self._build_tokenizer(
            backbone_name=self.backbone_name,
            image_size=self.image_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            hidden_dim=self.hidden_dim,
        )

        num_patches = self._infer_num_patches(
            image_size=self.image_size,
            in_channels=self.in_channels,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, self.hidden_dim)
        )
        self.dropout = nn.Dropout(self.dropout_rate)

        self.blocks = nn.ModuleList(
            [
                SwitchTransformerBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    ffn_dim=self.ffn_dim,
                    num_experts=self.num_experts,
                    dropout=self.dropout_rate,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.norm = nn.LayerNorm(self.hidden_dim)
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)

        self._init_parameters()

    def _build_tokenizer(
        self,
        backbone_name: str,
        image_size: int,
        patch_size: int,
        in_channels: int,
        hidden_dim: int,
    ) -> nn.Module:
        """
        构建图像 tokenizer。

        patch_embed:
            Conv2d 直接把图像切成 patch token。

        resnet18 / resnet34 / resnet50:
            CIFAR 版本 ResNet stem：
            conv1 改成 3x3 stride=1，去掉 maxpool，
            最后用 1x1 conv 投影到 hidden_dim。
        """
        if backbone_name == "patch_embed":
            if image_size % patch_size != 0:
                raise ValueError("image_size must be divisible by patch_size")
            return nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=patch_size,
                stride=patch_size,
            )

        resnet_builders = {
            "resnet18": (resnet18, 512),
            "resnet34": (resnet34, 512),
            "resnet50": (resnet50, 2048),
        }

        if backbone_name in resnet_builders:
            backbone_builder, feature_dim = resnet_builders[backbone_name]

            try:
                backbone = backbone_builder(weights=None)
            except TypeError:
                # 兼容旧版 torchvision。
                backbone = backbone_builder(pretrained=False)

            # CIFAR 小图像常用改法：3x3 stride=1，不用 maxpool。
            backbone.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            backbone.maxpool = nn.Identity()

            feature_extractor = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4,
                nn.Conv2d(
                    feature_dim,
                    hidden_dim,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            )
            return feature_extractor

        raise ValueError(f"Unsupported backbone_name: {backbone_name}")

    def _infer_num_patches(self, image_size: int, in_channels: int) -> int:
        """
        用 dummy input 自动推断 tokenizer 输出 token 数。
        """
        was_training = self.tokenizer.training
        self.tokenizer.eval()

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            token_map = self.tokenizer(dummy)

        self.tokenizer.train(was_training)
        return token_map.shape[-2] * token_map.shape[-1]

    def _init_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """
        图像转 token，并加入 CLS token 与位置编码。
        """
        x = self.tokenizer(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if x.size(1) != self.pos_embed.size(1):
            raise RuntimeError(
                f"Position embedding length mismatch: "
                f"x has {x.size(1)} tokens, pos_embed has {self.pos_embed.size(1)} tokens. "
                f"请检查 image_size / tokenizer 输出尺寸是否一致。"
            )

        return self.dropout(x + self.pos_embed)

    def _build_mpsl_output(
        self,
        logits: torch.Tensor,
        router_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        把 VSMC 原始 router_stats 转成 MPSL client.py 需要的 result dict。
        """
        zero = logits.new_tensor(0.0)

        expert_counts = router_stats.get("expert_counts")
        if expert_counts is None:
            expert_counts = torch.zeros(
                self.num_experts,
                device=logits.device,
                dtype=torch.float32,
            )

        router_probs = router_stats.get("router_probs")
        if router_probs is not None:
            avg_router_probs = router_probs.float().mean(dim=0)
        else:
            avg_router_probs = torch.zeros(
                self.num_experts,
                device=logits.device,
                dtype=torch.float32,
            )

        expert_stats_by_layer: Dict[str, Dict[str, Any]] = {}
        expert_activations_by_layer: Dict[str, torch.Tensor] = {}
        router_assignments_by_layer: Dict[str, Dict[str, torch.Tensor]] = {}

        expert_counts_by_layer = router_stats.get("expert_counts_by_layer")
        selected_experts_by_layer = router_stats.get("selected_experts_by_layer")
        router_probs_by_layer = router_stats.get("router_probs_by_layer")

        if expert_counts_by_layer is not None:
            for layer_id in range(expert_counts_by_layer.size(0)):
                layer_key = str(layer_id)
                layer_counts = expert_counts_by_layer[layer_id].float()

                if router_probs_by_layer is not None:
                    # router_probs_by_layer[layer_id]: [B*T, E]
                    layer_avg_probs = router_probs_by_layer[layer_id].float().mean(dim=0)
                else:
                    layer_avg_probs = torch.zeros_like(layer_counts)

                expert_stats_by_layer[layer_key] = {
                    "expert_activations": layer_counts,
                    "selected_counts": layer_counts,
                    "overflow_counts": torch.zeros_like(layer_counts),
                    "avg_router_probs": layer_avg_probs,
                    # VSMC 模型没有 capacity/drop token 机制，这里记为 0。
                    "capacity": 0,
                }
                expert_activations_by_layer[layer_key] = layer_counts

        if selected_experts_by_layer is not None:
            # selected_experts_by_layer: [L, B, T]
            for layer_id in range(selected_experts_by_layer.size(0)):
                router_assignments_by_layer[str(layer_id)] = {
                    "top1_expert_ids": selected_experts_by_layer[layer_id],
                }

        return {
            "logits": logits,
            "aux_loss": router_stats.get("aux_loss", zero),
            "router_aux_loss": router_stats.get("aux_loss", zero),
            "router_z_loss": zero,
            "expert_activations": expert_counts.float(),
            "avg_router_probs": avg_router_probs.float(),
            "expert_stats_by_layer": expert_stats_by_layer,
            "expert_activations_by_layer": expert_activations_by_layer,
            "router_assignments_by_layer": router_assignments_by_layer,
        }

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
        server_adapt_config: Optional[Any] = None,
        return_moe_inputs: bool = False,
    ) -> Dict[str, Any]:
        """
        MPSL 兼容 forward。

        参数 return_aux 保留只是为了兼容调用签名；
        不管 return_aux 是 True/False，这里都返回 dict。
        """
        x = self._tokenize(x)

        aux_losses = []
        expert_counts = []
        sample_expert_counts = []
        selected_experts_by_layer = []
        router_probs_by_layer = []
        router_logits_by_layer = []
        router_probs_tokens_by_layer = []
        router_logits_tokens_by_layer = []
        moe_inputs_by_layer = []
        block_residual_inputs_by_layer = []

        for block in self.blocks:
            x, stats = block(
                x,
                server_adapt_config=server_adapt_config,
                return_moe_inputs=return_moe_inputs,
            )

            aux_losses.append(stats.aux_loss)
            expert_counts.append(stats.expert_counts)
            sample_expert_counts.append(stats.sample_expert_counts)
            selected_experts_by_layer.append(stats.selected_experts)
            router_probs_by_layer.append(stats.router_probs_raw)
            router_logits_by_layer.append(stats.router_logits_raw)
            router_probs_tokens_by_layer.append(stats.router_probs_tokens_raw)
            router_logits_tokens_by_layer.append(stats.router_logits_tokens_raw)

            if return_moe_inputs:
                moe_inputs_by_layer.append(stats.moe_input_raw)
                block_residual_inputs_by_layer.append(stats.block_residual_input_raw)

        x = self.norm(x)
        logits = self.classifier(x[:, 0])

        aux_loss = torch.stack(aux_losses).mean() if aux_losses else logits.new_tensor(0.0)

        expert_counts_by_layer = torch.stack(expert_counts) if expert_counts else None
        sample_expert_counts_by_layer = (
            torch.stack(sample_expert_counts) if sample_expert_counts else None
        )
        selected_experts = (
            torch.stack(selected_experts_by_layer) if selected_experts_by_layer else None
        )
        router_probs_flat = (
            torch.cat(router_probs_by_layer, dim=0) if router_probs_by_layer else None
        )

        router_stats: Dict[str, Any] = {
            "aux_loss": aux_loss,
            "expert_counts": expert_counts_by_layer.sum(dim=0)
            if expert_counts_by_layer is not None
            else None,
            "expert_counts_by_layer": expert_counts_by_layer,
            "sample_expert_counts_by_layer": sample_expert_counts_by_layer,
            "selected_experts": selected_experts.reshape(-1)
            if selected_experts is not None
            else None,
            "selected_experts_by_layer": selected_experts,
            "router_probs": router_probs_flat,
            "router_probs_by_layer": torch.stack(router_probs_by_layer)
            if router_probs_by_layer
            else None,
            "router_logits_by_layer": torch.stack(router_logits_by_layer)
            if router_logits_by_layer
            else None,
            "router_probs_tokens_by_layer": torch.stack(router_probs_tokens_by_layer)
            if router_probs_tokens_by_layer
            else None,
            "router_logits_tokens_by_layer": torch.stack(router_logits_tokens_by_layer)
            if router_logits_tokens_by_layer
            else None,
        }

        if return_moe_inputs:
            router_stats["moe_inputs_by_layer"] = (
                torch.stack(moe_inputs_by_layer) if moe_inputs_by_layer else None
            )
            router_stats["block_residual_inputs_by_layer"] = (
                torch.stack(block_residual_inputs_by_layer)
                if block_residual_inputs_by_layer
                else None
            )

        return self._build_mpsl_output(logits=logits, router_stats=router_stats)

    @torch.no_grad()
    def collect_uoc_evidence(
        self,
        x: torch.Tensor,
        max_samples: Optional[int] = None,
        use_top1: bool = True,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        采集 UOC-FOGA / PISM 需要的 evidence。

        返回格式：
            {
                "0": {
                    "hidden": [B,T,D],
                    "residual": [B,T,D],
                    "top1_expert_ids": [B,T],
                    "top1_gates": [B,T],
                    "entropy": [B],
                },
                "1": {...}
            }

        说明：
        - hidden 是当前 MoE 层 FFN 输入，即 ffn_norm 后的 token 表示。
        - residual 是 FFN 残差相加前的 block 状态。
        - top1_expert_ids 是原 router 正常选择的 expert。
        - top1_gates 是 router 对 top1 expert 的概率。
        """
        if not use_top1:
            raise ValueError("VSMC collect_uoc_evidence currently supports use_top1=True only.")

        if max_samples is not None:
            x = x[:max_samples]

        was_training = self.training
        self.eval()

        try:
            tokens = self._tokenize(x)
            uoc_evidence: Dict[str, Dict[str, torch.Tensor]] = {}

            for layer_id, block in enumerate(self.blocks):
                tokens, stats = block(
                    tokens,
                    server_adapt_config=None,
                    return_moe_inputs=True,
                )

                router_probs_tokens = stats.router_probs_tokens_raw.detach()
                top1_expert_ids = stats.selected_experts.detach()
                top1_gates = router_probs_tokens.gather(
                    dim=-1,
                    index=top1_expert_ids.unsqueeze(-1),
                ).squeeze(-1)

                uoc_evidence[str(layer_id)] = {
                    "hidden": stats.moe_input_raw.detach(),
                    "residual": stats.block_residual_input_raw.detach(),
                    "top1_expert_ids": top1_expert_ids.detach(),
                    "top1_gates": top1_gates.detach(),
                }

            final_tokens = self.norm(tokens)
            logits = self.classifier(final_tokens[:, 0])
            probs = torch.softmax(logits.float(), dim=-1)
            entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).detach()

            for layer_evidence in uoc_evidence.values():
                layer_evidence["entropy"] = entropy.detach()

            return uoc_evidence

        finally:
            self.train(was_training)

    def forward_uoc_from_hidden(
        self,
        hidden: torch.Tensor,
        layer_id: int | str,
        force_expert_id: int,
        gate_mode: str = "one",
        residual: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        从某一层 MoE 的 hidden 开始，强制经过指定 expert，
        然后继续跑后续 Transformer block 和 classifier。

        这个接口只用于服务器端 UOC-FOGA / PISM 聚合。
        """
        layer_index = int(layer_id)
        if layer_index < 0 or layer_index >= len(self.blocks):
            raise ValueError(f"layer_id is outside model depth: {layer_id}")

        block = self.blocks[layer_index]

        x = block.forward_from_ffn_input(
            hidden=hidden,
            force_expert_id=force_expert_id,
            gate_mode=gate_mode,
            residual=residual,
        )

        for next_block in self.blocks[layer_index + 1:]:
            x, _ = next_block(
                x,
                server_adapt_config=None,
                return_moe_inputs=False,
            )

        x = self.norm(x)
        logits = self.classifier(x[:, 0])

        return {"logits": logits}

    def forward_from_block_output(
        self,
        hidden: torch.Tensor,
        start_block_index: int = 0,
        server_adapt_config: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        保留 VSMC 原项目的辅助接口。
        普通 FedAvg 不会用到。
        """
        x = hidden
        for block in self.blocks[start_block_index:]:
            x, _ = block(
                x,
                server_adapt_config=server_adapt_config,
                return_moe_inputs=False,
            )
        x = self.norm(x)
        return self.classifier(x[:, 0])
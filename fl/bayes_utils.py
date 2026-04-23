import collections
import math
from collections import OrderedDict

import torch


def parse_expert_ref(key):
    parts = key.split(".")
    if "blocks" not in parts or "experts" not in parts:
        return None

    blocks_idx = parts.index("blocks")
    experts_idx = parts.index("experts")
    if blocks_idx + 1 >= len(parts) or experts_idx + 1 >= len(parts):
        return None
    if not parts[blocks_idx + 1].isdigit() or not parts[experts_idx + 1].isdigit():
        return None

    return parts[blocks_idx + 1], parts[experts_idx + 1]


def group_expert_keys(state_dict):
    grouped = collections.OrderedDict()
    for key in state_dict.keys():
        expert_ref = parse_expert_ref(key)
        if expert_ref is None:
            continue
        grouped.setdefault(expert_ref, []).append(key)
    return grouped


def get_client_expert_evidence(client_evidence, layer_id, expert_id):
    if not isinstance(client_evidence, dict):
        return None
    return (
        client_evidence.get(str(layer_id), {})
        .get(str(expert_id))
    )


def get_bayes_expert_state(bayes_state, layer_id, expert_id):
    if not isinstance(bayes_state, dict):
        return None
    return (
        bayes_state.get("experts", {})
        .get(str(layer_id), {})
        .get(str(expert_id))
    )


def get_expert_parameter_keys(state_dict, layer_id, expert_id):
    prefix = f"blocks.{layer_id}.ffn.experts.{expert_id}."
    return [key for key in state_dict.keys() if key.startswith(prefix)]


def freeze_all_but_target_expert(model, layer_id, expert_id):
    prefix = f"blocks.{layer_id}.ffn.experts.{expert_id}."
    target_names = []
    target_params = []
    for name, param in model.named_parameters():
        is_target = name.startswith(prefix)
        param.requires_grad_(is_target)
        if is_target:
            target_names.append(name)
            target_params.append(param)
    return target_names, target_params


def vector_to_named_state(reference_state, keys, vector):
    named_state = OrderedDict()
    offset = 0
    for key in keys:
        reference_tensor = reference_state[key].detach().cpu()
        numel = reference_tensor.numel()
        named_state[key] = vector[offset: offset + numel].view_as(reference_tensor).clone()
        offset += numel

    if offset != vector.numel():
        raise ValueError("Vector size does not match the requested expert state layout")
    return named_state


def compute_optimal_local_posterior(
    local_mean,
    local_precision,
    prior_mean,
    prior_precision,
    prior_n0,
    min_precision=1e-6,
):
    # 对应 Word / NIWMeta 的闭式解：
    # gam*_ik = A_ik + n0_k * gam0_k
    # m*_ik = (A_ik * m_ik + n0_k * gam0_k * m0_k) / gam*_ik
    dtype = prior_mean.dtype
    device = prior_mean.device
    local_mean = local_mean.to(device=device, dtype=dtype)
    local_precision = local_precision.to(device=device, dtype=dtype).clamp(min=min_precision)
    prior_precision = prior_precision.to(device=device, dtype=dtype).clamp(min=min_precision)
    if torch.is_tensor(prior_n0):
        prior_n0 = prior_n0.to(device=device, dtype=dtype)
    else:
        prior_n0 = torch.tensor(prior_n0, device=device, dtype=dtype)
    prior_n0 = prior_n0.clamp(min=min_precision)

    posterior_precision = (local_precision + prior_n0 * prior_precision).clamp(min=min_precision)
    posterior_mean = (
        local_precision * local_mean
        + prior_n0 * prior_precision * prior_mean
    ) / posterior_precision
    posterior_variance = (1.0 / posterior_precision).clamp(min=min_precision)
    return posterior_mean, posterior_variance, posterior_precision


def compute_quadratic_meta_terms(
    local_mean,
    local_precision,
    prior_mean,
    prior_precision,
    prior_n0,
    posterior_mean,
    posterior_precision,
    min_precision=1e-6,
):
    # 用客户端上传的二次近似局部证据替代原始任务 loss：
    # f_i ≈ E_q[ 1/2 * A_i * (theta - m_i)^2 ]
    # g_i 采用 Word / NIWMeta 的对角化先验正则形式。
    dtype = prior_mean.dtype
    device = prior_mean.device
    local_mean = local_mean.to(device=device, dtype=dtype)
    local_precision = local_precision.to(device=device, dtype=dtype).clamp(min=min_precision)
    prior_precision = prior_precision.to(device=device, dtype=dtype).clamp(min=min_precision)
    posterior_precision = posterior_precision.to(device=device, dtype=dtype).clamp(min=min_precision)
    posterior_mean = posterior_mean.to(device=device, dtype=dtype)
    if torch.is_tensor(prior_n0):
        prior_n0 = prior_n0.to(device=device, dtype=dtype)
    else:
        prior_n0 = torch.tensor(prior_n0, device=device, dtype=dtype)
    prior_n0 = prior_n0.clamp(min=min_precision)

    fit_term = 0.5 * torch.sum(
        local_precision
        * (
            (posterior_mean - local_mean).square()
            + 1.0 / posterior_precision
        )
    )

    dim = local_mean.numel()
    digamma_term = torch.digamma(0.5 * prior_n0)
    regularizer = (
        -torch.log(prior_precision).sum()
        + torch.log(posterior_precision).sum()
        - dim * digamma_term
        + prior_n0 * (prior_precision / posterior_precision).sum()
        + prior_n0 * (prior_precision * (posterior_mean - prior_mean).square()).sum()
        - dim * (math.log(2.0) + 1.0)
    )
    return fit_term, regularizer


def run_expert_sgld_fit(
    model,
    batch_cache,
    criterion,
    layer_id,
    expert_id,
    device,
    steps,
    burnin,
    alp,
    ai_max,
):
    # 参考 fewshot_vit/niwmeta.py 中的 run_sgld_gaussian_fit：
    # 对 burn-in 后的参数样本做均值 / 方差统计，再把方差倒数当作 precision。
    if len(batch_cache) == 0:
        raise ValueError("SGLD evidence extraction requires at least one cached batch")

    target_names, target_params = freeze_all_but_target_expert(
        model=model,
        layer_id=layer_id,
        expert_id=expert_id,
    )
    if len(target_params) == 0:
        raise ValueError(f"Missing target expert parameters for layer {layer_id}, expert {expert_id}")

    model.to(device)
    model.train()

    total_samples = sum(int(labels.size(0)) for _, labels in batch_cache)
    total_samples = max(total_samples, 1)
    sgld_lr = float(alp) / float(total_samples)
    sgld_lr = max(sgld_lr, 1e-12)
    optim = torch.optim.Adam(params=target_params, lr=sgld_lr)

    moment1 = None
    moment2 = None
    sample_count = 0
    noise_scale = math.sqrt(1.0 / sgld_lr)

    fallback_precision = 1e-4

    for step_idx in range(steps):
        weighted_loss = None
        seen_samples = 0
        optim.zero_grad()

        for cached_inputs, cached_labels in batch_cache:
            inputs = cached_inputs.to(device, non_blocking=True)
            labels = cached_labels.to(device, non_blocking=True)
            result = model(inputs)
            logits = result["logits"]
            batch_loss = criterion(logits, labels)
            if not batch_loss.requires_grad:
                # 该 batch 中目标 expert 没有被实际路由到，loss 对它没有梯度；
                # 跳过这一批，继续在缓存 batch 里寻找真正能提供局部证据的样本。
                continue
            batch_weight = labels.size(0)
            weighted_term = batch_loss * batch_weight
            weighted_loss = weighted_term if weighted_loss is None else weighted_loss + weighted_term
            seen_samples += batch_weight

        if weighted_loss is None or seen_samples <= 0:
            break

        loss = weighted_loss / float(seen_samples)
        loss.backward()

        with torch.no_grad():
            for param in target_params:
                if param.grad is None:
                    continue
                param.grad.mul_(float(seen_samples) / 2.0)
                param.grad.add_(noise_scale * torch.randn_like(param))

        optim.step()

        with torch.no_grad():
            param_vector = torch.nn.utils.parameters_to_vector(
                [param.detach() for param in target_params]
            ).detach().cpu()

            if step_idx == burnin:
                moment1 = param_vector.clone()
                moment2 = param_vector.square()
                sample_count = 1
            elif step_idx > burnin:
                moment1 = (param_vector + sample_count * moment1) / (sample_count + 1)
                moment2 = (param_vector.square() + sample_count * moment2) / (sample_count + 1)
                sample_count += 1

    with torch.no_grad():
        reference_state = model.state_dict()
        if moment1 is None:
            mean_vector = torch.nn.utils.parameters_to_vector(
                [param.detach() for param in target_params]
            ).detach().cpu()
            precision_vector = torch.full_like(mean_vector, fill_value=fallback_precision)
        elif sample_count <= 1:
            mean_vector = moment1
            precision_vector = torch.full_like(mean_vector, fill_value=fallback_precision)
        else:
            unbiased_var = (sample_count / (sample_count - 1.0)) * (moment2 - moment1.square())
            precision_vector = (1.0 / unbiased_var.clamp(min=1e-12)).clamp(min=1e-4, max=ai_max)
            mean_vector = moment1

    mean_state = vector_to_named_state(reference_state, target_names, mean_vector)
    precision_state = vector_to_named_state(reference_state, target_names, precision_vector)
    return mean_state, precision_state

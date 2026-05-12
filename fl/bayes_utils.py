import collections
import math
import time
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


def _vector_segments(reference_state, keys):
    offset = 0
    segments = []
    for key in keys:
        numel = reference_state[key].numel()
        segments.append((key, offset, offset + numel))
        offset += numel
    return segments


def _calibrate_precision_vector(
    raw_var,
    reference_state,
    target_names,
    mode,
    var_floor,
    ai_max,
    temperature,
    target,
    min_value,
    max_value,
    eps,
):
    mode = str(mode or "floor_inverse").lower()
    eps = max(float(eps), 1e-12)
    if mode == "floor_inverse":
        effective_var_floor = max(float(var_floor), eps)
        precision = 1.0 / raw_var.clamp(min=effective_var_floor)
        return precision.clamp(min=1e-4, max=float(ai_max)), effective_var_floor

    temperature = max(float(temperature), 1e-8)
    target = max(float(target), eps)
    min_value = max(float(min_value), eps)
    max_value = max(float(max_value), min_value)

    if mode == "normalized_power":
        raw_precision = 1.0 / raw_var.clamp(min=eps)
        tempered = raw_precision.clamp(min=eps).pow(temperature)
        normalizer = tempered.mean().clamp(min=eps)
        precision = tempered / normalizer * target
        return precision.clamp(min=min_value, max=max_value), eps

    if mode == "scalar_normalized_power":
        segments = _vector_segments(reference_state, target_names)
        scalar_precisions = []
        scalar_weights = []
        for _, start, end in segments:
            raw_var_mean = raw_var[start:end].mean().clamp(min=eps)
            scalar_precisions.append(1.0 / raw_var_mean)
            scalar_weights.append(float(end - start))

        scalar_vector = torch.stack(scalar_precisions)
        tempered = scalar_vector.clamp(min=eps).pow(temperature)
        weight_vector = torch.tensor(scalar_weights, dtype=tempered.dtype, device=tempered.device)
        normalizer = (tempered * weight_vector).sum() / weight_vector.sum().clamp(min=eps)
        normalizer = normalizer.clamp(min=eps)
        calibrated_scalars = (tempered / normalizer * target).clamp(min=min_value, max=max_value)

        precision = torch.empty_like(raw_var)
        for scalar, (_, start, end) in zip(calibrated_scalars, segments):
            precision[start:end] = scalar
        return precision, eps

    raise ValueError(
        "Unknown bayes_precision_mode: "
        f"{mode}. Expected one of: floor_inverse, normalized_power, scalar_normalized_power"
    )


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
    var_floor=0.0,
    precision_mode="floor_inverse",
    precision_temperature=0.25,
    precision_target=100.0,
    precision_min=20.0,
    precision_max=300.0,
    precision_eps=1.0e-12,
    sgld_concat_cache=False,
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

    def _first_param_device(module):
        for param in module.parameters():
            return param.device
        return torch.device("cpu")

    def _normalize_device(device_like):
        resolved = torch.device(device_like)
        if resolved.type == "cuda" and resolved.index is None and torch.cuda.is_available():
            resolved = torch.device(f"cuda:{torch.cuda.current_device()}")
        return resolved

    target_device = _normalize_device(device)
    current_device = _first_param_device(model)
    if current_device != target_device:
        model.to(target_device)
    model.train()

    total_samples = sum(int(labels.size(0)) for _, labels in batch_cache)
    total_samples = max(total_samples, 1)
    # Match fewshot_vit/niwmeta.py as closely as the MoE setting allows:
    # Adam uses alp / N, while Langevin noise is injected into gradients.
    sgld_lr = float(alp) / float(total_samples)
    sgld_lr = max(sgld_lr, 1e-12)
    var_floor = max(float(var_floor), 0.0)
    optim = torch.optim.Adam(params=target_params, lr=sgld_lr)
    noise_scale = math.sqrt(1.0 / sgld_lr)

    prepare_start = time.perf_counter()
    prepared_batch_cache = []
    for cached_inputs, cached_labels in batch_cache:
        if cached_inputs.device != target_device:
            cached_inputs = cached_inputs.to(target_device, non_blocking=True)
        if cached_labels.device != target_device:
            cached_labels = cached_labels.to(target_device, non_blocking=True)
        prepared_batch_cache.append((cached_inputs, cached_labels))

    concat_cached_samples = int(total_samples)
    if sgld_concat_cache:
        concat_inputs = torch.cat([cached_inputs for cached_inputs, _ in prepared_batch_cache], dim=0)
        concat_labels = torch.cat([cached_labels for _, cached_labels in prepared_batch_cache], dim=0)
        prepared_batch_cache = [(concat_inputs, concat_labels)]
        concat_cached_samples = int(concat_labels.size(0))
    prepare_cache_time_sec = time.perf_counter() - prepare_start

    moment1 = None
    moment2 = None
    sample_count = 0

    fallback_precision = 1e-4
    last_seen_samples = 0
    forward_backward_time_sec = 0.0
    sample_moment_time_sec = 0.0

    for step_idx in range(steps):
        weighted_loss = None
        seen_samples = 0
        optim.zero_grad(set_to_none=True)

        step_fb_start = time.perf_counter()
        for cached_inputs, cached_labels in prepared_batch_cache:
            inputs = cached_inputs
            labels = cached_labels
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
            forward_backward_time_sec += time.perf_counter() - step_fb_start
            break

        last_seen_samples = int(seen_samples)
        loss = weighted_loss / float(seen_samples)
        loss.backward()

        with torch.no_grad():
            grad_scale = float(seen_samples) / 2.0
            for param in target_params:
                if param.grad is None:
                    continue
                param.grad.mul_(grad_scale)
                param.grad.add_(noise_scale * torch.randn_like(param))

        optim.step()
        forward_backward_time_sec += time.perf_counter() - step_fb_start

        step_sample_start = time.perf_counter()
        with torch.no_grad():
            param_vector = torch.nn.utils.parameters_to_vector(
                [param.detach() for param in target_params]
            ).detach()

            if step_idx == burnin:
                moment1 = param_vector.clone()
                moment2 = param_vector.square()
                sample_count = 1
            elif step_idx > burnin:
                moment1 = (param_vector + sample_count * moment1) / (sample_count + 1)
                moment2 = (param_vector.square() + sample_count * moment2) / (sample_count + 1)
                sample_count += 1
        sample_moment_time_sec += time.perf_counter() - step_sample_start

    with torch.no_grad():
        reference_state = model.state_dict()
        sgld_diag = {
            "sample_count": int(sample_count),
            "total_cached_samples": int(total_samples),
            "last_seen_samples": int(last_seen_samples),
            "sgld_lr": float(sgld_lr),
            "sgld_var_floor": float(var_floor),
            "precision_mode": str(precision_mode),
            "precision_temperature": float(precision_temperature),
            "precision_target": float(precision_target),
            "raw_var_mean": None,
            "raw_var_min": None,
            "raw_var_max": None,
            "raw_var_under_floor_pct": None,
            "unclipped_precision_mean": None,
            "unclipped_precision_min": None,
            "unclipped_precision_max": None,
            "unclipped_precision_over_ai_max_pct": None,
            "sgld_fit_time_sec": None,
            "sgld_forward_backward_time_sec": round(float(forward_backward_time_sec), 6),
            "sgld_sample_moment_time_sec": round(float(sample_moment_time_sec), 6),
            "sgld_prepare_cache_time_sec": round(float(prepare_cache_time_sec), 6),
            "sgld_device": str(target_device),
            "sgld_cache_device": str(prepared_batch_cache[0][0].device) if prepared_batch_cache else str(target_device),
            "sgld_param_vector_device": None,
            "sgld_concat_cache": bool(sgld_concat_cache),
            "concat_cached_samples": int(concat_cached_samples),
        }
        if moment1 is None:
            mean_vector = torch.nn.utils.parameters_to_vector(
                [param.detach() for param in target_params]
            ).detach()
            precision_vector = torch.full_like(mean_vector, fill_value=fallback_precision)
        elif sample_count <= 1:
            mean_vector = moment1
            precision_vector = torch.full_like(mean_vector, fill_value=fallback_precision)
        else:
            unbiased_var = (sample_count / (sample_count - 1.0)) * (moment2 - moment1.square())
            raw_var = unbiased_var.clamp(min=0.0)
            precision_vector, effective_var_floor = _calibrate_precision_vector(
                raw_var=raw_var,
                reference_state=reference_state,
                target_names=target_names,
                mode=precision_mode,
                var_floor=var_floor,
                ai_max=ai_max,
                temperature=precision_temperature,
                target=precision_target,
                min_value=precision_min,
                max_value=precision_max,
                eps=precision_eps,
            )
            precision_before_clip = 1.0 / raw_var.clamp(min=max(float(precision_eps), 1e-12))
            sgld_diag.update(
                {
                    "raw_var_mean": round(float(raw_var.mean().item()), 12),
                    "raw_var_min": round(float(raw_var.min().item()), 12),
                    "raw_var_max": round(float(raw_var.max().item()), 12),
                    "raw_var_under_floor_pct": round(
                        float((raw_var <= effective_var_floor).float().mean().item()),
                        6,
                    ),
                    "unclipped_precision_mean": round(float(precision_before_clip.mean().item()), 6),
                    "unclipped_precision_min": round(float(precision_before_clip.min().item()), 6),
                    "unclipped_precision_max": round(float(precision_before_clip.max().item()), 6),
                    "unclipped_precision_over_ai_max_pct": round(
                        float((precision_before_clip >= float(ai_max)).float().mean().item()),
                        6,
                    ),
                }
            )
            mean_vector = moment1

    sgld_diag["sgld_fit_time_sec"] = round(
        float(prepare_cache_time_sec + forward_backward_time_sec + sample_moment_time_sec),
        6,
    )
    sgld_diag["sgld_param_vector_device"] = str(mean_vector.device)

    mean_state = vector_to_named_state(reference_state, target_names, mean_vector.detach().cpu())
    precision_state = vector_to_named_state(reference_state, target_names, precision_vector.detach().cpu())
    return mean_state, precision_state, sgld_diag

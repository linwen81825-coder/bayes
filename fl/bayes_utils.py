import collections
import logging
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
    mean_vector=None,
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

    if mode == "relative_normalized_power":
        if mean_vector is None:
            raise ValueError("relative_normalized_power requires mean_vector for parameter scaling")
        mean_vector = mean_vector.to(device=raw_var.device, dtype=raw_var.dtype)
        param_scale2 = mean_vector.detach().pow(2).mean().clamp(min=eps)
        relative_var = raw_var / param_scale2
        raw_precision = 1.0 / (relative_var + eps)
        raw_precision = torch.nan_to_num(
            raw_precision,
            nan=1.0,
            posinf=1.0 / eps,
            neginf=1.0,
        ).clamp_min(eps)
        tempered = raw_precision.pow(temperature)
        normalizer = tempered.mean().clamp(min=eps)
        precision = tempered / normalizer * target
        return precision.clamp(min=min_value, max=max_value), eps

    raise ValueError(
        "Unknown bayes_precision_mode: "
        f"{mode}. Expected one of: floor_inverse, normalized_power, "
        "scalar_normalized_power, relative_normalized_power"
    )


def _safe_round(value, digits=6):
    if value is None:
        return None
    return round(float(value), digits)


def _mean_or_none(values):
    if not values:
        return None
    return sum(values) / float(len(values))


def _compute_cached_average_loss(model, batch_cache, criterion, device):
    weighted_loss = None
    seen_samples = 0

    for cached_inputs, cached_labels in batch_cache:
        inputs = cached_inputs.to(device, non_blocking=True)
        labels = cached_labels.to(device, non_blocking=True)
        result = model(inputs)
        logits = result["logits"]
        batch_loss = criterion(logits, labels)
        if not batch_loss.requires_grad:
            continue
        batch_weight = labels.size(0)
        weighted_term = batch_loss * batch_weight
        weighted_loss = weighted_term if weighted_loss is None else weighted_loss + weighted_term
        seen_samples += batch_weight

    if weighted_loss is None or seen_samples <= 0:
        return None, 0

    return weighted_loss / float(seen_samples), int(seen_samples)


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


def _run_expert_sgld_fit_adam_noise(
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
            population_var = (moment2 - moment1.square()).clamp(min=0.0)
            unbiased_var = (sample_count / (sample_count - 1.0)) * population_var
            if str(precision_mode or "floor_inverse").lower() == "relative_normalized_power":
                raw_var = population_var
            else:
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
                mean_vector=moment1,
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
    sgld_diag["precision_mean"] = round(float(precision_vector.detach().float().mean().item()), 6)
    sgld_diag["precision_min"] = round(float(precision_vector.detach().float().min().item()), 6)
    sgld_diag["precision_max"] = round(float(precision_vector.detach().float().max().item()), 6)

    mean_state = vector_to_named_state(reference_state, target_names, mean_vector.detach().cpu())
    precision_state = vector_to_named_state(reference_state, target_names, precision_vector.detach().cpu())
    return mean_state, precision_state, sgld_diag


def _emit_sgld_diag_log(sgld_diag, layer_id, expert_id):
    if sgld_diag.get("sgld_fit_mode") != "two_stage_plain":
        return

    message = (
        f"--bayes_sgld_fit_diag --layer:{layer_id} --expert:{expert_id} "
        f"--sgld_fit_mode:'{sgld_diag.get('sgld_fit_mode')}' "
        f"--map_steps:{sgld_diag.get('map_steps')} "
        f"--map_lr:{sgld_diag.get('map_lr')} "
        f"--plain_sgld_steps:{sgld_diag.get('plain_sgld_steps')} "
        f"--plain_sgld_burnin:{sgld_diag.get('plain_sgld_burnin')} "
        f"--plain_sgld_lr:{sgld_diag.get('plain_sgld_lr')} "
        f"--sgld_temperature:{sgld_diag.get('sgld_temperature')} "
        f"--plain_sgld_noise_scale:{sgld_diag.get('plain_sgld_noise_scale')} "
        f"--plain_sgld_loss_scale:{sgld_diag.get('plain_sgld_loss_scale')} "
        f"--plain_sgld_prior_precision:{sgld_diag.get('plain_sgld_prior_precision')} "
        f"--plain_sgld_sample_interval:{sgld_diag.get('plain_sgld_sample_interval')} "
        f"--collected_samples:{sgld_diag.get('collected_samples')} "
        f"--raw_var_mean:{sgld_diag.get('raw_var_mean')} "
        f"--raw_var_min:{sgld_diag.get('raw_var_min')} "
        f"--raw_var_max:{sgld_diag.get('raw_var_max')} "
        f"--precision_mode:{sgld_diag.get('precision_mode')} "
        f"--precision_mean:{sgld_diag.get('precision_mean')} "
        f"--precision_min:{sgld_diag.get('precision_min')} "
        f"--precision_max:{sgld_diag.get('precision_max')} "
        f"--noise_update_norm_mean:{sgld_diag.get('noise_update_norm_mean')} "
        f"--grad_update_norm_mean:{sgld_diag.get('grad_update_norm_mean')} "
        f"--noise_to_grad_ratio_mean:{sgld_diag.get('noise_to_grad_ratio_mean')} "
        f"--sample_pairwise_distance_mean:{sgld_diag.get('sample_pairwise_distance_mean')} "
        f"--mean_state_delta_rel:{sgld_diag.get('mean_state_delta_rel')} "
        f"--final_param_delta_rel:{sgld_diag.get('final_param_delta_rel')}"
    )

    emitted = False
    for logger_obj in logging.Logger.manager.loggerDict.values():
        if not isinstance(logger_obj, logging.Logger):
            continue
        if not any(isinstance(handler, logging.FileHandler) for handler in logger_obj.handlers):
            continue
        logger_obj.info(message)
        emitted = True

    if not emitted:
        logging.getLogger(__name__).info(message)


def _run_expert_sgld_fit_two_stage_plain(
    model,
    batch_cache,
    criterion,
    layer_id,
    expert_id,
    device,
    alp,
    ai_max,
    var_floor=0.0,
    precision_mode="floor_inverse",
    precision_temperature=0.25,
    precision_target=100.0,
    precision_min=20.0,
    precision_max=300.0,
    precision_eps=1.0e-12,
    map_steps=None,
    map_lr=None,
    plain_sgld_steps=None,
    plain_sgld_burnin=None,
    plain_sgld_lr=None,
    sgld_temperature=1.0,
    plain_sgld_noise_scale=1.0,
    plain_sgld_loss_scale=1.0,
    plain_sgld_prior_precision=0.0,
    plain_sgld_sample_interval=1,
):
    if len(batch_cache) == 0:
        raise ValueError("SGLD evidence extraction requires at least one cached batch")

    target_names, target_params = freeze_all_but_target_expert(
        model=model,
        layer_id=layer_id,
        expert_id=expert_id,
    )
    if len(target_params) == 0:
        raise ValueError(f"Missing target expert parameters for layer {layer_id}, expert {expert_id}")

    target_device = torch.device(device)
    model.to(target_device)
    model.train()

    map_steps = 8 if map_steps is None else int(map_steps)
    map_steps = max(map_steps, 0)
    map_lr = float(alp if map_lr is None else map_lr)
    map_lr = max(map_lr, 1e-12)
    plain_sgld_steps = 32 if plain_sgld_steps is None else int(plain_sgld_steps)
    plain_sgld_steps = max(plain_sgld_steps, 1)
    plain_sgld_burnin = 16 if plain_sgld_burnin is None else int(plain_sgld_burnin)
    plain_sgld_burnin = min(max(plain_sgld_burnin, 0), plain_sgld_steps - 1)
    plain_sgld_lr = 1.0e-6 if plain_sgld_lr is None else float(plain_sgld_lr)
    plain_sgld_lr = max(plain_sgld_lr, 1e-12)
    sgld_temperature = max(float(sgld_temperature), 0.0)
    plain_sgld_noise_scale = 1.0 if plain_sgld_noise_scale is None else float(plain_sgld_noise_scale)
    plain_sgld_noise_scale = max(plain_sgld_noise_scale, 0.0)
    plain_sgld_loss_scale = 1.0 if plain_sgld_loss_scale is None else float(plain_sgld_loss_scale)
    plain_sgld_loss_scale = max(plain_sgld_loss_scale, 0.0)
    plain_sgld_prior_precision = (
        0.0 if plain_sgld_prior_precision is None else float(plain_sgld_prior_precision)
    )
    plain_sgld_prior_precision = max(plain_sgld_prior_precision, 0.0)
    plain_sgld_sample_interval = 1 if plain_sgld_sample_interval is None else int(plain_sgld_sample_interval)
    plain_sgld_sample_interval = max(plain_sgld_sample_interval, 1)

    prepared_batch_cache = []
    for cached_inputs, cached_labels in batch_cache:
        if cached_inputs.device != target_device:
            cached_inputs = cached_inputs.to(target_device, non_blocking=True)
        if cached_labels.device != target_device:
            cached_labels = cached_labels.to(target_device, non_blocking=True)
        prepared_batch_cache.append((cached_inputs, cached_labels))

    total_samples = sum(int(labels.size(0)) for _, labels in prepared_batch_cache)
    total_samples = max(total_samples, 1)
    var_floor = max(float(var_floor), 0.0)
    ai_max = max(float(ai_max), 1e-4)
    fallback_precision = 1e-4
    last_seen_samples = 0

    with torch.no_grad():
        initial_vector = torch.nn.utils.parameters_to_vector(
            [param.detach() for param in target_params]
        ).detach().cpu()
        initial_param_norm = float(initial_vector.norm().item())
        relative_norm_floor = max(initial_param_norm, 1.0e-12)

    if map_steps > 0:
        map_optim = torch.optim.Adam(params=target_params, lr=map_lr)
        for _ in range(map_steps):
            map_optim.zero_grad(set_to_none=True)
            loss, seen_samples = _compute_cached_average_loss(
                model=model,
                batch_cache=prepared_batch_cache,
                criterion=criterion,
                device=target_device,
            )
            if loss is None:
                break
            last_seen_samples = int(seen_samples)
            loss.backward()
            map_optim.step()

    with torch.no_grad():
        theta_map_params = [param.detach().clone() for param in target_params]
        theta_map = torch.nn.utils.parameters_to_vector(theta_map_params).detach().cpu()
        map_param_delta_rel = float((theta_map - initial_vector).norm().item() / relative_norm_floor)

    sample_vectors = []
    data_losses = []
    prior_losses = []
    sgld_losses = []
    grad_update_norms = []
    noise_update_norms = []
    noise_to_grad_ratios = []
    map_pullback_norms = []
    param_step_delta_norms = []
    noise_scale = plain_sgld_noise_scale * math.sqrt(2.0 * plain_sgld_lr * sgld_temperature)

    for step_idx in range(plain_sgld_steps):
        for param in target_params:
            param.grad = None

        data_loss, seen_samples = _compute_cached_average_loss(
            model=model,
            batch_cache=prepared_batch_cache,
            criterion=criterion,
            device=target_device,
        )
        if data_loss is None:
            break
        last_seen_samples = int(seen_samples)

        prior_loss = data_loss.new_tensor(0.0)
        if plain_sgld_prior_precision > 0.0:
            for param, theta_map_param in zip(target_params, theta_map_params):
                prior_loss = prior_loss + (param - theta_map_param).square().sum()
            prior_loss = 0.5 * plain_sgld_prior_precision * prior_loss

        sgld_loss = plain_sgld_loss_scale * data_loss + prior_loss
        data_losses.append(float(data_loss.detach().item()))
        prior_losses.append(float(prior_loss.detach().item()))
        sgld_losses.append(float(sgld_loss.detach().item()))
        sgld_loss.backward()

        with torch.no_grad():
            grad_update_sq = 0.0
            noise_update_sq = 0.0
            map_pullback_sq = 0.0
            step_delta_sq = 0.0
            for param, theta_map_param in zip(target_params, theta_map_params):
                grad = param.grad
                if grad is None:
                    grad = torch.zeros_like(param)

                grad_update = -plain_sgld_lr * grad
                noise_update = noise_scale * torch.randn_like(param)
                param_delta = grad_update + noise_update
                map_pullback = plain_sgld_prior_precision * (param.detach() - theta_map_param)
                param.add_(param_delta)

                grad_update_sq += float(grad_update.square().sum().item())
                noise_update_sq += float(noise_update.square().sum().item())
                map_pullback_sq += float(map_pullback.square().sum().item())
                step_delta_sq += float(param_delta.square().sum().item())

            grad_update_norm = math.sqrt(max(grad_update_sq, 0.0))
            noise_update_norm = math.sqrt(max(noise_update_sq, 0.0))
            grad_update_norms.append(grad_update_norm)
            noise_update_norms.append(noise_update_norm)
            noise_to_grad_ratios.append(noise_update_norm / max(grad_update_norm, 1.0e-12))
            map_pullback_norms.append(math.sqrt(max(map_pullback_sq, 0.0)))
            param_step_delta_norms.append(math.sqrt(max(step_delta_sq, 0.0)))

            should_collect = (
                step_idx >= plain_sgld_burnin
                and (step_idx - plain_sgld_burnin) % plain_sgld_sample_interval == 0
            )
            if should_collect:
                sample_vectors.append(
                    torch.nn.utils.parameters_to_vector(
                        [param.detach() for param in target_params]
                    ).detach().cpu()
                )

    with torch.no_grad():
        reference_state = model.state_dict()
        final_vector = torch.nn.utils.parameters_to_vector(
            [param.detach() for param in target_params]
        ).detach().cpu()
        final_param_delta_rel = float((final_vector - initial_vector).norm().item() / relative_norm_floor)

        sample_count = len(sample_vectors)
        effective_var_floor = max(var_floor, max(float(precision_eps), 1.0e-12))
        sgld_diag = {
            "sgld_fit_mode": "two_stage_plain",
            "sample_count": int(sample_count),
            "collected_samples": int(sample_count),
            "total_cached_samples": int(total_samples),
            "last_seen_samples": int(last_seen_samples),
            "sgld_lr": float(plain_sgld_lr),
            "sgld_var_floor": float(var_floor),
            "precision_mode": str(precision_mode),
            "precision_temperature": float(precision_temperature),
            "precision_target": float(precision_target),
            "map_steps": int(map_steps),
            "map_lr": float(map_lr),
            "plain_sgld_steps": int(plain_sgld_steps),
            "plain_sgld_burnin": int(plain_sgld_burnin),
            "plain_sgld_lr": float(plain_sgld_lr),
            "sgld_temperature": float(sgld_temperature),
            "plain_sgld_noise_scale": float(plain_sgld_noise_scale),
            "plain_sgld_loss_scale": float(plain_sgld_loss_scale),
            "plain_sgld_prior_precision": float(plain_sgld_prior_precision),
            "plain_sgld_sample_interval": int(plain_sgld_sample_interval),
            "initial_param_norm": _safe_round(initial_param_norm, 12),
            "map_param_delta_rel": _safe_round(map_param_delta_rel, 12),
            "final_param_delta_rel": _safe_round(final_param_delta_rel, 12),
            "mean_state_delta_rel": None,
            "sample_pairwise_distance_mean": None,
            "sample_pairwise_distance_max": None,
            "data_loss_mean": _safe_round(_mean_or_none(data_losses), 12),
            "prior_loss_mean": _safe_round(_mean_or_none(prior_losses), 12),
            "sgld_loss_mean": _safe_round(_mean_or_none(sgld_losses), 12),
            "grad_update_norm_mean": _safe_round(_mean_or_none(grad_update_norms), 12),
            "noise_update_norm_mean": _safe_round(_mean_or_none(noise_update_norms), 12),
            "noise_to_grad_ratio_mean": _safe_round(_mean_or_none(noise_to_grad_ratios), 12),
            "map_pullback_norm_mean": _safe_round(_mean_or_none(map_pullback_norms), 12),
            "param_step_delta_norm_mean": _safe_round(_mean_or_none(param_step_delta_norms), 12),
            "raw_var_mean": None,
            "raw_var_min": None,
            "raw_var_max": None,
            "raw_var_under_floor_pct": None,
            "unclipped_precision_mean": None,
            "unclipped_precision_min": None,
            "unclipped_precision_max": None,
            "unclipped_precision_over_ai_max_pct": None,
            "precision_mean": None,
            "precision_min": None,
            "precision_max": None,
        }

        if sample_count == 0:
            mean_vector = final_vector
            precision_vector = torch.full_like(mean_vector, fill_value=fallback_precision)
        elif sample_count == 1:
            mean_vector = sample_vectors[0]
            precision_vector = torch.full_like(mean_vector, fill_value=fallback_precision)
            raw_var = torch.zeros_like(mean_vector)
            precision_before_clip = 1.0 / raw_var.clamp(min=effective_var_floor)
            sgld_diag.update(
                {
                    "raw_var_mean": 0.0,
                    "raw_var_min": 0.0,
                    "raw_var_max": 0.0,
                    "raw_var_under_floor_pct": 1.0,
                    "unclipped_precision_mean": round(float(precision_before_clip.mean().item()), 6),
                    "unclipped_precision_min": round(float(precision_before_clip.min().item()), 6),
                    "unclipped_precision_max": round(float(precision_before_clip.max().item()), 6),
                    "unclipped_precision_over_ai_max_pct": round(
                        float((precision_before_clip >= ai_max).float().mean().item()),
                        6,
                    ),
                }
            )
        else:
            sample_matrix = torch.stack(sample_vectors).float()
            mean_vector = sample_matrix.mean(dim=0)
            if str(precision_mode or "floor_inverse").lower() == "relative_normalized_power":
                raw_var = sample_matrix.var(dim=0, unbiased=False).clamp(min=0.0)
            else:
                raw_var = sample_matrix.var(dim=0, unbiased=True).clamp(min=0.0)
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
                mean_vector=mean_vector,
            )
            precision_vector = torch.nan_to_num(
                precision_vector,
                nan=ai_max,
                posinf=ai_max,
                neginf=1e-4,
            ).clamp(min=1e-4, max=float(ai_max))
            precision_before_clip = 1.0 / raw_var.clamp(min=effective_var_floor)
            pairwise_distances = torch.pdist(sample_matrix)
            sgld_diag.update(
                {
                    "sample_pairwise_distance_mean": round(float(pairwise_distances.mean().item()), 12),
                    "sample_pairwise_distance_max": round(float(pairwise_distances.max().item()), 12),
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
                        float((precision_before_clip >= ai_max).float().mean().item()),
                        6,
                    ),
                }
            )

        precision_vector = torch.nan_to_num(
            precision_vector,
            nan=ai_max,
            posinf=ai_max,
            neginf=1e-4,
        ).clamp(min=1e-4, max=float(ai_max))
        theta_map_norm_floor = max(float(theta_map.norm().item()), 1.0e-12)
        mean_state_delta_rel = float((mean_vector - theta_map).norm().item() / theta_map_norm_floor)
        sgld_diag["mean_state_delta_rel"] = _safe_round(mean_state_delta_rel, 12)
        sgld_diag["precision_mean"] = round(float(precision_vector.mean().item()), 6)
        sgld_diag["precision_min"] = round(float(precision_vector.min().item()), 6)
        sgld_diag["precision_max"] = round(float(precision_vector.max().item()), 6)

    mean_state = vector_to_named_state(reference_state, target_names, mean_vector)
    precision_state = vector_to_named_state(reference_state, target_names, precision_vector)
    return mean_state, precision_state, sgld_diag


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
    *,
    sgld_fit_mode="adam_noise",
    map_steps=None,
    map_lr=None,
    plain_sgld_steps=None,
    plain_sgld_burnin=None,
    plain_sgld_lr=None,
    sgld_temperature=1.0,
    plain_sgld_noise_scale=1.0,
    plain_sgld_loss_scale=1.0,
    plain_sgld_prior_precision=0.0,
    plain_sgld_sample_interval=1,
):
    mode = str(sgld_fit_mode or "adam_noise").lower()
    if mode == "adam_noise":
        mean_state, precision_state, sgld_diag = _run_expert_sgld_fit_adam_noise(
            model=model,
            batch_cache=batch_cache,
            criterion=criterion,
            layer_id=layer_id,
            expert_id=expert_id,
            device=device,
            steps=steps,
            burnin=burnin,
            alp=alp,
            ai_max=ai_max,
            var_floor=var_floor,
            precision_mode=precision_mode,
            precision_temperature=precision_temperature,
            precision_target=precision_target,
            precision_min=precision_min,
            precision_max=precision_max,
            precision_eps=precision_eps,
            sgld_concat_cache=sgld_concat_cache,
        )
        sgld_diag.update(
            {
                "sgld_fit_mode": "adam_noise",
                "map_steps": None,
                "map_lr": None,
                "plain_sgld_steps": None,
                "plain_sgld_burnin": None,
                "plain_sgld_lr": None,
                "sgld_temperature": None,
                "plain_sgld_noise_scale": None,
                "plain_sgld_loss_scale": None,
                "plain_sgld_prior_precision": None,
                "plain_sgld_sample_interval": None,
                "collected_samples": int(sgld_diag.get("sample_count", 0)),
                "grad_update_norm_mean": None,
                "noise_update_norm_mean": None,
                "noise_to_grad_ratio_mean": None,
                "sample_pairwise_distance_mean": None,
                "mean_state_delta_rel": None,
                "final_param_delta_rel": None,
            }
        )
        return mean_state, precision_state, sgld_diag

    if mode == "two_stage_plain":
        mean_state, precision_state, sgld_diag = _run_expert_sgld_fit_two_stage_plain(
            model=model,
            batch_cache=batch_cache,
            criterion=criterion,
            layer_id=layer_id,
            expert_id=expert_id,
            device=device,
            alp=alp,
            ai_max=ai_max,
            var_floor=var_floor,
            precision_mode=precision_mode,
            precision_temperature=precision_temperature,
            precision_target=precision_target,
            precision_min=precision_min,
            precision_max=precision_max,
            precision_eps=precision_eps,
            map_steps=map_steps,
            map_lr=map_lr,
            plain_sgld_steps=plain_sgld_steps,
            plain_sgld_burnin=plain_sgld_burnin,
            plain_sgld_lr=plain_sgld_lr,
            sgld_temperature=sgld_temperature,
            plain_sgld_noise_scale=plain_sgld_noise_scale,
            plain_sgld_loss_scale=plain_sgld_loss_scale,
            plain_sgld_prior_precision=plain_sgld_prior_precision,
            plain_sgld_sample_interval=plain_sgld_sample_interval,
        )
        _emit_sgld_diag_log(sgld_diag, layer_id=layer_id, expert_id=expert_id)
        return mean_state, precision_state, sgld_diag

    raise ValueError(
        f"Unknown bayes_sgld_fit_mode: {sgld_fit_mode!r}. "
        "Expected one of: adam_noise, two_stage_plain"
    )


_BAYES_SGLD_FIT_CONFIG_TO_PARAM = {
    "bayes_sgld_fit_mode": "sgld_fit_mode",
    "bayes_map_steps": "map_steps",
    "bayes_map_lr": "map_lr",
    "bayes_plain_sgld_steps": "plain_sgld_steps",
    "bayes_plain_sgld_burnin": "plain_sgld_burnin",
    "bayes_plain_sgld_lr": "plain_sgld_lr",
    "bayes_sgld_temperature": "sgld_temperature",
    "bayes_plain_sgld_noise_scale": "plain_sgld_noise_scale",
    "bayes_plain_sgld_loss_scale": "plain_sgld_loss_scale",
    "bayes_plain_sgld_prior_precision": "plain_sgld_prior_precision",
    "bayes_plain_sgld_sample_interval": "plain_sgld_sample_interval",
}


def _coerce_bayes_sgld_default_value(param_name, value):
    if value is None:
        return None

    if param_name == "sgld_fit_mode":
        return str(value).strip().lower()

    if param_name in {
        "map_steps",
        "plain_sgld_steps",
        "plain_sgld_burnin",
        "plain_sgld_sample_interval",
    }:
        return int(value)

    if param_name in {
        "map_lr",
        "plain_sgld_lr",
        "sgld_temperature",
        "plain_sgld_noise_scale",
        "plain_sgld_loss_scale",
        "plain_sgld_prior_precision",
    }:
        return float(value)

    return value


def configure_bayes_sgld_fit_defaults(config):
    """Apply YAML-level SGLD fit defaults without changing client.py call sites."""

    kwdefaults = dict(run_expert_sgld_fit.__kwdefaults__ or {})
    for config_key, param_name in _BAYES_SGLD_FIT_CONFIG_TO_PARAM.items():
        if isinstance(config, dict):
            if config_key not in config:
                continue
            value = config[config_key]
        else:
            if not hasattr(config, config_key):
                continue
            value = getattr(config, config_key)
        kwdefaults[param_name] = _coerce_bayes_sgld_default_value(param_name, value)

    run_expert_sgld_fit.__kwdefaults__ = kwdefaults

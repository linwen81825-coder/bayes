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
    return client_evidence.get(str(layer_id), {}).get(str(expert_id))


def get_bayes_expert_state(bayes_state, layer_id, expert_id):
    if not isinstance(bayes_state, dict):
        return None
    return (
        bayes_state.get("experts", {})
        .get(str(layer_id), {})
        .get(str(expert_id))
    )


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


def _normalize_torch_device(device_like):
    resolved = torch.device(device_like)
    if resolved.type == "cuda" and resolved.index is None and torch.cuda.is_available():
        resolved = torch.device(f"cuda:{torch.cuda.current_device()}")
    return resolved


def _parameters_to_vector(params):
    return torch.nn.utils.parameters_to_vector([param.detach() for param in params]).detach()


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
    var_floor=0.0,
    precision_eps=1.0e-12,
    *,
    precision_source="sgld_variance",
    sgld_fit_mode="adam_noise",
    precision_mode="floor_inverse",
):
    precision_source = str(precision_source or "sgld_variance").lower()
    if precision_source != "sgld_variance":
        raise ValueError("bayes_precision_source now only supports: sgld_variance")

    sgld_fit_mode = str(sgld_fit_mode or "adam_noise").lower()
    if sgld_fit_mode not in {"adam_noise", "sgd_noise"}:
        raise ValueError("bayes_sgld_fit_mode must be one of: adam_noise, sgd_noise")

    precision_mode = str(precision_mode or "floor_inverse").lower()
    if precision_mode != "floor_inverse":
        raise ValueError("bayes_precision_mode now only supports: floor_inverse")

    if len(batch_cache) == 0:
        raise ValueError("SGLD evidence extraction requires at least one cached batch")

    target_names, target_params = freeze_all_but_target_expert(
        model=model,
        layer_id=layer_id,
        expert_id=expert_id,
    )
    if len(target_params) == 0:
        raise ValueError(f"Missing target expert parameters for layer {layer_id}, expert {expert_id}")

    steps = max(int(steps), 1)
    burnin = min(max(int(burnin), 0), steps - 1)
    var_floor = max(float(var_floor), 0.0)
    precision_eps = max(float(precision_eps), 1.0e-12)
    target_device = _normalize_torch_device(device)
    model.to(target_device)
    model.train()

    prepare_start = time.perf_counter()
    prepared_batch_cache = []
    for cached_inputs, cached_labels in batch_cache:
        prepared_batch_cache.append((
            cached_inputs.to(target_device, non_blocking=True),
            cached_labels.to(target_device, non_blocking=True),
        ))

    total_samples = max(sum(int(labels.size(0)) for _, labels in prepared_batch_cache), 1)
    prepare_cache_time_sec = time.perf_counter() - prepare_start

    sgld_lr = max(float(alp) / float(total_samples), 1.0e-12)
    optimizer = None
    if sgld_fit_mode == "adam_noise":
        optimizer = torch.optim.Adam(params=target_params, lr=sgld_lr)
    noise_scale = math.sqrt(1.0 / sgld_lr)
    moment1 = None
    moment2 = None
    sample_count = 0
    last_seen_samples = 0
    forward_backward_time_sec = 0.0

    for step_idx in range(steps):
        if optimizer is None:
            for param in target_params:
                param.grad = None
        else:
            optimizer.zero_grad(set_to_none=True)
        weighted_loss = None
        seen_samples = 0
        step_start = time.perf_counter()
        for inputs, labels in prepared_batch_cache:
            result = model(inputs)
            batch_loss = criterion(result["logits"], labels)
            if not batch_loss.requires_grad:
                continue
            batch_weight = labels.size(0)
            weighted_term = batch_loss * batch_weight
            weighted_loss = weighted_term if weighted_loss is None else weighted_loss + weighted_term
            seen_samples += batch_weight

        if weighted_loss is None or seen_samples <= 0:
            forward_backward_time_sec += time.perf_counter() - step_start
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
                noise = noise_scale * torch.randn_like(param)
                if sgld_fit_mode == "adam_noise":
                    param.grad.add_(noise)
                else:
                    param.add_(param.grad + noise, alpha=-sgld_lr)
        if optimizer is not None:
            optimizer.step()
        forward_backward_time_sec += time.perf_counter() - step_start

        if step_idx >= burnin:
            with torch.no_grad():
                param_vector = _parameters_to_vector(target_params)
                if sample_count == 0:
                    moment1 = param_vector.clone()
                    moment2 = param_vector.square()
                else:
                    moment1 = (param_vector + sample_count * moment1) / (sample_count + 1)
                    moment2 = (param_vector.square() + sample_count * moment2) / (sample_count + 1)
                sample_count += 1

    with torch.no_grad():
        reference_state = model.state_dict()
        if moment1 is None:
            mean_vector = _parameters_to_vector(target_params)
            raw_variance = torch.zeros_like(mean_vector)
        else:
            mean_vector = moment1
            if sample_count <= 1:
                raw_variance = torch.zeros_like(mean_vector)
            else:
                population_variance = (moment2 - moment1.square()).clamp(min=0.0)
                raw_variance = (sample_count / (sample_count - 1.0)) * population_variance

        floored_variance = raw_variance.clamp(min=var_floor)
        precision_vector = 1.0 / (floored_variance + precision_eps)

    diag = {
        "precision_source": "sgld_variance",
        "sgld_noise_mode": sgld_fit_mode,
        "precision_method": "floor_inverse",
        "mean_state_source": "sgld_sample_mean",
        "precision_state_source": "sgld_variance_floor_inverse",
        "sample_count": int(sample_count),
        "total_cached_samples": int(total_samples),
        "last_seen_samples": int(last_seen_samples),
        "sgld_lr": float(sgld_lr),
        "sgld_var_floor": float(var_floor),
        "precision_eps": float(precision_eps),
        "raw_var_mean": round(float(raw_variance.mean().item()), 12),
        "raw_var_min": round(float(raw_variance.min().item()), 12),
        "raw_var_max": round(float(raw_variance.max().item()), 12),
        "precision_mean": round(float(precision_vector.mean().item()), 6),
        "precision_min": round(float(precision_vector.min().item()), 6),
        "precision_max": round(float(precision_vector.max().item()), 6),
        "sgld_prepare_cache_time_sec": round(float(prepare_cache_time_sec), 6),
        "sgld_forward_backward_time_sec": round(float(forward_backward_time_sec), 6),
        "sgld_fit_time_sec": round(float(prepare_cache_time_sec + forward_backward_time_sec), 6),
    }
    mean_state = vector_to_named_state(reference_state, target_names, mean_vector.detach().cpu())
    precision_state = vector_to_named_state(reference_state, target_names, precision_vector.detach().cpu())
    return mean_state, precision_state, diag

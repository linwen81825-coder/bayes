import math

import torch


def flatten_float_tensors(tensor_dict, device=None):
    if torch.is_tensor(tensor_dict):
        if not torch.is_floating_point(tensor_dict):
            return None
        tensor = tensor_dict.to(device) if device is not None else tensor_dict
        return tensor.reshape(-1)

    if not isinstance(tensor_dict, dict):
        return None

    flattened_tensors = []
    for key in sorted(tensor_dict.keys()):
        tensor = tensor_dict[key]
        if tensor is None or not torch.is_tensor(tensor):
            continue
        if not torch.is_floating_point(tensor):
            continue
        tensor = tensor.to(device) if device is not None else tensor
        flattened_tensors.append(tensor.reshape(-1))

    if not flattened_tensors:
        return None

    return torch.cat(flattened_tensors, dim=0)


def delta_to_negative_grad_score(
    delta_state,
    grad_state,
    metric="cosine",
    eps=1e-12,
    device=None,
):
    # cosine 只比较方向；dot 同时受到方向和更新/梯度幅度影响。
    delta_vector = flatten_float_tensors(delta_state, device=device)
    grad_vector = flatten_float_tensors(grad_state, device=device)
    if delta_vector is None or grad_vector is None:
        return None
    if delta_vector.shape != grad_vector.shape:
        return None
    if delta_vector.numel() == 0 or grad_vector.numel() == 0:
        return None
    if delta_vector.device != grad_vector.device:
        grad_vector = grad_vector.to(delta_vector.device)

    delta_vector = delta_vector.float()
    grad_vector = grad_vector.float()
    raw_dot = torch.dot(delta_vector, -grad_vector)

    if metric == "dot":
        if not torch.isfinite(raw_dot):
            return None
        return float(raw_dot.item())

    if metric == "cosine":
        delta_norm = torch.linalg.vector_norm(delta_vector)
        grad_norm = torch.linalg.vector_norm(grad_vector)
        if delta_norm.item() <= eps or grad_norm.item() <= eps:
            return None

        score = raw_dot / (delta_norm * grad_norm)
        return float(score.item())

    raise ValueError(f"Unknown UOC-FOGA score metric: {metric!r}")


def dot_grad_to_grad(query_grad_state, client_grad_state, device=None):
    """
    计算源码风格 FOGA 梯度内积分数：
        score = <g_query, g_client>

    注意：
    - 不加负号。
    - query_grad_state 和 client_grad_state 都是 dict。
    - 只使用两个 dict 共同拥有的参数 key。
    - 按 key 排序后 flatten 拼接。
    - 如果没有有效 tensor，返回 None。
    - 返回 Python float。
    """
    if not isinstance(query_grad_state, dict) or not isinstance(client_grad_state, dict):
        return None

    query_chunks = []
    client_chunks = []
    common_keys = set(query_grad_state.keys()) & set(client_grad_state.keys())
    for key in sorted(common_keys):
        query_tensor = query_grad_state.get(key)
        client_tensor = client_grad_state.get(key)
        if query_tensor is None or client_tensor is None:
            continue
        if not torch.is_tensor(query_tensor) or not torch.is_tensor(client_tensor):
            continue

        query_vec = query_tensor.detach()
        client_vec = client_tensor.detach()
        if device is not None:
            query_vec = query_vec.to(device)
            client_vec = client_vec.to(device)
        query_vec = query_vec.reshape(-1)
        client_vec = client_vec.reshape(-1)
        if query_vec.shape != client_vec.shape:
            continue

        query_chunks.append(query_vec.float())
        client_chunks.append(client_vec.float())

    if not query_chunks:
        return None

    query_vec = torch.cat(query_chunks, dim=0)
    client_vec = torch.cat(client_chunks, dim=0)
    score = torch.dot(query_vec, client_vec)
    if not torch.isfinite(score):
        return None
    return float(score.item())


def cosine_delta_to_negative_grad(delta_state, grad_state, eps=1e-12, device=None):
    return delta_to_negative_grad_score(
        delta_state,
        grad_state,
        metric="cosine",
        eps=eps,
        device=device,
    )


def l2_norm_state(tensor_dict, eps=1e-12, device=None):
    vector = flatten_float_tensors(tensor_dict, device=device)
    if vector is None:
        return 0.0

    norm = torch.linalg.vector_norm(vector.float())
    if norm.item() <= eps:
        return 0.0
    return float(norm.item())


def _empty_query_result(num_classes, fallback_reason, extra_stats=None):
    result = {
        "hidden": None,
        "labels": None,
        "gates": None,
        "residual": None,
        "has_residual": False,
        "class_hist": [0 for _ in range(num_classes)],
        "query_size": 0,
        "query_num_classes": 0,
        "fallback_reason": fallback_reason,
    }
    if extra_stats:
        result.update(extra_stats)
    return result


def _get_layer_evidence(client_evidence, layer_id):
    if not isinstance(client_evidence, dict):
        return None
    if layer_id in client_evidence:
        return client_evidence[layer_id]

    layer_key = str(layer_id)
    if layer_key in client_evidence:
        return client_evidence[layer_key]

    for key, value in client_evidence.items():
        if str(key) == layer_key:
            return value
    return None


def _as_cpu_tensor(value):
    if not torch.is_tensor(value):
        return None
    return value.detach().cpu()


def _select_expert_samples(layer_evidence, expert_id, client_id=0):
    hidden = _as_cpu_tensor(layer_evidence.get("hidden"))
    labels = _as_cpu_tensor(layer_evidence.get("labels"))
    top1_expert_ids = _as_cpu_tensor(layer_evidence.get("top1_expert_ids"))
    top1_gates = _as_cpu_tensor(layer_evidence.get("top1_gates"))
    residual = _as_cpu_tensor(layer_evidence.get("residual"))
    entropy = _as_cpu_tensor(layer_evidence.get("entropy"))
    if (
        hidden is None
        or labels is None
        or top1_expert_ids is None
        or top1_gates is None
    ):
        return None
    if (
        hidden.dim() == 0
        or labels.dim() == 0
        or top1_expert_ids.dim() == 0
        or top1_gates.dim() == 0
    ):
        return None
    if residual is not None and residual.dim() == 0:
        residual = None
    if entropy is not None and entropy.dim() == 0:
        entropy = None

    sample_count_values = [
        hidden.size(0),
        labels.size(0),
        top1_expert_ids.size(0),
        top1_gates.size(0),
    ]
    if residual is not None:
        sample_count_values.append(residual.size(0))
    if entropy is not None:
        sample_count_values.append(entropy.size(0))
    sample_count = min(sample_count_values)
    if sample_count <= 0:
        return None

    hidden = hidden[:sample_count]
    labels = labels[:sample_count]
    top1_expert_ids = top1_expert_ids[:sample_count].long()
    top1_gates = top1_gates[:sample_count].float()
    if residual is not None:
        residual = residual[:sample_count]
        if residual.shape[1:] != hidden.shape[1:]:
            residual = None
    if entropy is not None:
        entropy = entropy[:sample_count].float()
        if entropy.dim() > 1:
            entropy = entropy.reshape(sample_count, -1).mean(dim=1)
    expert_id = int(expert_id)

    if top1_expert_ids.dim() == 1:
        sample_mask = top1_expert_ids == expert_id
        expert_token_count = sample_mask.long()
        expert_token_ratio = sample_mask.float()
        if top1_gates.dim() == 1:
            gate_per_sample = top1_gates
        else:
            gate_per_sample = top1_gates.reshape(sample_count, -1).mean(dim=1)
    else:
        # token-level：任意 token 命中 expert_id，该样本就进入 query pool。
        expert_ids_flat = top1_expert_ids.reshape(sample_count, -1)
        gates_flat = top1_gates.reshape(sample_count, -1)
        if expert_ids_flat.shape != gates_flat.shape:
            return None
        token_mask = expert_ids_flat == expert_id
        token_count = token_mask.sum(dim=1)
        sample_mask = token_count > 0
        expert_token_count = token_count
        token_total = max(int(expert_ids_flat.size(1)), 1)
        expert_token_ratio = token_count.float() / float(token_total)
        gate_sum = gates_flat.masked_fill(~token_mask, 0.0).sum(dim=1)
        gate_per_sample = gate_sum / token_count.clamp_min(1).to(gates_flat.dtype)

    if sample_mask.sum().item() == 0:
        return None

    selected = {
        "hidden": hidden[sample_mask],
        "labels": labels[sample_mask].long(),
        "gates": gate_per_sample[sample_mask],
        "residual": None,
        "entropy": entropy[sample_mask] if entropy is not None else None,
        "client_ids": torch.full(
            (int(sample_mask.sum().item()),),
            int(client_id),
            dtype=torch.long,
        ),
        "expert_token_ratio": expert_token_ratio[sample_mask].float(),
        "expert_token_count": expert_token_count[sample_mask].long(),
    }
    if residual is not None:
        # residual 和 hidden 必须使用同一个 sample_mask，否则 server 无法恢复 block 输出。
        selected["residual"] = residual[sample_mask]
    return selected


def _build_random_class_balanced_query_from_pool(
    pool_hidden,
    pool_labels,
    pool_gates,
    pool_residual,
    num_classes,
    query_per_class,
    min_query_samples,
    min_query_classes,
    seed,
):
    class_hist = [0 for _ in range(num_classes)]
    selected_indices = []
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    for class_id in range(num_classes):
        class_indices = torch.nonzero(pool_labels == class_id, as_tuple=False).flatten()
        if class_indices.numel() == 0:
            continue

        take_count = min(max(query_per_class, 0), class_indices.numel())
        if take_count == 0:
            continue

        perm = torch.randperm(class_indices.numel(), generator=generator)
        chosen = class_indices[perm[:take_count]]
        selected_indices.append(chosen)
        class_hist[class_id] = int(take_count)

    if selected_indices:
        selected_indices = torch.cat(selected_indices, dim=0)
        query_hidden = pool_hidden[selected_indices].detach().cpu()
        query_labels = pool_labels[selected_indices].detach().cpu()
        query_gates = pool_gates[selected_indices].detach().cpu()
        # residual 必须和 hidden 使用同一个 final_indices，才能恢复：
        # x_after_block = residual + forced_expert(hidden)。
        query_residual = (
            pool_residual[selected_indices].detach().cpu()
            if pool_residual is not None
            else None
        )
        query_size = int(query_labels.size(0))
        query_num_classes = sum(1 for count in class_hist if count > 0)
    else:
        query_hidden = None
        query_labels = None
        query_gates = None
        query_residual = None
        query_size = 0
        query_num_classes = 0

    fallback_reason = None
    if query_size < min_query_samples:
        fallback_reason = "query_size_too_small"
    elif query_num_classes < min_query_classes:
        fallback_reason = "query_classes_too_few"

    has_residual = query_residual is not None
    if fallback_reason is not None:
        # fallback 后清空 query tensor，防止后续误用不可靠的 D_query。
        query_hidden = None
        query_labels = None
        query_gates = None
        query_residual = None
        has_residual = False

    return {
        "hidden": query_hidden,
        "labels": query_labels,
        "gates": query_gates,
        "residual": query_residual,
        "has_residual": has_residual,
        "class_hist": class_hist,
        "query_size": query_size,
        "query_num_classes": query_num_classes,
        "fallback_reason": fallback_reason,
    }


def _small_tensor_stats(values):
    if values is None or values.numel() == 0:
        return None, None, None

    values = values.float()
    return (
        float(values.mean().item()),
        float(values.min().item()),
        float(values.max().item()),
    )


def _add_query_stats(result, stats):
    result.update(stats)
    return result


def _fallback_random_with_stats(
    stats,
    pool_hidden,
    pool_labels,
    pool_gates,
    pool_residual,
    num_classes,
    query_per_class,
    min_query_samples,
    min_query_classes,
    seed,
):
    stats["fallback_to_random_used"] = True
    random_result = _build_random_class_balanced_query_from_pool(
        pool_hidden=pool_hidden,
        pool_labels=pool_labels,
        pool_gates=pool_gates,
        pool_residual=pool_residual,
        num_classes=num_classes,
        query_per_class=query_per_class,
        min_query_samples=min_query_samples,
        min_query_classes=min_query_classes,
        seed=seed,
    )
    return _add_query_stats(random_result, stats)


def _build_expert_ratio_entropy_query_from_pool(
    pool_hidden,
    pool_labels,
    pool_gates,
    pool_residual,
    pool_client_ids,
    pool_entropy,
    pool_expert_token_ratio,
    pool_expert_token_count,
    num_classes,
    query_per_class,
    min_query_samples,
    min_query_classes,
    min_expert_token_ratio,
    max_samples_per_client_per_class,
    fallback_to_random,
    seed,
):
    min_expert_token_ratio = float(min_expert_token_ratio)
    max_samples_per_client_per_class = int(max_samples_per_client_per_class)
    keep = pool_expert_token_ratio > min_expert_token_ratio
    filtered_indices = torch.nonzero(keep, as_tuple=False).flatten()
    ratio_values = pool_expert_token_ratio[filtered_indices]
    ratio_mean, ratio_min, ratio_max = _small_tensor_stats(ratio_values)
    stats = {
        "query_select_mode": "expert_ratio_entropy",
        "token_ratio_threshold": min_expert_token_ratio,
        "pool_size_before_filter": int(pool_hidden.size(0)),
        "pool_size_after_token_ratio_filter": int(filtered_indices.numel()),
        "selected_by_entropy": 0,
        "fallback_to_random_used": False,
        "entropy_missing": pool_entropy is None,
        "expert_token_ratio_mean": ratio_mean,
        "expert_token_ratio_min": ratio_min,
        "expert_token_ratio_max": ratio_max,
        "query_entropy_mean": None,
        "max_samples_per_client_per_class": max_samples_per_client_per_class,
    }

    if filtered_indices.numel() == 0:
        if fallback_to_random:
            return _fallback_random_with_stats(
                stats,
                pool_hidden,
                pool_labels,
                pool_gates,
                pool_residual,
                num_classes,
                query_per_class,
                min_query_samples,
                min_query_classes,
                seed,
            )
        return _empty_query_result(
            num_classes,
            "no_samples_after_token_ratio_filter",
            extra_stats=stats,
        )

    if pool_entropy is None:
        if fallback_to_random:
            return _fallback_random_with_stats(
                stats,
                pool_hidden,
                pool_labels,
                pool_gates,
                pool_residual,
                num_classes,
                query_per_class,
                min_query_samples,
                min_query_classes,
                seed,
            )
        return _empty_query_result(
            num_classes,
            "missing_entropy_for_query_selection",
            extra_stats=stats,
        )

    selected_indices_by_class = []
    class_hist = [0 for _ in range(num_classes)]
    filtered_labels = pool_labels[filtered_indices]
    filtered_client_ids = pool_client_ids[filtered_indices]

    for class_id in range(num_classes):
        class_positions = torch.nonzero(
            filtered_labels == class_id,
            as_tuple=False,
        ).flatten()
        if class_positions.numel() == 0:
            continue

        class_indices = filtered_indices[class_positions]
        class_client_ids = filtered_client_ids[class_positions]
        client_limited_indices = []

        for client_value in sorted(int(value) for value in torch.unique(class_client_ids).tolist()):
            client_mask = class_client_ids == client_value
            client_indices = class_indices[client_mask]
            if client_indices.numel() == 0:
                continue

            client_entropy = pool_entropy[client_indices].float()
            order = torch.argsort(client_entropy, descending=True)
            if max_samples_per_client_per_class > 0:
                take_count = min(max_samples_per_client_per_class, client_indices.numel())
            else:
                take_count = client_indices.numel()
            client_limited_indices.append(client_indices[order[:take_count]])

        if not client_limited_indices:
            continue

        class_candidates = torch.cat(client_limited_indices, dim=0)
        take_count = min(max(query_per_class, 0), class_candidates.numel())
        if take_count == 0:
            continue

        class_entropy = pool_entropy[class_candidates].float()
        order = torch.argsort(class_entropy, descending=True)
        chosen = class_candidates[order[:take_count]]
        selected_indices_by_class.append(chosen)
        class_hist[class_id] = int(take_count)

    if selected_indices_by_class:
        selected_indices = torch.cat(selected_indices_by_class, dim=0)
        selected_entropy = pool_entropy[selected_indices].float()
        stats["selected_by_entropy"] = int(selected_indices.numel())
        stats["query_entropy_mean"] = float(selected_entropy.mean().item())
        query_hidden = pool_hidden[selected_indices].detach().cpu()
        query_labels = pool_labels[selected_indices].detach().cpu()
        query_gates = pool_gates[selected_indices].detach().cpu()
        query_residual = (
            pool_residual[selected_indices].detach().cpu()
            if pool_residual is not None
            else None
        )
        query_size = int(query_labels.size(0))
        query_num_classes = sum(1 for count in class_hist if count > 0)
    else:
        query_hidden = None
        query_labels = None
        query_gates = None
        query_residual = None
        query_size = 0
        query_num_classes = 0

    fallback_reason = None
    if query_size < min_query_samples:
        fallback_reason = "query_size_too_small_after_entropy_filter"
    elif query_num_classes < min_query_classes:
        fallback_reason = "query_classes_too_few_after_entropy_filter"

    if fallback_reason is not None and fallback_to_random:
        return _fallback_random_with_stats(
            stats,
            pool_hidden,
            pool_labels,
            pool_gates,
            pool_residual,
            num_classes,
            query_per_class,
            min_query_samples,
            min_query_classes,
            seed,
        )

    has_residual = query_residual is not None
    if fallback_reason is not None:
        # fallback 关闭时，保持旧行为：只保留轻量统计，不返回不合格 query tensor。
        query_hidden = None
        query_labels = None
        query_gates = None
        query_residual = None
        has_residual = False

    return _add_query_stats(
        {
            "hidden": query_hidden,
            "labels": query_labels,
            "gates": query_gates,
            "residual": query_residual,
            "has_residual": has_residual,
            "class_hist": class_hist,
            "query_size": query_size,
            "query_num_classes": query_num_classes,
            "fallback_reason": fallback_reason,
        },
        stats,
    )


def build_client_grad_query_for_expert(
    client_evidence,
    layer_id,
    expert_id,
    query_per_class,
    min_query_samples,
    min_classes,
    query_select_mode=None,
    min_expert_token_ratio=0.0,
    max_samples_per_client_per_class=0,
    fallback_to_random=True,
):
    """
    从单个 client 的 UOC evidence 中，为某个 layer/expert 构造 D_client,i,l,e。
    这个 set 后续用于计算：
        g_client,i,l,e = ∇ CE(D_client,i,l,e; θ_global)

    第一版允许 D_client 和全局 D_query 有重叠，不处理 sample_id 排除。
    """
    layer_evidence = _get_layer_evidence(client_evidence, layer_id)
    if not isinstance(layer_evidence, dict):
        return None

    client_id = 0
    if isinstance(client_evidence, dict) and "client_id" in client_evidence:
        client_id = client_evidence["client_id"]
    if "client_id" in layer_evidence:
        client_id = layer_evidence["client_id"]

    selected = _select_expert_samples(layer_evidence, expert_id, client_id=client_id)
    if selected is None:
        return None

    pool_hidden = selected["hidden"]
    pool_labels = selected["labels"].long()
    pool_gates = selected["gates"]
    pool_residual = selected["residual"]
    if pool_hidden is None or pool_labels is None or pool_hidden.size(0) == 0:
        return None

    query_per_class = int(query_per_class)
    min_query_samples = int(min_query_samples)
    min_classes = int(min_classes)
    max_samples_per_client_per_class = int(max_samples_per_client_per_class)
    if query_per_class <= 0:
        return None

    inferred_num_classes = int(pool_labels.max().item()) + 1 if pool_labels.numel() > 0 else 0
    num_classes = max(inferred_num_classes, min_classes, 1)
    select_mode = query_select_mode or "class_balanced_random"
    effective_query_per_class = query_per_class
    if max_samples_per_client_per_class > 0:
        effective_query_per_class = min(
            effective_query_per_class,
            max_samples_per_client_per_class,
        )

    if select_mode == "class_balanced_random":
        result = _build_random_class_balanced_query_from_pool(
            pool_hidden=pool_hidden,
            pool_labels=pool_labels,
            pool_gates=pool_gates,
            pool_residual=pool_residual,
            num_classes=num_classes,
            query_per_class=effective_query_per_class,
            min_query_samples=min_query_samples,
            min_query_classes=min_classes,
            seed=None,
        )
    elif select_mode == "expert_ratio_entropy":
        result = _build_expert_ratio_entropy_query_from_pool(
            pool_hidden=pool_hidden,
            pool_labels=pool_labels,
            pool_gates=pool_gates,
            pool_residual=pool_residual,
            pool_client_ids=selected["client_ids"],
            pool_entropy=selected["entropy"],
            pool_expert_token_ratio=selected["expert_token_ratio"],
            pool_expert_token_count=selected["expert_token_count"],
            num_classes=num_classes,
            query_per_class=effective_query_per_class,
            min_query_samples=min_query_samples,
            min_query_classes=min_classes,
            min_expert_token_ratio=min_expert_token_ratio,
            max_samples_per_client_per_class=max_samples_per_client_per_class,
            fallback_to_random=bool(fallback_to_random),
            seed=None,
        )
    else:
        raise ValueError(f"Unknown client grad query select mode: {select_mode!r}")

    hidden = result.get("hidden")
    labels = result.get("labels")
    if hidden is None or labels is None or hidden.size(0) == 0:
        return None

    sample_count = int(labels.size(0))
    num_selected_classes = int(torch.unique(labels.long()).numel())
    if sample_count < min_query_samples or num_selected_classes < min_classes:
        return None

    return {
        "hidden": hidden,
        "residual": result.get("residual"),
        "labels": labels.long(),
        "sample_count": sample_count,
        "num_classes": num_selected_classes,
    }


def build_stratified_query_for_expert(
    uoc_evidences,
    layer_id,
    expert_id,
    num_classes,
    query_per_class,
    min_query_samples,
    min_query_classes,
    use_top1=True,
    seed=None,
    query_select_mode="class_balanced_random",
    min_expert_token_ratio=0.0,
    max_samples_per_client_per_class=0,
    fallback_to_random=True,
):
    if not use_top1:
        raise ValueError("build_stratified_query_for_expert currently supports use_top1=True only")
    query_stats = {
        "query_select_mode": query_select_mode,
        "token_ratio_threshold": float(min_expert_token_ratio),
        "pool_size_before_filter": 0,
        "pool_size_after_token_ratio_filter": None,
        "selected_by_entropy": 0,
        "fallback_to_random_used": False,
        "entropy_missing": False,
        "expert_token_ratio_mean": None,
        "expert_token_ratio_min": None,
        "expert_token_ratio_max": None,
        "query_entropy_mean": None,
        "max_samples_per_client_per_class": int(max_samples_per_client_per_class),
    }
    if not uoc_evidences:
        return _empty_query_result(num_classes, "no_uoc_evidence", extra_stats=query_stats)

    query_per_class = int(query_per_class)
    min_query_samples = int(min_query_samples)
    min_query_classes = int(min_query_classes)
    has_any_evidence = False
    has_layer_evidence = False
    pool_hidden_chunks = []
    pool_label_chunks = []
    pool_gate_chunks = []
    pool_residual_chunks = []
    pool_entropy_chunks = []
    pool_client_id_chunks = []
    pool_expert_token_ratio_chunks = []
    pool_expert_token_count_chunks = []
    missing_selected_residual = False
    missing_selected_entropy = False
    hidden_tail_shape = None

    for evidence_index, client_evidence in enumerate(uoc_evidences):
        if client_evidence:
            has_any_evidence = True

        layer_evidence = _get_layer_evidence(client_evidence, layer_id)
        if not isinstance(layer_evidence, dict):
            continue
        has_layer_evidence = True

        client_id = evidence_index
        if isinstance(client_evidence, dict) and "client_id" in client_evidence:
            client_id = client_evidence["client_id"]
        if "client_id" in layer_evidence:
            client_id = layer_evidence["client_id"]

        selected = _select_expert_samples(layer_evidence, expert_id, client_id=client_id)
        if selected is None:
            continue

        if hidden_tail_shape is None:
            hidden_tail_shape = selected["hidden"].shape[1:]
        elif selected["hidden"].shape[1:] != hidden_tail_shape:
            # 不同模型/层形状混入时跳过，避免拼接时报错。
            continue

        pool_hidden_chunks.append(selected["hidden"])
        pool_label_chunks.append(selected["labels"])
        pool_gate_chunks.append(selected["gates"])
        if selected["residual"] is None:
            missing_selected_residual = True
        else:
            pool_residual_chunks.append(selected["residual"])
        if selected["entropy"] is None:
            missing_selected_entropy = True
        else:
            pool_entropy_chunks.append(selected["entropy"])
        pool_client_id_chunks.append(selected["client_ids"])
        pool_expert_token_ratio_chunks.append(selected["expert_token_ratio"])
        pool_expert_token_count_chunks.append(selected["expert_token_count"])

    if not has_any_evidence:
        return _empty_query_result(num_classes, "no_uoc_evidence", extra_stats=query_stats)
    if not has_layer_evidence:
        return _empty_query_result(num_classes, "missing_layer_evidence", extra_stats=query_stats)
    if not pool_hidden_chunks:
        return _empty_query_result(num_classes, "no_samples_for_expert", extra_stats=query_stats)

    pool_hidden = torch.cat(pool_hidden_chunks, dim=0)
    pool_labels = torch.cat(pool_label_chunks, dim=0).long()
    pool_gates = torch.cat(pool_gate_chunks, dim=0).float()
    pool_has_residual = (
        not missing_selected_residual
        and len(pool_residual_chunks) == len(pool_hidden_chunks)
    )
    pool_residual = torch.cat(pool_residual_chunks, dim=0) if pool_has_residual else None
    pool_entropy = (
        torch.cat(pool_entropy_chunks, dim=0).float()
        if not missing_selected_entropy and len(pool_entropy_chunks) == len(pool_hidden_chunks)
        else None
    )
    pool_client_ids = torch.cat(pool_client_id_chunks, dim=0).long()
    pool_expert_token_ratio = torch.cat(pool_expert_token_ratio_chunks, dim=0).float()
    pool_expert_token_count = torch.cat(pool_expert_token_count_chunks, dim=0).long()

    query_stats["pool_size_before_filter"] = int(pool_hidden.size(0))
    ratio_mean, ratio_min, ratio_max = _small_tensor_stats(pool_expert_token_ratio)
    query_stats["expert_token_ratio_mean"] = ratio_mean
    query_stats["expert_token_ratio_min"] = ratio_min
    query_stats["expert_token_ratio_max"] = ratio_max

    if query_select_mode == "class_balanced_random":
        query_stats["pool_size_after_token_ratio_filter"] = None
        query_stats["entropy_missing"] = pool_entropy is None
        result = _build_random_class_balanced_query_from_pool(
            pool_hidden=pool_hidden,
            pool_labels=pool_labels,
            pool_gates=pool_gates,
            pool_residual=pool_residual,
            num_classes=num_classes,
            query_per_class=query_per_class,
            min_query_samples=min_query_samples,
            min_query_classes=min_query_classes,
            seed=seed,
        )
        return _add_query_stats(result, query_stats)

    if query_select_mode == "expert_ratio_entropy":
        return _build_expert_ratio_entropy_query_from_pool(
            pool_hidden=pool_hidden,
            pool_labels=pool_labels,
            pool_gates=pool_gates,
            pool_residual=pool_residual,
            pool_client_ids=pool_client_ids,
            pool_entropy=pool_entropy,
            pool_expert_token_ratio=pool_expert_token_ratio,
            pool_expert_token_count=pool_expert_token_count,
            num_classes=num_classes,
            query_per_class=query_per_class,
            min_query_samples=min_query_samples,
            min_query_classes=min_query_classes,
            min_expert_token_ratio=min_expert_token_ratio,
            max_samples_per_client_per_class=max_samples_per_client_per_class,
            fallback_to_random=bool(fallback_to_random),
            seed=seed,
        )

    raise ValueError(f"Unknown UOC-FOGA query_select_mode: {query_select_mode!r}")

def positive_score_to_weights(client_scores, mode="relu", eps=1e-12):
    if mode != "relu":
        raise ValueError(f"Unknown score weighting mode: {mode!r}")

    positive_scores = {}
    for client_id, score in client_scores.items():
        if score is None:
            continue
        positive_scores[client_id] = max(float(score), 0.0)

    if not positive_scores:
        return {}, "no_valid_scores"

    total_score = sum(positive_scores.values())
    if total_score > eps:
        weights = {
            client_id: score / total_score
            for client_id, score in positive_scores.items()
        }
        return weights, None

    uniform_weight = 1.0 / len(positive_scores)
    weights = {client_id: uniform_weight for client_id in positive_scores}
    return weights, "all_scores_non_positive_or_zero"


def expert_weight_entropy(weights, eps=1e-12):
    if not weights:
        return None

    entropy = 0.0
    for weight in weights.values():
        weight = float(weight)
        if weight <= eps:
            continue
        entropy -= weight * math.log(weight)
    return float(entropy)


def summarize_scores(client_scores):
    scores = [
        float(score)
        for score in client_scores.values()
        if score is not None
    ]
    if not scores:
        return {
            "score_mean": None,
            "score_min": None,
            "score_max": None,
            "score_pos_frac": None,
            "score_abs_mean": None,
            "score_abs_max": None,
            "valid_score_count": 0,
        }

    abs_scores = [abs(score) for score in scores]
    return {
        "score_mean": sum(scores) / len(scores),
        "score_min": min(scores),
        "score_max": max(scores),
        "score_pos_frac": sum(1 for score in scores if score > 0.0) / len(scores),
        "score_abs_mean": sum(abs_scores) / len(abs_scores),
        "score_abs_max": max(abs_scores),
        "valid_score_count": len(scores),
    }


def extract_expert_delta_state(client_state, global_state, expert_keys, device=None):
    delta_state = {}
    for key in expert_keys:
        if key not in client_state or key not in global_state:
            continue

        client_tensor = client_state[key]
        global_tensor = global_state[key]
        if not torch.is_tensor(client_tensor) or not torch.is_tensor(global_tensor):
            continue
        if (
            not torch.is_floating_point(client_tensor)
            or not torch.is_floating_point(global_tensor)
        ):
            continue

        if device is not None:
            client_tensor = client_tensor.to(device)
            global_tensor = global_tensor.to(device)
        delta_state[key] = client_tensor - global_tensor

    return delta_state


def make_uniform_weights(client_ids):
    if not client_ids:
        return {}

    weight = 1.0 / len(client_ids)
    return {client_id: weight for client_id in client_ids}

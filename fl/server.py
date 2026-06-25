import os
import time
from types import SimpleNamespace

import torch
from torch import nn
from tqdm import tqdm

from data.loader import (
    build_client_train_loader,
    build_global_eval_loader,
    build_server_query_loader,
    get_client_train_size,
    load_partition_meta,
)
from fl.aggregators import build_aggregator
from fl.client import Client
from model import build_model_from_args
from utils.utils import (
    capture_rng_state,
    init_result_csv,
    init_server_result_csv,
    record_server_result,
    restore_rng_state,
)


class Server:
    def __init__(self, args: SimpleNamespace, logger):
        self.args = args
        self.aggregator = build_aggregator(self.args)

        self.num_clients = self.args.num_clients
        self.server_epochs = self.args.server_epochs
        self.clientsID_list = [i + 1 for i in range(self.num_clients)]

        self.device = self.args.device
        self.logger = logger

        self.model_path = os.path.join(self.args.model_save_path, "server.pth")
        self.resume_enabled = bool(getattr(self.args, "resume", False))
        self.start_round = 0

        self.checkpoint_dir = os.path.join(self.args.model_save_path, "checkpoints")
        os.makedirs(self.args.model_save_path, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.partition_meta = load_partition_meta(self.args)

        self.global_test_loader = build_global_eval_loader(
            args=self.args,
            split="global_test",
            meta=self.partition_meta,
        )

        self.server_query_loader = None
        if getattr(self.args, "uoc_foga_query_select_mode", "") == "server_query":
            self.server_query_loader = build_server_query_loader(
                args=self.args,
                meta=self.partition_meta,
            )
            self.logger.info(
                f"--server_query_loader_enabled : true "
                f"--server_query_loader_size : {len(self.server_query_loader.dataset)}\n"
            )

        self.cache_client_train_loaders = bool(
            getattr(self.args, "cache_client_train_loaders", False)
        )

        self.client_train_loader_cache = {}
        if self.cache_client_train_loaders:
            self.client_train_loader_cache = {
                client_id: build_client_train_loader(
                    args=self.args,
                    client_id=client_id,
                    meta=self.partition_meta,
                )
                for client_id in self.clientsID_list
            }

        self.logger.info(
            f"--cache_client_train_loaders : "
            f"{str(self.cache_client_train_loaders).lower()}\n"
        )
        self.logger.info(
            f"--num_cached_client_loaders : "
            f"{len(self.client_train_loader_cache)}\n"
        )

        self.num_experts = self.args.num_experts
        self.criterion = nn.CrossEntropyLoss()

        if self.resume_enabled:
            self.init_resume_training_state()
            init_result_csv(self.args, overwrite=False)
            init_server_result_csv(self.args, overwrite=False)
        else:
            self.clear_old_checkpoints()
            self.init_fresh_training_state()
            init_result_csv(self.args, overwrite=True)
            init_server_result_csv(self.args, overwrite=True)

    def init_fresh_training_state(self):
        self.model = build_model_from_args(self.args)
        self.best_test_acc = -1.0
        self.best_test_loss = float("inf")
        self.best_round = 0
        self.best_state_dict = None
        self.start_round = 0

        self.save_server_model()

        if not bool(getattr(self.args, "in_memory_client_updates", True)):
            self.sync_clients_model()

    def init_resume_training_state(self):
        self.model = build_model_from_args(self.args)

        checkpoint_path = self.resolve_resume_checkpoint_path()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.validate_training_checkpoint(checkpoint, checkpoint_path)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.best_test_acc = float(checkpoint["best_test_acc"])
        self.best_test_loss = float(checkpoint["best_test_loss"])
        self.best_round = int(checkpoint["best_round"])
        self.best_state_dict = checkpoint.get("best_state_dict")
        self.start_round = int(checkpoint["round_completed"])

        restore_rng_state(checkpoint.get("rng_state"))

        aggregator_state = checkpoint.get("aggregator_state", None)
        if aggregator_state is not None and hasattr(self.aggregator, "load_checkpoint_state"):
            self.aggregator.load_checkpoint_state(
                aggregator_state,
                map_location=self.device,
            )
            self.logger.info("--aggregator_checkpoint_loaded : true\n")

            pism_steps = getattr(self.aggregator, "pism_update_steps", None)
            if pism_steps is not None:
                self.logger.info(f"--pism_update_steps_loaded : {int(pism_steps)}\n")
        else:
            self.logger.info("--aggregator_checkpoint_loaded : false\n")

        self.save_server_model()

        if not bool(getattr(self.args, "in_memory_client_updates", True)):
            self.sync_clients_model()

        self.logger.info(f"[Resume] Loaded checkpoint from {checkpoint_path}")
        self.logger.info(
            f"[Resume] Completed rounds: {self.start_round}, "
            f"next round: {self.start_round + 1}"
        )

    def clear_old_checkpoints(self):
        if not os.path.isdir(self.checkpoint_dir):
            return

        for filename in os.listdir(self.checkpoint_dir):
            if filename == "latest.pth" or (
                filename.startswith("round_") and filename.endswith(".pth")
            ):
                os.remove(os.path.join(self.checkpoint_dir, filename))

    def resolve_resume_checkpoint_path(self):
        resume_checkpoint = getattr(self.args, "resume_checkpoint", "latest")

        if resume_checkpoint == "latest":
            checkpoint_path = os.path.join(self.checkpoint_dir, "latest.pth")
        else:
            checkpoint_path = resume_checkpoint

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"resume=true but checkpoint not found: {checkpoint_path}. "
                "If you want to start a fresh run, set resume: false."
            )

        return checkpoint_path

    def validate_training_checkpoint(self, checkpoint, checkpoint_path):
        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"Training checkpoint must be a dict: {checkpoint_path}, "
                f"got {type(checkpoint).__name__}."
            )

        required_keys = {
            "round_completed",
            "model_state_dict",
            "best_test_acc",
            "best_test_loss",
            "best_round",
            "rng_state",
        }
        missing_keys = required_keys - set(checkpoint.keys())

        if missing_keys:
            raise ValueError(
                f"Training checkpoint {checkpoint_path} is missing keys: "
                f"{sorted(missing_keys)}"
            )

        round_completed = int(checkpoint["round_completed"])

        if round_completed < 0:
            raise ValueError(
                f"Training checkpoint {checkpoint_path} has negative "
                f"round_completed: {round_completed}"
            )

        if round_completed > self.server_epochs:
            raise ValueError(
                f"Training checkpoint round_completed={round_completed} exceeds "
                f"server_epochs={self.server_epochs}. Please check the config."
            )

    def save_training_checkpoint(self, round_completed):
        aggregator_state = None
        if hasattr(self.aggregator, "get_checkpoint_state"):
            aggregator_state = self.aggregator.get_checkpoint_state()

        checkpoint = {
            "round_completed": int(round_completed),
            "model_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in self.model.state_dict().items()
            },
            "best_test_acc": float(self.best_test_acc),
            "best_test_loss": float(self.best_test_loss),
            "best_round": int(self.best_round),
            "best_state_dict": (
                None
                if self.best_state_dict is None
                else {
                    key: value.detach().cpu().clone()
                    for key, value in self.best_state_dict.items()
                }
            ),
            "rng_state": capture_rng_state(),
            "config": dict(vars(self.args)),
            "aggregator_state": aggregator_state,
        }

        round_path = os.path.join(
            self.checkpoint_dir,
            f"round_{round_completed:04d}.pth",
        )
        latest_path = os.path.join(self.checkpoint_dir, "latest.pth")

        torch.save(checkpoint, round_path)
        torch.save(checkpoint, latest_path)

        self.logger.info(f"--checkpoint_saved : {round_path}\n")
        self.logger.info(f"--aggregator_checkpoint_saved : {aggregator_state is not None}\n")

        pism_steps = getattr(self.aggregator, "pism_update_steps", None)
        if pism_steps is not None:
            self.logger.info(f"--pism_update_steps : {int(pism_steps)}\n")

    def save_server_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def get_cpu_state_dict(self):
        return {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

    def sync_clients_model(self):
        server_state_dict = self.get_cpu_state_dict()

        for client_id in self.clientsID_list:
            model_path = os.path.join(self.args.model_save_path, f"{client_id}.pth")
            torch.save(server_state_dict, model_path)

    def collect_server_query_evidence(self):
        """
        服务端从 official test set 划出的 server_query_loader 采集 UOC evidence。
        """
        if self.server_query_loader is None:
            return {}

        if not hasattr(self.model, "collect_uoc_evidence"):
            return {}

        was_training = self.model.training
        self.model.to(self.device)
        self.model.eval()

        evidence_chunks_by_layer = {}
        use_top1 = bool(getattr(self.args, "uoc_foga_use_top1", True))

        non_blocking = (
            str(self.device).startswith("cuda")
            and bool(getattr(self.args, "pin_memory", False))
        )

        inference_context = (
            torch.inference_mode
            if hasattr(torch, "inference_mode")
            else torch.no_grad
        )

        try:
            with inference_context():
                for images, labels in self.server_query_loader:
                    images = images.to(self.device, non_blocking=non_blocking)
                    labels = labels.to(self.device, non_blocking=non_blocking)

                    if labels.size(0) == 0:
                        continue

                    batch_evidence = self.model.collect_uoc_evidence(
                        images,
                        max_samples=labels.size(0),
                        use_top1=use_top1,
                    )

                    for layer_id, layer_evidence in batch_evidence.items():
                        layer_key = str(layer_id)

                        hidden = layer_evidence["hidden"].detach()
                        num_samples = int(hidden.size(0))

                        top1_expert_ids = layer_evidence["top1_expert_ids"][
                            :num_samples
                        ].detach()
                        top1_gates = layer_evidence["top1_gates"][
                            :num_samples
                        ].detach()
                        layer_labels = labels[:num_samples].detach()

                        residual = layer_evidence.get("residual")
                        if residual is not None:
                            residual = residual[:num_samples].detach()

                        entropy = layer_evidence.get("entropy")
                        if entropy is not None:
                            entropy = entropy[:num_samples].detach()

                        if layer_key not in evidence_chunks_by_layer:
                            evidence_chunks_by_layer[layer_key] = {
                                "hidden": [],
                                "labels": [],
                                "top1_expert_ids": [],
                                "top1_gates": [],
                            }

                        evidence_chunks_by_layer[layer_key]["hidden"].append(
                            hidden.cpu()
                        )
                        evidence_chunks_by_layer[layer_key]["labels"].append(
                            layer_labels.cpu()
                        )
                        evidence_chunks_by_layer[layer_key]["top1_expert_ids"].append(
                            top1_expert_ids.cpu()
                        )
                        evidence_chunks_by_layer[layer_key]["top1_gates"].append(
                            top1_gates.cpu()
                        )

                        if residual is not None:
                            evidence_chunks_by_layer[layer_key].setdefault("residual", [])
                            evidence_chunks_by_layer[layer_key]["residual"].append(
                                residual.cpu()
                            )

                        if entropy is not None:
                            evidence_chunks_by_layer[layer_key].setdefault("entropy", [])
                            evidence_chunks_by_layer[layer_key]["entropy"].append(
                                entropy.cpu()
                            )

            server_query_evidence = {}

            for layer_id, layer_chunks in evidence_chunks_by_layer.items():
                server_query_evidence[layer_id] = {}

                for stat_key, chunks in layer_chunks.items():
                    if len(chunks) == 0:
                        continue

                    server_query_evidence[layer_id][stat_key] = torch.cat(
                        chunks,
                        dim=0,
                    ).detach().cpu()

            if server_query_evidence:
                counts_by_layer = {
                    layer_id: int(layer_evidence["hidden"].size(0))
                    for layer_id, layer_evidence in server_query_evidence.items()
                    if isinstance(layer_evidence, dict) and "hidden" in layer_evidence
                }
                self.logger.info(
                    f"[ServerQueryEvidence] counts_by_layer={counts_by_layer}\n"
                )
            else:
                self.logger.info("[ServerQueryEvidence] empty evidence\n")

            return server_query_evidence

        finally:
            self.model.train(was_training)

    def _summarize_uoc_foga_stats(self, uoc_foga_stats):
        if not isinstance(uoc_foga_stats, dict):
            return None

        expert_metrics = [
            metric
            for layer_stats in uoc_foga_stats.values()
            if isinstance(layer_stats, dict)
            for metric in layer_stats.values()
            if isinstance(metric, dict)
        ]

        total_experts = len(expert_metrics)
        fallback_reason_counts = {}
        updated_experts = 0
        score_means = []
        weight_entropies = []
        query_modes = []
        query_fallback_to_random_count = 0

        for metric in expert_metrics:
            fallback_reason = metric.get("fallback_reason")
            reason = fallback_reason if fallback_reason is not None else "none"
            fallback_reason_counts[str(reason)] = fallback_reason_counts.get(str(reason), 0) + 1

            if fallback_reason is None:
                updated_experts += 1

            score_mean = metric.get("score_mean")
            if score_mean is not None:
                score_means.append(float(score_mean))

            weight_entropy = metric.get("weight_entropy")
            if weight_entropy is not None:
                weight_entropies.append(float(weight_entropy))

            query_mode = metric.get("query_select_mode")
            if query_mode is not None:
                query_modes.append(str(query_mode))

            if bool(metric.get("fallback_to_random_used", False)):
                query_fallback_to_random_count += 1

        query_select_mode = (
            query_modes[0]
            if query_modes
            else getattr(self.args, "uoc_foga_query_select_mode", "class_balanced_random")
        )

        return {
            "uoc_foga_updated_experts": updated_experts,
            "uoc_foga_fallback_experts": total_experts - updated_experts,
            "uoc_foga_fallback_reason_counts": fallback_reason_counts,
            "uoc_foga_weight_entropy_mean": (
                sum(weight_entropies) / len(weight_entropies)
                if weight_entropies
                else None
            ),
            "uoc_foga_score_mean_mean": (
                sum(score_means) / len(score_means)
                if score_means
                else None
            ),
            "uoc_foga_query_select_mode": query_select_mode,
            "uoc_foga_query_fallback_to_random_count": query_fallback_to_random_count,
        }

    def train(self):
        if self.start_round >= self.server_epochs:
            self.logger.info(
                f"[Resume] Checkpoint already reached "
                f"server_epochs={self.server_epochs}. Nothing to train."
            )
            return

        total_client_steps = self.server_epochs * len(self.clientsID_list)
        initial_client_steps = self.start_round * len(self.clientsID_list)

        progress_bar = tqdm(
            total=total_client_steps,
            initial=initial_client_steps,
            desc="Experiment progress",
            unit="client",
            dynamic_ncols=True,
            leave=True,
            ascii=False,
            mininterval=0.1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )

        try:
            for c_T in range(self.start_round, self.server_epochs):
                round_start_time = time.perf_counter()

                self.logger.info(
                    f"============================== T:{c_T+1} start !!! ===============================\n"
                )

                use_in_memory_updates = bool(
                    getattr(self.args, "in_memory_client_updates", True)
                )

                if use_in_memory_updates:
                    server_state_dict = self.get_cpu_state_dict()
                else:
                    server_state_dict = None
                    self.save_server_model()
                    self.sync_clients_model()

                round_expert_usage_summary = torch.zeros(self.args.num_experts)
                round_layer_stats = {}
                round_client_expert_usages = []
                round_client_uoc_evidences = []
                round_client_stats = []
                client_states = []

                client_train_start_time = time.perf_counter()

                for client_id in self.clientsID_list:
                    train_loader = (
                        self.client_train_loader_cache.get(client_id)
                        if self.cache_client_train_loaders
                        else None
                    )

                    client_stats = Client(
                        args=self.args,
                        client_id=client_id,
                        logger=self.logger,
                        c_T=c_T,
                        partition_meta=self.partition_meta,
                        initial_state_dict=(
                            server_state_dict if use_in_memory_updates else None
                        ),
                        save_model_to_disk=not use_in_memory_updates,
                        train_loader=train_loader,
                    ).train()

                    progress_bar.update(1)
                    progress_bar.set_postfix_str(
                        f"round={c_T + 1}/{self.server_epochs}, client={client_id}"
                    )
                    progress_bar.refresh()

                    if use_in_memory_updates:
                        if "model_state_dict" not in client_stats:
                            raise KeyError(
                                "in_memory_client_updates=True requires "
                                "Client.train() to return model_state_dict"
                            )
                        client_states.append(client_stats.pop("model_state_dict"))

                    uoc_evidence = client_stats.get("uoc_evidence_by_layer", {})
                    round_client_uoc_evidences.append(uoc_evidence)

                    round_client_stats.append(
                        {
                            "client_id": client_id,
                            "client_loss": client_stats.get(
                                "client_loss",
                                client_stats.get("train_loss", 0.0),
                            ),
                            "train_loss": client_stats.get("train_loss", None),
                            "train_acc": client_stats.get("train_acc", None),
                            "expert_activations": client_stats.get(
                                "expert_activations",
                                None,
                            ),
                            "expert_stats_by_layer": client_stats.get(
                                "expert_stats_by_layer",
                                None,
                            ),
                            "expert_activations_by_layer": client_stats.get(
                                "expert_activations_by_layer",
                                None,
                            ),
                        }
                    )

                    client_expert_usage = client_stats["expert_activations"].float().cpu()
                    round_client_expert_usages.append(client_stats)
                    round_expert_usage_summary += client_expert_usage

                    for layer_id, stats in client_stats.get("expert_stats_by_layer", {}).items():
                        if layer_id not in round_layer_stats:
                            round_layer_stats[layer_id] = {
                                "expert_activations": torch.zeros(self.args.num_experts),
                                "overflow_counts": torch.zeros(self.args.num_experts),
                                "capacity": stats.get("capacity", 0),
                            }

                        round_layer_stats[layer_id]["expert_activations"] += (
                            stats["expert_activations"].float().cpu()
                        )
                        round_layer_stats[layer_id]["overflow_counts"] += (
                            stats["overflow_counts"].float().cpu()
                        )
                        round_layer_stats[layer_id]["capacity"] = stats.get(
                            "capacity",
                            round_layer_stats[layer_id]["capacity"],
                        )

                round_client_train_seconds = time.perf_counter() - client_train_start_time

                usage_list = [int(v) for v in round_expert_usage_summary.tolist()]
                self.logger.info(f"--round_expert_usage_summary : {usage_list}\n")

                client_usage_list = [
                    [int(v) for v in stats["expert_activations"].tolist()]
                    for stats in round_client_expert_usages
                ]

                layer_stats_log = {
                    layer_id: {
                        "expert_activations": [
                            int(v)
                            for v in stats["expert_activations"].tolist()
                        ],
                        "overflow_counts": [
                            int(v)
                            for v in stats["overflow_counts"].tolist()
                        ],
                        "capacity": int(stats["capacity"]),
                    }
                    for layer_id, stats in round_layer_stats.items()
                }

                client_uoc_evidence_counts = []
                for uoc_evidence in round_client_uoc_evidences:
                    if not uoc_evidence:
                        client_uoc_evidence_counts.append(0)
                        continue

                    client_uoc_evidence_counts.append(
                        {
                            str(layer_id): int(layer_evidence["hidden"].shape[0])
                            for layer_id, layer_evidence in uoc_evidence.items()
                            if isinstance(layer_evidence, dict)
                            and "hidden" in layer_evidence
                        }
                    )

                self.logger.info(f"--client_expert_usage_summary : {client_usage_list}\n")
                self.logger.info(f"--round_expert_stats_by_layer : {layer_stats_log}\n")
                self.logger.info(
                    f"--client_uoc_evidence_counts : {client_uoc_evidence_counts}\n"
                )

                aggregation_start_time = time.perf_counter()

                if getattr(self.args, "uoc_foga_query_select_mode", "") == "server_query":
                    server_query_evidence = self.collect_server_query_evidence()
                    uoc_evidence_for_aggregation = server_query_evidence
                else:
                    uoc_evidence_for_aggregation = round_client_uoc_evidences

                self.aggregation(
                    client_states=client_states if use_in_memory_updates else None,
                    uoc_evidence=uoc_evidence_for_aggregation,
                    client_stats=round_client_stats,
                )

                round_aggregation_seconds = time.perf_counter() - aggregation_start_time

                round_completed = c_T + 1
                eval_every = max(1, int(getattr(self.args, "eval_every", 1)))
                should_eval = (
                    round_completed % eval_every == 0
                    or round_completed >= self.server_epochs
                )

                round_eval_seconds = None

                if should_eval:
                    eval_start_time = time.perf_counter()
                    test_loss, test_acc = self.evaluate_global_model(
                        self.global_test_loader
                    )
                    round_eval_seconds = time.perf_counter() - eval_start_time

                    self.logger.info(
                        f"--server_global_test_loss : {test_loss:.4f} "
                        f"--server_global_test_acc : {test_acc:.4f}\n"
                    )

                    is_best = self.update_best_model(
                        test_acc=test_acc,
                        test_loss=test_loss,
                        round_id=round_completed,
                    )

                    record_server_result(
                        {
                            "phase": "test",
                            "round": round_completed,
                            "test_loss": test_loss,
                            "test_acc": test_acc,
                            "best_test_acc": self.best_test_acc,
                            "best_test_loss": self.best_test_loss,
                            "is_best": int(is_best),
                        },
                        self.args,
                    )
                else:
                    self.logger.info(
                        f"--server_global_test_skipped : true "
                        f"--eval_every : {eval_every}\n"
                    )

                if self.args.expert_agg_method in {
                    "uoc_foga_expert_align",
                    "uoc_foga_pism_expert_align",
                }:
                    self.model.to("cpu")

                save_server_each_round = bool(
                    getattr(self.args, "save_server_model_each_round", False)
                )
                if (not use_in_memory_updates) or save_server_each_round:
                    self.save_server_model()

                round_checkpoint_seconds = None
                checkpoint_every = max(1, int(getattr(self.args, "checkpoint_every", 1)))
                should_save_checkpoint = (
                    checkpoint_every <= 1
                    or round_completed % checkpoint_every == 0
                    or round_completed >= self.server_epochs
                )

                if should_save_checkpoint:
                    checkpoint_start_time = time.perf_counter()
                    self.save_training_checkpoint(round_completed)
                    round_checkpoint_seconds = time.perf_counter() - checkpoint_start_time
                else:
                    self.logger.info(
                        f"--checkpoint_skipped : true "
                        f"--checkpoint_every : {checkpoint_every}\n"
                    )

                round_total_seconds = time.perf_counter() - round_start_time

                self.logger.info(
                    f"--round_client_train_seconds : "
                    f"{round_client_train_seconds:.4f}\n"
                )
                self.logger.info(
                    f"--round_aggregation_seconds : "
                    f"{round_aggregation_seconds:.4f}\n"
                )

                if round_eval_seconds is None:
                    self.logger.info("--round_eval_seconds : None\n")
                else:
                    self.logger.info(f"--round_eval_seconds : {round_eval_seconds:.4f}\n")

                if round_checkpoint_seconds is None:
                    self.logger.info("--round_checkpoint_seconds : None\n")
                else:
                    self.logger.info(
                        f"--round_checkpoint_seconds : "
                        f"{round_checkpoint_seconds:.4f}\n"
                    )

                self.logger.info(f"--round_total_seconds : {round_total_seconds:.4f}\n")

        finally:
            progress_bar.close()

        self.logger.info(
            f"--best_global_test_acc : {self.best_test_acc:.4f} "
            f"--best_global_test_loss : {self.best_test_loss:.4f} "
            f"--best_round : {self.best_round}\n"
        )

    def evaluate_global_model(self, data_loader):
        self.model.to(self.device)
        self.model.eval()

        running_loss = 0.0
        running_corrects = 0

        non_blocking = (
            str(self.device).startswith("cuda")
            and bool(getattr(self.args, "pin_memory", False))
        )

        inference_context = (
            torch.inference_mode
            if hasattr(torch, "inference_mode")
            else torch.no_grad
        )

        with inference_context():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device, non_blocking=non_blocking)
                labels = labels.to(self.device, non_blocking=non_blocking)

                result = self.model(inputs)
                outputs = result["logits"]

                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

        average_loss = running_loss / len(data_loader.dataset)
        accuracy = running_corrects.double() / len(data_loader.dataset)

        self.model.to("cpu")

        return average_loss, accuracy.item()

    def update_best_model(self, test_acc, test_loss, round_id):
        is_better = (
            test_acc > self.best_test_acc
            or (test_acc == self.best_test_acc and test_loss < self.best_test_loss)
        )

        if not is_better:
            return False

        self.best_test_acc = test_acc
        self.best_test_loss = test_loss
        self.best_round = round_id
        self.best_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

        best_model_path = os.path.join(self.args.model_save_path, "best_server.pth")

        torch.save(
            {
                "model_state_dict": self.best_state_dict,
                "best_round": self.best_round,
                "best_test_acc": self.best_test_acc,
                "best_test_loss": self.best_test_loss,
            },
            best_model_path,
        )

        self.logger.info(
            f"--best_global_test_acc : {self.best_test_acc:.4f} "
            f"--best_global_test_loss : {self.best_test_loss:.4f} "
            f"--best_round : {self.best_round}\n"
        )

        return True

    def get_client_train_size(self, client_id):
        return get_client_train_size(
            self.args,
            client_id,
            meta=self.partition_meta,
        )

    def aggregation_by_method(self, client_states=None, uoc_evidence=None, client_stats=None):
        client_sizes = []

        if client_states is None:
            loaded_client_states = []

            for client_id in self.clientsID_list:
                client_state_dict = torch.load(
                    os.path.join(self.args.model_save_path, f"{client_id}.pth"),
                    map_location="cpu",
                )
                loaded_client_states.append(client_state_dict)
                client_sizes.append(self.get_client_train_size(client_id))

            client_states = loaded_client_states

        else:
            for client_id in self.clientsID_list:
                client_sizes.append(self.get_client_train_size(client_id))

        use_uoc_foga = self.args.expert_agg_method in {
            "uoc_foga_expert_align",
            "uoc_foga_pism_expert_align",
        }

        if use_uoc_foga:
            self.model.to(self.device)

        try:
            if use_uoc_foga:
                aggregated_state = self.aggregator.aggregate(
                    client_updates=client_states,
                    client_weights=client_sizes,
                    global_model=self.model,
                    uoc_evidence=uoc_evidence,
                    client_stats=client_stats,
                )
            else:
                aggregated_state = self.aggregator.aggregate(
                    client_updates=client_states,
                    client_weights=client_sizes,
                )

            self.model.load_state_dict(aggregated_state)

        finally:
            if use_uoc_foga and not bool(
                getattr(self.args, "keep_uoc_model_on_device_until_eval", True)
            ):
                self.model.to("cpu")

        aggregation_metrics = getattr(self.aggregator, "last_aggregation_metrics", {})
        uoc_foga_stats = aggregation_metrics.get("uoc_foga_stats")
        pism_summary = aggregation_metrics.get("uoc_foga_pism_summary", None)

        if uoc_foga_stats is not None and bool(getattr(self.args, "uoc_foga_log_detail", False)):
            self.logger.info(f"--uoc_foga_stats : {uoc_foga_stats}\n")

        if uoc_foga_stats is not None:
            uoc_summary = self._summarize_uoc_foga_stats(uoc_foga_stats)
            if uoc_summary is not None:
                for key, value in uoc_summary.items():
                    self.logger.info(f"--{key} : {value}\n")

        if pism_summary is not None:
            for key, value in pism_summary.items():
                if isinstance(value, (int, float, str, bool, type(None), dict, list)):
                    self.logger.info(f"--{key} : {value}\n")

        self.logger.info(
            f"--non_expert_agg_method : {self.args.non_expert_agg_method} "
            f"--expert_agg_method : {self.args.expert_agg_method}\n"
        )
        self.logger.info(f"--client_train_sizes : {client_sizes}\n")

    def aggregation(self, client_states=None, uoc_evidence=None, client_stats=None):
        self.aggregation_by_method(
            client_states=client_states,
            uoc_evidence=uoc_evidence,
            client_stats=client_stats,
        )
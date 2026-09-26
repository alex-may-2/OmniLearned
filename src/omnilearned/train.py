import json
import numpy as np
import torch
import torch.nn as nn
from omnilearned.network import PET2, DeepSets, MLPStudent, ACT_LAYERS
from omnilearned.dataloader import load_data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pytorch_optimizer import Lion
from diffusers.optimization import get_cosine_schedule_with_warmup

from omnilearned.utils import (
    is_master_node,
    ddp_setup,
    get_param_groups,
    CLIPLoss,
    get_checkpoint_name,
    get_last_checkpoint_name,
    shadow_copy,
    get_loss,
    get_distill_loss,
    get_distill_cls_loss,
    save_checkpoint,
    restore_checkpoint,
    get_model_parameters,
    get_deepsets_parameters,
    get_mlp_parameters,
)

import time
import os
import torch.amp as amp

torch.set_float32_matmul_precision("high")
torch._dynamo.config.verbose = False


def _log_heartbeat(epoch, batch_idx):
    # Cheap per-rank progress+resource probe (no extra deps) to help localize
    # the 2026-08-07 recurring mid-training NCCL hang: if a worker's RSS/open
    # file-handle count is climbing toward the node ceiling right before a
    # rank stops responding, this shows it without needing a live debugger
    # attach during the hang window.
    try:
        with open("/proc/self/status") as f:
            status = f.read()
        rss_kb = int(
            next(l for l in status.splitlines() if l.startswith("VmRSS:")).split()[1]
        )
    except Exception:
        rss_kb = -1
    try:
        open_fds = len(os.listdir("/proc/self/fd"))
    except Exception:
        open_fds = -1
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(
        f"[heartbeat] rank={rank} epoch={epoch} iter={batch_idx} "
        f"rss_mb={rss_kb / 1024:.0f} open_fds={open_fds}",
        flush=True,
    )


def get_logs(device):
    logs_buff = torch.zeros((8), dtype=torch.float32, device=device)
    logs = {}
    logs["loss"] = logs_buff[0].view(-1)
    logs["loss_class"] = logs_buff[1].view(-1)
    logs["loss_gen"] = logs_buff[2].view(-1)
    logs["loss_clip"] = logs_buff[3].view(-1)
    logs["loss_class_event"] = logs_buff[4].view(-1)
    logs["loss_kd"] = logs_buff[5].view(-1)
    logs["loss_cls_mse"] = logs_buff[6].view(-1)
    # Standardization-independent KD diagnostic (plain KL at distill_T, no
    # logit standardization) -- lets an A/B on --distill-standardize be
    # compared on a common metric, since loss_kd itself changes definition.
    logs["loss_kd_ref"] = logs_buff[7].view(-1)
    return logs


def train_step(
    model,
    dataloader,
    class_cost,
    gen_cost,
    optimizer,
    scheduler,
    mode,
    device,
    clip_loss=CLIPLoss(),
    use_clip=False,
    use_event_loss=False,
    iterations_per_epoch=-1,
    use_amp=False,
    amp_dtype=torch.float16,
    gscaler=None,
    ema_model=None,
    ema_decay=0.9999,
    distill=False,
    distill_alpha=0.5,
    distill_beta=0.5,
    distill_T=4.0,
    distill_teacher_slice=None,
    distill_standardize=False,
    distill_cls=False,
    distill_gamma=0.5,
    distill_cls_num_tokens=4,
    epoch=None,
):
    model.train()

    logs = get_logs(device)

    if iterations_per_epoch < 0:
        iterations_per_epoch = len(dataloader)

    data_iter = iter(dataloader)

    for batch_idx in range(iterations_per_epoch):
        if batch_idx % 100 == 0:
            _log_heartbeat(epoch, batch_idx)

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()  # Zero the gradients

        X, y = batch["X"].to(device, dtype=torch.float), batch["y"].to(device)
        model_kwargs = {
            key: (batch[key].to(device) if batch[key] is not None else None)
            for key in ["cond", "pid", "add_info"]
            if key in batch
        }

        if batch.get("data_pid") is not None:
            data_pid = batch["data_pid"].to(device)
        else:
            data_pid = None

        if batch.get("weight") is not None:
            sample_weight = batch["weight"].to(device, dtype=torch.float)
        else:
            sample_weight = None

        with amp.autocast(
            "cuda:{}".format(device) if torch.cuda.is_available() else "cpu",
            enabled=use_amp,
            dtype=amp_dtype,
        ):
            outputs = model(X, y, **model_kwargs)
            loss = get_loss(
                outputs,
                y,
                mode,
                class_cost,
                gen_cost,
                use_event_loss,
                use_clip,
                clip_loss,
                logs,
                data_pid=data_pid,
                sample_weight=sample_weight,
            )

            if distill:
                if outputs["y_pred"] is None:
                    raise ValueError(
                        "Distillation requires classifier-style logits in outputs['y_pred']."
                    )
                if batch.get("teacher_logits") is None:
                    raise ValueError(
                        "Distillation enabled but no teacher_logits in batch. "
                        "Check --teacher_labels_dir / --teacher_tag and that "
                        "generate-labels was run for this split."
                    )
                teacher_logits = batch["teacher_logits"].to(device)
                if distill_teacher_slice is not None:
                    teacher_logits = teacher_logits[:, distill_teacher_slice[0]:distill_teacher_slice[1]]
                loss_kd = get_distill_loss(
                    outputs["y_pred"], teacher_logits, distill_T=distill_T,
                    standardize=distill_standardize,
                )
                logs["loss_kd"] += loss_kd.detach()
                loss = distill_alpha * loss + distill_beta * loss_kd

                if distill_cls:
                    if outputs["x_body"] is None:
                        raise ValueError(
                            "--distill-cls requires the model's forward to "
                            "return x_body (classifier/ftag/regression/pretrain modes)."
                        )
                    if batch.get("teacher_cls_embed") is None:
                        raise ValueError(
                            "Distillation with --distill-cls enabled but no "
                            "teacher_cls_embed in batch. Check that "
                            "tools/preprocess/build_teacher_h5.py was run with --include-cls-embed "
                            "and --teacher_labels_dir points at those companions."
                        )
                    student_embed = outputs["x_body"][
                        :, :distill_cls_num_tokens
                    ].reshape(outputs["x_body"].shape[0], -1)
                    teacher_embed = batch["teacher_cls_embed"].to(device)
                    loss_cls = get_distill_cls_loss(
                        student_embed, teacher_embed, model.module.cls_projector
                    )
                    logs["loss_cls_mse"] += loss_cls.detach()
                    loss = loss + distill_gamma * loss_cls

        if use_amp and gscaler is not None:
            gscaler.scale(loss).backward()
            gscaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            gscaler.step(optimizer)
            gscaler.update()
        else:
            loss.backward()  # Backward pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()  # Update parameters
        scheduler.step()

        if ema_model is not None:
            with torch.no_grad():
                for ema_p, model_p in zip(
                    ema_model.parameters(), model.module.parameters()
                ):
                    ema_p.mul_(ema_decay).add_(model_p, alpha=1.0 - ema_decay)

    if dist.is_initialized():
        for key in logs:
            dist.all_reduce(logs[key].detach())
            logs[key] = float(logs[key] / dist.get_world_size() / iterations_per_epoch)

    return logs


def val_step(
    model,
    dataloader,
    class_cost,
    gen_cost,
    mode,
    device,
    clip_loss=CLIPLoss(),
    use_clip=False,
    use_event_loss=False,
    iterations_per_epoch=-1,
    distill=False,
    distill_alpha=0.5,
    distill_beta=0.5,
    distill_T=4.0,
    distill_teacher_slice=None,
    distill_standardize=False,
    distill_cls=False,
    distill_gamma=0.5,
    distill_cls_num_tokens=4,
):
    model.eval()

    logs = get_logs(device)

    if iterations_per_epoch < 0:
        iterations_per_epoch = len(dataloader)

    data_iter = iter(dataloader)
    for batch_idx in range(iterations_per_epoch):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # for batch_idx, batch in enumerate(dataloader):
        X, y = batch["X"].to(device, dtype=torch.float), batch["y"].to(device)
        model_kwargs = {
            key: (batch[key].to(device) if batch[key] is not None else None)
            for key in ["cond", "pid", "add_info"]
            if key in batch
        }

        if batch.get("data_pid") is not None:
            data_pid = batch["data_pid"].to(device)
        else:
            data_pid = None

        if batch.get("weight") is not None:
            sample_weight = batch["weight"].to(device, dtype=torch.float)
        else:
            sample_weight = None

        with torch.no_grad():
            outputs = model(X, y, **model_kwargs)
            get_loss(
                outputs,
                y,
                mode,
                class_cost,
                gen_cost,
                use_event_loss,
                use_clip,
                clip_loss,
                logs,
                data_pid=data_pid,
                sample_weight=sample_weight,
            )

            if distill:
                if outputs["y_pred"] is None:
                    raise ValueError(
                        "Distillation requires classifier-style logits in outputs['y_pred']."
                    )
                if batch.get("teacher_logits") is None:
                    raise ValueError(
                        "Validation distillation enabled but no teacher_logits in batch."
                    )
                teacher_logits = batch["teacher_logits"].to(device)
                if distill_teacher_slice is not None:
                    teacher_logits = teacher_logits[:, distill_teacher_slice[0]:distill_teacher_slice[1]]
                loss_kd = get_distill_loss(
                    outputs["y_pred"], teacher_logits, distill_T=distill_T,
                    standardize=distill_standardize,
                )
                logs["loss_kd"] += loss_kd.detach()
                # Common-metric diagnostic: plain KL at distill_T regardless of
                # the standardize flag, so std vs no-std runs are comparable.
                if distill_standardize:
                    logs["loss_kd_ref"] += get_distill_loss(
                        outputs["y_pred"], teacher_logits, distill_T=distill_T,
                        standardize=False,
                    ).detach()
                else:
                    logs["loss_kd_ref"] += loss_kd.detach()

                if distill_cls:
                    if outputs["x_body"] is None:
                        raise ValueError(
                            "--distill-cls requires the model's forward to "
                            "return x_body (classifier/ftag/regression/pretrain modes)."
                        )
                    if batch.get("teacher_cls_embed") is None:
                        raise ValueError(
                            "Validation distillation with --distill-cls enabled "
                            "but no teacher_cls_embed in batch."
                        )
                    student_embed = outputs["x_body"][
                        :, :distill_cls_num_tokens
                    ].reshape(outputs["x_body"].shape[0], -1)
                    teacher_embed = batch["teacher_cls_embed"].to(device)
                    loss_cls = get_distill_cls_loss(
                        student_embed, teacher_embed, model.module.cls_projector
                    )
                    logs["loss_cls_mse"] += loss_cls.detach()

    if dist.is_initialized():
        for key in logs:
            dist.all_reduce(logs[key].detach())
            logs[key] = float(logs[key] / dist.get_world_size() / iterations_per_epoch)

    return logs


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    lr_scheduler,
    mode,
    num_epochs=1,
    device="cpu",
    patience=500,
    loss_class=nn.CrossEntropyLoss(),
    loss_gen=nn.MSELoss(),
    use_clip=False,
    use_event_loss=False,
    output_dir="",
    save_tag="",
    iterations_per_epoch=-1,
    epoch_init=0,
    loss_init=np.inf,
    best_epoch_init=None,
    use_amp=False,
    amp_dtype=torch.float16,
    run=None,
    ema_model=None,
    ema_decay=0.999,
    distill=False,
    distill_alpha=0.5,
    distill_beta=0.5,
    distill_T=4.0,
    distill_teacher_slice=None,
    distill_standardize=False,
    distill_cls=False,
    distill_gamma=0.5,
    distill_cls_num_tokens=4,
):
    checkpoint_name = get_checkpoint_name(save_tag)
    last_checkpoint_name = get_last_checkpoint_name(save_tag)

    losses = {
        "train_loss": [],
        "val_loss": [],
    }

    tracker = {
        "bestValLoss": loss_init,
        "bestEpoch": epoch_init if best_epoch_init is None else best_epoch_init,
    }
    if use_amp and amp_dtype == torch.float16:
        # GradScaler's loss-scaling only guards against fp16 underflow; bf16's
        # wider exponent range doesn't need it and scaling can push bf16
        # values toward overflow instead, so skip it for that dtype.
        gscaler = amp.GradScaler()
    else:
        gscaler = None
    for epoch in range(int(epoch_init), num_epochs):
        if isinstance(
            train_loader.sampler, torch.utils.data.distributed.DistributedSampler
        ):
            train_loader.sampler.set_epoch(epoch)

        start = time.time()
        train_logs = train_step(
            model,
            train_loader,
            loss_class,
            loss_gen,
            optimizer,
            lr_scheduler,
            mode,
            device,
            use_clip=use_clip,
            use_event_loss=use_event_loss,
            iterations_per_epoch=iterations_per_epoch,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            gscaler=gscaler,
            ema_model=ema_model,
            ema_decay=ema_decay,
            distill=distill,
            distill_alpha=distill_alpha,
            distill_beta=distill_beta,
            distill_T=distill_T,
            distill_teacher_slice=distill_teacher_slice,
            distill_standardize=distill_standardize,
            distill_cls=distill_cls,
            distill_gamma=distill_gamma,
            distill_cls_num_tokens=distill_cls_num_tokens,
            epoch=epoch,
        )
        val_logs = val_step(
            model,
            val_loader,
            loss_class,
            loss_gen,
            mode,
            device,
            use_clip=use_clip,
            use_event_loss=use_event_loss,
            iterations_per_epoch=iterations_per_epoch,
            distill=distill,
            distill_alpha=distill_alpha,
            distill_beta=distill_beta,
            distill_T=distill_T,
            distill_teacher_slice=distill_teacher_slice,
            distill_standardize=distill_standardize,
            distill_cls=distill_cls,
            distill_gamma=distill_gamma,
            distill_cls_num_tokens=distill_cls_num_tokens,
        )

        losses["train_loss"].append(train_logs["loss"])
        losses["val_loss"].append(val_logs["loss"])

        if is_master_node():
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] Loss: {losses['train_loss'][-1]:.4f}, Val Loss: {losses['val_loss'][-1]:.4f} , lr: {lr_scheduler.get_last_lr()[0]}"
            )
            print(
                f"Class Loss: {train_logs['loss_class']:.4f}, Class Val Loss: {val_logs['loss_class']:.4f}"
            )
            if use_event_loss:
                print(
                    f"Class Event Loss: {train_logs['loss_class_event']:.4f}, Class Event Val Loss: {val_logs['loss_class_event']:.4f}"
                )
            print(
                f"Gen Loss: {train_logs['loss_gen']:.4f}, Gen Val Loss: {val_logs['loss_gen']:.4f}"
            )
            if use_clip:
                print(
                    f"CLIP loss: {train_logs['loss_clip']:.4f}, CLIP Val Loss: {val_logs['loss_clip']:.4f}"
                )
            if distill:
                print(
                    f"KD Loss: {train_logs['loss_kd']:.4f}, KD Val Loss: {val_logs['loss_kd']:.4f}"
                )
                if distill_cls:
                    print(
                        f"CLS-MSE Loss: {train_logs['loss_cls_mse']:.4f}, "
                        f"CLS-MSE Val Loss: {val_logs['loss_cls_mse']:.4f}"
                    )
            print(
                "Time taken for epoch {} is {} sec".format(epoch, time.time() - start)
            )

        if losses["val_loss"][-1] < tracker["bestValLoss"]:
            tracker["bestValLoss"] = losses["val_loss"][-1]
            tracker["bestEpoch"] = epoch

            if is_master_node():
                print("replacing best checkpoint ...")
                save_checkpoint(
                    model,
                    ema_model,
                    epoch + 1,
                    optimizer,
                    losses["val_loss"][-1],
                    lr_scheduler,
                    output_dir,
                    checkpoint_name,
                )

        if is_master_node():
            save_checkpoint(
                model,
                ema_model,
                epoch + 1,
                optimizer,
                losses["val_loss"][-1],
                lr_scheduler,
                output_dir,
                last_checkpoint_name,
                best_loss=tracker["bestValLoss"],
                best_epoch=tracker["bestEpoch"],
            )

        if run is not None and is_master_node():
            run.log(
                {
                    **{f"train {key}": train_logs[key] for key in train_logs},
                    **{f"val {key}": val_logs[key] for key in val_logs},
                },
                step=epoch,
            )

        if epoch - tracker["bestEpoch"] > patience:
            print(f"breaking on device: {device}")
            break

    if is_master_node():
        print(
            f"Training Complete, best loss: {tracker['bestValLoss']:.5f} at epoch {tracker['bestEpoch']}!"
        )
        # save losses
        json.dump(losses, open(f"{output_dir}/training_{save_tag}.json", "w"))


def run(
    outdir: str = "",
    save_tag: str = "",
    pretrain_tag: str = "pretrain",
    dataset: str = "top",
    path: str = "/pscratch/sd/v/vmikuni/datasets",
    wandb=False,
    fine_tune: bool = False,
    resuming: bool = False,
    seed: int = -1,
    num_feat: int = 4,
    model_size: str = "small",
    interaction: bool = False,
    local_interaction: bool = False,
    num_coord: int = 2,
    K: int = 10,
    interaction_type: str = "lhc",
    conditional: bool = False,
    num_cond: bool = 3,
    use_pid: bool = False,
    pid_idx: int = -1,
    pid_dim: int = 9,
    use_add: bool = False,
    num_add: int = 4,
    zero_add: bool = False,
    use_clip: bool = False,
    use_event_loss: bool = False,
    num_classes: int = 2,
    num_gen_classes: int = 1,
    mode: str = "classifier",
    batch: int = 64,
    iterations: int = -1,
    epoch: int = 15,
    warmup_epoch: int = 1,
    use_amp: bool = False,
    amp_dtype: str = "fp16",
    optim: str = "lion",
    sched: str = "cosine",
    b1: float = 0.95,
    b2: float = 0.98,
    lr: float = 5e-4,
    lr_factor: float = 10.0,
    wd: float = 0.3,
    nevts: int = -1,
    attn_drop: float = 0.1,
    mlp_drop: float = 0.1,
    feature_drop: float = 0.0,
    num_workers: int = 16,
    clip_inputs: bool = False,
    distill: bool = False,
    teacher_labels_dir: str = "",
    teacher_tag: str = "",
    distill_alpha: float = 0.5,
    distill_beta: float = 0.5,
    distill_T: float = 4.0,
    distill_teacher_slice: str = "",
    distill_standardize: bool = False,
    distill_cls: bool = False,
    distill_gamma: float = 0.5,
    distill_cls_teacher_dim: int = 1024,
    arch: str = "pet2",
    energy_weighted_pool: bool = False,
    num_interaction_layers: int = 0,
    interaction_k: int = 0,
    act_layer: str = "gelu",
    deepsets_fixed_n: int = 0,
):
    if energy_weighted_pool and arch != "deep-sets":
        raise ValueError("--energy-weighted-pool requires --arch deep-sets.")

    if (num_interaction_layers or interaction_k) and arch != "deep-sets":
        raise ValueError(
            "--num-interaction-layers / --interaction-k require --arch deep-sets."
        )

    if act_layer not in ACT_LAYERS:
        raise ValueError(
            f"--act-layer must be one of {list(ACT_LAYERS)}, got '{act_layer}'"
        )
    if act_layer != "gelu" and arch != "deep-sets":
        raise ValueError("--act-layer is only wired for --arch deep-sets.")
    _act_layer_cls = ACT_LAYERS[act_layer]

    if deepsets_fixed_n:
        if arch != "deep-sets":
            raise ValueError("--deepsets-fixed-n requires --arch deep-sets.")
        if num_interaction_layers:
            raise ValueError(
                "--deepsets-fixed-n is not supported with "
                "--num-interaction-layers (message passing needs the mask)."
            )
        if energy_weighted_pool:
            raise ValueError(
                "--deepsets-fixed-n is incompatible with --energy-weighted-pool."
            )

    if distill and (not teacher_labels_dir or not teacher_tag):
        raise ValueError(
            "--distill requires both --teacher_labels_dir and --teacher_tag."
        )

    if distill_cls and arch != "pet2":
        raise ValueError(
            "--distill-cls requires --arch pet2 (needs a token-based body "
            "embedding via outputs['x_body']; DeepSets/MLP students have no "
            "equivalent representation)."
        )

    _teacher_slice = None
    if distill_teacher_slice:
        parts = distill_teacher_slice.split(":")
        _teacher_slice = (int(parts[0]), int(parts[1]))

    amp_dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16}
    if amp_dtype not in amp_dtypes:
        raise ValueError(f"--amp-dtype must be one of {list(amp_dtypes)}, got '{amp_dtype}'")
    _amp_dtype = amp_dtypes[amp_dtype]

    local_rank, rank, size = ddp_setup()

    if seed >= 0:
        # Rank-offset so DDP replicas still see different data draws, but the
        # whole run is reproducible given `seed` -- needed for a controlled A/B
        # (e.g. --distill-standardize on vs off with identical init + data order).
        import random as _random

        _s = seed + rank
        _random.seed(_s)
        np.random.seed(_s)
        torch.manual_seed(_s)
        torch.cuda.manual_seed_all(_s)
        if rank == 0:
            print(f"[seed] deterministic run, base seed {seed} (+rank)")

    # set up model
    if arch == "pet2":
        model_params = get_model_parameters(model_size)
        model = PET2(
            input_dim=num_feat,
            use_int=interaction,
            local_int=local_interaction,
            int_type=interaction_type,
            conditional=conditional,
            cond_dim=num_cond,
            pid=use_pid,
            pid_dim=pid_dim,
            add_info=use_add,
            add_dim=num_add,
            mode=mode,
            num_classes=num_classes,
            num_gen_classes=num_gen_classes,
            mlp_drop=mlp_drop,
            attn_drop=attn_drop,
            feature_drop=feature_drop,
            num_coord=num_coord,
            K=K,
            **model_params,
        )
    elif arch == "deep-sets":
        ds_params = get_deepsets_parameters(model_size)
        model = DeepSets(
            input_dim=num_feat,
            num_classes=num_classes,
            pid=use_pid,
            pid_dim=pid_dim,
            add_info=use_add,
            add_dim=num_add,
            conditional=conditional,
            cond_dim=num_cond,
            mode=mode,
            mlp_drop=mlp_drop,
            energy_weighted_pool=energy_weighted_pool,
            num_interaction_layers=num_interaction_layers,
            interaction_k=interaction_k,
            act_layer=_act_layer_cls,
            fixed_n=deepsets_fixed_n,
            **ds_params,
        )
        # Saved into the checkpoint so the exact network can be rebuilt for FPGA
        # conversion: activation and masking have no parameters, so they cannot
        # be recovered from the weights.
        model.arch_config = {
            "arch": arch,
            "size": model_size,
            "dims": ds_params,
            "input_dim": num_feat,
            "num_classes": num_classes,
            "act_layer": act_layer,
            "fixed_n": deepsets_fixed_n,  # 0 = masked mean over valid particles
            "num_interaction_layers": num_interaction_layers,
            "interaction_k": interaction_k,
            "energy_weighted_pool": energy_weighted_pool,
            "pid": use_pid,
            "add_info": use_add,
            "conditional": conditional,
            "quant": None,
        }
    elif arch == "mlp":
        mlp_params = get_mlp_parameters(model_size)
        model = MLPStudent(
            input_dim=num_feat,
            num_classes=num_classes,
            mode=mode,
            **mlp_params,
        )
    else:
        raise ValueError(f"Unknown arch '{arch}'. Choose from: pet2, deep-sets, mlp")

    if distill_cls:
        # Attached as a submodule (not a free-standing module passed around)
        # so it rides along for free with get_param_groups/DDP-wrap/EMA/
        # checkpoint save-restore -- all of which already operate on `model`.
        # Trained jointly with the student, discarded at inference.
        student_flat_dim = model_params["num_tokens"] * model_params["base_dim"]
        teacher_flat_dim = model_params["num_tokens"] * distill_cls_teacher_dim
        model.cls_projector = nn.Linear(student_flat_dim, teacher_flat_dim)

    if rank == 0:
        d = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("**** Setup ****")
        print(
            "Total params: %.2fM"
            % (sum(p.numel() for p in model.parameters()) / 1000000.0)
        )
        print(f"Training on device: {d}, with {size} GPUs")
        print("************")

    # load in train data
    train_loader = load_data(
        dataset,
        dataset_type="train",
        use_cond=conditional,
        use_pid=use_pid,
        pid_idx=pid_idx,
        use_add=use_add,
        num_add=num_add,
        path=path,
        batch=batch,
        num_workers=num_workers,
        rank=rank,
        size=size,
        clip_inputs=clip_inputs,
        mode=mode,
        nevts=nevts,
        teacher_labels_dir=teacher_labels_dir if distill else None,
        teacher_tag=teacher_tag if distill else None,
    )
    if rank == 0:
        print("**** Setup ****")
        print(f"Train dataset len: {len(train_loader)}")
        print("************")

    val_loader = load_data(
        dataset,
        dataset_type="val",
        use_cond=conditional,
        use_pid=use_pid,
        pid_idx=pid_idx,
        use_add=use_add,
        num_add=num_add,
        path=path,
        batch=batch,
        num_workers=num_workers,
        rank=rank,
        size=size,
        clip_inputs=clip_inputs,
        mode=mode,
        teacher_labels_dir=teacher_labels_dir if distill else None,
        teacher_tag=teacher_tag if distill else None,
    )

    param_groups = get_param_groups(
        model, wd, lr, lr_factor=lr_factor, fine_tune=fine_tune
    )

    if optim not in ["adam", "lion"]:
        raise ValueError(
            f"Optimizer '{optim}' not supported. Choose from adam or lion."
        )
    if sched not in ["cosine", "onecycle"]:
        raise ValueError(
            f"Scheduler '{sched}' not supported. Choose from cosine or onecycle."
        )

    if optim == "lion":
        optimizer = Lion(param_groups, betas=(b1, b2))
    elif optim == "adam":
        optimizer = torch.optim.AdamW(param_groups)

    train_steps = len(train_loader) if iterations < 0 else iterations

    if sched == "onecycle":
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            total_steps=(train_steps * epoch),
            max_lr=lr,
            pct_start=0.1,
        )
    elif sched == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=train_steps * warmup_epoch,
            num_training_steps=(train_steps * epoch),
        )

    # Transfer model to GPU if available
    kwarg = {}
    if torch.cuda.is_available():
        device = local_rank
        model.to(local_rank)
        kwarg["device_ids"] = [device]
    else:
        model.cpu()
        device = "cpu"

    model = DDP(
        model,
        **kwarg,
    )

    # Set up EMA model
    ema_model = shadow_copy(model.module)

    epoch_init = 0
    loss_init = np.inf
    best_epoch_init = None
    checkpoint_name = None

    last_checkpoint_path = os.path.join(outdir, get_last_checkpoint_name(save_tag))
    best_checkpoint_path = os.path.join(outdir, get_checkpoint_name(save_tag))

    if resuming and os.path.isfile(last_checkpoint_path):
        if is_master_node():
            print(f"Continue training with checkpoint from {last_checkpoint_path}")
        checkpoint_name = get_last_checkpoint_name(save_tag)
        fine_tune = False

    elif resuming and os.path.isfile(best_checkpoint_path):
        if is_master_node():
            print(
                f"No last-epoch checkpoint found, continuing training with best "
                f"checkpoint from {best_checkpoint_path}"
            )
        checkpoint_name = get_checkpoint_name(save_tag)
        fine_tune = False

    elif fine_tune:
        if is_master_node():
            print(
                f"Will fine-tune using checkpoint {os.path.join(outdir, get_checkpoint_name(pretrain_tag))}"
            )
        checkpoint_name = get_checkpoint_name(pretrain_tag)

    if checkpoint_name is not None:
        epoch_init, loss_init, best_epoch_init = restore_checkpoint(
            model,
            outdir,
            checkpoint_name,
            local_rank,
            is_main_node=is_master_node(),
            ema_model=ema_model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            fine_tune=fine_tune,
        )

    if wandb and num_workers > 0:
        # Force the persistent DataLoader worker pool to fork now, before wandb.init()
        # (below) starts its background async-service thread. If workers fork *after*
        # wandb is live, they inherit wandb's Python objects but not the background
        # thread that services them -- when Python's GC later finalizes one of those
        # inherited objects inside a worker (a matter of allocation-pressure timing,
        # not anything data-dependent), the finalizer calls back into wandb expecting
        # that thread to respond and blocks forever, freezing the worker mid-read and
        # hanging the whole DDP job (all other ranks then hang waiting on this rank's
        # next collective). Confirmed via a live py-spy capture -- see
        # distill-lazy-teacher-progress memory, 2026-08-08 root-cause entry. Discarding
        # the iterator here is safe: with persistent_workers=True, the DataLoader caches
        # and reuses the same worker pool on every later `iter()` call.
        next(iter(train_loader))
        next(iter(val_loader))

    if wandb:
        import wandb

        if is_master_node():
            mode_wandb = None
            wandb.login()
        else:
            mode_wandb = "disabled"

        run = wandb.init(
            # Set the project where this run will be logged
            project="OmniBoone",
            name=save_tag,
            mode=mode_wandb,
            # Track hyperparameters and run metadata
            config={
                "learning_rate": lr,
                "epochs": epoch,
                "batch size": batch,
                "mode": mode,
                "size": model_size,
                "fine_tune": fine_tune,
                "distill": distill,
                "distill_alpha": distill_alpha,
                "distill_beta": distill_beta,
                "distill_T": distill_T,
                "distill_standardize": distill_standardize,
                "distill_cls": distill_cls,
                "distill_gamma": distill_gamma,
            },
        )
    else:
        run = None

    if mode == "regression":
        loss_class = nn.MSELoss(reduction="none")
    else:
        loss_class = nn.CrossEntropyLoss(reduction="none")

    if mode == "ftag":
        loss_gen = nn.CrossEntropyLoss(reduction="none")
    else:
        loss_gen = nn.MSELoss(reduction="none")

    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        lr_scheduler,
        mode=mode,
        num_epochs=epoch,
        device=device,
        loss_class=loss_class,
        loss_gen=loss_gen,
        output_dir=outdir,
        save_tag=save_tag,
        use_clip=use_clip,
        use_event_loss=use_event_loss,
        iterations_per_epoch=iterations,
        epoch_init=epoch_init,
        loss_init=loss_init,
        best_epoch_init=best_epoch_init,
        use_amp=use_amp,
        amp_dtype=_amp_dtype,
        run=run,
        ema_model=ema_model,
        distill=distill,
        distill_alpha=distill_alpha,
        distill_beta=distill_beta,
        distill_T=distill_T,
        distill_teacher_slice=_teacher_slice,
        distill_standardize=distill_standardize,
        distill_cls=distill_cls,
        distill_gamma=distill_gamma,
    )

    dist.destroy_process_group()

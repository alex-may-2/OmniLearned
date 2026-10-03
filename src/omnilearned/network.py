import torch
import torch.nn as nn


from omnilearned.layers import (
    NoScaleDropout,
    InteractionBlock,
    LocalEmbeddingBlock,
    MLP,
    AttBlock,
    DynamicTanh,
    InputBlock,
    TokenAttBlock,
    get_mass,
    get_dr,
    get_kt,
)
from omnilearned.diffusion import MPFourier, perturb, get_logsnr_alpha_sigma

# String -> activation class, shared by the CLI and the QAT/export tools so a
# single spelling ("gelu" | "relu") drives every entry point. ReLU is the
# hls4ml-friendly choice; GELU has no ONNX/hls4ml lowering.
ACT_LAYERS = {"gelu": nn.GELU, "relu": nn.ReLU}


class PET2(nn.Module):
    def __init__(
        self,
        input_dim,
        use_int=True,
        local_int=True,
        int_type="lhc",
        conditional=False,
        cond_dim=3,
        pid=False,
        pid_dim=9,
        add_info=False,
        add_dim=4,
        mode="classifier",
        num_classes=2,
        num_gen_classes=1,
        base_dim=128,
        num_transformers=2,
        num_transformers_head=2,
        num_tokens=4,
        num_heads=4,
        mlp_ratio=2,
        norm_layer=DynamicTanh,
        act_layer=nn.GELU,
        mlp_drop=0.0,
        attn_drop=0.0,
        feature_drop=0.0,
        K=15,
        skip=False,
        num_coord=3,
    ):
        super().__init__()
        self.mode = mode
        if self.mode not in [
            "classifier",
            "generator",
            "regression",
            "segmentation",
            "ftag",
            "pretrain",
        ]:
            raise ValueError(f"Mode '{self.mode}' not supported.")

        self.body = PET_body(
            input_dim,
            base_dim,
            num_transformers=num_transformers,
            num_transf_local=num_transformers_head,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            act_layer=act_layer,
            mlp_drop=mlp_drop,
            attn_drop=attn_drop,
            feature_drop=feature_drop,
            num_tokens=num_tokens,
            K=K,
            use_int=use_int,
            local_int=local_int,
            int_type=int_type,
            conditional=conditional,
            cond_dim=cond_dim,
            pid=pid,
            pid_dim=pid_dim,
            add_info=add_info,
            add_dim=add_dim,
            use_time=self.mode in ["generator", "pretrain"],
            skip=skip,
            num_coord=num_coord,
        )

        self.num_add = self.body.num_add
        self.num_tokens = num_tokens
        self.classifier = None
        self.generator = None

        use_classifier = self.mode in ["classifier", "ftag", "regression", "pretrain"]
        use_generator = self.mode in ["generator", "segmentation", "ftag", "pretrain"]
        if use_classifier:
            self.classifier = PET_classifier(
                base_dim,
                num_transformers=num_transformers_head,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_drop=mlp_drop,
                attn_drop=attn_drop,
                num_tokens=num_tokens,
                num_classes=num_classes,
            )

        if use_generator:
            self.generator = PET_generator(
                input_dim
                if num_gen_classes == 1
                else num_gen_classes,  # diffusion or segmentation
                base_dim,
                num_transformers=num_transformers_head,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_drop=mlp_drop,
                attn_drop=attn_drop,
                num_tokens=num_tokens,
                num_add=self.num_add,
                num_classes=num_classes,
                skip_pid=(num_classes == 1) | (self.mode == "segmentation"),
            )

        self.initialize_weights()

    def initialize_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)

    def no_weight_decay(self):
        # Specify parameters that should not be decayed
        return {"norm", "token"}

    def forward(self, x, y, cond=None, pid=None, add_info=None):
        y_pred, y_perturb, z_pred, v, v_weight, x_body, z_body = (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

        time = torch.rand(size=(x.shape[0],)).to(x.device)
        _, alpha, sigma = get_logsnr_alpha_sigma(time)

        if self.mode in ["generator", "pretrain"]:
            z, v, v_weight = perturb(x, time)
            z_body = self.body(z, cond, pid, add_info, time)
            z_pred = self.generator(z_body, y)

        if self.mode in [
            "classifier",
            "regression",
            "segmentation",
            "pretrain",
            "ftag",
        ]:
            x_body = self.body(x, cond, pid, add_info, torch.zeros_like(time))

            if self.mode in ["classifier", "pretrain", "regression", "ftag"]:
                y_pred = self.classifier(x_body)
            if self.mode == "pretrain":
                y_perturb = self.classifier(z_body)
            if self.mode == "ftag" or self.mode == "segmentation":
                z_pred = self.generator(x_body, y)

        return {
            "y_pred": y_pred,
            "y_perturb": y_perturb,
            "z_pred": z_pred,
            "v": v,
            "v_weight": v_weight,
            "x_body": x_body,
            "z_body": z_body,
            "alpha": alpha**2,
        }


class PET_classifier(nn.Module):
    def __init__(
        self,
        base_dim,
        num_transformers=2,
        num_heads=4,
        mlp_ratio=2,
        norm_layer=DynamicTanh,
        act_layer=nn.GELU,
        mlp_drop=0.1,
        attn_drop=0.1,
        num_tokens=4,
        num_classes=2,
    ):
        super().__init__()
        self.num_tokens = num_tokens

        self.in_blocks = nn.ModuleList(
            [
                TokenAttBlock(
                    dim=base_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    mlp_drop=mlp_drop,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    num_tokens=self.num_tokens,
                    skip=False,
                )
                for _ in range(num_transformers)
            ]
        )

        self.fc = MLP(
            base_dim * self.num_tokens,
            int(mlp_ratio * self.num_tokens * base_dim),
            act_layer=act_layer,
            drop=mlp_drop,
            norm_layer=norm_layer,
        )

        self.out = nn.Linear(self.num_tokens * base_dim, num_classes)

        self.initialize_weights()

    def initialize_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)

    def forward(self, x):
        B = x.shape[0]
        mask = x[:, self.num_tokens :, 2:3] != 0
        for ib, blk in enumerate(self.in_blocks):
            x = blk(x, mask=mask)

        x = self.fc(x[:, : self.num_tokens].reshape(B, -1))
        return self.out(x)


class PET_generator(nn.Module):
    def __init__(
        self,
        output_size,
        base_dim,
        num_transformers=2,
        num_heads=4,
        mlp_ratio=2,
        norm_layer=DynamicTanh,
        act_layer=nn.GELU,
        mlp_drop=0.1,
        attn_drop=0.1,
        num_tokens=4,
        num_add=1,
        num_classes=2,
        skip_pid=False,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_add = num_add
        self.num_classes = num_classes
        self.skip_pid = skip_pid

        if not self.skip_pid:
            self.pid_embed = nn.Sequential(
                nn.Embedding(num_classes, base_dim),
                MLP(
                    base_dim,
                    int(mlp_ratio * base_dim),
                    out_features=base_dim,
                    act_layer=act_layer,
                    drop=mlp_drop,
                ),
            )
            self.num_add = self.num_add + 1

        self.in_blocks = nn.ModuleList(
            [
                AttBlock(
                    dim=base_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    mlp_drop=mlp_drop,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    num_tokens=num_tokens,
                    skip=False,
                    use_int=False,
                )
                for _ in range(num_transformers)
            ]
        )

        self.fc = nn.Sequential(
            MLP(
                base_dim,
                int(mlp_ratio * base_dim),
                act_layer=act_layer,
                drop=mlp_drop,
            ),
        )

        self.out = nn.Linear(base_dim, output_size)

        self.initialize_weights()

    def initialize_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)

    def forward(self, x, y=None):
        # Add tokens and label embedding
        mask = x[:, :, 2:3] != 0

        if not self.skip_pid and y is not None:
            mask = torch.cat([torch.ones_like(mask[:, :1]), mask], 1)
            x = torch.cat([self.pid_embed(y).unsqueeze(1), x], 1) * mask

        for ib, blk in enumerate(self.in_blocks):
            x = blk(x, mask=mask)
        x = (
            self.fc(x[:, self.num_add + self.num_tokens :])
            * mask[:, self.num_add + self.num_tokens :]
        )
        return self.out(x) * mask[:, self.num_add + self.num_tokens :]


class PET_body(nn.Module):
    def __init__(
        self,
        input_dim,
        base_dim,
        num_transformers=2,
        num_transf_local=2,
        num_heads=4,
        mlp_ratio=2,
        norm_layer=DynamicTanh,
        act_layer=nn.GELU,
        mlp_drop=0.1,
        attn_drop=0.1,
        feature_drop=0.0,
        num_tokens=4,
        K=10,
        use_int=True,
        local_int=True,
        int_type="lhc",
        conditional=False,
        cond_dim=3,
        pid=False,
        pid_dim=9,
        add_info=False,
        add_dim=4,
        use_time=False,
        skip=False,
        num_coord=3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.use_int = use_int
        self.use_time = use_time
        self.conditional = conditional
        self.pid = pid
        self.add_info = add_info
        self.skip = skip
        self.num_coord = num_coord
        self.embed = InputBlock(
            in_features=input_dim,
            hidden_features=int(mlp_ratio * base_dim),
            out_features=base_dim,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

        if self.use_int:
            self.interaction = InteractionBlock(
                hidden_features=base_dim,
                out_features=num_heads,
                # mlp_drop=mlp_drop,
                act_layer=act_layer,
                norm_layer=norm_layer,
                int_type=int_type,
                feature_drop=feature_drop,
            )

        self.local_physics = LocalEmbeddingBlock(
            in_features=input_dim,
            hidden_features=mlp_ratio * base_dim,
            out_features=base_dim,
            act_layer=act_layer,
            mlp_drop=mlp_drop,
            attn_drop=attn_drop,
            norm_layer=norm_layer,
            K=K,
            num_heads=num_heads,
            local_int=local_int,
            int_type=int_type,
            num_transformers=num_transf_local,
            feature_drop=feature_drop,
        )

        self.num_add = 0
        if self.conditional:
            self.cond_embed = nn.Sequential(
                MLP(
                    in_features=cond_dim,
                    hidden_features=base_dim,
                    out_features=base_dim,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
            )
            self.num_add += 1

        if self.use_time:
            # Time embedding module for diffusion timesteps
            self.MPFourier = MPFourier(base_dim)
            self.time_embed = MLP(
                in_features=base_dim,
                hidden_features=int(mlp_ratio * base_dim),
                out_features=base_dim,
                norm_layer=norm_layer,
                act_layer=act_layer,
                bias=False,
            )
            self.num_add += 1

        if self.add_info:
            self.add_embed = nn.Sequential(
                MLP(
                    in_features=add_dim,
                    hidden_features=int(mlp_ratio * base_dim),
                    out_features=base_dim,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    bias=False,
                ),
                NoScaleDropout(feature_drop),
            )

        if self.pid:
            # Will assume PIDs are just a list of integers and use the embedding layer, notice that zero_pid_idx is used to map zero-padded entries
            self.pid_embed = nn.Sequential(
                nn.Embedding(pid_dim, base_dim, padding_idx=0),
                NoScaleDropout(feature_drop),
            )

        self.num_tokens = num_tokens
        self.token = nn.Parameter(1e-3 * torch.ones(1, self.num_tokens, base_dim))

        self.num_heads = num_heads

        if self.skip:
            self.in_blocks = nn.ModuleList(
                [
                    AttBlock(
                        dim=base_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_drop=attn_drop,
                        mlp_drop=mlp_drop,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        num_tokens=num_tokens + self.num_add,
                        skip=False,
                        use_int=use_int,
                    )
                    for _ in range(num_transformers // 2)
                ]
            )

            self.out_blocks = nn.ModuleList(
                [
                    AttBlock(
                        dim=base_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_drop=attn_drop,
                        mlp_drop=mlp_drop,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        num_tokens=num_tokens + self.num_add,
                        skip=True,
                        use_int=use_int,
                    )
                    for _ in range(num_transformers // 2)
                ]
            )

        else:
            self.in_blocks = nn.ModuleList(
                [
                    AttBlock(
                        dim=base_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_drop=attn_drop,
                        mlp_drop=mlp_drop,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        num_tokens=num_tokens + self.num_add,
                        skip=False,
                        use_int=use_int,
                    )
                    for _ in range(num_transformers)
                ]
            )

        self.norm = norm_layer(base_dim)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.token, mean=0.0, std=0.02, a=-2.0, b=2.0)

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)

    def forward(self, x, cond=None, pid=None, add_info=None, time=None):
        B = x.shape[0]
        mask = x[:, :, 2:3] != 0
        token = self.token.expand(B, -1, -1)

        x_embed, x = self.embed(x, mask)

        # Move away zero-padded entries
        coord_shift = 999.0 * (~mask).float()
        local_features, indices = self.local_physics(
            coord_shift + x[:, :, : self.num_coord], x, mask
        )

        x_int = None
        if self.use_int:
            x_int = self.interaction(x, mask, indices)

        # Combine local + global info
        x = x_embed + local_features
        # Add classification tokens

        if pid is not None and self.pid:
            # Encode the PID info
            x = x + self.pid_embed(pid) * mask
        if add_info is not None and self.add_info:
            x = x + self.add_embed(add_info) * mask

        if cond is not None and self.conditional:
            # Conditional information: jet level quantities for example
            x = torch.cat([self.cond_embed(cond).unsqueeze(1), x], 1)

        if self.use_time and time is not None:
            # Create time token
            x = torch.cat([self.time_embed(self.MPFourier(time)).unsqueeze(1), x], 1)

        x = torch.cat([token, x], 1)

        # Create a new mask based on the updated point cloud with additional tokens
        mask = x[:, :, 2:3] != 0

        attn_mask = mask.float() @ mask.float().transpose(-1, -2)
        attn_mask = ~(attn_mask.bool()).repeat_interleave(self.num_heads, dim=0)
        attn_mask = attn_mask.float() * -1e9

        if self.use_int:
            # Add the information of the interaction matrix
            attn_mask[
                :, self.num_tokens + self.num_add :, self.num_tokens + self.num_add :
            ] = (
                x_int
                + attn_mask[
                    :,
                    self.num_tokens + self.num_add :,
                    self.num_tokens + self.num_add :,
                ]
            )

        skips = []
        for ib, blk in enumerate(self.in_blocks):
            x = blk(x, mask=mask, attn_mask=attn_mask)
            skips.append(x)
        if self.skip:
            for ib, blk in enumerate(self.out_blocks):
                x = blk(x, mask=mask, attn_mask=attn_mask, skip=skips.pop())

        x = self.norm(x) * mask
        return x


class DeepSetsInteraction(nn.Module):
    """One particle-particle message-passing layer for the DeepSets body.

    For every ordered pair (i, j) of constituents a shared edge MLP maps
    ``[h_i, h_j, e_ij]`` to a message, where ``e_ij`` is a 3-vector of
    physics edge features (log invariant mass, log ΔR, log k_T) -- the same
    quantities the PET2 teacher's InteractionBlock uses. Messages are
    mean-pooled over valid neighbours j; the caller adds the result back to
    h_i as a pre-norm residual. This is the EdgeConv / Interaction-Network
    style of graph layer: it makes the body permutation-equivariant while
    the downstream masked pool keeps the whole model permutation-invariant.

    Cost is O(N²) in the constituent count, so pair it with ``interaction_k``
    truncation on wide inputs.
    """

    def __init__(self, base_dim, mlp_ratio=2, mlp_drop=0.0, act_layer=nn.GELU):
        super().__init__()
        self.edge_mlp = MLP(
            2 * base_dim + 3,
            int(mlp_ratio * base_dim),
            out_features=base_dim,
            act_layer=act_layer,
            drop=mlp_drop,
        )

    def forward(self, h, edge_feats, edge_mask):
        # h: (B, N, D), pre-normed by caller
        # edge_feats: (B, N, N, 3); edge_mask: (B, N, N, 1)
        n = h.shape[1]
        hi = h.unsqueeze(2).expand(-1, -1, n, -1)
        hj = h.unsqueeze(1).expand(-1, n, -1, -1)
        msg = self.edge_mlp(torch.cat([hi, hj, edge_feats], dim=-1)) * edge_mask
        denom = edge_mask.sum(dim=2).clamp(min=1.0)  # (B, N, 1)
        return msg.sum(dim=2) / denom                 # (B, N, D)


class DeepSetsBody(nn.Module):
    """Per-particle φ MLP + masked mean pooling.

    With ``num_interaction_layers > 0`` a stack of :class:`DeepSetsInteraction`
    message-passing layers runs between the embedding and the φ blocks, adding
    particle-particle context. ``num_interaction_layers == 0`` (the default)
    builds no extra modules and is bit-for-bit the original Deep Sets body.
    """

    def __init__(
        self,
        input_dim,
        base_dim=128,
        num_layers=3,
        mlp_ratio=2,
        mlp_drop=0.0,
        pid=False,
        pid_dim=9,
        add_info=False,
        add_dim=4,
        conditional=False,
        cond_dim=3,
        norm_layer=DynamicTanh,
        act_layer=nn.GELU,
        energy_weighted_pool=False,
        num_interaction_layers=0,
        interaction_k=0,
        fixed_n=0,
    ):
        super().__init__()
        self.pid = pid
        self.add_info = add_info
        self.conditional = conditional
        self.energy_weighted_pool = energy_weighted_pool
        # 0 => plain Deep Sets; >0 => leading-pT truncation before the body
        # (constituents are stored pT-descending, so this is a plain slice).
        self.interaction_k = interaction_k if num_interaction_layers > 0 else 0
        # >0 => hls4ml-friendly body: fixed leading-pT N slots, no in-graph
        # validity mask, plain mean pool. Not compatible with the O(N^2)
        # message-passing path (which needs the mask), so ignore it there.
        self.fixed_n = int(fixed_n) if num_interaction_layers == 0 else 0

        self.embed = MLP(
            input_dim,
            int(mlp_ratio * base_dim),
            out_features=base_dim,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

        # pre-norm residual φ blocks
        self.phi_norms = nn.ModuleList(
            [norm_layer(base_dim) for _ in range(num_layers - 1)]
        )
        self.phi_blocks = nn.ModuleList(
            [
                MLP(
                    base_dim,
                    int(mlp_ratio * base_dim),
                    out_features=base_dim,
                    act_layer=act_layer,
                    drop=mlp_drop,
                )
                for _ in range(num_layers - 1)
            ]
        )

        # Optional particle-particle message passing before the φ blocks.
        # Only instantiated when requested, so state_dict keys (and existing
        # checkpoints) are unchanged when num_interaction_layers == 0.
        self.interaction_norms = nn.ModuleList(
            [norm_layer(base_dim) for _ in range(num_interaction_layers)]
        )
        self.interaction_blocks = nn.ModuleList(
            [
                DeepSetsInteraction(
                    base_dim,
                    mlp_ratio=mlp_ratio,
                    mlp_drop=mlp_drop,
                    act_layer=act_layer,
                )
                for _ in range(num_interaction_layers)
            ]
        )

        if pid:
            self.pid_embed = nn.Embedding(pid_dim, base_dim, padding_idx=0)

        if add_info:
            self.add_embed = MLP(
                add_dim,
                int(mlp_ratio * base_dim),
                out_features=base_dim,
                act_layer=act_layer,
            )

        if conditional:
            self.cond_embed = MLP(
                cond_dim,
                int(mlp_ratio * base_dim),
                out_features=base_dim,
                act_layer=act_layer,
            )

    def _forward_fixed_n(self, x, cond=None, pid=None, add_info=None):
        """hls4ml-friendly forward: assume the leading-pT ``fixed_n`` slots,
        drop the ``x[..., 2] != 0`` validity mask, and plain-mean-pool over
        all N slots. The model is trained this way, so it learns to tolerate
        the zero-padded rows folded into the mean. Removes the Equal/Not/Cast
        and masked-ReduceSum ops that block hls4ml ingestion; the mean is a
        GlobalAveragePool on export.
        """
        n = self.fixed_n
        if x.shape[1] > n:
            x = x[:, :n, :]
            if pid is not None:
                pid = pid[:, :n]
            if add_info is not None:
                add_info = add_info[:, :n, :]

        h = self.embed(x)  # (B, n, D)
        if pid is not None and self.pid:
            h = h + self.pid_embed(pid)
        if add_info is not None and self.add_info:
            h = h + self.add_embed(add_info)

        for norm, phi in zip(self.phi_norms, self.phi_blocks):
            h = h + phi(norm(h))  # pre-norm residual, no mask

        # plain mean over the fixed N slots, written as a 1-D adaptive pool so
        # ONNX export emits GlobalAveragePool (hls4ml-supported), not ReduceMean
        z = nn.functional.adaptive_avg_pool1d(h.transpose(1, 2), 1).flatten(1)

        if cond is not None and self.conditional:
            z = z + self.cond_embed(cond)
        return z

    def forward(self, x, cond=None, pid=None, add_info=None):
        if self.fixed_n:
            return self._forward_fixed_n(x, cond=cond, pid=pid, add_info=add_info)

        if self.interaction_k and x.shape[1] > self.interaction_k:
            # Keep the leading-pT K constituents. Inputs are stored
            # pT-descending, so this is a plain slice -- no sort needed.
            k = self.interaction_k
            x = x[:, :k, :]
            if pid is not None:
                pid = pid[:, :k]
            if add_info is not None:
                add_info = add_info[:, :k, :]

        mask = (x[:, :, 2:3] != 0).float()  # (B, N, 1)

        h = self.embed(x) * mask  # (B, N, D)

        if pid is not None and self.pid:
            h = h + self.pid_embed(pid) * mask
        if add_info is not None and self.add_info:
            h = h + self.add_embed(add_info) * mask

        if self.interaction_blocks:
            n = x.shape[1]
            xi = x.unsqueeze(2).expand(-1, -1, n, -1)
            xj = x.unsqueeze(1).expand(-1, n, -1, -1)
            edge_mask = (mask @ mask.transpose(-1, -2)).unsqueeze(-1)  # (B, N, N, 1)
            edge_feats = torch.cat(
                [
                    get_mass(xi, xj, edge_mask, is_log=True),
                    get_dr(xi, xj, edge_mask, is_log=True),
                    get_kt(xi, xj, edge_mask, is_log=True),
                ],
                dim=-1,
            )
            for norm, blk in zip(self.interaction_norms, self.interaction_blocks):
                h = h + blk(norm(h), edge_feats, edge_mask) * mask

        for norm, phi in zip(self.phi_norms, self.phi_blocks):
            h = h + phi(norm(h)) * mask  # pre-norm residual; padded stays 0

        if self.energy_weighted_pool:
            # weight each particle by its raw pT (feature index 2 is log(pT);
            # exponentiate then re-mask since exp(0) = 1, not 0)
            weight = torch.exp(x[:, :, 2:3]) * mask
            denom = weight.sum(dim=1).clamp(min=1e-8)  # (B, 1)
            z = (h * weight).sum(dim=1) / denom         # pT-weighted pool → (B, D)
        else:
            n_valid = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
            z = (h * mask).sum(dim=1) / n_valid        # masked mean pool → (B, D)

        if cond is not None and self.conditional:
            z = z + self.cond_embed(cond)

        return z


class DeepSetsHead(nn.Module):
    """Global ρ MLP + linear classifier output."""

    def __init__(
        self,
        base_dim=128,
        num_layers=2,
        mlp_ratio=2,
        mlp_drop=0.0,
        num_classes=2,
        norm_layer=DynamicTanh,
        act_layer=nn.GELU,
    ):
        super().__init__()

        self.rho_norms = nn.ModuleList(
            [norm_layer(base_dim) for _ in range(num_layers)]
        )
        self.rho = nn.ModuleList(
            [
                MLP(
                    base_dim,
                    int(mlp_ratio * base_dim),
                    out_features=base_dim,
                    act_layer=act_layer,
                    drop=mlp_drop,
                )
                for _ in range(num_layers)
            ]
        )
        self.out = nn.Linear(base_dim, num_classes)

    def forward(self, z):
        for norm, rho in zip(self.rho_norms, self.rho):
            z = z + rho(norm(z))
        return self.out(z)


class DeepSets(nn.Module):
    """
    Deep Sets classifier student for distillation from PET2.

    Drop-in replacement for PET2 in classifier/pretrain mode: same forward()
    output dict and same .body / .classifier / .generator attributes so
    save_checkpoint / restore_checkpoint work unchanged.
    """

    def __init__(
        self,
        input_dim,
        num_classes=2,
        base_dim=128,
        num_phi_layers=3,
        num_rho_layers=2,
        mlp_ratio=2,
        mlp_drop=0.0,
        mode="classifier",
        pid=False,
        pid_dim=9,
        add_info=False,
        add_dim=4,
        conditional=False,
        cond_dim=3,
        norm_layer=DynamicTanh,
        act_layer=nn.GELU,
        energy_weighted_pool=False,
        num_interaction_layers=0,
        interaction_k=0,
        fixed_n=0,
        rho_mlp_ratio=None,  # rho (head) blocks' mlp_ratio; None = mlp_ratio
    ):
        super().__init__()
        if mode not in ["classifier", "pretrain"]:
            raise ValueError(
                f"DeepSets supports classifier and pretrain modes, got '{mode}'"
            )
        self.mode = mode
        self.generator = None  # checkpoint compatibility with PET2 interface

        self.body = DeepSetsBody(
            input_dim=input_dim,
            base_dim=base_dim,
            num_layers=num_phi_layers,
            mlp_ratio=mlp_ratio,
            mlp_drop=mlp_drop,
            pid=pid,
            pid_dim=pid_dim,
            add_info=add_info,
            add_dim=add_dim,
            conditional=conditional,
            cond_dim=cond_dim,
            norm_layer=norm_layer,
            act_layer=act_layer,
            energy_weighted_pool=energy_weighted_pool,
            num_interaction_layers=num_interaction_layers,
            interaction_k=interaction_k,
            fixed_n=fixed_n,
        )

        self.classifier = DeepSetsHead(
            base_dim=base_dim,
            num_layers=num_rho_layers,
            mlp_ratio=rho_mlp_ratio or mlp_ratio,
            mlp_drop=mlp_drop,
            num_classes=num_classes,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

        self.initialize_weights()

    def initialize_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)

    def no_weight_decay(self):
        return {"norm"}

    def forward(self, x, y, cond=None, pid=None, add_info=None):
        z = self.body(x, cond=cond, pid=pid, add_info=add_info)  # (B, D)
        y_pred = self.classifier(z)                               # (B, num_classes)
        return {
            "y_pred": y_pred,
            "y_perturb": None,
            "z_pred": None,
            "v": None,
            "v_weight": None,
            "x_body": None,
            "z_body": None,
            "alpha": torch.ones(x.shape[0], device=x.device),
        }


class MLPStudentBody(nn.Module):
    """Masked mean-pool over raw per-particle features, then one hidden
    layer. Unlike DeepSetsBody, there is no per-particle embedding -- the
    mean is taken directly over the raw kinematic features."""

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()

    def forward(self, x, cond=None, pid=None, add_info=None):
        mask = (x[:, :, 2:3] != 0).float()  # (B, N, 1)
        n_valid = mask.sum(dim=1).clamp(min=1.0)
        pooled = (x * mask).sum(dim=1) / n_valid  # masked mean pool -> (B, input_dim)
        return self.act(self.fc1(pooled))


class MLPStudent(nn.Module):
    """
    Minimal MLP classifier student for distillation from PET2: masked
    mean-pool over raw features -> one hidden layer -> linear classifier.
    No per-particle embedding (unlike DeepSets/PET2) -- intended as a
    lower-bound baseline, not a competitive architecture.

    Drop-in replacement for PET2/DeepSets in classifier mode: same
    forward() output dict and same .body / .classifier / .generator
    attributes so save_checkpoint / restore_checkpoint work unchanged.
    """

    def __init__(
        self,
        input_dim,
        num_classes=2,
        hidden_dim=64,
        mode="classifier",
    ):
        super().__init__()
        if mode != "classifier":
            raise ValueError(f"MLPStudent supports classifier mode only, got '{mode}'")
        self.mode = mode
        self.generator = None  # checkpoint compatibility with PET2 interface

        self.body = MLPStudentBody(input_dim=input_dim, hidden_dim=hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.initialize_weights()

    def initialize_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)

    def no_weight_decay(self):
        return set()

    def forward(self, x, y, cond=None, pid=None, add_info=None):
        z = self.body(x, cond=cond, pid=pid, add_info=add_info)  # (B, D)
        y_pred = self.classifier(z)                               # (B, num_classes)
        return {
            "y_pred": y_pred,
            "y_perturb": None,
            "z_pred": None,
            "v": None,
            "v_weight": None,
            "x_body": None,
            "z_body": None,
            "alpha": torch.ones(x.shape[0], device=x.device),
        }


class MLPGEN(nn.Module):
    def __init__(
        self,
        input_dim,
        base_dim,
        mlp_ratio=2,
        norm_layer=DynamicTanh,
        act_layer=nn.GELU,
        num_layers=3,
        mlp_drop=0.0,
        conditional=False,
        cond_dim=1,
    ):
        super().__init__()
        self.embed = MLP(
            in_features=input_dim,
            hidden_features=int(mlp_ratio * base_dim),
            out_features=base_dim // 2,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

        self.cond = MLP(
            in_features=cond_dim,
            hidden_features=int(mlp_ratio * base_dim),
            out_features=base_dim // 2,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

        self.in_blocks = nn.ModuleList(
            [
                MLP(
                    in_features=base_dim,
                    hidden_features=int(mlp_ratio * base_dim),
                    out_features=base_dim,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for _ in range(num_layers)
            ]
        )

        # Time embedding module for diffusion timesteps
        self.MPFourier = MPFourier(base_dim)
        self.time_embed = MLP(
            in_features=base_dim,
            hidden_features=int(mlp_ratio * base_dim),
            out_features=base_dim // 2,
            act_layer=act_layer,
        )

        self.out = nn.Linear(base_dim, input_dim)

    def forward(self, z, time, cond=None):
        z = self.embed(z)
        z = z + self.cond(cond)
        z = torch.cat([self.time_embed(self.MPFourier(time)), z], 1)
        for ib, blk in enumerate(self.in_blocks):
            z = blk(z)
        z = self.out(z)
        return z

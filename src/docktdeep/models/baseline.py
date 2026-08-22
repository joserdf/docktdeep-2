import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torchmetrics.regression import MeanAbsoluteError

__all__ = ["Baseline"]

# ESM-2 model key -> embedding dim (matches tools/precompute_esm2.py).
ESM2_DIMS = {"esm2-650M": 1280, "esm2-150M": 640, "esm2-35M": 480, "esm2-8M": 320}
E_LIG_DIM = 768  # ChemBERTa (zinc-base)


class ConvGroup(torch.nn.Sequential):
    def __init__(self, in_c, out_c, kernel_size, **kwargs):
        super().__init__(
            torch.nn.Conv3d(
                in_c,
                out_c,
                kernel_size=kernel_size,
                padding="same",
                bias=False,
                **kwargs,
            ),
            torch.nn.BatchNorm3d(out_c),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool3d((2, 2, 2)),
        )


class ConvGroupDepthwise(torch.nn.Sequential):
    def __init__(self, in_c, out_c, kernel_size, **kwargs):
        super().__init__(
            self.depthwise_separable_conv(in_c, out_c, kernel_size, **kwargs),
            torch.nn.BatchNorm3d(out_c),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool3d((2, 2, 2)),
        )

    def depthwise_separable_conv(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 2,
        kernels_per_layer: int = 1,
    ):
        """3D depthwise conv layer."""
        conv3d = torch.nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels * kernels_per_layer,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        pointwise_conv = torch.nn.Conv3d(
            in_channels=in_channels * kernels_per_layer,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        return torch.nn.Sequential(conv3d, pointwise_conv)


class FCGroup(torch.nn.Sequential):
    def __init__(self, in_c, out_c, dropout_rate, **kwargs):
        super().__init__(
            torch.nn.Linear(in_c, out_c, bias=False),
            torch.nn.BatchNorm1d(1000),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout_rate),
        )


class Baseline(pl.LightningModule):
    def __init__(self, input_size: tuple[int], **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.loss_fn = torch.nn.MSELoss() # Try: L1Loss, SmoothL1Loss, HuberLoss
        self.mae = MeanAbsoluteError()
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.validation_logs = []

        use_esm2 = bool(self.hparams.get("use_esm2", False))
        use_chemberta = bool(self.hparams.get("use_chemberta", False))
        semi = bool(self.hparams.get("semi", False))
        no_cnn = bool(self.hparams.get("no_cnn", False))
        f_dim = int(self.hparams.get("f_dim", 512))
        emb_proj_dim = int(self.hparams.get("emb_proj_dim", 128))
        proj_dim = int(self.hparams.get("proj_dim", 128))

        # `no_cnn` ablates the whole convolutional branch: the latent `f` then
        # comes from the projected embeddings instead of the voxel grid, so the
        # head, the semi head and the dataloader all lose the structural input.
        self.no_cnn = no_cnn
        if no_cnn:
            if not (use_esm2 or use_chemberta):
                raise ValueError("--no-cnn requires --use-esm2 and/or --use-chemberta: "
                                 "without the CNN and without embeddings the model has no input.")
            self.conv_layers = None
            self.flatten = None
            self.f_proj = None
            base_dim = emb_proj_dim * (int(use_esm2) + int(use_chemberta))
        else:
            conv = ConvGroup if not self.hparams.depthwise_convs else ConvGroupDepthwise
            conv1 = conv(input_size[0], 64, 5)
            conv2 = conv(64, 128, 5)
            conv3 = conv(128, 256, 5)
            self.conv_layers = torch.nn.Sequential(conv1, conv2, conv3)

            polling = torch.nn.AdaptiveAvgPool3d((2, 2, 2))
            flatten = torch.nn.Flatten()
            self.flatten = (
                torch.nn.Sequential(polling, flatten)
                if self.hparams.adaptive_pooling
                else flatten
            )

            # compact deterministic latent `f`
            flat_dim = 256 * (2**3 if self.hparams.adaptive_pooling else 3**3)
            self.f_proj = torch.nn.Sequential(
                torch.nn.Linear(flat_dim, f_dim, bias=False),
                torch.nn.BatchNorm1d(f_dim),
                torch.nn.ReLU(inplace=True),
            )
            base_dim = f_dim

        # embedding conditioning projections (factors A / B)
        self.emb_proj_dim = emb_proj_dim
        self.proj_prot = None
        self.proj_lig = None
        if use_esm2:
            e_prot_dim = ESM2_DIMS.get(self.hparams.get("esm2_model", "esm2-650M"), 1280)
            self.proj_prot = torch.nn.Sequential(
                torch.nn.Linear(e_prot_dim, emb_proj_dim, bias=False),
                torch.nn.BatchNorm1d(emb_proj_dim),
                torch.nn.ReLU(inplace=True),
            )
        if use_chemberta:
            self.proj_lig = torch.nn.Sequential(
                torch.nn.Linear(E_LIG_DIM, emb_proj_dim, bias=False),
                torch.nn.BatchNorm1d(emb_proj_dim),
                torch.nn.ReLU(inplace=True),
            )

        # prediction head over f (+ conditioned embeddings)
        head_in = base_dim if no_cnn else (
            base_dim + (emb_proj_dim if use_esm2 else 0) + (emb_proj_dim if use_chemberta else 0))
        fc_units = list(self.hparams.num_fc_units)
        self.head = torch.nn.Sequential()
        prev = head_in
        for u in fc_units:
            self.head.append(torch.nn.Linear(prev, u, bias=False))
            self.head.append(torch.nn.BatchNorm1d(u))
            self.head.append(torch.nn.ReLU(inplace=True))
            self.head.append(torch.nn.Dropout(self.hparams.dropout))
            prev = u
        self.head.append(torch.nn.Linear(prev, 1))

        # projection head p(f) for the semi-supervised objective (factor C)
        self.proj_head = None
        self.proj_target = None
        if semi:
            self.proj_head = torch.nn.Sequential(
                torch.nn.Linear(base_dim, proj_dim, bias=False),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(0.1),  # stochasticity for R-Drop consistency
                torch.nn.Linear(proj_dim, proj_dim, bias=False),
            )
            target_in = 0
            if use_esm2:
                target_in += ESM2_DIMS.get(self.hparams.get("esm2_model", "esm2-650M"), 1280)
            if use_chemberta:
                target_in += E_LIG_DIM
            if target_in > 0:
                self.proj_target = torch.nn.Linear(target_in, proj_dim, bias=False)

        # affinity loss (Huber robust to experimental noise) + label smoothing
        if self.hparams.get("loss", "mse") == "huber":
            self.loss_fn = torch.nn.SmoothL1Loss(beta=self.hparams.get("huber_beta", 1.0))
        else:
            self.loss_fn = torch.nn.MSELoss()
        self.label_smoothing = float(self.hparams.get("label_smoothing", 0.0))
        self.lambda_semi = float(self.hparams.get("lambda_semi", 1.0))
        self.tau = float(self.hparams.get("semi_tau", 0.1))
        self.yaware = bool(self.hparams.get("yaware", False))
        self.yaware_sigma = float(self.hparams.get("yaware_sigma", 1.0))

    def forward(self, x, e_prot=None, e_lig=None):
        f = self.forward_base_f(x, e_prot, e_lig)
        if not self.no_cnn:  # sem CNN o `f` ja e a concatenacao das projecoes
            if self.proj_prot is not None and e_prot is not None:
                f = torch.cat([f, self.proj_prot(e_prot)], dim=1)
            if self.proj_lig is not None and e_lig is not None:
                f = torch.cat([f, self.proj_lig(e_lig)], dim=1)
        return self.head(f)

    @staticmethod
    def add_specific_args(parent_parser):
        """Add model specific arguments to the parser; accessible with self.hparams."""
        # fmt: off
        parser = parent_parser.add_argument_group("Model args")
        parser.add_argument("--optim", type=str, default="Adam")
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--beta2", type=float, default=0.999)
        parser.add_argument("--eps", type=float, default=1e-8)
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--wdecay", type=float, default=0.0)
        parser.add_argument("--num-fc-units", type=int, nargs="+", default=[1000], help="Number of neurons in each fc layer")
        # parser.add_argument("--num-fc-layers", type=int, default=1, help="Number of fc layers (not useful if `--fc-layers` is already specified; can be used to specify the number of fc layers in a hyperparameter search).")
        # parser.add_argument("--num-conv-layers", type=int, default=1, help="Number of conv layers.")
        # parser.add_argument("--num-kernels", type=int, nargs="+", default=[16], help="Number of kernels in each conv layer.")
        # parser.add_argument("--kernel-sizes", type=int, nargs="+", default=[3], help="Kernel size in each conv layer.")
        parser.add_argument("--depthwise-convs", action="store_true", help="Use depthwise separable convolutions.")
        parser.add_argument("--adaptive-pooling", action="store_true", help="Use adaptive pooling before flattening.")
        parser.add_argument("--f-dim", type=int, default=512, help="Dimension of the compact latent f.")
        parser.add_argument("--emb-proj-dim", type=int, default=128, help="Dimension of embedding conditioning projections.")
        parser.add_argument("--semi", action="store_true", default=False, help="Enable factor C: semi-supervised + regularizers.")
        parser.add_argument("--no-cnn", action="store_true", default=False, help="Ablate the 3D CNN branch: predict from the frozen embeddings alone (requires --use-esm2 and/or --use-chemberta).")
        parser.add_argument("--loss", type=str, default="mse", choices=["mse", "huber"], help="Affinity regression loss.")
        parser.add_argument("--huber-beta", type=float, default=1.0, help="Huber loss beta (if --loss huber).")
        parser.add_argument("--label-smoothing", type=float, default=0.0, help="Shrink regression targets toward batch mean.")
        parser.add_argument("--lambda-semi", type=float, default=1.0, help="Weight of L_semi in the total loss.")
        parser.add_argument("--proj-dim", type=int, default=128, help="Projection head p(f) output dim (factor C).")
        parser.add_argument("--semi-tau", type=float, default=0.1, help="Temperature for contrastive L_semi.")
        parser.add_argument("--yaware", action="store_true", default=False, help="Use y-aware InfoNCE (anchored on affinity) instead of embedding-anchored contrastive.")
        parser.add_argument("--yaware-sigma", type=float, default=1.0, help="Affinity distance scale (pKd) for y-aware positive weights.")
        # fmt: on
        return parent_parser

    def configure_optimizers(self):
        return getattr(torch.optim, self.hparams.optim)(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            eps=self.hparams.eps,
            weight_decay=self.hparams.wdecay,
        )

    def _yaware_infonce(self, p, y):
        """Soft InfoNCE anchored on the affinity labels themselves.

        Samples with close pKd act as soft positives; far ones as negatives. This
        shapes p(f) so that projection-space proximity mirrors affinity ordering.
        """
        B = p.shape[0]
        sim = p @ p.T / self.tau  # (B, B) embedding similarity
        d = torch.abs(y[:, None] - y[None, :])  # (B, B) affinity distance
        tgt = torch.exp(-d / self.yaware_sigma) * (1.0 - torch.eye(B, device=p.device))
        tgt = tgt / (tgt.sum(dim=1, keepdim=True) + 1e-8)  # normalize rows, self excluded
        log_softmax = torch.log_softmax(sim, dim=1)
        return -(tgt * log_softmax).sum(dim=1).mean()

    def _semi_loss(self, x, e_prot, e_lig, y):
        """L_semi (factor C): consistency (R-Drop) + contrastive (y-aware or embedding-anchored)."""
        self.train()  # enable dropout for stochastic passes
        p1 = F.normalize(self.proj_head(self.forward_base_f(x, e_prot, e_lig)), dim=1)
        p2 = F.normalize(self.proj_head(self.forward_base_f(x, e_prot, e_lig)), dim=1)
        loss = F.mse_loss(p1, p2)  # consistency: align p(f) under two stochastic passes

        if self.yaware:
            loss = loss + self._yaware_infonce(p1, y)
            return loss

        parts = []
        if self.proj_prot is not None and e_prot is not None:
            parts.append(e_prot)
        if self.proj_lig is not None and e_lig is not None:
            parts.append(e_lig)
        if self.proj_target is not None and parts:
            t = F.normalize(self.proj_target(torch.cat(parts, dim=1)), dim=1)
            sim = p1 @ t.T / self.tau
            labels = torch.arange(len(p1), device=p1.device)
            loss = loss + F.cross_entropy(sim, labels)  # InfoNCE anchoring p(f) to (e_prot,e_lig)
        return loss

    def forward_base_f(self, x, e_prot=None, e_lig=None):
        """Forward up to the base latent f (before embedding conditioning)."""
        if self.no_cnn:
            parts = []
            if self.proj_prot is not None and e_prot is not None:
                parts.append(self.proj_prot(e_prot))
            if self.proj_lig is not None and e_lig is not None:
                parts.append(self.proj_lig(e_lig))
            f = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        else:
            f = self.f_proj(self.flatten(self.conv_layers(x)))
        self.last_f = f
        return f

    def shared_step(self, batch, batch_idx, stage):
        if len(batch) == 4:
            x, e_prot, e_lig, y = batch
        else:
            x, y = batch
            e_prot = e_lig = None
        y_pred = self(x, e_prot, e_lig)

        log_params = {
            "on_step": False,
            "on_epoch": True,
            "prog_bar": True,
            "logger": True,
        }

        y_target = y.reshape(y.shape[0], 1)
        if self.label_smoothing > 0.0 and stage == "train":
            y_target = y_target * (1.0 - self.label_smoothing) + y_target.mean() * self.label_smoothing
        loss = self.loss_fn(y_pred, y_target)
        if self.proj_head is not None and stage == "train":
            semi = self._semi_loss(x, e_prot, e_lig, y)
            loss = loss + self.lambda_semi * semi
            self.log("train_semi", semi.detach(), **log_params)
        self.log(f"{stage}_loss", loss, **log_params)

        if stage == "train":
            pearsonr = torch.corrcoef(torch.stack((y_pred.squeeze(), y)))[0][1]
            self.log("train_pearsonr", pearsonr, **log_params)

        out = {
            f"{stage}_loss": loss,
            # for calculating metrics on validation and test set:
            "preds": y_pred,
            "labels": y,
        }

        return out

    def training_step(self, batch, batch_idx):
        out = self.shared_step(batch, batch_idx, stage="train")
        return out["train_loss"]

    def validation_step(self, batch, batch_idx):
        out = self.shared_step(batch, batch_idx, stage="val")
        self.validation_step_outputs.append(out)
        return out["val_loss"]

    def test_step(self, batch, batch_idx):
        out = self.shared_step(batch, batch_idx, stage="test")
        self.test_step_outputs.append(out)
        return out["test_loss"]

    def on_validation_epoch_end(self) -> None:
        out = self.validation_step_outputs
        preds = torch.cat([x["preds"] for x in out]).squeeze()
        labels = torch.cat([x["labels"] for x in out])

        pearsonr = torch.corrcoef(torch.stack((preds, labels)))[0][1]
        loss = torch.stack([x["val_loss"] for x in out]).mean()
        mae = self.mae(preds, labels)

        log = {"val_pearsonr": pearsonr, "val_loss": loss, "val_mae": mae}
        self.log_dict(log, prog_bar=True, logger=True)
        self.validation_logs.append(log)

        self.validation_step_outputs.clear()

    def on_train_end(self) -> None:
        best_pearsonr = max(self.validation_logs, key=lambda x: x["val_pearsonr"])
        best_loss = min(self.validation_logs, key=lambda x: x["val_loss"])
        best_mae = min(self.validation_logs, key=lambda x: x["val_mae"])

        self.logger.experiment.track(
            {
                "best_val_pearsonr": best_pearsonr["val_pearsonr"],
                "best_val_loss": best_loss["val_loss"],
                "best_val_mae": best_loss["val_mae"],
            },
            context={"subset": "val"},
        )

    def on_test_epoch_end(self) -> None:
        out = self.test_step_outputs
        preds = torch.cat([x["preds"] for x in out]).squeeze()
        labels = torch.cat([x["labels"] for x in out])

        self.log_dict(
            {
                "test_pearsonr": torch.corrcoef(torch.stack((preds, labels)))[0][1],
                "test_loss": torch.stack([x["test_loss"] for x in out]).mean(),
                "test_mae": self.mae(preds, labels),
            },
            prog_bar=True,
            logger=True,
        )

        self.test_step_outputs.clear()

import csv
import os

import numpy as np
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torchmetrics.functional.regression import (
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    spearman_corrcoef,
    symmetric_mean_absolute_percentage_error,
)
from torchmetrics.regression import MeanAbsoluteError

__all__ = ["Baseline"]

# ESM-2 model key -> embedding dim (matches tools/precompute_esm2.py).
ESM2_DIMS = {
    "esm2-650M": 1280,
    "esm2-3B": 2560,
    "esm2-15B": 5120,
    "esm2-150M": 640,
    "esm2-35M": 480,
    "esm2-8M": 320,
}
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
        self.train_step_outputs = []
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
        self.ifp_tau = float(self.hparams.get("ifp_tau", 0.3))
        # Anchor of the y-aware InfoNCE: how the soft-positive target is built.
        #   affinity  (arr05) tgt = exp(-|dy|/sigma)                    (pure yaware)
        #   gate  (N1)         tgt = exp(-|dy|/sigma) * [IFP_sim >= tau] (running now)
        #   ifp   (N2)         tgt = IFP_sim                             (pure IFP)
        #   hybrid(N3)         tgt = exp(-|dy|/sigma) * IFP_sim          (continuous weight)
        #   struct(10.5)       tgt = exp(-|dy|/sigma) * voxel_sim        (on-the-fly, no IFP)
        anchor = str(self.hparams.get("anchor_mode", "affinity"))
        if bool(self.hparams.get("ifp_aware", False)):
            anchor = "gate"  # alias for backwards compat with the running N1 grid
        self.anchor_mode = anchor
        self.lambda_aff = float(self.hparams.get("lambda_aff", 1.0))
        self.lambda_ifp = float(self.hparams.get("lambda_ifp", 1.0))
        self.lambda_prot = float(self.hparams.get("lambda_prot", 0.0))
        self.lambda_lig = float(self.hparams.get("lambda_lig", 0.0))
        self.auto_scale_loss = bool(self.hparams.get("auto_scale_loss", True))

        # factor C decomposition: independent similarity terms (ifp/aff/prot/lig).
        # When --sim-terms is non-empty this path replaces the single y-aware
        # InfoNCE; the matrices are COMMON attributes (not register_buffer) so
        # ~370 MB never enter a .ckpt, and row indices come from the datamodule.
        self.sim_terms = list(self.hparams.get("sim_terms", []))
        self.sim_lambda = float(self.hparams.get("sim_lambda", 0.025))
        self.sim_lambda_max = float(self.hparams.get("sim_lambda_max", 0.125))
        self.sim_kendall = bool(self.hparams.get("sim_kendall", False))
        self.sim_mat_dir = str(self.hparams.get("sim_mat_dir", ""))
        self.S_prot = None
        self.S_lig = None
        if self.sim_terms:
            if self.sim_kendall:
                raise NotImplementedError(
                    "--sim-kendall is the phase-4.3 fallback (learned Kendall uncertainty "
                    "weighting) and is not implemented yet; run the plain ablation first.")
            if self.yaware:
                raise ValueError("--sim-terms is incompatible with --yaware: the decomposed "
                                 "similarity terms replace the y-aware InfoNCE.")
            if bool(self.hparams.get("ifp_aware", False)):
                raise ValueError("--sim-terms is incompatible with --ifp-aware "
                                 "(alias that forces --anchor-mode gate).")
            if self.anchor_mode != "affinity":
                raise ValueError(
                    f"--sim-terms is incompatible with --anchor-mode '{self.anchor_mode}': "
                    "keep the default 'affinity' (the anchor is unused in the decomposed path).")
            if "ifp" in self.sim_terms and not self.hparams.get("ifp_path"):
                raise ValueError(
                    "--sim-terms ifp requires --ifp-path: without it every ifp slot is None and "
                    "L_ifp would be identically zero (a dead term) while the run looks healthy.")
            # non-dominance budget: lambda_semi (RDrop) + one lambda per term must
            # stay within the total block (D5: 5 x 0.025 = 0.125).
            total = self.lambda_semi + len(self.sim_terms) * self.sim_lambda
            if total > self.sim_lambda_max + 1e-9:
                raise ValueError(
                    f"similarity budget {total:.3f} (lambda_semi + |K|*sim_lambda) exceeds "
                    f"--sim-lambda-max {self.sim_lambda_max}; lower --sim-lambda or --lambda-semi.")
            self._load_sim_matrices()

    def _load_sim_matrices(self):
        """Load the precomputed similarity matrices as plain numpy arrays.

        Kept as common attributes (NOT register_buffer) so the ~370 MB never end
        up in a .ckpt. Only the (B, B) sub-block needed per batch is moved to the
        device (see _prot_target/_lig_target), never the whole matrix.
        """
        if "prot" in self.sim_terms:
            self.S_prot = np.load(os.path.join(self.sim_mat_dir, "S_prot.npz"))["S"]
        if "lig" in self.sim_terms:
            self.S_lig = np.load(os.path.join(self.sim_mat_dir, "S_lig.npz"))["S"]

    def forward(self, x, e_prot=None, e_lig=None):
        if e_prot is not None and isinstance(e_prot, torch.Tensor) and e_prot.device != self.device:
            e_prot = e_prot.to(self.device)
        if e_lig is not None and isinstance(e_lig, torch.Tensor) and e_lig.device != self.device:
            e_lig = e_lig.to(self.device)
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
        parser.add_argument("--preds-dir", type=str, default="preds", help="Directory for the per-complex test predictions CSV (id,y_true,y_pred).")
        parser.add_argument("--loss", type=str, default="mse", choices=["mse", "huber"], help="Affinity regression loss.")
        parser.add_argument("--huber-beta", type=float, default=1.0, help="Huber loss beta (if --loss huber).")
        parser.add_argument("--label-smoothing", type=float, default=0.0, help="Shrink regression targets toward batch mean.")
        parser.add_argument("--lambda-semi", type=float, default=1.0, help="Weight of L_semi in the total loss.")
        parser.add_argument("--proj-dim", type=int, default=128, help="Projection head p(f) output dim (factor C).")
        parser.add_argument("--semi-tau", type=float, default=0.1, help="Temperature for contrastive L_semi.")
        parser.add_argument("--yaware", action="store_true", default=False, help="Use y-aware InfoNCE (anchored on affinity) instead of embedding-anchored contrastive.")
        parser.add_argument("--yaware-sigma", type=float, default=1.0, help="Affinity distance scale (pKd) for y-aware positive weights.")
        parser.add_argument("--ifp-aware", action="store_true", default=False, help="Alias for --anchor-mode gate (backwards compat with arr09).")
        parser.add_argument("--ifp-tau", type=float, default=0.3, help="IFP Dice threshold for the IFP gate in the y-aware anchor.")
        parser.add_argument("--anchor-mode", type=str, default="affinity",
                            choices=["affinity", "gate", "ifp", "hybrid", "struct", "dual"],
                            help="Anchor of the y-aware InfoNCE (requires --yaware). affinity: tgt=exp(-|dy|/s) (arr05). gate (N1): x [IFP_sim>=tau]. ifp (N2): tgt=IFP_sim. hybrid (N3): x IFP_sim (continuous). struct (10.5): x on-the-fly voxel similarity. dual: separate L_aff and L_ifp terms with scale balancing.")
        parser.add_argument("--lambda-aff", type=float, default=1.0, help="Weight of affinity contrastive term in dual mode.")
        parser.add_argument("--lambda-ifp", type=float, default=1.0, help="Weight of IFP contrastive term in dual mode.")
        parser.add_argument("--lambda-prot", type=float, default=0.0, help="Weight of ESM-2 protein embedding cosine contrastive term.")
        parser.add_argument("--lambda-lig", type=float, default=0.0, help="Weight of ChemBERTa ligand embedding cosine contrastive term.")
        parser.add_argument("--auto-scale-loss", action="store_true", default=True, help="Equalize magnitudes of loss terms via scale balancing before applying weights.")
        parser.add_argument("--eval-test-per-epoch", action="store_true", default=False, help="Evaluate test set and log test_pearsonr, test_pearsonr_ood, test_pearsonr_casf at each epoch end.")
        parser.add_argument("--sim-terms", nargs="+", default=[],
                            choices=["ifp", "aff", "prot", "lig"],
                            help="Factor C decomposition: independent similarity terms, each an InfoNCE over the same projection p(f). Replaces the y-aware InfoNCE. Mutually exclusive with --yaware/--ifp-aware/non-default --anchor-mode.")
        parser.add_argument("--sim-lambda", type=float, default=0.025,
                            help="Weight of each active similarity term (lambda_0 in the proposal).")
        parser.add_argument("--sim-lambda-max", type=float, default=0.125,
                            help="Budget cap for the whole similarity block = lambda_semi + |K|*sim_lambda (D5: 5 x 0.025). With the default --lambda-semi 1.0 the budget always trips; set --lambda-semi 0.025 alongside --sim-terms.")
        parser.add_argument("--sim-mat-dir", type=str, default="",
                            help="Dir with S_prot.npz / S_lig.npz (loaded as common attributes, not buffers).")
        parser.add_argument("--sim-kendall", action="store_true", default=False,
                            help="Use learned Kendall uncertainty weighting (phase-4.3 fallback). NOT implemented yet.")
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

    def _ifp_sim(self, ifp):
        """Pairwise Dice similarity of binary interaction fingerprints (B, 4096) -> (B, B)."""
        b = ifp.float()
        inter = b @ b.T  # (B, B) shared bits
        pop = b.sum(dim=1)  # (B,)
        return 2.0 * inter / (pop[:, None] + pop[None, :] + 1e-8)

    def _voxel_sim(self, x):
        """On-the-fly structural similarity from the input voxel grids (10.5).

        Coarse per-sample 3D-occupancy descriptor via adaptive average pooling to
        (4,4,4); pairwise cosine similarity. Cheap and does not depend on the
        precomputed IFP (the pre-IFP alternative of section 10.5).
        """
        B = x.shape[0]
        desc = F.adaptive_avg_pool3d(x, (4, 4, 4))  # (B, C, 4, 4, 4)
        desc = desc.reshape(B, -1)  # (B, C*64)
        desc = F.normalize(desc, dim=1)
        return desc @ desc.T  # (B, B)

    def _sim_infonce(self, p, tgt):
        """InfoNCE against a fixed soft target (factor C decomposition).

        ``tgt`` is row-normalized; rows with no positive partner (rowsum ~ 0)
        stay at zero and contribute no gradient, so the loss only shapes ranking
        *within* each sample's selected context.
        """
        sim = p @ p.T / self.tau  # (B, B) embedding similarity
        # The target is a fixed weighting (labels / IFP / similarity) — detach so
        # gradient only flows through the projection similarity sim(p,p), never
        # through the raw inputs.
        tgt = tgt.detach()
        rowsum = tgt.sum(dim=1, keepdim=True)
        tgt = torch.where(rowsum > 1e-8, tgt / (rowsum + 1e-8), torch.zeros_like(tgt))
        log_softmax = torch.log_softmax(sim, dim=1)
        return -(tgt * log_softmax).sum(dim=1).mean()

    def _yaware_infonce(self, p, y, ifp=None, x=None, e_prot=None, e_lig=None, reg_loss=None):
        """Soft InfoNCE anchored on affinity, IFP, structure, or embedding similarities.

        Shapes p(f) so that projection-space proximity mirrors target similarity ordering.
        """
        B = p.shape[0]
        eye = torch.eye(B, device=p.device)
        d = torch.abs(y[:, None] - y[None, :])  # (B, B) affinity distance
        aff = torch.exp(-d / self.yaware_sigma) * (1.0 - eye)
        needs_ifp = self.anchor_mode in ("gate", "ifp", "hybrid", "dual")

        if needs_ifp and ifp is not None:
            ifp_sim = self._ifp_sim(ifp) * (1.0 - eye)
        if self.anchor_mode == "dual" and ifp is not None:
            l_aff = self._sim_infonce(p, aff)
            l_ifp = self._sim_infonce(p, ifp_sim)
            if self.auto_scale_loss and reg_loss is not None:
                reg_scale = reg_loss.detach() + 1e-8
                l_aff_s = l_aff * (reg_scale / (l_aff.detach() + 1e-8))
                l_ifp_s = l_ifp * (reg_scale / (l_ifp.detach() + 1e-8))
            elif self.auto_scale_loss:
                l_aff_s = l_aff / (l_aff.detach() + 1e-8)
                l_ifp_s = l_ifp / (l_ifp.detach() + 1e-8)
            else:
                l_aff_s, l_ifp_s = l_aff, l_ifp
            base_loss = self.lambda_aff * l_aff_s + self.lambda_ifp * l_ifp_s
        elif self.anchor_mode == "affinity" or (needs_ifp and ifp is None):
            tgt = aff
            base_loss = self._sim_infonce(p, tgt)
        elif self.anchor_mode == "gate":
            gate = (self._ifp_sim(ifp) >= self.ifp_tau).float() * (1.0 - eye)
            tgt = aff * gate
            base_loss = self._sim_infonce(p, tgt)
        elif self.anchor_mode == "ifp":
            tgt = ifp_sim
            base_loss = self._sim_infonce(p, tgt)
        elif self.anchor_mode == "hybrid":
            tgt = aff * ifp_sim
            base_loss = self._sim_infonce(p, tgt)
        elif self.anchor_mode == "struct":
            vsim = self._voxel_sim(x) * (1.0 - eye) if x is not None else None
            tgt = aff * vsim if vsim is not None else aff
            base_loss = self._sim_infonce(p, tgt)
        else:
            raise ValueError(f"unknown anchor_mode: {self.anchor_mode}")

        total_loss = base_loss

        # Optional ESM-2 protein embedding cosine similarity term
        if self.lambda_prot > 0.0 and e_prot is not None:
            e_prot_norm = F.normalize(e_prot, dim=1)
            tgt_prot = torch.relu(e_prot_norm @ e_prot_norm.T) * (1.0 - eye)
            l_prot = self._sim_infonce(p, tgt_prot)
            if self.auto_scale_loss and reg_loss is not None:
                reg_scale = reg_loss.detach() + 1e-8
                l_prot_s = l_prot * (reg_scale / (l_prot.detach() + 1e-8))
            elif self.auto_scale_loss:
                l_prot_s = l_prot / (l_prot.detach() + 1e-8)
            else:
                l_prot_s = l_prot
            total_loss = total_loss + self.lambda_prot * l_prot_s

        # Optional ChemBERTa ligand embedding cosine similarity term
        if self.lambda_lig > 0.0 and e_lig is not None:
            e_lig_norm = F.normalize(e_lig, dim=1)
            tgt_lig = torch.relu(e_lig_norm @ e_lig_norm.T) * (1.0 - eye)
            l_lig = self._sim_infonce(p, tgt_lig)
            if self.auto_scale_loss and reg_loss is not None:
                reg_scale = reg_loss.detach() + 1e-8
                l_lig_s = l_lig * (reg_scale / (l_lig.detach() + 1e-8))
            elif self.auto_scale_loss:
                l_lig_s = l_lig / (l_lig.detach() + 1e-8)
            else:
                l_lig_s = l_lig
            total_loss = total_loss + self.lambda_lig * l_lig_s

        return total_loss

    # --- target builders for the similarity-term decomposition (factor C) ----
    def _aff_target(self, y):
        """Soft-positive target from affinity proximity: exp(-|dy| / sigma), off-diagonal."""
        B = y.shape[0]
        d = torch.abs(y[:, None] - y[None, :])
        return torch.exp(-d / self.yaware_sigma) * (1.0 - torch.eye(B, device=y.device))

    def _ifp_target(self, ifp):
        """Soft-positive target from PLEC IFP Dice, off-diagonal."""
        B = ifp.shape[0]
        return self._ifp_sim(ifp) * (1.0 - torch.eye(B, device=ifp.device))

    def _prot_target(self, prot_idx):
        """Soft-positive target from precomputed PSI (S_prot), off-diagonal.

        Only the (B, B) sub-block is gathered on CPU and moved to the device; the
        full matrix stays a numpy array. Values are PSI/100 in [0, 1]; the all-zero
        sentinel row yields a zero target row (no gradient).
        """
        pidx = prot_idx.detach().cpu().numpy()
        sub = self.S_prot[np.ix_(pidx, pidx)].astype(np.float32) / 100.0
        tgt = torch.as_tensor(sub, device=prot_idx.device)
        return tgt * (1.0 - torch.eye(pidx.shape[0], device=prot_idx.device))

    def _lig_target(self, lig_idx):
        """Soft-positive target from precomputed Morgan Tanimoto (S_lig), off-diagonal."""
        lidx = lig_idx.detach().cpu().numpy()
        sub = self.S_lig[np.ix_(lidx, lidx)].astype(np.float32) / 100.0
        tgt = torch.as_tensor(sub, device=lig_idx.device)
        return tgt * (1.0 - torch.eye(lidx.shape[0], device=lig_idx.device))

    def _sim_terms_loss(self, p, prot_idx, lig_idx, ifp, y):
        """Weighted sum of the active similarity terms over the shared projection p.

        Returns ``(weighted_total, per_term)`` where ``per_term[k] = (L_k, row_frac)``
        with ``L_k`` the unweighted loss and ``row_frac`` the fraction of batch rows
        that have at least one positive partner for that term.
        """
        total = torch.zeros((), device=p.device)
        per_term = {}
        for k in self.sim_terms:
            if k == "ifp":
                tgt = self._ifp_target(ifp)
            elif k == "aff":
                tgt = self._aff_target(y)
            elif k == "prot":
                tgt = self._prot_target(prot_idx)
            elif k == "lig":
                tgt = self._lig_target(lig_idx)
            else:
                raise ValueError(f"unknown sim term: {k}")
            row_frac = (tgt.sum(dim=1) > 1e-8).float().mean()
            Lk = self._sim_infonce(p, tgt)
            per_term[k] = (Lk, row_frac)
            total = total + Lk
        return self.sim_lambda * total, per_term

    def _semi_loss(self, x, e_prot, e_lig, y, ifp=None, prot_idx=None, lig_idx=None, reg_loss=None):
        """L_semi (factor C): consistency (R-Drop) + contrastive (y-aware or embedding-anchored).

        With --sim-terms active it returns ``(rdrop, sim_block)``; otherwise a
        scalar ``rdrop [+ contrastive]`` (legacy paths untouched).
        """
        self.train()  # enable dropout for stochastic passes
        p1 = F.normalize(self.proj_head(self.forward_base_f(x, e_prot, e_lig)), dim=1)
        p2 = F.normalize(self.proj_head(self.forward_base_f(x, e_prot, e_lig)), dim=1)
        rdrop = F.mse_loss(p1, p2)  # consistency: align p(f) under two stochastic passes

        if self.sim_terms:
            return rdrop, self._sim_terms_loss(p1, prot_idx, lig_idx, ifp, y)

        loss = rdrop
        if self.yaware:
            loss = loss + self._yaware_infonce(p1, y, ifp, x=x, e_prot=e_prot, e_lig=e_lig, reg_loss=reg_loss)
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
        if e_prot is not None and isinstance(e_prot, torch.Tensor) and e_prot.device != self.device:
            e_prot = e_prot.to(self.device)
        if e_lig is not None and isinstance(e_lig, torch.Tensor) and e_lig.device != self.device:
            e_lig = e_lig.to(self.device)
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
        if len(batch) == 7:
            # similarity-term ablation: (voxs, e_prot, e_lig, ifp, prot_idx, lig_idx, y)
            x, e_prot, e_lig, ifp, prot_idx, lig_idx, y = batch
        elif len(batch) == 5:
            x, e_prot, e_lig, ifp, y = batch
            prot_idx = lig_idx = None
        elif len(batch) == 4:
            x, e_prot, e_lig, y = batch
            ifp = prot_idx = lig_idx = None
        else:
            x, y = batch
            e_prot = e_lig = ifp = prot_idx = lig_idx = None
        if e_prot is not None and isinstance(e_prot, torch.Tensor) and e_prot.device != self.device:
            e_prot = e_prot.to(self.device)
        if e_lig is not None and isinstance(e_lig, torch.Tensor) and e_lig.device != self.device:
            e_lig = e_lig.to(self.device)
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
            if self.sim_terms:
                rdrop, (sim_block, per_term) = self._semi_loss(
                    x, e_prot, e_lig, y, ifp, prot_idx, lig_idx, reg_loss=loss)
                loss = loss + self.lambda_semi * rdrop + sim_block
                self.log("train_semi", rdrop.detach(), **log_params)
                for k, (Lk, row_frac) in per_term.items():
                    self.log(f"train_sim_{k}", Lk.detach(), **log_params)
                    self.log(f"train_sim_{k}_rows", row_frac.detach(), **log_params)
            else:
                semi = self._semi_loss(x, e_prot, e_lig, y, ifp, reg_loss=loss)
                loss = loss + self.lambda_semi * semi
                self.log("train_semi", semi.detach(), **log_params)
        self.log(f"{stage}_loss", loss, **log_params)

        # as metricas de treino saem em on_train_epoch_end, sobre a epoca inteira:
        # a media de correlacoes por batch de 64 pontos nao e o mesmo estimador

        out = {
            f"{stage}_loss": loss,
            # for calculating metrics on validation and test set:
            "preds": y_pred,
            "labels": y,
        }

        return out

    def training_step(self, batch, batch_idx):
        out = self.shared_step(batch, batch_idx, stage="train")
        # detach obrigatorio: guardar o tensor com grafo manteria o backward da
        # epoca inteira vivo na memoria
        self.train_step_outputs.append(
            {"preds": out["preds"].detach(), "labels": out["labels"].detach()}
        )
        return out["train_loss"]

    def validation_step(self, batch, batch_idx):
        out = self.shared_step(batch, batch_idx, stage="val")
        self.validation_step_outputs.append(out)
        return out["val_loss"]

    def test_step(self, batch, batch_idx):
        out = self.shared_step(batch, batch_idx, stage="test")
        self.test_step_outputs.append(out)
        return out["test_loss"]

    def on_train_epoch_end(self) -> None:
        out = self.train_step_outputs
        if not out:
            return
        preds = torch.cat([x["preds"] for x in out]).squeeze()
        labels = torch.cat([x["labels"] for x in out])

        # train_loss ja e agregado pelo self.log do shared_step; repetir a chave
        # aqui levantaria conflito de configuracao no Lightning
        self.log_dict(self._regression_metrics(preds, labels, "train"), logger=True)

        self.train_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        out = self.validation_step_outputs
        preds = torch.cat([x["preds"] for x in out]).squeeze()
        labels = torch.cat([x["labels"] for x in out])

        log = self._regression_metrics(preds, labels, "val")
        log["val_loss"] = torch.stack([x["val_loss"] for x in out]).mean()

        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None and getattr(dm, "val_strata", None) is not None:
            val_strata = np.array(dm.val_strata)
            if len(val_strata) == preds.numel():
                for st in np.unique(val_strata):
                    mask = (val_strata == st)
                    if mask.sum() >= 2:
                        sub_p = preds[mask]
                        sub_l = labels[mask]
                        sub_m = self._regression_metrics(sub_p, sub_l, f"val_{st}")
                        log.update(sub_m)

        if bool(self.hparams.get("eval_test_per_epoch", False)) and dm is not None and hasattr(dm, "test_dataloader"):
            was_training = self.training
            self.eval()
            test_loader = dm.test_dataloader()
            t_preds, t_labels = [], []
            with torch.no_grad():
                for batch in test_loader:
                    # loop manual: o Lightning so move o batch p/ a GPU nos
                    # *_step que ele mesmo chama, entao aqui e por nossa conta
                    batch = self.transfer_batch_to_device(batch, self.device, 0)
                    out = self.shared_step(batch, 0, stage="test")
                    t_preds.append(out["preds"].detach().cpu())
                    t_labels.append(out["labels"].detach().cpu())
            if t_preds:
                t_preds = torch.cat(t_preds).squeeze()
                t_labels = torch.cat(t_labels)
                t_metrics = self._regression_metrics(t_preds, t_labels, "test")
                log.update(t_metrics)
                if getattr(dm, "test_strata", None) is not None:
                    test_strata = np.array(dm.test_strata)
                    if len(test_strata) == t_preds.numel():
                        for st in np.unique(test_strata):
                            mask = (test_strata == st)
                            if mask.sum() >= 2:
                                sub_p = t_preds[mask]
                                sub_l = t_labels[mask]
                                sub_m = self._regression_metrics(sub_p, sub_l, f"test_{st}")
                                log.update(sub_m)
            # sem isto o modelo seguiria em eval() no resto do treino (dropout
            # e batchnorm desligados) — fatal numa busca que tuna dropout
            self.train(was_training)

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

        log = self._regression_metrics(preds, labels, "test")
        log["test_loss"] = torch.stack([x["test_loss"] for x in out]).mean()
        self.log_dict(log, prog_bar=True, logger=True)

        # o dump nunca pode derrubar um treino que ja terminou: sem ele o run
        # perde o CSV, com a excecao propagando perde tambem a linha de metricas
        try:
            self._dump_predictions(preds, labels)
        except Exception as e:  # noqa: BLE001
            print(f"[preds] falha ao gravar o CSV: {e}", flush=True)

        self.test_step_outputs.clear()

    def _regression_metrics(self, preds, labels, stage: str) -> dict:
        """Metricas de regressao sobre o fold inteiro, para validacao e teste.

        R2 nao e r²: `1 - SS_res/SS_tot` so coincide com o quadrado do Pearson sob
        ajuste linear por minimos quadrados. Aqui os dois divergem, e a divergencia
        e informativa (R2 pune vies de escala, r nao), entao os dois sao reportados.
        """
        mse = mean_squared_error(preds, labels)
        return {
            f"{stage}_pearsonr": torch.corrcoef(torch.stack((preds, labels)))[0][1],
            f"{stage}_mae": self.mae(preds, labels),
            f"{stage}_mse": mse,
            f"{stage}_rmse": torch.sqrt(mse),
            # MAPE/sMAPE dividem por |y|, que nao tem significado em pKi (grandeza
            # intervalar); o epsilon do torchmetrics (1.17e-6) so evita o ZeroDivision.
            f"{stage}_mape": mean_absolute_percentage_error(preds, labels),
            f"{stage}_smape": symmetric_mean_absolute_percentage_error(preds, labels),
            f"{stage}_r2": r2_score(preds, labels),
            f"{stage}_spearman": spearman_corrcoef(preds, labels),
        }

    def _dump_predictions(self, preds, labels) -> None:
        """Grava id,y_true,y_pred do fold de teste antes do descarte das saidas.

        Sem os ids nao da para separar os estratos ood/casf, nem refazer analise
        alguma sem re-treinar.
        """
        dataset = getattr(getattr(self, "trainer", None), "datamodule", None)
        ids = getattr(getattr(dataset, "test_dataset", None), "ids", None)
        if ids is None:
            print("[preds] test_dataset sem ids: dump de predicoes ignorado", flush=True)
            return
        # unico anteparo contra um join silenciosamente errado
        if len(ids) != preds.numel():
            print(f"[preds] {len(ids)} ids para {preds.numel()} predicoes: dump ignorado "
                  "para nao gravar um CSV desalinhado", flush=True)
            return

        preds_dir = self.hparams.get("preds_dir") or "preds"
        experiment = self.hparams.get("experiment") or "unknown"
        split_column = self.hparams.get("split_column") or "unknown"
        seed = self.hparams.get("seed", 0)
        os.makedirs(preds_dir, exist_ok=True)
        path = os.path.join(preds_dir, f"{experiment}__{split_column}__seed{seed}.csv")

        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id", "y_true", "y_pred"])
            writer.writerows(
                zip(ids, labels.detach().cpu().tolist(), preds.detach().cpu().tolist())
            )

        # contrato de stdout com worker/agent.py::_parse_preds_path
        print(f"[preds] {os.path.abspath(path)} ({len(ids)} linhas)", flush=True)

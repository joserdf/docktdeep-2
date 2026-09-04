import csv
import os

import numpy as np
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torchmetrics.functional.regression import (
    mean_squared_error,
    r2_score,
    spearman_corrcoef,
)
from torchmetrics.regression import MeanAbsoluteError

from . import losses
from .cnn import build_voxel_encoder
from .embeddings import E_LIG_DIM, build_embedding_projection, esm2_dim
from .heads import build_projection_head, build_regression_head
from .latent import build_latent_projection

__all__ = ["Baseline"]

# Piso de amostras para reportar metrica de um estrato. Pearson sobre 2 pontos e
# +-1 por construcao, entao um piso baixo enche o log de correlacoes perfeitas
# que nao dizem nada. Cada estrato tambem publica o proprio n, para que o piso
# nunca precise ser adivinhado a partir da tabela.
MIN_STRATUM_N = 10


class Baseline(pl.LightningModule):
    """Regressor de afinidade sobre grade de voxels + embeddings congelados.

    A arquitetura e montada por construtores externos, e cada um deles vira um
    atributo de primeiro nivel:

        conv_layers / flatten  (cnn.py)         grade 3D -> features
        proj_prot / proj_lig   (embeddings.py)  ESM-2 / ChemBERTa condicionados
        latent_proj            (latent.py)      concat `u` -> latente `z`
        head                   (heads.py)       cabeca de regressao sobre `z`
        proj_head              (heads.py)       p(z), objetivo do fator C

    O caminho e sempre o mesmo:

        u = concat(flatten(conv(x)), proj_prot(e_p), proj_lig(e_l))
        z = latent_proj(u)        # ou z = u, com --latent-dim 0
        y = head(z)               # contrastivo em p(z)

    Os nomes desses atributos sao as chaves do `state_dict`: renomea-los ou
    aninha-los num submodulo invalida todos os checkpoints ja treinados.
    """

    def __init__(self, input_size: tuple[int], **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.mae = MeanAbsoluteError()
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.validation_logs = []

        self._build_architecture(input_size)
        self._configure_objective()

    # ----------------------------------------------------------------- #
    # construcao
    # ----------------------------------------------------------------- #
    def _build_architecture(self, input_size: tuple[int]) -> None:
        hp = self.hparams
        use_esm2 = bool(hp.get("use_esm2", False))
        use_chemberta = bool(hp.get("use_chemberta", False))
        self.use_esm2 = use_esm2
        self.use_chemberta = use_chemberta

        # `no_cnn` ablates the whole convolutional branch: `u` then comes from
        # the projected embeddings instead of the voxel grid, so the head, the
        # semi head and the dataloader all lose the structural input.
        self.no_cnn = bool(hp.get("no_cnn", False))

        if self.no_cnn:
            if not (use_esm2 or use_chemberta):
                raise ValueError("--no-cnn requires --use-esm2 and/or --use-chemberta: "
                                 "without the CNN and without embeddings the model has no input.")
            self.conv_layers = None
            self.flatten = None
            cnn_dim = 0
        else:
            self.conv_layers, self.flatten, cnn_dim = build_voxel_encoder(
                in_channels=input_size[0],
                depthwise_convs=hp.depthwise_convs,
                adaptive_pooling=hp.adaptive_pooling,
            )

        # embedding conditioning projections (factors A / B)
        #
        # Cada ramo tem a sua largura. Uma largura unica forcava ESM-2 (1280 ou
        # 2560) e ChemBERTa (768) ao mesmo destino, e com o default de 128 isso
        # era 10x a 20x de compressao contra os 6912 dims que a CNN despeja em
        # `u` -- os embeddings ficavam com 1.8% a 3.6% da concatenacao.
        #
        # Largura 0 remove a projecao: o embedding congelado entra cru em `u`.
        # Sem a BatchNorm do `linear_bn_relu` ele chega numa escala que nao e a
        # da saida da CNN (pos-BN, pos-ReLU); e uma opcao de busca legitima, nao
        # um default seguro.
        e_prot_dim = esm2_dim(hp.get("esm2_model", "esm2-650M"))
        d_prot = self._branch_proj_dim(hp, "prot")
        d_lig = self._branch_proj_dim(hp, "lig")
        self.emb_proj_dim_prot = d_prot
        self.emb_proj_dim_lig = d_lig
        self.proj_prot = (
            build_embedding_projection(e_prot_dim, d_prot)
            if use_esm2 and d_prot > 0 else None
        )
        self.proj_lig = (
            build_embedding_projection(E_LIG_DIM, d_lig)
            if use_chemberta and d_lig > 0 else None
        )

        # `u`: concatenacao dos ramos ligados. `z`: o espaco latente onde o
        # contrastivo age e de onde a cabeca de regressao le. Com --latent-dim 0
        # nao ha gargalo e z == u, o que sem embeddings reproduz o upstream.
        self.u_dim = (
            cnn_dim
            + (0 if not use_esm2 else (d_prot or e_prot_dim))
            + (0 if not use_chemberta else (d_lig or E_LIG_DIM))
        )
        latent_dim = int(hp.get("latent_dim", 512))
        self.latent_proj = (
            build_latent_projection(self.u_dim, latent_dim) if latent_dim > 0 else None
        )
        self.z_dim = latent_dim if latent_dim > 0 else self.u_dim

        self.head = build_regression_head(
            self.z_dim, list(hp.num_fc_units), hp.dropout
        )

        # projection head p(z) for the semi-supervised objective (factor C)
        #
        # Quais embeddings o contrastivo enxerga e uma pergunta separada de quais
        # ramos o modelo consome. Um embedding pode ancorar `p(z)` sem nunca
        # entrar em `u`: o termo de cosseno so precisa do vetor congelado. Por
        # isso `proj_target` e dimensionado pelos lambdas, e nao por
        # use_esm2/use_chemberta -- o dataloader aplica exatamente a mesma regra
        # (PDBbind.need_e_prot / need_e_lig), entao as larguras casam.
        semi = bool(hp.get("semi", False))
        self.contrastive_prot = use_esm2 or (
            semi and float(hp.get("lambda_prot", 0.0)) > 0.0
        )
        self.contrastive_lig = use_chemberta or (
            semi and float(hp.get("lambda_lig", 0.0)) > 0.0
        )
        # Com o ramo ligado a ancora reusa o e_prot dele; so com o ramo desligado
        # a escolha do ESM-2 e livre (o dataloader rejeita as duas de uma vez).
        anchor_esm2 = (
            hp.get("esm2_model", "esm2-650M") if use_esm2
            else (hp.get("contrastive_esm2_model", "") or hp.get("esm2_model", "esm2-650M"))
        )
        self.proj_head = None
        self.proj_target = None
        if semi:
            proj_dim = int(hp.get("proj_dim", 128))
            self.proj_head = build_projection_head(self.z_dim, proj_dim)
            target_in = (esm2_dim(anchor_esm2) if self.contrastive_prot else 0) + (
                E_LIG_DIM if self.contrastive_lig else 0
            )
            if target_in > 0:
                self.proj_target = torch.nn.Linear(target_in, proj_dim, bias=False)

    @staticmethod
    def _branch_proj_dim(hp, sufixo: str) -> int:
        """Largura da projecao de um ramo. -1 herda o `--emb-proj-dim` comum.

        O sentinela existe para separar "nao pedi nada" de "pedi 0": 0 e uma
        escolha (sem projecao), e o default compartilhado de 128 e o que todos
        os trials ja rodados usaram.
        """
        v = int(hp.get(f"emb_proj_dim_{sufixo}", -1))
        return int(hp.get("emb_proj_dim", 128)) if v < 0 else v

    def _configure_objective(self) -> None:
        """Perda de afinidade, pesos dos termos e validacao da configuracao."""
        hp = self.hparams

        # affinity loss (Huber robust to experimental noise) + label smoothing
        if hp.get("loss", "mse") == "huber":
            self.loss_fn = torch.nn.SmoothL1Loss(beta=hp.get("huber_beta", 1.0))
        else:
            self.loss_fn = torch.nn.MSELoss()
        self.label_smoothing = float(hp.get("label_smoothing", 0.0))
        self.lambda_semi = float(hp.get("lambda_semi", 1.0))
        # Peso proprio do R-Drop. None mantem o acoplamento historico (a
        # consistencia e escalada junto com o bloco contrastivo por
        # --lambda-semi); um valor explicito a separa, e 0.0 a desliga sem
        # mexer nos termos contrastivos.
        lam_rdrop = hp.get("lambda_rdrop", None)
        self.lambda_rdrop = None if lam_rdrop is None else float(lam_rdrop)
        # Ruido no latente `z`, aplicado so nas passagens contrastivas: e o que
        # faz a consistencia do R-Drop restringir a representacao, e nao apenas
        # as mascaras internas do proj_head.
        self.rdrop_dropout = float(hp.get("rdrop_dropout", 0.0))
        self.tau = float(hp.get("semi_tau", 0.1))
        self.yaware = bool(hp.get("yaware", False))
        self.yaware_sigma = float(hp.get("yaware_sigma", 1.0))
        self.ifp_tau = float(hp.get("ifp_tau", 0.3))
        # Anchor of the y-aware InfoNCE: how the soft-positive target is built.
        #   affinity  (arr05) tgt = exp(-|dy|/sigma)                    (pure yaware)
        #   gate  (N1)         tgt = exp(-|dy|/sigma) * [IFP_sim >= tau] (running now)
        #   ifp   (N2)         tgt = IFP_sim                             (pure IFP)
        #   hybrid(N3)         tgt = exp(-|dy|/sigma) * IFP_sim          (continuous weight)
        #   struct(10.5)       tgt = exp(-|dy|/sigma) * voxel_sim        (on-the-fly, no IFP)
        anchor = str(hp.get("anchor_mode", "affinity"))
        if bool(hp.get("ifp_aware", False)):
            anchor = "gate"  # alias for backwards compat with the running N1 grid
        self.anchor_mode = anchor
        self.lambda_aff = float(hp.get("lambda_aff", 1.0))
        self.lambda_ifp = float(hp.get("lambda_ifp", 1.0))
        self.lambda_prot = float(hp.get("lambda_prot", 0.0))
        self.lambda_lig = float(hp.get("lambda_lig", 0.0))
        self.auto_scale_loss = bool(hp.get("auto_scale_loss", True))

        # factor C decomposition: independent similarity terms (ifp/aff/prot/lig).
        # When --sim-terms is non-empty this path replaces the single y-aware
        # InfoNCE; the matrices are COMMON attributes (not register_buffer) so
        # ~370 MB never enter a .ckpt, and row indices come from the datamodule.
        self.sim_terms = list(hp.get("sim_terms", []))
        self.sim_lambda = float(hp.get("sim_lambda", 0.025))
        self.sim_lambda_max = float(hp.get("sim_lambda_max", 0.125))
        self.sim_kendall = bool(hp.get("sim_kendall", False))
        self.sim_mat_dir = str(hp.get("sim_mat_dir", ""))
        self.S_prot = None
        self.S_lig = None
        if self.sim_terms:
            self._validate_sim_terms()
            self._load_sim_matrices()

    @property
    def rdrop_weight(self) -> float:
        """Peso que de fato multiplica o R-Drop na loss total."""
        return self.lambda_semi if self.lambda_rdrop is None else self.lambda_rdrop

    def _validate_sim_terms(self) -> None:
        """Recusa combinacoes que produziriam um termo morto ou dominante."""
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
        # non-dominance budget: the R-Drop weight + one lambda per term must
        # stay within the total block (D5: 5 x 0.025 = 0.125).
        total = self.rdrop_weight + len(self.sim_terms) * self.sim_lambda
        if total > self.sim_lambda_max + 1e-9:
            raise ValueError(
                f"similarity budget {total:.3f} (rdrop_weight + |K|*sim_lambda) exceeds "
                f"--sim-lambda-max {self.sim_lambda_max}; lower --sim-lambda, --lambda-semi "
                f"or --lambda-rdrop.")

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

    # ----------------------------------------------------------------- #
    # forward
    # ----------------------------------------------------------------- #
    def _to_device(self, *tensors):
        """Traz para o device do modelo os embeddings que vieram soltos na CPU."""
        return tuple(
            t.to(self.device)
            if isinstance(t, torch.Tensor) and t.device != self.device
            else t
            for t in tensors
        )

    def _concat_branches(self, x, e_prot, e_lig):
        """`u`: concatenacao dos ramos ligados, antes do gargalo latente."""
        parts = []
        if not self.no_cnn:
            parts.append(self.flatten(self.conv_layers(x)))
        # `proj_* is None` com o ramo ligado significa largura 0: o embedding
        # entra cru, sem projecao.
        if self.use_esm2 and e_prot is not None:
            parts.append(self.proj_prot(e_prot) if self.proj_prot is not None else e_prot)
        if self.use_chemberta and e_lig is not None:
            parts.append(self.proj_lig(e_lig) if self.proj_lig is not None else e_lig)
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

    def forward_latent(self, x, e_prot=None, e_lig=None):
        """Forward ate `z`, o espaco latente que alimenta as duas cabecas.

        E aqui que o contrastivo age. Como `z` vem depois da concatenacao, o
        gradiente dele atravessa o concat e chega em `proj_prot`/`proj_lig` —
        o que nao acontecia quando o gargalo ficava antes.
        """
        e_prot, e_lig = self._to_device(e_prot, e_lig)
        u = self._concat_branches(x, e_prot, e_lig)
        z = self.latent_proj(u) if self.latent_proj is not None else u
        self.last_z = z
        return z

    def forward(self, x, e_prot=None, e_lig=None):
        return self.head(self.forward_latent(x, e_prot, e_lig))

    # ----------------------------------------------------------------- #
    # perdas (fator C) — a matematica vive em losses.py; aqui so o estado
    # ----------------------------------------------------------------- #
    def _contrastive_cfg(self) -> losses.ContrastiveConfig:
        """Fotografa os hiperparametros contrastivos no estado atual.

        Montado na hora, e nao no __init__, porque a suite de testes altera
        `model.anchor_mode` entre chamadas.
        """
        return losses.ContrastiveConfig(
            tau=self.tau,
            yaware_sigma=self.yaware_sigma,
            ifp_tau=self.ifp_tau,
            anchor_mode=self.anchor_mode,
            lambda_aff=self.lambda_aff,
            lambda_ifp=self.lambda_ifp,
            lambda_prot=self.lambda_prot,
            lambda_lig=self.lambda_lig,
            auto_scale_loss=self.auto_scale_loss,
        )

    def _ifp_sim(self, ifp):
        return losses.ifp_dice_similarity(ifp)

    def _voxel_sim(self, x):
        return losses.voxel_similarity(x)

    def _sim_infonce(self, p, tgt):
        return losses.soft_infonce(p, tgt, self.tau)

    def _aff_target(self, y):
        return losses.affinity_target(y, self.yaware_sigma)

    def _ifp_target(self, ifp):
        return losses.ifp_target(ifp)

    def _prot_target(self, prot_idx):
        return losses.precomputed_target(prot_idx, self.S_prot)

    def _lig_target(self, lig_idx):
        return losses.precomputed_target(lig_idx, self.S_lig)

    def _yaware_infonce(self, p, y, ifp=None, x=None, e_prot=None, e_lig=None,
                        reg_loss=None):
        return losses.yaware_infonce(
            p, y, self._contrastive_cfg(), ifp=ifp, x=x,
            e_prot=e_prot, e_lig=e_lig, reg_loss=reg_loss,
        )

    def _similarity_targets(self, prot_idx, lig_idx, ifp, y) -> dict:
        """Alvo de cada termo ativo, na ordem em que `--sim-terms` os declarou."""
        builders = {
            "ifp": lambda: self._ifp_target(ifp),
            "aff": lambda: self._aff_target(y),
            "prot": lambda: self._prot_target(prot_idx),
            "lig": lambda: self._lig_target(lig_idx),
        }
        targets = {}
        for k in self.sim_terms:
            if k not in builders:
                raise ValueError(f"unknown sim term: {k}")
            targets[k] = builders[k]()
        return targets

    def _sim_terms_loss(self, p, prot_idx, lig_idx, ifp, y):
        """Weighted sum of the active similarity terms over the shared projection p.

        Returns ``(weighted_total, per_term)`` where ``per_term[k] = (L_k, row_frac)``
        with ``L_k`` the unweighted loss and ``row_frac`` the fraction of batch rows
        that have at least one positive partner for that term.
        """
        targets = self._similarity_targets(prot_idx, lig_idx, ifp, y)
        return losses.similarity_terms_loss(p, targets, self.tau, self.sim_lambda)

    def _semi_loss(self, x, e_prot, e_lig, y, ifp=None, prot_idx=None, lig_idx=None,
                   reg_loss=None):
        """L_semi (factor C): consistency (R-Drop) + contrastive (y-aware or embedding-anchored).

        Devolve SEMPRE ``(rdrop, resto)``: com --sim-terms `resto` e o par
        ``(sim_block, per_term)``, nos demais casos e o escalar contrastivo.
        Separar os dois e o que permite pesar a consistencia sozinha
        (--lambda-rdrop); quem recompoe a soma e o `shared_step`.
        """
        self.train()  # enable dropout for stochastic passes
        z1 = self.forward_latent(x, e_prot, e_lig)
        z2 = self.forward_latent(x, e_prot, e_lig)

        if self.rdrop_dropout > 0.0:
            # As duas vistas passam a diferir ja em `z`, entao a consistencia
            # restringe a representacao e nao so o dropout interno do
            # proj_head. Fica aqui, e nao em `forward_latent`, porque vale so
            # para as passagens contrastivas: a de predicao (L_reg) continua
            # deterministica.
            pa = F.normalize(
                self.proj_head(F.dropout(z1, self.rdrop_dropout, True)), dim=1)
            pb = F.normalize(
                self.proj_head(F.dropout(z2, self.rdrop_dropout, True)), dim=1)
            # Os termos contrastivos leem uma projecao LIMPA, para que
            # --rdrop-dropout continue sendo um eixo que mexe apenas no R-Drop.
            # Reusa `z1`: e mais uma passada pelo proj_head, nao pela CNN.
            p = F.normalize(self.proj_head(z1), dim=1)
        else:
            pa = F.normalize(self.proj_head(z1), dim=1)
            pb = F.normalize(self.proj_head(z2), dim=1)
            p = pa

        rdrop = F.mse_loss(pa, pb)  # consistency between the two stochastic views

        if self.sim_terms:
            return rdrop, self._sim_terms_loss(p, prot_idx, lig_idx, ifp, y)

        if self.yaware:
            return rdrop, self._yaware_infonce(
                p, y, ifp, x=x, e_prot=e_prot, e_lig=e_lig, reg_loss=reg_loss)

        return rdrop, self._embedding_anchored_loss(p, e_prot, e_lig)

    def _embedding_anchored_loss(self, p, e_prot, e_lig):
        """InfoNCE que ancora p(f) no par (e_prot, e_lig) do proprio complexo.

        Positivo e a diagonal: cada projecao contra o seu proprio embedding.
        Sem embeddings disponiveis nao ha ancora e o termo desaparece.

        A condicao e `contrastive_*`, nao `proj_* is not None`: um embedding
        ancora a projecao mesmo quando o ramo que o consumiria esta desligado.
        """
        parts = []
        if self.contrastive_prot and e_prot is not None:
            parts.append(e_prot)
        if self.contrastive_lig and e_lig is not None:
            parts.append(e_lig)
        if self.proj_target is None or not parts:
            return torch.zeros((), device=p.device)
        t = F.normalize(self.proj_target(torch.cat(parts, dim=1)), dim=1)
        sim = p @ t.T / self.tau
        labels = torch.arange(len(p), device=p.device)
        return F.cross_entropy(sim, labels)

    # ----------------------------------------------------------------- #
    # CLI e otimizador
    # ----------------------------------------------------------------- #
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
        parser.add_argument("--latent-dim", type=int, default=512, help="Width of the shared latent z, applied after the branch concatenation. 0 disables the bottleneck: the head reads the concatenation directly, which with no embeddings reproduces the upstream docktdeep architecture.")
        parser.add_argument("--emb-proj-dim", type=int, default=128, help="Default width of the embedding conditioning projections, used by any branch that does not override it.")
        parser.add_argument("--emb-proj-dim-prot", type=int, default=-1, help="Width of the ESM-2 projection before the concatenation. 0 removes the projection and feeds the frozen embedding raw into u (no BatchNorm, so it arrives on a different scale than the CNN output). -1 follows --emb-proj-dim.")
        parser.add_argument("--emb-proj-dim-lig", type=int, default=-1, help="Width of the ChemBERTa projection before the concatenation. 0 removes the projection and feeds the frozen embedding raw into u. -1 follows --emb-proj-dim.")
        parser.add_argument("--semi", action="store_true", default=False, help="Enable factor C: semi-supervised + regularizers.")
        parser.add_argument("--no-cnn", action="store_true", default=False, help="Ablate the 3D CNN branch: predict from the frozen embeddings alone (requires --use-esm2 and/or --use-chemberta).")
        parser.add_argument("--preds-dir", type=str, default="preds", help="Directory for the per-complex test predictions CSV (id,y_true,y_pred).")
        parser.add_argument("--loss", type=str, default="mse", choices=["mse", "huber"], help="Affinity regression loss.")
        parser.add_argument("--huber-beta", type=float, default=1.0, help="Huber loss beta (if --loss huber).")
        parser.add_argument("--label-smoothing", type=float, default=0.0, help="Shrink regression targets toward batch mean.")
        parser.add_argument("--lambda-semi", type=float, default=1.0, help="Weight of L_semi in the total loss.")
        parser.add_argument("--proj-dim", type=int, default=128, help="Projection head p(f) output dim (factor C).")
        parser.add_argument("--semi-tau", type=float, default=0.1, help="Temperature for contrastive L_semi.")
        parser.add_argument("--lambda-rdrop", type=float, default=None, help="Weight of the R-Drop consistency term ALONE. Unset (default) keeps the historical coupling: R-Drop is scaled by --lambda-semi together with the contrastive block, so every run measured so far is reproduced bit for bit. Set it to weight the consistency independently; 0 disables R-Drop while keeping the contrastive terms, which is what isolates its individual contribution.")
        parser.add_argument("--rdrop-dropout", type=float, default=0.0, help="Dropout applied to the latent z in the TWO CONTRASTIVE PASSES ONLY, so the R-Drop consistency constrains the representation instead of just the proj_head masks. The prediction pass stays deterministic and L_reg is untouched; the contrastive terms keep reading a clean projection. 0 (default) reproduces the previous behaviour.")
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
    # ----------------------------------------------------------------- #
    # laco de treino, metricas e dump
    # ----------------------------------------------------------------- #
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
            rdrop, rest = self._semi_loss(
                x, e_prot, e_lig, y, ifp, prot_idx, lig_idx, reg_loss=loss)
            if self.sim_terms:
                sim_block, per_term = rest
                loss = loss + self.rdrop_weight * rdrop + sim_block
                self.log("train_semi", rdrop.detach(), **log_params)
                for k, (Lk, row_frac) in per_term.items():
                    self.log(f"train_sim_{k}", Lk.detach(), **log_params)
                    self.log(f"train_sim_{k}_rows", row_frac.detach(), **log_params)
            else:
                semi = rdrop + rest
                # Sem --lambda-rdrop a soma inteira e escalada por lambda_semi,
                # exatamente como nas campanhas ja rodadas; com ele, cada parte
                # ganha o seu peso.
                if self.lambda_rdrop is None:
                    loss = loss + self.lambda_semi * semi
                else:
                    loss = loss + self.lambda_rdrop * rdrop + self.lambda_semi * rest
                self.log("train_semi", semi.detach(), **log_params)
            self.log("train_rdrop", rdrop.detach(), **log_params)
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
                    if mask.sum() >= MIN_STRATUM_N:
                        sub_p = preds[mask]
                        sub_l = labels[mask]
                        sub_m = self._regression_metrics(sub_p, sub_l, f"val_{st}")
                        sub_m[f"val_{st}_n"] = float(mask.sum())
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
                            if mask.sum() >= MIN_STRATUM_N:
                                sub_p = t_preds[mask]
                                sub_l = t_labels[mask]
                                sub_m = self._regression_metrics(sub_p, sub_l, f"test_{st}")
                                sub_m[f"test_{st}_n"] = float(mask.sum())
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
                "best_val_mae": best_mae["val_mae"],
                "val_mae_at_best_loss": best_loss["val_mae"],
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

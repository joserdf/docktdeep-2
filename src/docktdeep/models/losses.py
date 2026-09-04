"""Termos contrastivos do fator C, como funcoes puras.

Nada aqui tem estado nem parametros treinaveis: entram tensores e escalares,
sai um tensor. O `Baseline` continua dono dos hiperparametros e das matrizes de
similaridade; este modulo so sabe transformar uns nos outros.

O vocabulario e sempre o mesmo:

* *similaridade* — matriz (B, B) crua, com a diagonal ainda presente;
* *alvo* (`*_target`) — a mesma matriz com a diagonal zerada, pronta para o
  InfoNCE. A diagonal sai porque um exemplo nao pode ser positivo de si mesmo;
* *InfoNCE soft* — entropia cruzada contra um alvo normalizado por linha.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "ContrastiveConfig",
    "ifp_dice_similarity",
    "voxel_similarity",
    "off_diagonal",
    "affinity_target",
    "ifp_target",
    "precomputed_target",
    "cosine_target",
    "soft_infonce",
    "balance_scale",
    "yaware_infonce",
    "similarity_terms_loss",
]

_EPS = 1e-8


@dataclass(frozen=True)
class ContrastiveConfig:
    """Hiperparametros lidos por `yaware_infonce`.

    Montado a cada chamada a partir dos atributos do `Baseline`, e nao guardado,
    para que mexer em `model.anchor_mode` em tempo de execucao continue valendo.
    """

    tau: float
    yaware_sigma: float
    ifp_tau: float
    anchor_mode: str
    lambda_aff: float
    lambda_ifp: float
    lambda_prot: float
    lambda_lig: float
    auto_scale_loss: bool


# --------------------------------------------------------------------------- #
# similaridades pareadas
# --------------------------------------------------------------------------- #
def ifp_dice_similarity(ifp: torch.Tensor) -> torch.Tensor:
    """Dice pareado de fingerprints de interacao binarios (B, 4096) -> (B, B)."""
    b = ifp.float()
    inter = b @ b.T  # (B, B) shared bits
    pop = b.sum(dim=1)  # (B,)
    return 2.0 * inter / (pop[:, None] + pop[None, :] + _EPS)


def voxel_similarity(x: torch.Tensor) -> torch.Tensor:
    """Similaridade estrutural calculada na hora, a partir das grades (10.5).

    Descritor grosseiro de ocupacao 3D por amostra via average pooling adaptativo
    para (4,4,4), seguido de cosseno pareado. Barato e independente do IFP
    precalculado (a alternativa pre-IFP da secao 10.5).
    """
    B = x.shape[0]
    desc = F.adaptive_avg_pool3d(x, (4, 4, 4))  # (B, C, 4, 4, 4)
    desc = desc.reshape(B, -1)  # (B, C*64)
    desc = F.normalize(desc, dim=1)
    return desc @ desc.T  # (B, B)


# --------------------------------------------------------------------------- #
# alvos soft-positive
# --------------------------------------------------------------------------- #
def off_diagonal(sim: torch.Tensor) -> torch.Tensor:
    """Zera a diagonal de uma matriz (B, B) de similaridade."""
    eye = torch.eye(sim.shape[0], device=sim.device)
    return sim * (1.0 - eye)


def affinity_target(y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Proximidade em afinidade: exp(-|dy| / sigma), fora da diagonal."""
    d = torch.abs(y[:, None] - y[None, :])
    return off_diagonal(torch.exp(-d / sigma))


def ifp_target(ifp: torch.Tensor) -> torch.Tensor:
    """Dice do IFP PLEC, fora da diagonal."""
    return off_diagonal(ifp_dice_similarity(ifp))


def precomputed_target(idx: torch.Tensor, S: np.ndarray) -> torch.Tensor:
    """Alvo vindo de uma matriz de similaridade precalculada, fora da diagonal.

    So o sub-bloco (B, B) e coletado na CPU e movido para o device; a matriz
    inteira continua sendo um array numpy (sao ~370 MB que nao podem entrar num
    .ckpt). Os valores estao em centesimos ([0, 100]) e a linha-sentinela, toda
    zero, produz uma linha de alvo zerada — sem gradiente, por construcao.
    """
    rows = idx.detach().cpu().numpy()
    sub = S[np.ix_(rows, rows)].astype(np.float32) / 100.0
    return off_diagonal(torch.as_tensor(sub, device=idx.device))


def cosine_target(e: torch.Tensor) -> torch.Tensor:
    """Cosseno positivo entre embeddings congelados, fora da diagonal."""
    e_norm = F.normalize(e, dim=1)
    return off_diagonal(torch.relu(e_norm @ e_norm.T))


# --------------------------------------------------------------------------- #
# InfoNCE e balanceamento
# --------------------------------------------------------------------------- #
def soft_infonce(p: torch.Tensor, tgt: torch.Tensor, tau: float) -> torch.Tensor:
    """InfoNCE contra um alvo soft fixo.

    `tgt` e normalizado por linha; linhas sem nenhum parceiro positivo
    (rowsum ~ 0) ficam zeradas e nao contribuem gradiente, entao a perda so
    ordena *dentro* do contexto selecionado de cada amostra.
    """
    sim = p @ p.T / tau  # (B, B) embedding similarity
    # O alvo e uma ponderacao fixa (rotulos / IFP / similaridade) — detach para
    # que o gradiente flua so pela similaridade das projecoes sim(p,p), nunca
    # pelas entradas cruas.
    tgt = tgt.detach()
    rowsum = tgt.sum(dim=1, keepdim=True)
    tgt = torch.where(rowsum > _EPS, tgt / (rowsum + _EPS), torch.zeros_like(tgt))
    log_softmax = torch.log_softmax(sim, dim=1)
    return -(tgt * log_softmax).sum(dim=1).mean()


def balance_scale(
    loss: torch.Tensor, reg_loss: torch.Tensor | None, enabled: bool
) -> torch.Tensor:
    """Reescala `loss` para a magnitude de `reg_loss` sem mexer no gradiente.

    O fator e construido so com tensores destacados, entao a direcao do
    gradiente e preservada e apenas o passo muda de tamanho. Sem `reg_loss` a
    perda vai para magnitude 1.
    """
    if not enabled:
        return loss
    denom = loss.detach() + _EPS
    if reg_loss is None:
        return loss / denom
    return loss * ((reg_loss.detach() + _EPS) / denom)


# --------------------------------------------------------------------------- #
# ancoras
# --------------------------------------------------------------------------- #
def _anchor_target(
    cfg: ContrastiveConfig,
    y: torch.Tensor,
    ifp: torch.Tensor | None,
    x: torch.Tensor | None,
) -> torch.Tensor:
    """Alvo do InfoNCE y-aware, conforme `cfg.anchor_mode`.

    Sem IFP disponivel os modos que dependem dele degradam para `affinity` em
    vez de quebrar — o run continua, so perde o termo estrutural.
    """
    aff = affinity_target(y, cfg.yaware_sigma)
    needs_ifp = cfg.anchor_mode in ("gate", "ifp", "hybrid", "dual")

    if cfg.anchor_mode == "affinity" or (needs_ifp and ifp is None):
        return aff
    if cfg.anchor_mode == "gate":
        gate = off_diagonal((ifp_dice_similarity(ifp) >= cfg.ifp_tau).float())
        return aff * gate
    if cfg.anchor_mode == "ifp":
        return ifp_target(ifp)
    if cfg.anchor_mode == "hybrid":
        return aff * ifp_target(ifp)
    if cfg.anchor_mode == "struct":
        if x is None:
            return aff
        return aff * off_diagonal(voxel_similarity(x))
    raise ValueError(f"unknown anchor_mode: {cfg.anchor_mode}")


def yaware_infonce(
    p: torch.Tensor,
    y: torch.Tensor,
    cfg: ContrastiveConfig,
    ifp: torch.Tensor | None = None,
    x: torch.Tensor | None = None,
    e_prot: torch.Tensor | None = None,
    e_lig: torch.Tensor | None = None,
    reg_loss: torch.Tensor | None = None,
) -> torch.Tensor:
    """InfoNCE soft ancorado em afinidade, IFP, estrutura ou embeddings.

    Molda p(f) para que a proximidade no espaco de projecao espelhe a ordenacao
    da similaridade escolhida como ancora.
    """
    if cfg.anchor_mode == "dual" and ifp is not None:
        # dois termos separados, cada um trazido a escala da perda de regressao
        # antes de receber o proprio peso — senao o de maior magnitude domina.
        l_aff = soft_infonce(p, affinity_target(y, cfg.yaware_sigma), cfg.tau)
        l_ifp = soft_infonce(p, ifp_target(ifp), cfg.tau)
        total = (
            cfg.lambda_aff * balance_scale(l_aff, reg_loss, cfg.auto_scale_loss)
            + cfg.lambda_ifp * balance_scale(l_ifp, reg_loss, cfg.auto_scale_loss)
        )
    else:
        total = soft_infonce(p, _anchor_target(cfg, y, ifp, x), cfg.tau)

    # termos opcionais de cosseno sobre os embeddings congelados
    for weight, emb in ((cfg.lambda_prot, e_prot), (cfg.lambda_lig, e_lig)):
        if weight > 0.0 and emb is not None:
            term = soft_infonce(p, cosine_target(emb), cfg.tau)
            total = total + weight * balance_scale(
                term, reg_loss, cfg.auto_scale_loss
            )

    return total


# --------------------------------------------------------------------------- #
# decomposicao em termos de similaridade
# --------------------------------------------------------------------------- #
def similarity_terms_loss(
    p: torch.Tensor,
    targets: dict[str, torch.Tensor],
    tau: float,
    weight: float,
) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """Soma ponderada de termos independentes sobre a mesma projecao `p`.

    Devolve `(total_ponderado, por_termo)`, onde `por_termo[k] = (L_k, row_frac)`
    com `L_k` a perda sem peso e `row_frac` a fracao de linhas do batch que tem
    ao menos um parceiro positivo naquele termo — a metrica que denuncia um
    termo morto antes de ele passar despercebido como zero saudavel.
    """
    total = torch.zeros((), device=p.device)
    per_term = {}
    for name, tgt in targets.items():
        row_frac = (tgt.sum(dim=1) > _EPS).float().mean()
        Lk = soft_infonce(p, tgt, tau)
        per_term[name] = (Lk, row_frac)
        total = total + Lk
    return weight * total, per_term

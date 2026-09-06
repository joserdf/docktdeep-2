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
    "row_fraction",
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


def row_fraction(tgt: torch.Tensor) -> torch.Tensor:
    """Fracao das linhas de `tgt` com ao menos um parceiro positivo.

    `soft_infonce` zera as linhas cuja soma e ~0, e uma linha zerada nao
    contribui gradiente. Sem esta fracao um termo morto e indistinguivel de um
    termo saudavel que por acaso deu perda baixa: os dois aparecem como um
    numero pequeno. E o mesmo diagnostico que `similarity_terms_loss` ja
    publicava; aqui ele fica disponivel tambem para o caminho y-aware.
    """
    return (tgt.sum(dim=1) > _EPS).float().mean()


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

    A DIAGONAL SAI DO DENOMINADOR. `off_diagonal` ja zerava a diagonal do ALVO
    (uma amostra nao e positiva de si mesma), mas `sim_ii` continuava dentro do
    softmax -- e com `p` normalizado por L2 ele vale 1/tau, a maior logit
    possivel da matriz. Medido a B=256: a diagonal levava 99,9% do denominador
    em tau=0,08 e 100% em tau=0,02, sobrando 0,1% de repulsao para os negativos
    reais. Como d(loss_i)/d(sim_ik) = softmax_ik - tgt_ik e tgt_ii == 0, toda
    essa massa caia sobre a propria amostra, onde nao faz nada (p_i . p_i == 1
    por construcao): a perda degenerava em alinhamento puro com os positivos, e
    o tamanho do batch -- que E o conjunto de negativos -- deixava de importar.

    Isto vale so aqui. `_embedding_anchored_loss` faz `p @ t.T` com t de OUTRA
    projecao, e la a diagonal e o par positivo legitimo.
    """
    # B == 1 nao tem negativo nenhum: com a diagonal mascarada a linha fica toda
    # -inf e o log_softmax devolve nan, que o alvo zerado NAO neutraliza
    # (0 * nan == nan). Um batch final de tamanho 1 mataria o run inteiro.
    if p.shape[0] < 2:
        return torch.zeros((), device=p.device, dtype=p.dtype)
    diag = torch.eye(p.shape[0], dtype=torch.bool, device=p.device)
    sim = p @ p.T / tau  # (B, B) embedding similarity
    sim = sim.masked_fill(diag, float("-inf"))
    # O alvo e uma ponderacao fixa (rotulos / IFP / similaridade) — detach para
    # que o gradiente flua so pela similaridade das projecoes sim(p,p), nunca
    # pelas entradas cruas.
    tgt = tgt.detach()
    rowsum = tgt.sum(dim=1, keepdim=True)
    tgt = torch.where(rowsum > _EPS, tgt / (rowsum + _EPS), torch.zeros_like(tgt))
    log_softmax = torch.log_softmax(sim, dim=1)
    # A diagonal do log_softmax e -inf pela mascara, e a do alvo e 0: o produto
    # daria 0 * -inf == nan em vez do 0 que a matematica pede. Zerar a diagonal
    # DEPOIS do softmax e exato -- ela ja nao contribuia termo nenhum.
    log_softmax = log_softmax.masked_fill(diag, 0.0)
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
    diag: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """InfoNCE soft ancorado em afinidade, IFP, estrutura ou embeddings.

    Molda p(f) para que a proximidade no espaco de projecao espelhe a ordenacao
    da similaridade escolhida como ancora.

    `diag`, quando passado, e preenchido com a perda crua e a fracao de linhas
    vivas de cada termo montado — o mesmo par que `similarity_terms_loss` ja
    devolvia. E a unica forma de distinguir, neste caminho, um termo com pouco
    sinal de um termo cujo alvo nao tem positivo nenhum.
    """
    def _rec(name, loss, tgt):
        if diag is not None:
            diag[f"{name}"] = loss.detach()
            diag[f"{name}_rows"] = row_fraction(tgt).detach()

    if cfg.anchor_mode == "dual" and ifp is not None:
        # dois termos separados, cada um trazido a escala da perda de regressao
        # antes de receber o proprio peso — senao o de maior magnitude domina.
        t_aff = affinity_target(y, cfg.yaware_sigma)
        t_ifp = ifp_target(ifp)
        l_aff = soft_infonce(p, t_aff, cfg.tau)
        l_ifp = soft_infonce(p, t_ifp, cfg.tau)
        _rec("aff", l_aff, t_aff)
        _rec("ifp", l_ifp, t_ifp)
        total = (
            cfg.lambda_aff * balance_scale(l_aff, reg_loss, cfg.auto_scale_loss)
            + cfg.lambda_ifp * balance_scale(l_ifp, reg_loss, cfg.auto_scale_loss)
        )
    else:
        tgt = _anchor_target(cfg, y, ifp, x)
        total = soft_infonce(p, tgt, cfg.tau)
        _rec("anchor", total, tgt)

    # termos opcionais de cosseno sobre os embeddings congelados
    for name, weight, emb in (
        ("prot", cfg.lambda_prot, e_prot), ("lig", cfg.lambda_lig, e_lig)
    ):
        if weight > 0.0 and emb is not None:
            t_emb = cosine_target(emb)
            term = soft_infonce(p, t_emb, cfg.tau)
            _rec(name, term, t_emb)
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
        row_frac = row_fraction(tgt)
        Lk = soft_infonce(p, tgt, tau)
        per_term[name] = (Lk, row_frac)
        total = total + Lk
    return weight * total, per_term

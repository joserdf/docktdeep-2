"""A regra de nomes do Aim: o nome e a quantidade, o resto e contexto.

Cobre as quatro familias de eixo que o levantamento de 2026-09-11 achou
embutidas nos nomes de 2355 runs: subset, stratum, term/family e branch, mais
a reducao sobre as epocas.
"""

import pytest

from src.docktdeep.aim_naming import (AUX_LOSS, AUX_ROWS, ESTRATOS, GRAD_NORM,
                                      POOLED, RAMOS, TERMOS, e_curva_pooled,
                                      nome_e_contexto)

METRICAS = ("pearsonr", "spearman", "mae", "mse", "rmse", "r2", "loss")


# --- eixo: subset e stratum -------------------------------------------------
@pytest.mark.parametrize("chave,nome,ctx", [
    ("val_pearsonr", "pearsonr", {"subset": "val", "stratum": "all"}),
    ("val_ood_pearsonr", "pearsonr", {"subset": "val", "stratum": "ood"}),
    ("val_in_pearsonr", "pearsonr", {"subset": "val", "stratum": "in"}),
    ("test_casf_r2", "r2", {"subset": "test", "stratum": "casf"}),
    ("test_dev_spearman", "spearman", {"subset": "test", "stratum": "dev"}),
    ("train_loss", "loss", {"subset": "train", "stratum": "all"}),
    ("val_in_n", "n", {"subset": "val", "stratum": "in"}),
])
def test_metrica_de_dataset(chave, nome, ctx):
    assert nome_e_contexto(chave) == (nome, ctx)


def test_pooled_e_estratos_compartilham_o_nome():
    """O ponto do exercicio: um grafico so, agrupado por stratum."""
    nomes = {nome_e_contexto(k)[0]
             for k in ("val_pearsonr", "val_in_pearsonr", "val_ood_pearsonr",
                       "test_pearsonr", "test_casf_pearsonr")}
    assert nomes == {"pearsonr"}


# --- eixo: reducao sobre as epocas ------------------------------------------
@pytest.mark.parametrize("chave,nome,reducao", [
    ("val_best_pearsonr", "pearsonr", "best"),
    ("val_best_loss", "loss", "best"),
    ("val_best_mae", "mae", "best"),
    ("val_mae_at_best_loss", "mae", "at_best_loss"),
])
def test_resumo_e_a_mesma_quantidade(chave, nome, reducao):
    """`best_val_pearsonr` trazia `val` no nome E no contexto, e criava um
    nome proprio para uma quantidade que ja existe como `pearsonr`."""
    assert nome_e_contexto(chave) == (
        nome, {"subset": "val", "stratum": POOLED, "reduction": reducao})


def test_resumo_nao_e_curva():
    """Uma serie de um ponto nao pode ser lida como a curva por epoca."""
    _, curva = nome_e_contexto("val_pearsonr")
    _, resumo = nome_e_contexto("val_best_pearsonr")
    assert e_curva_pooled(curva)
    assert not e_curva_pooled(resumo)


# --- eixo: termo e familia da perda auxiliar --------------------------------
@pytest.mark.parametrize("k", TERMOS)
def test_os_dois_contrastivos_sob_um_nome(k):
    """`--sim-terms` e `--yaware` usam os MESMOS nomes de termo: sem `family`
    as series colidiriam."""
    sim = nome_e_contexto(f"train_sim_{k}")
    yaw = nome_e_contexto(f"train_yaw_{k}")
    assert sim == (AUX_LOSS, {"subset": "train", "term": k, "family": "sim"})
    assert yaw == (AUX_LOSS, {"subset": "train", "term": k, "family": "yaware"})


@pytest.mark.parametrize("familia", ["sim", "yaw"])
def test_rows_e_outra_quantidade(familia):
    """`_rows` e fracao de linhas, nao perda: nome proprio, mesmos eixos."""
    nome, ctx = nome_e_contexto(f"train_{familia}_lig_rows")
    assert nome == AUX_ROWS
    assert ctx["term"] == "lig"


def test_blocos_nao_tem_familia():
    """`semi` e o bloco inteiro e `rdrop` e a sua parte: nenhum e modalidade."""
    for termo in ("semi", "rdrop"):
        nome, ctx = nome_e_contexto(f"train_{termo}")
        assert nome == AUX_LOSS
        assert ctx == {"subset": "train", "term": termo}


# --- eixo: ramo da rede -----------------------------------------------------
@pytest.mark.parametrize("ramo", RAMOS)
def test_norma_de_gradiente(ramo):
    assert nome_e_contexto(f"train_grad_{ramo}") == (
        GRAD_NORM, {"subset": "train", "branch": ramo})


# --- limites do parser ------------------------------------------------------
def test_internos_de_treino_nao_tem_estrato():
    """`stratum` responde "que fatia de exemplos". Para um componente da perda
    a pergunta nao se aplica, e um `all` constante sugeriria um `ood` irmao."""
    for chave in ("train_semi", "train_sim_lig", "train_grad_cnn",
                  "train_yaw_aff_rows"):
        assert "stratum" not in nome_e_contexto(chave)[1]


def test_vocabulario_fechado_protege_o_parser():
    """Termo ou ramo desconhecido cai na metrica de dataset em vez de virar um
    eixo inventado -- e o que impede um nome novo de sumir num contexto."""
    nome, ctx = nome_e_contexto("train_grad_desconhecido")
    assert nome == "grad_desconhecido"
    assert ctx == {"subset": "train", "stratum": POOLED}


def test_chave_sem_subset_sai_intacta():
    """O Lightning passa metricas que nao sao do modelo pelo mesmo canal."""
    assert nome_e_contexto("hp_metric") == ("hp_metric", {})
    assert not e_curva_pooled({})


def test_nenhuma_colisao_no_conjunto_real():
    """57 chaves reais -> 57 series. Duas chaves no mesmo (nome, contexto)
    fariam uma sobrescrever a outra em silencio."""
    chaves = [f"{s}_{m}" for s in ("train", "val", "test") for m in METRICAS]
    chaves += [f"{st}_{m}" for st in ("val_in", "val_ood", "test_ood",
                                      "test_casf", "test_dev")
               for m in ("pearsonr", "n")]
    chaves += ["val_best_pearsonr", "val_best_loss", "val_best_mae",
               "val_mae_at_best_loss", "train_semi", "train_rdrop"]
    chaves += [f"train_{f}_{k}{r}" for f in ("sim", "yaw") for k in TERMOS
               for r in ("", "_rows")]
    chaves += [f"train_grad_{b}" for b in RAMOS]

    series = [(n, tuple(sorted(c.items())))
              for n, c in map(nome_e_contexto, chaves)]
    assert len(set(series)) == len(chaves)
    # 11 quantidades: aux_loss, aux_rows, grad_norm, loss, mae, mse, n,
    # pearsonr, r2, rmse, spearman. Eram ~40 nomes distintos no Aim.
    assert len({n for n, _ in series}) == 11


def test_estratos_cobrem_as_duas_colunas_do_dataset():
    """grp_stratum da ood/casf/dev; grp_mixval_stratum_o<N> da in/ood."""
    assert set(ESTRATOS) == {"in", "ood", "casf", "dev"}

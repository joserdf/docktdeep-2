"""A regra de nomes do Aim: o nome e a quantidade, o resto e contexto."""

import pytest

from src.docktdeep.aim_naming import ESTRATOS, POOLED, nome_e_contexto


@pytest.mark.parametrize("chave,nome,ctx", [
    ("val_pearsonr", "pearsonr", {"subset": "val", "stratum": "all"}),
    ("val_ood_pearsonr", "pearsonr", {"subset": "val", "stratum": "ood"}),
    ("val_in_pearsonr", "pearsonr", {"subset": "val", "stratum": "in"}),
    ("test_casf_r2", "r2", {"subset": "test", "stratum": "casf"}),
    ("test_dev_spearman", "spearman", {"subset": "test", "stratum": "dev"}),
    ("train_loss", "loss", {"subset": "train", "stratum": "all"}),
    ("val_in_n", "n", {"subset": "val", "stratum": "in"}),
])
def test_traducao(chave, nome, ctx):
    assert nome_e_contexto(chave) == (nome, ctx)


def test_pooled_e_estrato_compartilham_o_nome():
    """O ponto do exercicio: um grafico so, agrupado por stratum."""
    nomes = {nome_e_contexto(k)[0]
             for k in ("val_pearsonr", "val_in_pearsonr", "val_ood_pearsonr",
                       "test_pearsonr", "test_casf_pearsonr")}
    assert nomes == {"pearsonr"}


def test_contextos_sao_distintos():
    """Mesmo nome exige contextos distintos, ou uma serie sobrescreve a outra."""
    ctxs = [tuple(sorted(nome_e_contexto(k)[1].items()))
            for k in ("val_pearsonr", "val_in_pearsonr", "val_ood_pearsonr",
                      "test_pearsonr", "test_casf_pearsonr")]
    assert len(set(ctxs)) == len(ctxs)


@pytest.mark.parametrize("chave", ["train_sim_lig", "train_sim_lig_rows",
                                   "train_grad_cnn", "train_yaw_aff",
                                   "train_semi", "train_rdrop"])
def test_termos_de_perda_nao_viram_estrato(chave):
    """`sim`, `grad` e `yaw` ocupam a posicao do estrato sem serem estratos.

    E por isso que ESTRATOS e um vocabulario fechado: um parser generico leria
    `train_sim_lig` como estrato 'sim', metrica 'lig'.
    """
    nome, ctx = nome_e_contexto(chave)
    assert nome == chave[len("train_"):]
    assert ctx["stratum"] == POOLED


def test_chave_sem_subset_sai_intacta():
    """O Lightning passa metricas que nao sao do modelo pelo mesmo canal."""
    assert nome_e_contexto("hp_metric") == ("hp_metric", {})


def test_sumario_perde_o_subset_duplicado():
    """`best_val_pearsonr` trazia `val` no nome E no contexto."""
    assert nome_e_contexto("val_best_pearsonr") == (
        "best_pearsonr", {"subset": "val", "stratum": "all"})


def test_estratos_cobrem_as_duas_colunas_do_dataset():
    """grp_stratum da ood/casf/dev; grp_mixval_stratum_o<N> da in/ood."""
    assert set(ESTRATOS) == {"in", "ood", "casf", "dev"}

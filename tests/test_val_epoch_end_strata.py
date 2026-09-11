"""O lado PRODUTOR das metricas de validacao: `on_validation_epoch_end`.

Os testes de `emit_metrics_line` fabricam o dicionario da epoca, entao nao veem
o que este arquivo cobre: o nome que o produtor da a cada estrato, e quais
epocas entram em `validation_logs`. Os dois defeitos que motivaram o arquivo
passaram exatamente por esse vao.

A instancia e falsa de proposito: construir um `Baseline` de verdade exige GPU,
grade de voxels e embeddings. O que esta sob teste e a logica do metodo, entao o
metodo e chamado desligado da classe, com o minimo de atributos que ele le.
"""
import types

import torch

from src.docktdeep.models.baseline import Baseline

# valores reais das colunas de estrato do index-grouped-nocl1.csv
ESTRATOS = ["val_in"] * 40 + ["val_ood"] * 60


def fake(strata, sanity=False, n=100):
    torch.manual_seed(0)
    saidas = [{"preds": torch.rand(n, 1), "labels": torch.rand(n),
               "val_loss": torch.tensor(1.0)}]
    return types.SimpleNamespace(
        validation_step_outputs=saidas,
        validation_logs=[],
        _regression_metrics=lambda p, l, s: Baseline._regression_metrics(
            types.SimpleNamespace(mae=lambda a, b: (a - b).abs().mean()), p, l, s),
        trainer=types.SimpleNamespace(
            datamodule=types.SimpleNamespace(val_strata=strata), sanity_checking=sanity),
        hparams={},
        log_dict=lambda *a, **k: None,
    )


def rodar(m):
    Baseline.on_validation_epoch_end(m)
    return m.validation_logs


def test_estrato_nao_ganha_prefixo_dobrado():
    """`grp_mixval_stratum_o<N>` ja traz o valor prefixado ("val_in").

    Concatenar sem verificar produzia `val_val_in_pearsonr`, chave que nenhum
    leitor procura -- o estrato sumia do metrics_json sem erro algum.
    """
    log = rodar(fake(ESTRATOS))[0]

    assert "val_in_pearsonr" in log and "val_ood_pearsonr" in log
    assert log["val_in_n"] == 40.0 and log["val_ood_n"] == 60.0
    assert not [k for k in log if k.startswith("val_val_")]


def test_sanity_check_nao_entra_no_log():
    """Pesos aleatorios sobre parte do split nao sao uma epoca.

    Entrando em `validation_logs`, a sanity check disputa o argmax de
    `best_val_pearsonr` -- e ganha justamente nos runs ruins.
    """
    assert rodar(fake(ESTRATOS, sanity=True)) == []


def test_estrato_menor_que_o_piso_e_omitido():
    """Abaixo de MIN_STRATUM_N a correlacao nao e publicada, nem como zero."""
    log = rodar(fake(["val_in"] * 5 + ["val_ood"] * 95))[0]

    assert "val_in_pearsonr" not in log
    assert log["val_ood_n"] == 95.0


def test_split_sem_coluna_de_estrato():
    log = rodar(fake(None))[0]

    assert "val_pearsonr" in log
    assert not [k for k in log if k.startswith("val_in") or k.startswith("val_ood")]

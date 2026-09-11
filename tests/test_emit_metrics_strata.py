import json
import types

import pytest
import torch

from src.docktdeep.models.baseline import _chave_estrato
from train import emit_metrics_line

# `val_ood` e o estrato que carrega o sinal de generalizacao; `val_in` e a
# metade in-domain. Valores distintos de proposito: um bug que publicasse o
# pooled no lugar do estrato passaria despercebido com numeros iguais.
EPOCA_BOA = {"val_pearsonr": 0.63, "val_loss": 0.98, "val_mae": 1.07,
             "val_in_pearsonr": torch.tensor(0.73), "val_in_n": 187.0,
             "val_ood_pearsonr": torch.tensor(0.57), "val_ood_n": 402.0}
EPOCA_RUIM = {"val_pearsonr": 0.41, "val_loss": 1.40, "val_mae": 1.30,
              "val_in_pearsonr": torch.tensor(0.50), "val_in_n": 187.0,
              "val_ood_pearsonr": torch.tensor(0.30), "val_ood_n": 402.0}


def emitir(logs, capsys):
    model = types.SimpleNamespace(validation_logs=logs)
    trainer = types.SimpleNamespace(current_epoch=len(logs), callback_metrics={})
    args = types.SimpleNamespace(experiment="t", seed=0)
    emit_metrics_line(trainer, model, args)
    saida = capsys.readouterr().out.strip()
    return json.loads(saida)["metrics"] if saida else None


def test_estratos_saem_da_epoca_do_melhor_pooled(capsys):
    """A epoca do checkpoint e a que define TODOS os numeros da linha.

    Publicar o estrato da ultima epoca ao lado do pooled da melhor daria duas
    procedencias dentro do mesmo JSON. A epoca ruim vem por ultimo justamente
    para que a ordem nao possa mascarar o erro.
    """
    m = emitir([EPOCA_BOA, EPOCA_RUIM], capsys)

    assert m["best_val_pearsonr"] == 0.63
    assert m["val_in_pearsonr"] == pytest.approx(0.73)
    assert m["val_ood_pearsonr"] == pytest.approx(0.57)
    # a ultima epoca segue publicada, e continua sendo a ruim
    assert m["final_val_pearsonr"] == 0.41


def test_estrato_ausente_nao_vira_zero(capsys):
    """Split sem coluna de estrato, ou estrato abaixo de MIN_STRATUM_N.

    A chave tem de SUMIR. Um zero silencioso entraria na busca multiobjetivo
    como 'correlacao nula medida', que e uma afirmacao forte sobre um fato que
    ninguem mediu.
    """
    m = emitir([{"val_pearsonr": 0.63, "val_loss": 0.98, "val_mae": 1.07}], capsys)

    assert m["best_val_pearsonr"] == 0.63
    assert "val_in_pearsonr" not in m
    assert "val_ood_pearsonr" not in m


def test_um_estrato_so(capsys):
    """So `val_ood` passou do piso de amostras: publica um, omite o outro."""
    epoca = {"val_pearsonr": 0.63, "val_loss": 0.98, "val_mae": 1.07,
             "val_ood_pearsonr": torch.tensor(0.57), "val_ood_n": 402.0}
    m = emitir([epoca], capsys)

    assert m["val_ood_pearsonr"] == pytest.approx(0.57)
    assert "val_in_pearsonr" not in m


# --- lado do produtor -------------------------------------------------------
# Os testes acima fabricam as chaves que o leitor espera, e foi exatamente esse
# vao que escondeu `val_val_in_pearsonr` por tres campanhas: o produtor
# (`on_validation_epoch_end`) montava o nome concatenando o prefixo do split com
# o VALOR da coluna de estrato, que no caso do mixval ja vem prefixado. Daqui
# para baixo o nome vem da funcao de verdade, nunca de um literal.

# valores que cada coluna de estrato carrega no index-grouped-nocl1.csv
VALORES_MIXVAL = ("val_in", "val_ood")      # grp_mixval_stratum_o<N>
VALORES_GRP = ("dev", "ood", "casf")        # grp_stratum


def test_chave_do_estrato_nao_duplica_prefixo():
    assert [_chave_estrato("val", v) for v in VALORES_MIXVAL] == ["val_in", "val_ood"]
    assert [_chave_estrato("test", v) for v in VALORES_GRP] == [
        "test_dev", "test_ood", "test_casf"]


def test_produtor_e_leitor_usam_a_mesma_chave(capsys):
    """A epoca e montada com as chaves que o produtor realmente gera."""
    epoca = {"val_pearsonr": 0.63, "val_loss": 0.98, "val_mae": 1.07}
    for i, v in enumerate(VALORES_MIXVAL):
        nome = _chave_estrato("val", v)
        epoca[f"{nome}_pearsonr"] = torch.tensor(0.70 - 0.1 * i)
        epoca[f"{nome}_n"] = 200.0 + i

    m = emitir([epoca], capsys)

    assert m["val_in_pearsonr"] == pytest.approx(0.70)
    assert m["val_ood_pearsonr"] == pytest.approx(0.60)

import json
import types

import torch

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
    assert m["val_in_pearsonr"] == 0.73
    assert m["val_ood_pearsonr"] == 0.57
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

    assert m["val_ood_pearsonr"] == 0.57
    assert "val_in_pearsonr" not in m

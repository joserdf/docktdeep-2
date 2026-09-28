"""`_load_best_score`: o caminho de resume depois de um pause do broker.

eec2a67 apagou a funcao junto com a grade de sopas e deixou a chamada em
run(); toda tarefa retomada treinava ate o fim e morria com NameError antes do
test. O `import *` de train.py cega o pyflakes para nomes indefinidos, entao so
um teste pega a regressao.
"""
import torch

from train import _load_best_score


def test_formato_dict_lightning2(tmp_path):
    p = tmp_path / "pause.ckpt"
    torch.save({"callbacks": {"ModelCheckpoint{...}": {"best_model_score": torch.tensor(0.61)},
                              "EarlyStopping{...}": {"wait_count": 3}}}, p)
    assert abs(_load_best_score(str(p)) - 0.61) < 1e-6


def test_formato_lista_antigo(tmp_path):
    p = tmp_path / "pause.ckpt"
    torch.save({"callbacks": [{"best_model_score": 0.5}]}, p)
    assert _load_best_score(str(p)) == 0.5


def test_sem_score_ou_ilegivel(tmp_path):
    p = tmp_path / "pause.ckpt"
    torch.save({"callbacks": {}}, p)
    assert _load_best_score(str(p)) is None
    ruim = tmp_path / "ruim.ckpt"
    ruim.write_bytes(b"nao e checkpoint")
    assert _load_best_score(str(ruim)) is None

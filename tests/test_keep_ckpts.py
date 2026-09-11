"""`--keep-ckpts`: o que sobra em disco quando o run termina.

O defeito que motivou o flag nao foi um numero errado, foi ENOSPC: 201 tarefas
da campanha 4 morreram porque nada apagava os K checkpoints que `--soup` deixa
por trial. O teste cobre os tres niveis e o caso em que o melhor nao existe.
"""
import types

import pytest

from train import podar_ckpts


def cenario(tmp_path, n=4):
    paths = []
    for i in range(n):
        p = tmp_path / f"epoch={i}-step={i * 8}.ckpt"
        p.write_bytes(b"x" * 1024)
        paths.append(p)
    (tmp_path / "last.ckpt").write_bytes(b"x" * 1024)
    # um arquivo que NAO e checkpoint: a poda nao pode levar o diretorio junto
    (tmp_path / "metrics.json").write_text("{}")
    cb = types.SimpleNamespace(dirpath=str(tmp_path))
    return cb, str(paths[2])


def sobrou(tmp_path):
    return sorted(p.name for p in tmp_path.iterdir())


def test_all_nao_mexe(tmp_path):
    cb, melhor = cenario(tmp_path)
    podar_ckpts(cb, melhor, "all")
    assert len(sobrou(tmp_path)) == 6


def test_best_guarda_so_o_vencedor(tmp_path):
    cb, melhor = cenario(tmp_path)
    podar_ckpts(cb, melhor, "best")
    assert sobrou(tmp_path) == ["epoch=2-step=16.ckpt", "metrics.json"]


def test_none_nao_guarda_ckpt_nenhum(tmp_path):
    cb, melhor = cenario(tmp_path)
    podar_ckpts(cb, melhor, "none")
    assert sobrou(tmp_path) == ["metrics.json"]


def test_sem_melhor_conhecido(tmp_path):
    """`best_model_path` vazio (run que nao salvou): poda tudo, sem estourar."""
    cb, _ = cenario(tmp_path)
    podar_ckpts(cb, "", "best")
    assert sobrou(tmp_path) == ["metrics.json"]


def test_diretorio_inexistente_nao_estoura():
    podar_ckpts(types.SimpleNamespace(dirpath="/nao/existe"), "", "none")
    podar_ckpts(types.SimpleNamespace(dirpath=None), "", "none")

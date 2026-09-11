"""O canal: o que o LoggerComContexto de fato manda para o `track` do Aim.

`test_aim_naming.py` cobre a REGRA; este arquivo cobre a LIGACAO -- que o
logger aplica a regra a cada chave em vez de deixar o AimLogger de fabrica
partir o nome sozinho (que e o que produzia `ood_pearsonr`).

O `experiment` e um gravador, e nao um repo Aim de verdade, porque criar um
repo aqui exige `pkg_resources` (ausente neste venv) e porque o que esta sob
teste e a traducao, nao a camada de armazenamento do Aim. Que o Aim guarda e
devolve tres series sob um nome so, separadas por `stratum`, foi verificado
contra um repo real em 2026-09-11.
"""
import pytest

from train import LoggerComContexto

EPOCA = {
    "val_pearsonr": 0.62, "val_loss": 0.91,
    "val_in_pearsonr": 0.73, "val_in_n": 187.0,
    "val_ood_pearsonr": 0.58, "val_ood_n": 402.0,
    "train_loss": 0.88, "test_casf_pearsonr": 0.70,
    "epoch": 7,
}


class Gravador:
    def __init__(self):
        self.chamadas = []

    def track(self, valor, name, step=None, epoch=None, context=None):
        self.chamadas.append((name, tuple(sorted((context or {}).items())),
                              valor, step, epoch))


@pytest.fixture
def series(monkeypatch):
    gravador = Gravador()
    logger = LoggerComContexto.__new__(LoggerComContexto)
    monkeypatch.setattr(type(logger), "experiment",
                        property(lambda self: gravador))
    logger.log_metrics(EPOCA, step=0)
    return gravador.chamadas


def test_pearsonr_e_um_nome_so(series):
    """Pooled, val_in, val_ood e o estrato do teste: quatro series, um nome."""
    ctxs = {c for n, c, *_ in series if n == "pearsonr"}
    assert ctxs == {
        (("stratum", "all"), ("subset", "val")),
        (("stratum", "in"), ("subset", "val")),
        (("stratum", "ood"), ("subset", "val")),
        (("stratum", "casf"), ("subset", "test")),
    }


def test_nenhum_estrato_sobrou_no_nome(series):
    """O sintoma original: `ood_pearsonr` como nome proprio."""
    assert not [n for n, *_ in series
                if n.startswith(("in_", "ood_", "casf_", "dev_", "val_", "test_"))]


def test_train_convive_sem_estrato_proprio(series):
    nomes = {(n, c) for n, c, *_ in series}
    assert ("loss", (("stratum", "all"), ("subset", "train"))) in nomes
    assert ("loss", (("stratum", "all"), ("subset", "val"))) in nomes


def test_epoch_vira_eixo_e_nao_serie(series):
    """`epoch` e o eixo x de todas as series, nao uma serie."""
    assert "epoch" not in {n for n, *_ in series}
    assert {ep for *_, ep in series} == {7}


def test_valores_chegam_intactos(series):
    """A traducao mexe no nome e no contexto, nunca no numero."""
    v = {(n, c): val for n, c, val, *_ in series}
    assert v[("pearsonr", (("stratum", "ood"), ("subset", "val")))] == 0.58
    assert v[("n", (("stratum", "in"), ("subset", "val")))] == 187.0

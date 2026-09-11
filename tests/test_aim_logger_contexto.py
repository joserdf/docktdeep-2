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
    # internos de treino: componentes da perda e normas de gradiente
    "train_semi": 0.21, "train_rdrop": 0.05,
    "train_yaw_lig": 0.31, "train_yaw_lig_rows": 0.84,
    "train_grad_cnn": 1.7, "train_grad_prot": 0.4,
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


def test_perdas_auxiliares_sob_um_nome(series):
    """Quatro series de `train_*` viram duas quantidades, separadas por eixo."""
    por_nome = {}
    for n, c, *_ in series:
        por_nome.setdefault(n, set()).add(c)
    assert por_nome["aux_loss"] == {
        (("subset", "train"), ("term", "semi")),
        (("subset", "train"), ("term", "rdrop")),
        (("family", "yaware"), ("subset", "train"), ("term", "lig")),
    }
    assert por_nome["aux_rows"] == {
        (("family", "yaware"), ("subset", "train"), ("term", "lig")),
    }


def test_gradientes_sob_um_nome(series):
    """`grad_cnn` e `grad_prot` eram dois nomes para uma norma so."""
    ctxs = {c for n, c, *_ in series if n == "grad_norm"}
    assert ctxs == {(("branch", "cnn"), ("subset", "train")),
                    (("branch", "prot"), ("subset", "train"))}


def test_internos_nao_ganham_estrato(series):
    """Um `stratum: all` num componente da perda sugeriria um `ood` irmao."""
    for n, c, *_ in series:
        if n in ("aux_loss", "aux_rows", "grad_norm"):
            assert "stratum" not in dict(c)

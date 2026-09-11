import types

import train
from train import _avaliar_sopa_final, _receita_final

# Um plato em torno da epoca 50 (scores altos, epocas vizinhas) mais dois
# outliers longe no tempo: e o formato que motivou a janela. A epoca 2 tem
# score COMPETITIVO de proposito -- so a janela a separa, nao o score.
POOL = {f"epoch={e}.ckpt": s for e, s in
        ((50, 0.620), (48, 0.618), (52, 0.617), (45, 0.615), (47, 0.612),
         (10, 0.611), (2, 0.610), (49, 0.589), (60, 0.590), (5, 0.585))}
BEST = "epoch=50.ckpt"


def epocas(paths):
    return sorted(int(p.split("=")[1].split(".")[0]) for p in paths)


def receita(k=5, janela=25, tol=0.01, modo="abs"):
    return epocas(_receita_final(POOL, BEST, k, janela, tol, modo))


def test_janela_separa_bacias():
    """As epocas 2 e 10 tem score competitivo; so a janela as exclui.

    Sem esta guarda um run juntou as epocas 2, 5, 9, 12 e 47 e perdeu 0.1342 de
    val_pearsonr: media de pesos pressupoe mesma bacia, e score alto nao e
    evidencia de bacia comum.
    """
    assert 2 not in receita(k=9)
    assert 10 not in receita(k=9)
    assert {2, 10} <= set(receita(k=9, janela=0))


def test_k_e_tolerancia_sao_limites_independentes():
    """Cada um consegue ser o gargalo sozinho."""
    assert len(receita(k=5, tol=0.30)) == 5          # K limita
    assert receita(k=5, tol=0.001) == [50]           # tolerancia limita


def test_ordem_entre_tolerancia_e_k_nao_importa():
    """Truncar-depois-filtrar == filtrar-depois-truncar.

    As duas operacoes vivem no mesmo ranking de score, entao comutam. O codigo
    aplica na ordem escrita so por clareza; se alguem inverter, o resultado tem
    de ser identico.
    """
    ordenado = sorted(POOL.items(), key=lambda kv: -kv[1])
    dentro = [(p, v) for p, v in ordenado if abs(
        int(p.split("=")[1].split(".")[0]) - 50) <= 25]
    limite = dentro[0][1] - 0.008
    truncar_depois = [p for p, v in dentro[:4] if v >= limite]
    filtrar_depois = [p for p, v in dentro if v >= limite][:4]
    assert truncar_depois == filtrar_depois
    assert epocas(truncar_depois) == receita(k=4, tol=0.008)


def test_modo_rel_e_abs_coincidem_na_escala_equivalente():
    """rel 0.05 sobre um top-1 de 0.620 vale abs 0.031, nao abs 0.05.

    Os dois modos nao sao intercambiaveis; medido na grade v3-v6, foi por isso
    que a receita de tolerancia relativa engoliu o pool inteiro.
    """
    assert receita(k=9, tol=0.05, modo="rel") == receita(k=9, tol=0.031)
    # E com uma tolerancia que morde antes da janela, os mesmos "0.01" nos dois
    # modos separam conjuntos diferentes: rel 0.01 vale abs 0.0062.
    assert receita(k=9, tol=0.01, modo="rel") != receita(k=9, tol=0.01)
    assert receita(k=9, tol=0.01, modo="rel") == receita(k=9, tol=0.0062)


def test_melhor_checkpoint_sempre_entra():
    """A janela e centrada nele, e a tolerancia e medida a partir dele.

    Garante que a sopa nunca fica vazia com pool nao-vazio, e que `tol`
    significa sempre distancia ao melhor checkpoint do treino.
    """
    for janela in (0, 1, 25, 1000):
        for tol in (0.0, 0.001, 0.5):
            assert 50 in receita(k=1, janela=janela, tol=tol)


def test_pool_vazio_nao_inventa_selecao():
    assert _receita_final({}, BEST, 5, 25, 0.01, "abs") == []


def rodar(monkeypatch, log, **flags):
    args = types.SimpleNamespace(soup_k=5, soup_window=25, soup_tol=0.01,
                                 soup_tol_mode="abs")
    for k, v in flags.items():
        setattr(args, k, v)
    model = types.SimpleNamespace()
    monkeypatch.setattr(train, "_metricas_da_sopa",
                        lambda *a, **k: dict(log) if log else None)
    _avaliar_sopa_final(None, model, None, POOL, BEST, args)
    return getattr(model, "_topk_avg_metrics", {})


LOG = {"val_pearsonr": 0.6420, "val_loss": 0.95,
       "val_in_pearsonr": 0.7310, "val_in_n": 187.0,
       "val_ood_pearsonr": 0.5804, "val_ood_n": 402.0}


def test_publica_metricas_e_estratos(monkeypatch):
    m = rodar(monkeypatch, LOG)

    assert m["soup_val_pearsonr"] == 0.6420
    assert m["soup_k"] == 5.0
    assert m["soup_pool"] == float(len(POOL))
    assert m["soup_epoch_span"] == 7.0          # 45..52
    assert m["soup_val_in_pearsonr"] == 0.7310


def test_k1_publica_em_vez_de_se_abster(monkeypatch):
    """Diferenca deliberada em relacao a grade v3-v6.

    A receita v6 pulava quando a selecao degenerava para um checkpoint, e a
    coluna saia com buracos -- 17 runs de 19. Quem le nao conseguia separar
    "nao houve sopa" de "a sopa nao ajudou". Aqui o numero sai e `soup_k=1`
    diz o que aconteceu.
    """
    m = rodar(monkeypatch, LOG, soup_tol=0.0)

    assert m["soup_k"] == 1.0
    assert m["soup_epoch_span"] == 0.0
    assert m["soup_val_pearsonr"] == 0.6420


def test_estrato_ausente_nao_vira_zero(monkeypatch):
    m = rodar(monkeypatch, {"val_pearsonr": 0.6420, "val_loss": 0.95})

    assert "soup_val_in_pearsonr" not in m

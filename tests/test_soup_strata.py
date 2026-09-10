import types

import train
from train import _avaliar_sopas

# Pool com 5 checkpoints, scores decrescentes e epocas espalhadas: e o formato
# de `ModelCheckpoint.best_k_models` (caminho -> score monitorado).
POOL = {f"e{e}-epoch={e}.ckpt": s for e, s in
        ((10, 0.63), (12, 0.62), (14, 0.61), (30, 0.60), (48, 0.59))}
BEST = "e10-epoch=10.ckpt"

# O log que `on_validation_epoch_end` produz para a sopa. Numeros distintos do
# pooled de proposito: publicar o pooled no lugar do estrato passaria
# despercebido com valores iguais.
LOG_SOPA = {"val_pearsonr": 0.6420, "val_loss": 0.95,
            "val_in_pearsonr": 0.7310, "val_in_n": 187.0,
            "val_ood_pearsonr": 0.5804, "val_ood_n": 402.0}


def rodar(monkeypatch, log, **flags):
    args = types.SimpleNamespace(topk_avg=3, topk_window=0, topk_rel_tol=0.05,
                                 topk_abs_tol=0.01)
    for k, v in flags.items():
        setattr(args, k, v)
    model = types.SimpleNamespace()
    monkeypatch.setattr(train, "_metricas_da_sopa",
                        lambda *a, **k: dict(log) if log else None)
    _avaliar_sopas(None, model, None, POOL, BEST, args)
    return getattr(model, "_topk_avg_metrics", {})


def test_sopa_publica_os_estratos(monkeypatch):
    """Sem isto, ligar a sopa nao teria como mover um objetivo em val_in/val_ood.

    O eixo `avg_mode=soup` da campanha 4 le `soup_<receita>_val_in_pearsonr`; se
    a chave nao existisse, o nivel seria um gene neutro -- indistinguivel de
    `none` para o sampler, e consumindo populacao para nao medir nada.
    """
    m = rodar(monkeypatch, LOG_SOPA)

    assert m["soup_v3_val_pearsonr"] == 0.6420
    assert m["soup_v3_val_in_pearsonr"] == 0.7310
    assert m["soup_v3_val_ood_pearsonr"] == 0.5804
    assert m["soup_v3_val_in_n"] == 187.0


def test_alias_antigo_cobre_os_estratos(monkeypatch):
    """A v3 continua publicada com os nomes `topk_avg_*`.

    As ferramentas de analise ja construidas leem esses nomes, e renomea-las
    quebraria a comparacao com os bracos topk-test, v2 e v3 ja medidos. O alias
    tem que cobrir os estratos tambem, senao ele passa a ser uma vista PARCIAL
    da mesma receita -- e quem le pelo nome antigo veria o pooled da sopa ao
    lado de estrato nenhum.
    """
    m = rodar(monkeypatch, LOG_SOPA)

    assert m["topk_avg_val_pearsonr"] == m["soup_v3_val_pearsonr"]
    assert m["topk_avg_val_in_pearsonr"] == m["soup_v3_val_in_pearsonr"]
    assert m["topk_avg_val_ood_pearsonr"] == m["soup_v3_val_ood_pearsonr"]


def test_estrato_ausente_nao_vira_zero(monkeypatch):
    """Split sem coluna de estrato: a chave nao sai, e nao sai valendo 0.0.

    Mesma regra do `emit_metrics_line`. Um zero aqui entraria na fronteira de
    Pareto como um fato medido -- e `val_ood = 0.0` domina tudo em `sd`.
    """
    m = rodar(monkeypatch, {"val_pearsonr": 0.6420, "val_loss": 0.95})

    assert m["soup_v3_val_pearsonr"] == 0.6420
    assert "soup_v3_val_in_pearsonr" not in m
    assert "topk_avg_val_in_pearsonr" not in m


def test_pool_pequeno_nao_publica_nada(monkeypatch):
    """Menos de 2 checkpoints: nao ha o que mediar, e nenhuma chave e inventada."""
    args = types.SimpleNamespace(topk_avg=3, topk_window=0, topk_rel_tol=0.05,
                                 topk_abs_tol=0.01)
    model = types.SimpleNamespace()
    monkeypatch.setattr(train, "_metricas_da_sopa", lambda *a, **k: dict(LOG_SOPA))
    _avaliar_sopas(None, model, None, {BEST: 0.63}, BEST, args)

    assert not hasattr(model, "_topk_avg_metrics")

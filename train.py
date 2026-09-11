import argparse
import glob
import itertools
import json
import os
import re
import subprocess
import sys

import aim
import docktgrid
import dotenv
import lightning.pytorch as pl
import torch
from aim.pytorch_lightning import AimLogger
from docktgrid.view import BasicView, VolumeView
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn

from src.docktgrid_2.NewViewComplex import NewViewComplex
from src.docktgrid_2.NewViewLigProt import NewViewLigProt
from src.docktgrid_2.BasicViewComplex import BasicViewComplex
from src.docktgrid_2.BasicViewLigProt import BasicViewLigProt
from src.docktgrid_2.VolumeViewComplex import VolumeViewComplex
from src.docktgrid_2.VolumeViewLigProt import VolumeViewLigProt
from src.docktgrid_2.CustomVoxelGrid import CustomVoxelGrid

from src.docktdeep.dataset import PDBbind
from src.docktdeep.models import *
from src.docktdeep.transforms import MolecularDropout, Random90DegreesRotation


def run(args):
    if args.sema and args.ema:
        sys.exit("--sema e --ema sao exclusivos: um devolve a media ao modelo "
                 "otimizado, o outro nao.")
    if args.soup and args.topk_avg > 1:
        sys.exit("--soup e --topk-avg sao exclusivos: o primeiro e a receita "
                 "final, o segundo e a grade experimental v3-v6 que a "
                 "originou. Rodar os dois deixaria ambiguo qual conjunto de "
                 "pesos as metricas soup_* descrevem.")
    torch.set_float32_matmul_precision("medium")
    dotenv.load_dotenv()

    pl.seed_everything(args.seed)

    callbacks = configure_callbacks(
        args.early_stop_patience, args.val_monitor,
        early_stop_warmup=args.early_stop_warmup,
        sema=args.sema, sema_decay=args.sema_decay,
        ema=args.ema, ema_decay=args.ema_decay, topk_avg=args.topk_avg,
        topk_pool=args.topk_pool, soup=args.soup, soup_k=args.soup_k,
        soup_pool=args.soup_pool)
    logger = configure_logger(args)
    track_files(logger)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        precision=args.precision,
        detect_anomaly=args.detect_anomaly,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        callbacks=callbacks,
        logger=logger,
    )

    transforms = []
    if args.random_rotation:
        transforms.append(docktgrid.transforms.RandomRotation())
    if args.rotation_90_degrees:
        transforms.append(Random90DegreesRotation())

    voxel_grid = configure_voxel_grid(args)
    model = eval(args.model)(input_size=voxel_grid.shape, **vars(args))
    data_module = PDBbind(voxel_grid=voxel_grid, transforms=transforms, **vars(args))

    # ckpt_path: resume de task pausada pelo broker (checkpoint enviado pelo
    # worker anterior). Sem o argumento, comportamento identico ao original.
    trainer.fit(model, datamodule=data_module, ckpt_path=args.ckpt_path)

    ckpt_cb = next(c for c in trainer.callbacks if isinstance(c, ModelCheckpoint))
    # Snapshot dos K melhores AGORA, antes do trainer.test abaixo. O test roda
    # com ckpt_path="best", e carregar um checkpoint restaura o estado dos
    # callbacks gravado DENTRO dele -- o que rebobina `best_k_models` para o
    # que era quando aquela epoca foi salva. Medido: `topk_avg_k` saia igual a
    # posicao cronologica do melhor checkpoint (2 ou 3), nunca 5.
    # O pool inteiro, COM os scores: cada receita de sopa filtra e escolhe de
    # um jeito, entao filtrar aqui perderia informacao que uma delas precisa.
    topk_pool = {p: float(v) for p, v in ckpt_cb.best_k_models.items()
                 if os.path.exists(p)}
    topk_best = ckpt_cb.best_model_path
    for tag, path in (("last", ckpt_cb.last_model_path), ("best", ckpt_cb.best_model_path)):
        if path:
            # Contrato do worker do broker (agent.py::_parse_ckpt_path): a pausa
            # localiza os checkpoints por essas linhas de stdout.
            print(f"[ckpt] {tag}={path}", flush=True)

    # Report on the held-out fold with the checkpoint that ModelCheckpoint
    # selected on the (cluster-disjoint) validation slice. Skipped under
    # --merge-val-test, where the datamodule aliases validation to test and
    # testing would only re-report the selection metric.
    # Em resume, o "best" do 2o half pode ser pior que o best pre-pause
    # (--prior-best-path): testa o vencedor global, selecionado pelo val_pearsonr.
    prior_score = None
    if args.prior_best_path and os.path.exists(args.prior_best_path):
        prior_score = _load_best_score(args.prior_best_path)
        if prior_score is not None:
            model._prior_best_pearsonr = prior_score
    if not args.merge_val_test:
        winner = "best"
        if prior_score is not None and (ckpt_cb.best_model_score is None
                                        or prior_score > ckpt_cb.best_model_score):
            winner = args.prior_best_path
            print(f"[ckpt] winner=prior-best (val_pearsonr={prior_score:.4f})", flush=True)
        trainer.test(model, datamodule=data_module, ckpt_path=winner)
        # Congela as metricas de test AGORA. `trainer.callback_metrics` e
        # zerado no inicio de cada estagio, e _avaliar_sopas abaixo roda um
        # trainer.validate() -- que apagava todo `test_*` do dicionario. Efeito
        # medido no refit: os runs com --topk-avg entregaram sopa OU
        # test_pearsonr, nunca os dois, conforme a sopa tivesse rodado ou nao.
        model._test_metrics = {k: float(v) for k, v in
                               trainer.callback_metrics.items()
                               if k.startswith("test_")}

    if args.soup:
        _avaliar_sopa_final(trainer, model, data_module, topk_pool, topk_best, args)
    elif args.topk_avg > 1:
        _avaliar_sopas(trainer, model, data_module, topk_pool, topk_best, args)

    emit_metrics_line(trainer, model, args)

    return trainer


def _epoca_do_ckpt(path: str) -> int | None:
    m = re.search(r"epoch=(\d+)", os.path.basename(str(path)))
    return int(m.group(1)) if m else None


def _na_janela(pool: dict, best_path: str, janela: int) -> dict:
    """Mantem so os checkpoints a menos de `janela` epocas do melhor.

    Media de pesos pressupoe que os pontos estejam na MESMA bacia. Medido no
    topk-test-v2: um run cuja sopa juntou as epocas 2, 5, 9, 12 e 47 perdeu
    0.1342 de val_pearsonr, enquanto a mediana dos outros 14 runs era +0.0050.
    A janela e o unico parametro que separa "media de pesos vizinhos" de
    "media entre dois modelos diferentes".
    """
    if janela <= 0:
        return dict(pool)
    best_ep = _epoca_do_ckpt(best_path)
    if best_ep is None:
        return dict(pool)
    dentro = {p: v for p, v in pool.items()
              if (e := _epoca_do_ckpt(p)) is not None and abs(e - best_ep) <= janela}
    return dentro or dict(pool)


def _receita_final(pool: dict, best_path: str, k: int, janela: int,
                   tol: float, modo: str) -> list[str]:
    """A receita do estudo: janela, tolerancia ao top-1, e no maximo K.

    Tres restricoes, nesta leitura: dos checkpoints a menos de `janela` epocas
    do melhor, ficam os que estao a menos de `tol` do top-1, e desses os `k`
    melhores.

    A ordem entre a tolerancia e o corte em K nao importa: as duas operam sobre
    o mesmo ranking de score, entao filtrar-depois-truncar e truncar-depois-
    filtrar produzem o mesmo conjunto. Sao aplicadas na ordem escrita so por
    clareza.

    O top-1 da janela e sempre o top-1 global: `_na_janela` centra a janela no
    proprio melhor checkpoint, que por isso nunca e excluido. Isso garante que
    a sopa nunca fica vazia e que `tol` significa sempre "distancia ao melhor
    checkpoint do treino", nao "ao melhor da vizinhanca".

    `modo` e 'abs' (tol em unidades da metrica) ou 'rel' (fracao do top-1).
    Medido na grade v3-v6: para um val_pearsonr que vive perto de 0.62, um
    rel de 0.05 vale 0.031 -- tres vezes mais frouxo que um abs de 0.01. Os
    dois nao sao intercambiaveis, e e por isso que o modo e explicito.
    """
    dentro = _na_janela(pool, best_path, janela)
    ordenado = sorted(dentro.items(), key=lambda kv: -kv[1])
    if not ordenado:
        return []
    top1 = ordenado[0][1]
    limite = top1 - (abs(top1) * tol if modo == "rel" else tol)
    return [p for p, v in ordenado[:k] if v >= limite]


def _avaliar_sopa_final(trainer, model, data_module, pool: dict, best_path: str,
                        args) -> None:
    """Avalia a receita final e publica as metricas `soup_*`.

    Diferenca deliberada em relacao a grade v3-v6: aqui NAO se pula quando a
    selecao degenera para um unico checkpoint. A v6 se abstinha nesses casos e
    o resultado era uma coluna com buracos -- 17 runs de 19 --, o que obriga
    quem le a tratar "sem sopa" e "sopa que nao ajudou" como a mesma coisa.
    Com k=1 a sopa e o proprio melhor checkpoint; o numero e valido, e
    `soup_k` diz exatamente o que aconteceu.
    """
    paths = sorted(set(_receita_final(pool, best_path, args.soup_k,
                                      args.soup_window, args.soup_tol,
                                      args.soup_tol_mode)))
    if not paths:
        print("[sopa] pool vazio; pulando", flush=True)
        return
    r = _metricas_da_sopa(trainer, model, data_module, paths)
    if r is None:
        print("[sopa] validacao nao produziu log; pulando", flush=True)
        return
    eps = [e for e in map(_epoca_do_ckpt, paths) if e is not None]
    metricas = {
        "soup_val_pearsonr": r["val_pearsonr"],
        "soup_val_loss": r["val_loss"],
        "soup_k": float(len(paths)),
        "soup_epoch_span": float(max(eps) - min(eps)) if eps else -1.0,
        "soup_pool": float(len(pool)),
    }
    for st in ("val_in", "val_ood"):
        for chave in ("pearsonr", "n"):
            if f"{st}_{chave}" in r:
                metricas[f"soup_{st}_{chave}"] = r[f"{st}_{chave}"]
    # Mesmo canal de transporte da grade experimental: `emit_metrics_line` le
    # este atributo. So o conteudo muda.
    model._topk_avg_metrics = metricas
    print(f"[sopa] final: k={len(paths)} epocas={sorted(eps)} "
          f"span={metricas['soup_epoch_span']:.0f} "
          f"val_pearsonr={r['val_pearsonr']:.4f}", flush=True)


def _receitas(pool: dict, best_path: str, args) -> dict:
    """As receitas de sopa, todas sobre o MESMO treino e o mesmo pool.

    Elas se distinguem por DUAS decisoes independentes -- o que entra na
    janela, e como escolher dentro dela:

      v3  os `topk_avg` melhores do pool inteiro, depois filtrados pela janela.
          Reproduz o comportamento anterior; quando o pool tem exatamente
          `topk_avg` checkpoints, e literalmente ele.
      v4  a janela primeiro, os `topk_avg` melhores depois. Com um pool maior
          que `topk_avg`, ainda consegue juntar K vizinhos quando a v3 ficaria
          com menos.
      v5  na janela, todos os que estao a menos de `topk_rel_tol` do top-1, em
          termos RELATIVOS. O tamanho da sopa passa a ser lido dos dados: um
          plato largo junta muitos, um pico isolado junta poucos.
      v6  igual a v5, com tolerancia ABSOLUTA. Para um score que vive perto de
          0.62, uma fracao e um valor fixo nao sao a mesma coisa, e qual dos
          dois descreve melhor "epocas equivalentes" e o que se quer medir.
    """
    ordenado_global = sorted(pool.items(), key=lambda kv: -kv[1])
    dentro = _na_janela(pool, best_path, args.topk_window)
    ordenado = sorted(dentro.items(), key=lambda kv: -kv[1])
    receitas = {}

    v3 = _na_janela(dict(ordenado_global[:args.topk_avg]), best_path,
                    args.topk_window)
    receitas["v3"] = list(v3)
    receitas["v4"] = [p for p, _ in ordenado[:args.topk_avg]]
    if ordenado:
        top1 = ordenado[0][1]
        receitas["v5"] = [p for p, v in ordenado
                          if v >= top1 - abs(top1) * args.topk_rel_tol]
        receitas["v6"] = [p for p, v in ordenado if v >= top1 - args.topk_abs_tol]
    return receitas


def _metricas_da_sopa(trainer, model, data_module, paths: list[str]):
    """Media os pesos, valida, e devolve o log INTEIRO da validacao. None se nao der.

    Devolvia so `(val_pearsonr, val_loss)`. Agora devolve o dicionario todo
    porque `on_validation_epoch_end` poe os estratos (`val_in_*`, `val_ood_*`)
    na mesma entrada, e uma busca multiobjetivo em `val_in` x `val_ood` precisa
    deles para a sopa tambem -- sem isso, ligar a sopa nao teria como mover o
    objetivo, e o eixo seria um gene neutro consumindo populacao.

    A entrada que o trainer.validate acrescenta a validation_logs e removida em
    seguida: sem isso a sopa entraria no argmax que define best_val_pearsonr, e
    o run passaria a reportar como "melhor epoca" algo que nao e uma epoca.
    """
    model.load_state_dict(average_state_dicts(paths))
    logs = getattr(model, "validation_logs", [])
    n_antes = len(logs)
    trainer.validate(model, datamodule=data_module, verbose=False)
    extras = logs[n_antes:]
    del logs[n_antes:]
    if not extras:
        return None
    return {k: float(v) for k, v in extras[-1].items()}


def _avaliar_sopas(trainer, model, data_module, pool: dict, best_path: str,
                   args) -> None:
    """Avalia todas as receitas no mesmo run, uma validacao por conjunto DISTINTO.

    Receitas diferentes frequentemente selecionam o mesmo conjunto de
    checkpoints; validar de novo custaria uma passada inteira pela validacao
    para reproduzir um numero ja conhecido. O cache por conjunto e o que torna
    viavel medir quatro receitas pelo preco de um treino.
    """
    if len(pool) < 2:
        print(f"[sopa] pool com {len(pool)} checkpoint(s); pulando", flush=True)
        return
    receitas = _receitas(pool, best_path, args)
    metricas, cache = {}, {}
    for nome, paths in receitas.items():
        paths = sorted(set(paths))
        if len(paths) < 2:
            print(f"[sopa] {nome}: so {len(paths)} checkpoint(s) elegivel(is); "
                  "pulando", flush=True)
            continue
        chave = tuple(paths)
        if chave not in cache:
            cache[chave] = _metricas_da_sopa(trainer, model, data_module, paths)
        r = cache[chave]
        if r is None:
            print(f"[sopa] {nome}: validacao nao produziu log; pulando", flush=True)
            continue
        pearson, loss = r["val_pearsonr"], r["val_loss"]
        eps = [e for e in map(_epoca_do_ckpt, paths) if e is not None]
        metricas[f"soup_{nome}_val_pearsonr"] = pearson
        metricas[f"soup_{nome}_val_loss"] = loss
        # Estratos da sopa, quando o split os tem. Mesma regra do
        # `emit_metrics_line`: ausencia nao vira zero, a chave simplesmente nao
        # sai -- um estrato com menos de MIN_STRATUM_N pontos nao e logado, e um
        # split sem coluna de estrato nao produz nenhum.
        for st in ("val_in", "val_ood"):
            for k in ("pearsonr", "n"):
                if f"{st}_{k}" in r:
                    metricas[f"soup_{nome}_{st}_{k}"] = r[f"{st}_{k}"]
        metricas[f"soup_{nome}_k"] = float(len(paths))
        metricas[f"soup_{nome}_epoch_span"] = float(max(eps) - min(eps)) if eps else -1.0
        print(f"[sopa] {nome}: k={len(paths)} epocas={sorted(eps)} "
              f"val_pearsonr={pearson:.4f}", flush=True)

    # A v3 continua publicada com os nomes antigos: as ferramentas de analise
    # ja construidas leem `topk_avg_*`, e renomea-las quebraria a comparacao
    # com os bracos topk-test, v2 e v3 ja medidos.
    for sufixo in ("val_pearsonr", "val_loss", "k", "epoch_span",
                   "val_in_pearsonr", "val_in_n", "val_ood_pearsonr", "val_ood_n"):
        if f"soup_v3_{sufixo}" in metricas:
            alvo = "topk_avg_" + ("epoch_span" if sufixo == "epoch_span" else sufixo)
            metricas[alvo] = metricas[f"soup_v3_{sufixo}"]
    metricas["soup_pool"] = float(len(pool))
    model._topk_avg_metrics = metricas


def _load_best_score(ckpt_path: str):
    """val_pearsonr do melhor checkpoint lido do estado do ModelCheckpoint
    gravado no arquivo (ckpt['callbacks']). None se nao for possivel ler.

    Em Lightning >= 2.x, ckpt['callbacks'] e um dict {repr(callback): state};
    em versoes antigas, uma lista de states. Os dois formatos sao aceitos."""
    try:
        cbs = torch.load(ckpt_path, map_location="cpu",
                         weights_only=False).get("callbacks")
        states = list(cbs.values()) if isinstance(cbs, dict) else (cbs or [])
        for cb_state in states:
            if isinstance(cb_state, dict) and cb_state.get("best_model_score") is not None:
                return float(cb_state["best_model_score"])
    except Exception:
        return None
    return None


def emit_metrics_line(trainer, model, args) -> None:
    """Imprime a linha JSON de metricas que o worker do broker consome.

    Contrato de worker/agent.py::_parse_metrics: uma unica linha em stdout,
    objeto JSON com a chave 'metrics' mapeando para um dict de numeros. Sem
    ela o broker grava metrics_json vazio e os resultados so existem no Aim.
    """
    logs = getattr(model, "validation_logs", [])
    if not logs:
        return

    # mesmos criterios do on_train_end do modelo, para a linha bater com o Aim
    best_pearsonr = max(logs, key=lambda x: x["val_pearsonr"])
    best_loss = min(logs, key=lambda x: x["val_loss"])
    best_mae = min(logs, key=lambda x: x["val_mae"])
    # resume: o melhor val_pearsonr pode ter sido alcancado no 1o half (pre-pause);
    # model._prior_best_pearsonr e preenchido em run() quando --prior-best-path existe
    prior = getattr(model, "_prior_best_pearsonr", None)
    if prior is not None:
        best_pearsonr_val = max(float(best_pearsonr["val_pearsonr"]), prior)
    else:
        best_pearsonr_val = float(best_pearsonr["val_pearsonr"])
    metrics = {
        "best_val_pearsonr": best_pearsonr_val,
        # o objetivo da busca e um maximo sobre epocas: quanto mais tempo o trial
        # treina, mais sorteios ele tem. Publicar a ultima epoca ao lado torna
        # esse vies mensuravel em vez de suposto.
        "final_val_pearsonr": float(logs[-1]["val_pearsonr"]),
        "best_val_loss": float(best_loss["val_loss"]),
        "best_val_mae": float(best_mae["val_mae"]),
        "val_mae_at_best_loss": float(best_loss["val_mae"]),
        "epochs": trainer.current_epoch,
    }
    # Estratos da validacao, da MESMA epoca que definiu `best_val_pearsonr`.
    #
    # Ate aqui eles so existiam no Aim, e `trials_table.py` reconstruia a epoca
    # certa de la para poder imprimi-los -- dezenas de segundos por chamada, e
    # duas fontes para o mesmo fato. Uma busca multiobjetivo em `val_in` x
    # `val_ood` precisa deles DENTRO do laco, a cada trial, entao o custo deixa
    # de caber e a segunda fonte deixa de ser aceitavel.
    #
    # Sao lidos de `best_pearsonr`, e nao de `callback_metrics`, exatamente para
    # que a epoca seja a mesma: `callback_metrics` guarda a ULTIMA epoca, e
    # publicar o estrato da ultima ao lado do pooled da melhor produziria duas
    # procedencias dentro do mesmo JSON -- o defeito que tirou o
    # `--eval-test-per-epoch` do template da campanha 3.
    #
    # Ausencia nao e erro: um estrato com menos de MIN_STRATUM_N pontos nao e
    # logado, e um split sem coluna de estrato nao produz nenhum. Quem consome
    # decide o que fazer com a falta, em vez de receber um zero silencioso.
    for st in ("val_in", "val_ood"):
        for k in ("pearsonr", "n"):
            v = best_pearsonr.get(f"{st}_{k}")
            if v is not None:
                metrics[f"{st}_{k}"] = float(v)

    # `_test_metrics` e o snapshot tirado logo apos o trainer.test em run();
    # o fallback cobre os caminhos que nao testam (--merge-val-test).
    testes = getattr(model, "_test_metrics", None)
    if testes is None:
        testes = {k: float(v) for k, v in trainer.callback_metrics.items()
                  if k.startswith("test_")}
    metrics.update(testes)

    # media dos K melhores checkpoints (--topk-avg), quando houver
    metrics.update(getattr(model, "_topk_avg_metrics", {}))

    print(json.dumps({"experiment": args.experiment, "seed": args.seed,
                      "metrics": metrics}), flush=True)


def configure_voxel_grid(args):
    views = [eval(v)() for v in args.view]

    return CustomVoxelGrid(
        vox_size=args.vox_size,
        box_dims=args.box_dims,
        views=views,
        occupancy=args.occupancy,
        device=args.voxel_device,
    )


def configure_logger(args):
    logger = AimLogger(
        repo=os.environ.get("AIM_REPO") if args.remote else None,
        experiment=args.experiment,
        log_system_params=False,
    )

    # AimLogger.finalize() only calls run.close(); it never calls
    # report_successful_finish(), so the run is never marked "finished" and its
    # metrics are not queryable via the SDK. Patch the instance finalize to
    # report a successful finish (blocking until flushed) before closing.
    _orig_finalize = logger.finalize

    def _finalize(self, status: str = "") -> None:
        run = getattr(self, "_run", None)
        if run is not None and status == "success":
            try:
                run.report_successful_finish(block=True)
            except Exception:
                pass
        _orig_finalize(status)

    logger.finalize = _finalize.__get__(logger, AimLogger)
    return logger


class WarmupEarlyStopping(EarlyStopping):
    """EarlyStopping que so comeca a contar a paciencia depois de um warmup.

    O EarlyStopping padrao comeca a contar desde a epoca 0. Com curvas de
    validacao ruidosas o pico costuma aparecer cedo e o patience pode cortar o
    treino antes de o modelo amadurecer. `warmup_epochs` adia o inicio da
    contagem: antes dele o callback apenas observa e nao incrementa a
    paciencia nem atualiza o best (por isso o ModelCheckpoint, que monitora a
    mesma metrica, continua selecionando o argmax de sempre — este callback so
    muda *quando* o treino para, nao *o que* e guardado).
    """

    def __init__(self, warmup_epochs: int = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmup_epochs = warmup_epochs
        self._armed = warmup_epochs <= 0

    def on_validation_end(self, trainer, pl_module):
        if not self._armed and trainer.current_epoch >= self.warmup_epochs:
            # Armado: o best deve comecar a partir do valor da epoca de warmup,
            # nao do epoca 0 — senao o patience ja chega "queimado".
            self._armed = True
            logs = trainer.callback_metrics
            if self.monitor in logs:
                self.best_score = logs[self.monitor].squeeze().clone()
                self.wait_count = 0
        if not self._armed:
            return
        super().on_validation_end(trainer, pl_module)


class SwitchEMA(pl.Callback):
    """SEMA: Switch EMA (arXiv:2402.09240).

    EMA classico mantem uma copia suavizada dos pesos (a media) e usa-a so
    para validar/salvar; o modelo otimizado segue em paralelo e nunca recebe a
    media. SEMA inverte isso: a cada epoca a media EMA e *devolvida* ao modelo
    otimizado (o switch), de modo que a proxima epoca de otimizacao parte do
    ponto suavizado. Isso combina a regularizacao implicita do EMA (picos de
    ruido de validacao sao amortecidos, o checkpoint selecionado nao fica num
    pico espurio) com a convergencia do gradiente.

    Implementacao: usamos o `AveragedModel` do torch (com `get_ema_avg_fn`,
    disponivel em qualquer versao de Lightning) e fazemos o switch no ciclo:
      - `on_train_batch_end`: atualiza a media EMA com os pesos otimizados. O
        decay e uma constante POR PASSO; atualizar uma vez por epoca deixava a
        media presa nos pesos iniciais (0.999^50 ~ 0.95 depois de 50 epocas).
      - `on_train_epoch_end`: copia a media DE VOLTA para o modelo otimizado (o
        switch). O proximo epoch otimiza a partir do ponto suavizado.
      - nao sobrescrevemos validacao: apos o switch o modelo atual ja E a
        media, entao a validacao ja mede a media e o ModelCheckpoint (que
        monitora a mesma metrica) salva o checkpoint suavizado — exatamente o
        objetivo.
    """

    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self._average_model = None

    def setup(self, trainer, pl_module, stage):
        if stage == "fit":
            self._average_model = AveragedModel(
                model=pl_module, device=next(pl_module.parameters()).device,
                use_buffers=True, avg_fn=get_ema_avg_fn(decay=self.decay))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # incorpora os pesos otimizados na media EMA a cada PASSO: `decay` e
        # uma constante por passo, nao por epoca.
        if self._average_model is not None:
            self._average_model.update_parameters(pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if self._average_model is None:
            return
        # switch: o modelo otimizado parte da media suavizada (uma vez por epoca)
        self._copy_average_to_current(pl_module)

    def on_train_end(self, trainer, pl_module):
        if self._average_model is not None:
            self._copy_average_to_current(pl_module)

    def _copy_average_to_current(self, pl_module):
        avg_params = itertools.chain(self._average_model.module.parameters(),
                                     self._average_model.module.buffers())
        cur_params = itertools.chain(pl_module.parameters(), pl_module.buffers())
        for ap, cp in zip(avg_params, cur_params):
            cp.data.copy_(ap.data)


def _params_and_buffers(module):
    """Parametros e buffers na mesma ordem, para copiar um modelo no outro."""
    return itertools.chain(module.parameters(), module.buffers())


class ClassicEMA(pl.Callback):
    """EMA classico: a media suavizada e usada SO para validar e salvar.

    Diferenca para o `SwitchEMA`: o modelo otimizado nunca recebe a media. A
    troca acontece apenas em volta da validacao, de modo que `validation_logs`
    e o `ModelCheckpoint` medem a media, enquanto a proxima epoca de treino
    continua do ponto nao suavizado. E a variante em que o EMA e um estimador,
    nao uma intervencao no otimizador.

    A media e atualizada por PASSO (o `decay` e uma constante por passo: o
    horizonte e 1/(1-decay) passos). Com ~32 passos/epoca neste dataset,
    0.99 cobre ~3 epocas.

    A restauracao dos pesos acontece em `on_train_epoch_start`, e nao em
    `on_validation_end`: o Lightning reordena os ModelCheckpoint para o fim da
    lista de callbacks, entao restaurar em `on_validation_end` desfaria a troca
    ANTES de o checkpoint ser gravado — e o arquivo salvo nao seria a media.
    """

    def __init__(self, decay: float = 0.99):
        super().__init__()
        self.decay = decay
        self._average_model = None
        self._stash = None

    def setup(self, trainer, pl_module, stage):
        if stage == "fit":
            self._average_model = AveragedModel(
                model=pl_module, device=next(pl_module.parameters()).device,
                use_buffers=True, avg_fn=get_ema_avg_fn(decay=self.decay))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._average_model is not None:
            self._average_model.update_parameters(pl_module)

    def on_validation_start(self, trainer, pl_module):
        # guarda os pesos otimizados e coloca a media no lugar
        if self._average_model is None or self._stash is not None:
            return
        self._stash = [t.detach().clone() for t in _params_and_buffers(pl_module)]
        for ap, cp in zip(_params_and_buffers(self._average_model.module),
                          _params_and_buffers(pl_module)):
            cp.data.copy_(ap.data)

    def on_train_epoch_start(self, trainer, pl_module):
        # o checkpoint da validacao anterior ja foi gravado; devolve os pesos
        if self._stash is None:
            return
        for saved, cp in zip(self._stash, _params_and_buffers(pl_module)):
            cp.data.copy_(saved)
        self._stash = None


def average_state_dicts(paths: list[str]) -> dict:
    """Media aritmetica dos state_dicts de varios checkpoints (model soup).

    Buffers inteiros (`num_batches_tracked` das BatchNorm) sao acumulados em
    float e devolvidos ao dtype original: a media de um contador nao e um
    contador, mas manter o dtype evita que o load_state_dict recuse a chave.
    """
    acc, dtypes = None, None
    for path in paths:
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        if acc is None:
            dtypes = {k: v.dtype for k, v in sd.items()}
            acc = {k: v.double().clone() for k, v in sd.items()}
        else:
            for k in acc:
                acc[k] += sd[k].double()
    return {k: (v / len(paths)).to(dtypes[k]) for k, v in acc.items()}


def configure_callbacks(early_stop_patience: int = 0, val_monitor: str = "val_pearsonr",
                        early_stop_warmup: int = 0, sema: bool = False,
                        sema_decay: float = 0.999, ema: bool = False,
                        ema_decay: float = 0.99, topk_avg: int = 1,
                        topk_pool: int = 0, soup: bool = False,
                        soup_k: int = 5, soup_pool: int = 0):
    monitor, mode = val_monitor, "max"
    callbacks = [
        # save_last: essencial p/ o pause/migracao do broker — sem ele so o
        # "best" e guardado e o resume voltaria ate o melhor epoch (nao ao ultimo).
        # save_top_k: >1 so com --topk-avg, que precisa dos K melhores em disco.
        # Com --topk-pool o disco guarda MAIS que K: as receitas que filtram
        # por janela antes de escolher precisam de candidatos sobrando, senao
        # o filtro so consegue encolher a sopa, nunca trocar seus membros.
        ModelCheckpoint(monitor=monitor, mode=mode,
                        save_top_k=(max(1, soup_pool or soup_k) if soup
                                    else max(1, topk_pool or topk_avg)),
                        save_last=True),
    ]
    if sema:
        callbacks.append(SwitchEMA(decay=sema_decay))
    if ema:
        callbacks.append(ClassicEMA(decay=ema_decay))
    if early_stop_patience > 0:
        # Early stopping para cortar o rabo sobre-treinado
        callbacks.append(WarmupEarlyStopping(
            warmup_epochs=early_stop_warmup, monitor=monitor, mode=mode,
            patience=early_stop_patience))
    return callbacks


def get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"

def track_files(logger) -> None:
    files = [os.path.abspath(__file__), os.path.abspath("src/docktdeep/dataset.py")]
    files.extend([os.path.abspath(f) for f in glob.glob("src/docktdeep/models/*.py")])
    files.extend(
        [os.path.abspath(f) for f in glob.glob("src/docktdeep/transforms/*.py")]
    )
    for idx, file in enumerate(files):
        with open(file, "r") as f:
            file = aim.Text(f.read())
        logger.experiment.track(file, name=os.path.basename(files[idx]))

def get_parser():
    parser = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # script args
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--experiment", type=str, help="id of the experiment")
    parser.add_argument("--git-hash", type=str, default=get_git_revision_hash())
    parser.add_argument("--cmd", type=str, default=" ".join(sys.argv))

    # trainer args
    trainer_parser = parser.add_argument_group("Trainer args")
    trainer_parser.add_argument("--accelerator", type=str, default="gpu")
    trainer_parser.add_argument("--devices", default=1)
    trainer_parser.add_argument("--max-epochs", type=int, default=1000)
    trainer_parser.add_argument("--detect-anomaly", action="store_true", default=False)
    trainer_parser.add_argument("--gradient-clip-val", type=float, default=5.0)
    trainer_parser.add_argument("--gradient-clip-algorithm", type=str, default="norm")
    # Default preserva o fp32 de todas as campanhas anteriores; quem quiser
    # bf16 pede no comando. Separar a capacidade da mudanca de protocolo evita
    # que o refit em grp_final, ja em submissao, mude de numerica sozinho.
    # Os tres workers sao RTX 4090 (Ada), entao bf16 e nativo neles.
    trainer_parser.add_argument("--precision", type=str, default="32-true",
                                help="Lightning precision. '32-true' is the fp32 of every previous campaign; 'bf16-mixed' roughly halves activation memory (which is what pays for a larger --batch-size) and speeds up the 3D convolutions.")
    trainer_parser.add_argument("--early-stop-patience", type=int, default=50,
                                help="parar apos N epochs sem melhorar val_pearsonr "
                                     "(0 = desligado; default 50 — RESULTS-MEASURED §21)")
    trainer_parser.add_argument("--early-stop-warmup", type=int, default=0,
                                help="nao contar a paciencia do early stopping "
                                     "antes desta epoca (0 = desligado). So muda "
                                     "*quando* o treino para, nao o que o "
                                     "ModelCheckpoint guarda (continua o argmax).")
    trainer_parser.add_argument("--sema", action="store_true",
                                help="SEMA (Switch EMA): media exponencial dos "
                                     "pesos devolvida ao modelo a cada epoca, "
                                     "para suavizar picos de ruido de validacao.")
    trainer_parser.add_argument("--sema-decay", type=float, default=0.999,
                                help="decay da media EMA do --sema.")
    trainer_parser.add_argument("--ema", action="store_true",
                                help="EMA classico: a media suavizada e usada so "
                                     "para validar/salvar; o treino segue nos pesos "
                                     "otimizados. Exclusivo com --sema.")
    trainer_parser.add_argument("--ema-decay", type=float, default=0.99,
                                help="decay da media do --ema, POR PASSO. O horizonte "
                                     "e 1/(1-decay) passos; com ~32 passos/epoca, "
                                     "0.99 cobre ~3 epocas.")
    trainer_parser.add_argument("--soup", action="store_true",
                                help="RECEITA FINAL do estudo: media dos pesos "
                                     "dos ate --soup-k melhores checkpoints que "
                                     "estejam a menos de --soup-window epocas e "
                                     "a menos de --soup-tol do melhor. Publica "
                                     "soup_val_pearsonr ao lado de "
                                     "best_val_pearsonr. Exclusiva com --topk-avg.")
    trainer_parser.add_argument("--soup-k", type=int, default=5,
                                help="tamanho maximo da sopa (default 5)")
    trainer_parser.add_argument("--soup-window", type=int, default=25,
                                help="so mistura checkpoints a menos de N epocas "
                                     "do melhor (0 = sem janela). E a guarda de "
                                     "bacia: sem ela, um run juntou as epocas "
                                     "2, 5, 9, 12 e 47 e perdeu 0.1342.")
    trainer_parser.add_argument("--soup-tol", type=float, default=0.01,
                                help="diferenca maxima para o top-1 (default 0.01)")
    trainer_parser.add_argument("--soup-tol-mode", choices=("abs", "rel"),
                                default="abs",
                                help="--soup-tol em unidades da metrica (abs) ou "
                                     "como fracao do top-1 (rel). Nao sao "
                                     "intercambiaveis: para val_pearsonr ~0.62, "
                                     "rel 0.05 vale abs 0.031.")
    trainer_parser.add_argument("--soup-pool", type=int, default=0,
                                help="quantos checkpoints manter em disco "
                                     "(0 = usa --soup-k). Um pool maior que K da "
                                     "a janela candidatos para repor, em vez de "
                                     "so encolher a sopa.")
    trainer_parser.add_argument("--topk-pool", type=int, default=0,
                                help="quantos checkpoints manter em disco para "
                                     "as receitas de sopa (0 = usa --topk-avg)")
    trainer_parser.add_argument("--topk-rel-tol", type=float, default=0.05,
                                help="receita v5: fracao do top-1 abaixo da "
                                     "qual o checkpoint ainda entra na sopa")
    trainer_parser.add_argument("--topk-abs-tol", type=float, default=0.01,
                                help="receita v6: idem, em valor absoluto")
    trainer_parser.add_argument("--topk-window", type=int, default=0,
                                help="so mistura checkpoints a menos de N "
                                     "epocas do melhor (0 = sem janela)")
    trainer_parser.add_argument("--topk-avg", type=int, default=1,
                                help="media dos pesos dos K melhores checkpoints "
                                     "(model soup). K>1 faz o ModelCheckpoint guardar "
                                     "K arquivos e publica topk_avg_val_pearsonr ao "
                                     "lado de best_val_pearsonr.")
    trainer_parser.add_argument("--val-monitor", type=str, default="val_pearsonr",
                                help="Metric to monitor for ModelCheckpoint and EarlyStopping.")
    # resume (broker pause/migracao): o worker injeta esses argumentos quando
    # claima uma task pausada com checkpoint
    trainer_parser.add_argument("--ckpt-path", type=str, default=None,
                                help="retomar o fit deste checkpoint (caminho absoluto)")
    trainer_parser.add_argument("--prior-best-path", type=str, default=None,
                                help="best checkpoint do 1o half (pre-pause), p/ testar o vencedor global")

    # data args
    parser.add_argument(
        "--dataframe-path",
        type=str,
        default="data/index.csv",
        help="Path to the dataframe CSV file.",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default="data/processed",
        help="Root directory for processed data.",
    )
    parser = PDBbind.add_specific_args(parser)

    # model args
    tmp_args, _ = parser.parse_known_args()
    eval(tmp_args.model).add_specific_args(parser)

    parser.add_argument("--help", "-h", action="help", default=argparse.SUPPRESS)

    return parser

def parse_args():
    parser = get_parser()
    args = parser.parse_args()
    args.hostname = os.uname().nodename
    return args

if __name__ == "__main__":
    # DataLoader workers do CPU-only work (pickle load, voxelization) while the
    # model stays on GPU in the main process. We use 'fork' so the workers
    # inherit the (large, up-to-GB) in-memory dataset via copy-on-write instead
    # of being serialized over a pipe (spawn deadlocks on the full 17k dataset).
    # This is only safe because --voxel-device defaults to 'cpu': docktgrid
    # otherwise allocates the voxel grid on cuda (docktgrid.config.DEVICE), and
    # a forked child cannot re-initialize CUDA. Do not pass --voxel-device cuda
    # together with --num-workers > 0.
    import multiprocessing
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    args = parse_args()
    run(args)

import argparse
import glob
import json
import os
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
    torch.set_float32_matmul_precision("medium")
    dotenv.load_dotenv()

    pl.seed_everything(args.seed)

    callbacks = configure_callbacks(args.early_stop_patience, args.val_monitor)
    logger = configure_logger(args)
    track_files(logger)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
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

    emit_metrics_line(trainer, model, args)

    return trainer


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
    for name, value in trainer.callback_metrics.items():
        if name.startswith("test_"):
            metrics[name] = float(value)

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


def configure_callbacks(early_stop_patience: int = 0, val_monitor: str = "val_pearsonr"):
    monitor, mode = val_monitor, "max"
    callbacks = [
        # save_last: essencial p/ o pause/migracao do broker — sem ele so o
        # "best" e guardado e o resume voltaria ate o melhor epoch (nao ao ultimo).
        ModelCheckpoint(monitor=monitor, mode=mode, save_top_k=1, save_last=True),
    ]
    if early_stop_patience > 0:
        # Early stopping para cortar o rabo sobre-treinado
        callbacks.append(EarlyStopping(monitor=monitor, mode=mode,
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
    trainer_parser.add_argument("--early-stop-patience", type=int, default=50,
                                help="parar apos N epochs sem melhorar val_pearsonr "
                                     "(0 = desligado; default 50 — RESULTS-MEASURED §21)")
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

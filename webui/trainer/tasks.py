from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

from django.conf import settings
from django.utils import timezone

from pipeline import TrainingConfig, run_training_pipeline

from .models import TrainingRun

_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def enqueue_training(run_id: int) -> None:
    _EXECUTOR.submit(_run_training, run_id)


def _run_training(run_id: int) -> None:
    run = TrainingRun.objects.get(pk=run_id)
    _update_run(run, status=TrainingRun.Status.RUNNING, log="Starting data generation...")
    log_lines: List[str] = []

    def _append_log(line: str) -> None:
        log_lines.append(line)
        TrainingRun.objects.filter(pk=run_id).update(log="\n".join(log_lines), updated_at=timezone.now())

    try:
        config_kwargs: Dict[str, int] = {
            "num_confusers": run.config.get("num_confusers", 100),
            "num_positives": run.config.get("num_positives", 100),
            "num_inbetween": run.config.get("num_inbetween", 150),
            "num_plain_negatives": run.config.get("num_plain_negatives", 100),
            "growth_constant": run.config.get("growth_constant", 5),
        }
        # TODO add num_eleven_per, num_tps_random, num_bark_per, num_kokoro_per, num_piper_per as UI args
        config = TrainingConfig(
            key_phrase=run.key_phrase,
            num_piper_per=10,
            num_bark_per=0,
            num_kokoro_per=3,
            num_eleven_per=2,
            num_tps_random=0,
            artifact_dir=Path(settings.BASE_DIR).parent / "artifacts",
            **config_kwargs,
        )

        def _progress(epoch: int, total: int, train_loss: float, val_loss: float, val_f1: float) -> None:
            _append_log(
                f"Epoch {epoch}/{total}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
            )

        result = run_training_pipeline(config, progress_callback=_progress)
        final_log = "\n".join(log_lines + result.log_lines)
        _update_run(
            run,
            status=TrainingRun.Status.COMPLETED,
            log=final_log,
            model_path=str(result.model_path),
            train_loss=result.train_loss,
            val_loss=result.val_loss,
            macro_f1=result.val_f1,
        )
    except Exception as exc:
        _append_log(f"Error: {exc}")
        _update_run(run, status=TrainingRun.Status.FAILED)


def _update_run(run: TrainingRun, **kwargs) -> None:
    for field, value in kwargs.items():
        setattr(run, field, value)
    run.updated_at = timezone.now()
    run.save()


from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

from django.conf import settings
from django.utils import timezone

from pipeline import TrainingConfig, run_training_pipeline
from data_generation.main_audio_generator import plan_generation_with_augmentations

from .models import TrainingRun

_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def enqueue_training(run_id: int) -> None:
    try:
        run = TrainingRun.objects.get(pk=run_id)
    except TrainingRun.DoesNotExist:
        return
    if not run.generation_progress:
        _seed_initial_progress(run_id)
    _EXECUTOR.submit(_run_training, run_id)


def plan_generation(run_id: int) -> None:
    """
    Compute and store generation statistics only. Does NOT launch synthesis or training.
    """
    try:
        run = TrainingRun.objects.get(pk=run_id)
    except TrainingRun.DoesNotExist:
        return

    config_kwargs: Dict[str, int] = {
        "num_confusers": run.config.get("num_confusers", 100),
        "num_positives": run.config.get("num_positives", 100),
        "num_inbetween": run.config.get("num_inbetween", 150),
        "num_plain_negatives": run.config.get("num_plain_negatives", 100),
        "growth_constant": run.config.get("growth_constant", 5),
    }

    generation_state: Dict[str, Dict[str, Any]] = {}

    def _generation_progress(update: Dict[str, Any]) -> None:
        category = update.get("category")
        if not category:
            return
        generation_state[category] = update
        TrainingRun.objects.filter(pk=run_id).update(
            generation_progress=generation_state,
            updated_at=timezone.now(),
        )

    # Use the same per-API defaults as the training pipeline for consistent sample_multiplier.
    plan_generation_with_augmentations(
        key_phrase=run.key_phrase,
        num_confusers=config_kwargs["num_confusers"],
        num_positives=config_kwargs["num_positives"],
        num_inbetween=config_kwargs["num_inbetween"],
        num_plain_negatives=config_kwargs["num_plain_negatives"],
        growth_constant=config_kwargs["growth_constant"],
        num_piper_per=10,
        num_bark_per=0,
        num_kokoro_per=3,
        num_eleven_per=2,
        num_tps_random=0,
        progress_callback=_generation_progress,
    )

    TrainingRun.objects.filter(pk=run_id).update(
        status=TrainingRun.Status.QUEUED,
        log="Planned data counts only (no generation started).",
        updated_at=timezone.now(),
        generation_progress=generation_state,
    )


def _seed_initial_progress(run_id: int) -> None:
    try:
        run = TrainingRun.objects.get(pk=run_id)
    except TrainingRun.DoesNotExist:
        return
    config_kwargs: Dict[str, int] = {
        "num_confusers": run.config.get("num_confusers", 100),
        "num_positives": run.config.get("num_positives", 100),
        "num_inbetween": run.config.get("num_inbetween", 150),
        "num_plain_negatives": run.config.get("num_plain_negatives", 100),
        "growth_constant": run.config.get("growth_constant", 5),
    }
    # Keep in sync with defaults in _run_training
    per_api_total = 10 + 0 + 3 + 2  # piper + bark + kokoro + eleven defaults
    samples_per_phrase = max(1, per_api_total)
    gen_state: Dict[str, Dict[str, Any]] = {}
    for category, phrases in (
        ("positives", config_kwargs["num_positives"]),
        ("confusers", config_kwargs["num_confusers"]),
        ("inbetween", config_kwargs["num_inbetween"]),
        ("plain_negatives", config_kwargs["num_plain_negatives"]),
    ):
        target = max(0, int(phrases)) * samples_per_phrase
        gen_state[category] = {
            "category": category,
            "category_label": category.replace("_", " ").title(),
            "requested_phrases": int(phrases),
            "phrases_available": 0,
            "samples_per_phrase": samples_per_phrase,
            "target_clips": target,
            "db_clips_used": 0,
            "db_clips_available": 0,
            "generated_clips": 0,
            "completed_clips": 0,
            "growth_constant": int(config_kwargs["growth_constant"]),
            "completion_percent": 0,
            "reused_by_api": {},
            "generated_by_api": {},
            "phase": "queued",
        }
    TrainingRun.objects.filter(pk=run_id).update(
        generation_progress=gen_state,
        updated_at=timezone.now(),
    )


def _run_training(run_id: int) -> None:
    run = TrainingRun.objects.get(pk=run_id)
    _update_run(run, status=TrainingRun.Status.RUNNING, log="Starting data generation...")
    log_lines: List[str] = []
    generation_state: Dict[str, Dict[str, Any]] = dict(run.generation_progress or {})

    def _append_log(line: str) -> None:
        log_lines.append(line)
        TrainingRun.objects.filter(pk=run_id).update(log="\n".join(log_lines), updated_at=timezone.now())

    def _generation_progress(update: Dict[str, Any]) -> None:
        category = update.get("category")
        if not category:
            return
        generation_state[category] = update
        TrainingRun.objects.filter(pk=run_id).update(
            generation_progress=generation_state,
            updated_at=timezone.now(),
        )

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

        result = run_training_pipeline(
            config,
            progress_callback=_progress,
            generation_progress_callback=_generation_progress,
        )
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


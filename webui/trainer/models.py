from __future__ import annotations

from django.db import models


class TrainingRun(models.Model):
    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    key_phrase = models.CharField(max_length=255)
    status = models.CharField(max_length=32, choices=Status.choices, default=Status.QUEUED)
    config = models.JSONField(default=dict, blank=True)
    model_path = models.CharField(max_length=512, blank=True)
    log = models.TextField(blank=True)
    generation_progress = models.JSONField(default=dict, blank=True)
    train_loss = models.FloatField(null=True, blank=True)
    val_loss = models.FloatField(null=True, blank=True)
    macro_f1 = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    CATEGORY_ORDER = ("positives", "confusers", "inbetween", "plain_negatives", "tps_random")

    def __str__(self) -> str:  # pragma: no cover - admin helper
        return f"{self.key_phrase} ({self.status})"

    @property
    def is_download_ready(self) -> bool:
        return bool(self.model_path and self.status == self.Status.COMPLETED)

    @property
    def generation_progress_items(self):
        progress = self.generation_progress or {}
        ordered = []
        seen = set()
        for key in self.CATEGORY_ORDER:
            data = progress.get(key)
            if data:
                ordered.append(data)
                seen.add(key)
        for key, data in progress.items():
            if key not in seen:
                ordered.append(data)
        return ordered

    @property
    def generation_progress_totals(self) -> dict:
        items = self.generation_progress_items or []
        totals = {
            "target_clips": 0,
            "db_clips_used": 0,
            "generated_clips": 0,
            "generated_by_api": {},
            "reused_by_api": {},
        }

        def _merge_counts(dst: dict, src: object) -> None:
            if not isinstance(src, dict):
                return
            for key, value in src.items():
                try:
                    inc = int(value)
                except Exception:
                    continue
                dst[key] = int(dst.get(key, 0)) + inc

        for info in items:
            if not isinstance(info, dict):
                continue
            for field in ("target_clips", "db_clips_used", "generated_clips"):
                try:
                    totals[field] += int(info.get(field, 0) or 0)
                except Exception:
                    pass
            _merge_counts(totals["generated_by_api"], info.get("generated_by_api"))
            _merge_counts(totals["reused_by_api"], info.get("reused_by_api"))
        return totals


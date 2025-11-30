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
    train_loss = models.FloatField(null=True, blank=True)
    val_loss = models.FloatField(null=True, blank=True)
    macro_f1 = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:  # pragma: no cover - admin helper
        return f"{self.key_phrase} ({self.status})"

    @property
    def is_download_ready(self) -> bool:
        return bool(self.model_path and self.status == self.Status.COMPLETED)


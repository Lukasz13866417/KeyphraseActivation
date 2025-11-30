from django.contrib import admin

from .models import TrainingRun


@admin.register(TrainingRun)
class TrainingRunAdmin(admin.ModelAdmin):
    list_display = ("id", "key_phrase", "status", "train_loss", "val_loss", "macro_f1", "created_at")
    list_filter = ("status",)
    search_fields = ("key_phrase",)


from __future__ import annotations

from pathlib import Path

from django.http import FileResponse, Http404, HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse

from .forms import TrainingRequestForm
from .models import TrainingRun
from .tasks import plan_generation


def index(request: HttpRequest) -> HttpResponse:
    if request.method == "POST":
        form = TrainingRequestForm(request.POST)
        if form.is_valid():
            config = {
                "num_positives": form.cleaned_data["num_positives"],
                "num_confusers": form.cleaned_data["num_confusers"],
                "num_inbetween": form.cleaned_data["num_inbetween"],
                "num_plain_negatives": form.cleaned_data["num_plain_negatives"],
                "growth_constant": form.cleaned_data["growth_constant"],
            }
            run = TrainingRun.objects.create(
                key_phrase=form.cleaned_data["key_phrase"].strip(),
                config=config,
            )
            plan_generation(run.id)
            return redirect(reverse("trainer:index"))
    else:
        form = TrainingRequestForm()

    runs = TrainingRun.objects.order_by("-created_at")[:10]
    return render(
        request,
        "trainer/index.html",
        {
            "form": form,
            "runs": runs,
        },
    )


def download_model(request: HttpRequest, run_id: int) -> FileResponse:
    run = get_object_or_404(TrainingRun, pk=run_id)
    if not run.is_download_ready:
        raise Http404("Model not ready for download.")
    model_path = Path(run.model_path)
    if not model_path.exists():
        raise Http404("Model file missing.")
    return FileResponse(model_path.open("rb"), as_attachment=True, filename=model_path.name)


def run_progress(request: HttpRequest, run_id: int) -> JsonResponse:
    run = get_object_or_404(TrainingRun, pk=run_id)
    payload = {
        "id": run.id,
        "key_phrase": run.key_phrase,
        "status": run.status,
        "generation_progress": run.generation_progress or {},
        "generation_progress_items": run.generation_progress_items,
        "generation_progress_totals": run.generation_progress_totals,
        "is_download_ready": run.is_download_ready,
        "download_url": reverse("trainer:download", args=[run.id]) if run.is_download_ready else None,
        "train_loss": run.train_loss,
        "val_loss": run.val_loss,
        "macro_f1": run.macro_f1,
        "updated_at": run.updated_at.isoformat() if run.updated_at else None,
    }
    return JsonResponse(payload)


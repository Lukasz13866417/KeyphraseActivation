from __future__ import annotations

from pathlib import Path

from django.http import FileResponse, Http404, HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse

from .forms import TrainingRequestForm
from .models import TrainingRun
from .tasks import enqueue_training


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
            enqueue_training(run.id)
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


from django.urls import path

from . import views

app_name = "trainer"

urlpatterns = [
    path("", views.index, name="index"),
    path("download/<int:run_id>/", views.download_model, name="download"),
]


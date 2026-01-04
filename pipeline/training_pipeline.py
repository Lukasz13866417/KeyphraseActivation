from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data_generation.main_audio_generator import generate_with_augmentations
from model.dataset import AudioDataset, build_dataset_dataframe
from model.model import TinyCRNN
from model.spectrogram_config import SpectrogramConfig

ProgressCallback = Callable[[int, int, float, float, float], None]
GenerationProgressCallback = Callable[[Dict[str, Any]], None]


@dataclass
class TrainingConfig:
    key_phrase: str
    num_confusers: int = 100
    num_positives: int = 100
    num_inbetween: int = 150
    num_plain_negatives: int = 100
    num_piper_per: int = 10
    num_bark_per: int = 0
    num_kokoro_per: int = 3
    num_eleven_per: int = 0
    num_tps_random: int = 0
    growth_constant: int = 5
    batch_size: int = 8
    epochs: int = 1000
    train_split: float = 0.8
    shuffle_seed: int = 42
    artifact_dir: Optional[Path] = None


@dataclass
class TrainingResult:
    key_phrase: str
    train_loss: float
    val_loss: float
    val_f1: float
    train_samples: int
    val_samples: int
    epochs: int
    model_path: Path
    log_lines: List[str] = field(default_factory=list)


def run_training_pipeline(
    config: TrainingConfig,
    *,
    progress_callback: Optional[ProgressCallback] = None,
    generation_progress_callback: Optional[GenerationProgressCallback] = None,
) -> TrainingResult:
    start_time = time.time()
    pos_payload, neg_payload, generation_report = generate_with_augmentations(
        key_phrase=config.key_phrase,
        num_confusers=config.num_confusers,
        num_positives=config.num_positives,
        num_inbetween=config.num_inbetween,
        num_plain_negatives=config.num_plain_negatives,
        num_piper_per=config.num_piper_per,
        num_bark_per=config.num_bark_per,
        num_kokoro_per=config.num_kokoro_per,
        num_eleven_per=config.num_eleven_per,
        num_tps_random=config.num_tps_random,
        growth_constant=config.growth_constant,
        progress_callback=generation_progress_callback,
    )
    if generation_progress_callback and generation_report:
        for info in generation_report.values():
            generation_progress_callback(info)

    df = build_dataset_dataframe(pos_payload, neg_payload)
    if df.empty:
        raise RuntimeError("Data generation yielded zero samples.")

    df = df.sample(frac=1.0, random_state=config.shuffle_seed).reset_index(drop=True)
    split_idx = max(1, int(len(df) * config.train_split))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    if val_df.empty:
        val_df = train_df.copy()

    history, model_path = _train_model(
        train_df,
        val_df,
        batch_size=config.batch_size,
        epochs=config.epochs,
        artifact_dir=config.artifact_dir,
        key_phrase=config.key_phrase,
        progress_callback=progress_callback,
    )

    if not history:
        raise RuntimeError("Training loop produced no history.")

    last_epoch, train_loss, val_loss, val_f1 = history[-1]
    log_lines = [
        f"Training completed in {time.time() - start_time:.1f}s "
        f"| train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
    ]

    return TrainingResult(
        key_phrase=config.key_phrase,
        train_loss=train_loss,
        val_loss=val_loss,
        val_f1=val_f1,
        train_samples=len(train_df),
        val_samples=len(val_df),
        epochs=last_epoch,
        model_path=model_path,
        log_lines=log_lines,
    )


def _train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    batch_size: int,
    epochs: int,
    artifact_dir: Optional[Path],
    key_phrase: str,
    progress_callback: Optional[ProgressCallback],
) -> Tuple[List[Tuple[int, float, float, float]], Path]:
    spect_cfg = SpectrogramConfig()
    model_dir = Path(artifact_dir or (Path.cwd() / "artifacts"))
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = model_dir / "spec_cache"
    train_ds = AudioDataset(train_df, spect_cfg, augment=True, cache_dir=cache_dir, memory_cache_items=512)
    val_ds = AudioDataset(val_df, spect_cfg, augment=False, cache_dir=cache_dir, memory_cache_items=512)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCRNN(in_ch=1, n_mels=spect_cfg.n_mels).to(device)
    criterion = BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)

    history: List[Tuple[int, float, float, float]] = []
    for epoch in range(1, epochs + 1):
        train_loss, train_f1 = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1 = _run_epoch(model, val_loader, criterion, None, device)
        history.append((epoch, train_loss, val_loss, val_f1))
        log_line = (
            f"Epoch {epoch:04d}/{epochs} "
            f"- Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} "
            f"Train F1: {train_f1:.4f} Val F1: {val_f1:.4f}"
        )
        print(log_line, flush=True)
        if progress_callback:
            progress_callback(epoch, epochs, train_loss, val_loss, val_f1)

    safe_key = _slugify(key_phrase) or "keyphrase"
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    model_path = model_dir / f"{safe_key}_{timestamp}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "spect_cfg": spect_cfg.__dict__,
            "key_phrase": key_phrase,
        },
        model_path,
    )
    return history, model_path


def _collate_batch(batch):
    specs, labels = zip(*batch)
    max_len = max(spec.shape[-1] for spec in specs)
    padded = []
    for spec in specs:
        pad_width = max_len - spec.shape[-1]
        if pad_width > 0:
            spec = F.pad(spec, (0, pad_width), value=0.0)
        padded.append(spec.unsqueeze(0))
    spec_tensor = torch.stack(padded, dim=0)
    label_tensor = torch.stack(labels)
    return spec_tensor, label_tensor


def _macro_f1_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    targets_long = targets.long()
    tp = ((preds == 1) & (targets_long == 1)).sum().item()
    tn = ((preds == 0) & (targets_long == 0)).sum().item()
    fp = ((preds == 1) & (targets_long == 0)).sum().item()
    fn = ((preds == 0) & (targets_long == 1)).sum().item()

    def _safe_f1(t_pos: int, f_pos: int, f_neg: int) -> float:
        denom = (2 * t_pos + f_pos + f_neg)
        return (2 * t_pos / denom) if denom > 0 else 0.0

    f1_pos = _safe_f1(tp, fp, fn)
    f1_neg = _safe_f1(tn, fn, fp)
    return 0.5 * (f1_pos + f1_neg)


def _run_epoch(
    model: TinyCRNN,
    loader: DataLoader,
    criterion: BCEWithLogitsLoss,
    optimizer: Optional[AdamW],
    device: torch.device,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    total_loss = 0.0
    total_f1 = 0.0
    batches = 0
    model.train(mode=is_train)
    for specs, labels in loader:
        specs = specs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(is_train):
            window_logits, _ = model(specs)
            loss = criterion(window_logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item()
        total_f1 += _macro_f1_from_logits(window_logits.detach(), labels.detach())
        batches += 1
    if batches == 0:
        return 0.0, 0.0
    return total_loss / batches, total_f1 / batches


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return cleaned.strip("_")


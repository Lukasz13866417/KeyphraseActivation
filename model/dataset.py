from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

from .spectrogram_config import SpectrogramConfig
from .training_augmentations import apply_training_augmentations


def _normalize_payload(payload: Dict[str, Any], is_positive: bool) -> List[Dict[str, Any]]:
    """Normalize the payload to a list of dictionaries. Made by AI. TODO: review this"""
    label_value = 1 if is_positive else 0
    label_name = payload.get("label", "positive" if is_positive else "negative")
    rows: List[Dict[str, Any]] = []
    for rec in payload.get("records", []):
        path = rec.get("path")
        if not path:
            continue
        rows.append(
            {
                "path": path,
                "label": label_value,
                "category": rec.get("category", label_name),
                "api_name": rec.get("api_name"),
                "model_name": rec.get("model_name"),
                "text": rec.get("text"),
                "duration_sec": rec.get("duration_sec"),
                "sample_rate": rec.get("sample_rate"),
                "from_db": rec.get("from_db", False),
            }
        )
    return rows


def build_dataset_dataframe(
    positive_payload: Dict[str, Any], negative_payload: Dict[str, Any]
) -> pd.DataFrame:
    """
    Turn the tuple of JSON payloads returned by generate_with_augmentations into a DataFrame.
    """
    rows = _normalize_payload(positive_payload, True)
    rows.extend(_normalize_payload(negative_payload, False))
    columns = [
        "path",
        "label",
        "category",
        "api_name",
        "model_name",
        "text",
        "duration_sec",
        "sample_rate",
        "from_db",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        spectrogram_config: SpectrogramConfig,
        *,
        augment: bool = False,
        cache_dir: Optional[os.PathLike | str] = None,
        memory_cache_items: int = 256,
    ):
        if "path" not in dataframe or "label" not in dataframe:
            raise ValueError("DataFrame must contain 'path' and 'label' columns.")
        self.df = dataframe.reset_index(drop=True).copy()
        self.spectrogram_config = spectrogram_config
        self.augment = augment
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else None
        self.memory_cache_items = max(0, int(memory_cache_items))
        self._mem_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mel_transform = T.MelSpectrogram(
            sample_rate=spectrogram_config.sample_rate,
            n_fft=spectrogram_config.n_fft,
            hop_length=spectrogram_config.hop_length,
            n_mels=spectrogram_config.n_mels,
            power=2.0,
        )
        self._to_db = T.AmplitudeToDB(stype="power")

    def __len__(self):
        return len(self.df)

    def _cache_key(self, audio_path: str) -> str:
        try:
            st = os.stat(audio_path)
            stamp = f"{st.st_mtime_ns}:{st.st_size}"
        except OSError:
            stamp = "missing"
        cfg = self.spectrogram_config.to_dict()
        cfg_s = f"{cfg.get('sample_rate')}:{cfg.get('n_fft')}:{cfg.get('hop_length')}:{cfg.get('n_mels')}"
        raw = f"{audio_path}|{stamp}|{cfg_s}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _load_from_cache(self, key: str) -> Optional[torch.Tensor]:
        if self.memory_cache_items > 0:
            cached = self._mem_cache.get(key)
            if cached is not None:
                self._mem_cache.move_to_end(key)
                return cached

        if not self.cache_dir:
            return None
        cache_path = self.cache_dir / f"{key}.pt"
        if not cache_path.exists():
            return None
        try:
            obj = torch.load(cache_path, map_location="cpu")
            spec = obj["spec"] if isinstance(obj, dict) and "spec" in obj else obj
            if isinstance(spec, torch.Tensor):
                if self.memory_cache_items > 0:
                    self._mem_cache[key] = spec
                    self._mem_cache.move_to_end(key)
                    while len(self._mem_cache) > self.memory_cache_items:
                        self._mem_cache.popitem(last=False)
                return spec
        except Exception:
            # Corrupt cache entries should not crash training; we'll recompute.
            return None
        return None

    def _save_to_cache(self, key: str, spec: torch.Tensor) -> None:
        if self.memory_cache_items > 0:
            self._mem_cache[key] = spec
            self._mem_cache.move_to_end(key)
            while len(self._mem_cache) > self.memory_cache_items:
                self._mem_cache.popitem(last=False)

        if not self.cache_dir:
            return
        cache_path = self.cache_dir / f"{key}.pt"
        if cache_path.exists():
            return
        tmp_path = cache_path.with_suffix(".pt.tmp")
        try:
            torch.save({"spec": spec}, tmp_path)
            tmp_path.replace(cache_path)
        except Exception:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row["path"]
        is_keyphrase = bool(row["label"])
        target_sr = self.spectrogram_config.sample_rate

        key = self._cache_key(audio_path)
        mel_spec_db = self._load_from_cache(key)
        if mel_spec_db is None:
            waveform, sample_rate = torchaudio.load(audio_path)

            if sample_rate != target_sr:
                waveform = F.resample(waveform, sample_rate, target_sr)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            mel_spec = self._mel_transform(waveform)
            mel_spec_db = self._to_db(mel_spec).squeeze(0).float()
            self._save_to_cache(key, mel_spec_db)

        if self.augment:
            mel_spec_db = apply_training_augmentations(mel_spec_db.clone())

        label = torch.tensor(1.0 if is_keyphrase else 0.0, dtype=torch.float32)

        return mel_spec_db, label
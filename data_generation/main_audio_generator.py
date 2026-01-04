import hashlib
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# NOTE: Heavy dependencies (TTS drivers, TPS, torchaudio-based keyphrase finding)
# are imported lazily inside the functions that need them, so DB planning/progress
# updates can be emitted quickly.

# Text augmentation
from phrase_augmentation.augmenter import generate_augmented_phrases
from phrase_augmentation.util import get_word_base
from phrase_augmentation.punct_augmenter import add_punct

from db import db_api

API_ORDER = ("piper", "bark", "kokoro", "eleven")
CATEGORY_LABELS = {
    "positives": "Positives",
    "confusers": "Confusers",
    "inbetween": "In-between",
    "plain_negatives": "Plain Negatives",
    "tps_random": "TPS Samples",
}
_NONWORD_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")
_DB_READY = False
_MAX_SYNTH_WORKERS = max(1, min(8, int(os.getenv("AUDIO_GEN_MAX_WORKERS", "4"))))
_MAX_POSITIVE_DURATION_SEC = float(os.getenv("AUDIO_GEN_MAX_POSITIVE_DURATION_SEC", "4.0"))
_MIN_POSITIVE_SEGMENT_SEC = float(os.getenv("AUDIO_GEN_MIN_POSITIVE_SEGMENT_SEC", "0.6"))
_KEYPHRASE_PAD_PRE_SEC = float(os.getenv("AUDIO_GEN_KEYPHRASE_PAD_PRE_SEC", "0.15"))
_KEYPHRASE_PAD_POST_SEC = float(os.getenv("AUDIO_GEN_KEYPHRASE_PAD_POST_SEC", "0.15"))
_REUSED_CLIP_SUBDIR = os.getenv("AUDIO_GEN_REUSED_CLIP_SUBDIR", "reused_clips")


@dataclass
class GenerationPlan:
    key_phrase: str
    key_norm: str
    requirements: Dict[str, int]
    per_api_counts: Dict[str, int]
    sample_multiplier: int
    category_phrases: Dict[str, List[str]]
    sample_requirements: Dict[str, int]
    existing_records: Dict[str, List[dict]]
    growth_constant: int
    num_tps_random: int
    progress_summary: Dict[str, Dict[str, Any]]


GenerationProgressCallback = Callable[[Dict[str, Any]], None]


def _count_by_api(records: Sequence[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for rec in records or []:
        api = (rec.get("api_name") or "unknown").strip().lower() if isinstance(rec, dict) else "unknown"
        counts[api] = counts.get(api, 0) + 1
    return counts


def _build_generation_setup(
    key_phrase: str,
    *,
    num_confusers: int,
    num_positives: int,
    num_inbetween: int,
    num_plain_negatives: int,
    num_piper_per: int,
    num_bark_per: int,
    num_kokoro_per: int,
    num_eleven_per: int,
    confuser_inbetween_prob: float,
    p_pos_extra_start: float,
    p_pos_extra_end: float,
    max_inserts_per_gap: int,
    growth_constant: int,
) -> Tuple[
    Dict[str, int],
    int,
    Dict[str, int],
    Dict[str, List[str]],
    Dict[str, int],
]:
    per_api_counts = {
        "piper": max(0, int(num_piper_per)),
        "bark": max(0, int(num_bark_per)),
        "kokoro": max(0, int(num_kokoro_per)),
        "eleven": max(0, int(num_eleven_per)),
    }
    samples_per_phrase = sum(per_api_counts.values())
    sample_multiplier = samples_per_phrase if samples_per_phrase > 0 else 1

    requirements = {
        "positives": max(0, int(num_positives)),
        "confusers": max(0, int(num_confusers)),
        "inbetween": max(0, int(num_inbetween)),
        "plain_negatives": max(0, int(num_plain_negatives)),
    }

    aug = generate_augmented_phrases(
        key_phrase=key_phrase,
        num_confusers=max(1, requirements["confusers"]),
        num_positives=max(1, requirements["positives"]),
        num_inbetween=max(1, requirements["inbetween"]),
        confuser_inbetween_prob=confuser_inbetween_prob,
        p_pos_extra_start=p_pos_extra_start,
        p_pos_extra_end=p_pos_extra_end,
        max_inserts_per_gap=max_inserts_per_gap,
    )

    def _merge_phrases(primary: str, extras: Sequence[str]) -> List[str]:
        seen: set[str] = set()
        merged: List[str] = []
        for candidate in [primary, *extras]:
            norm = _normalize_text(candidate)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            merged.append(candidate)
        return merged

    positive_phrases = _merge_phrases(key_phrase, aug.get("positives", []))
    confuser_phrases = aug.get("confusers", [])
    inbetween_phrases = aug.get("inbetween", [])
    plain_negative_phrases = _generate_plain_negative_phrases(
        key_phrase,
        max(
            requirements["plain_negatives"],
            (growth_constant // sample_multiplier) + 1 if growth_constant > 0 else 0,
            1,
        ),
    )

    category_phrases = {
        "positives": positive_phrases,
        "confusers": confuser_phrases,
        "inbetween": inbetween_phrases,
        "plain_negatives": plain_negative_phrases,
    }

    sample_requirements = {
        bucket: requirements[bucket] * sample_multiplier for bucket in requirements
    }

    return per_api_counts, sample_multiplier, requirements, category_phrases, sample_requirements


def _project_root() -> Path:
    return _REPO_ROOT


def _ensure_db_initialized() -> None:
    """Initialize the SQLite DB (idempotent)."""
    global _DB_READY
    if _DB_READY:
        return
    if not getattr(db_api, "is_initialized", False):
        try:
            db_api.init_db()
        except Warning:
            pass
    _DB_READY = True


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    lowered = text.lower()
    no_punct = _NONWORD_RE.sub(" ", lowered)
    collapsed = _WHITESPACE_RE.sub(" ", no_punct)
    return collapsed.strip()


def _file_sha256(path: str) -> Optional[str]:
    try:
        hasher = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        return None
    except OSError as exc:
        print(f"[main_audio_generator] Failed to hash {path}: {exc}")
        return None


def _insert_records(
    records: Sequence[dict],
    *,
    default_text: str = "",
) -> None:
    if not records:
        return
    _ensure_db_initialized()
    with db_api.get_db_connection() as conn:
        for rec in records:
            raw_path = rec.get("path")
            if not raw_path:
                continue
            path_str = str(Path(raw_path).resolve())
            if not Path(path_str).exists():
                continue
            text_original = rec.get("text") or default_text or ""
            text_norm = _normalize_text(text_original)
            sha_value = _file_sha256(path_str)
            if not sha_value:
                continue
            rec["path"] = path_str
            conn.execute(
                """
                INSERT OR IGNORE INTO audio_sample
                (path, api_name, model_name, text_original, text_normalized, audio_sha256,
                 duration_sec, sample_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    path_str,
                    rec.get("api_name", ""),
                    rec.get("model_name", ""),
                    text_original,
                    text_norm,
                    sha_value,
                    rec.get("duration_sec"),
                    rec.get("sample_rate"),
                ),
            )
        conn.commit()


def _fetch_db_records(
    text_norm: str,
    limit: int,
    *,
    substring: bool = False,
) -> List[dict]:
    if limit <= 0:
        return []
    _ensure_db_initialized()
    needle = (text_norm or "").strip()
    if not needle:
        return []
    pattern = needle
    if substring:
        pattern = f"%{needle}%"
    records: List[dict] = []
    with db_api.get_db_connection() as conn:
        cur = conn.execute(
            """
            SELECT id, path, api_name, model_name, duration_sec, sample_rate, text_original
            FROM audio_sample
            WHERE text_normalized LIKE ?
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (pattern, limit),
        )
        rows = cur.fetchall()
        missing_ids = []
        for row in rows:
            path_str = row["path"]
            if not Path(path_str).exists():
                missing_ids.append((row["id"],))
                continue
            records.append(
                {
                    "path": path_str,
                    "api_name": row["api_name"],
                    "model_name": row["model_name"],
                    "duration_sec": row["duration_sec"],
                    "sample_rate": row["sample_rate"],
                    "text": row["text_original"],
                    "from_db": True,
                }
            )
        if missing_ids:
            conn.executemany("DELETE FROM audio_sample WHERE id=?", missing_ids)
        conn.commit()
    return records


def _fetch_plain_negative_records(
    key_norm: str,
    limit: int,
) -> List[dict]:
    if limit <= 0:
        return []
    _ensure_db_initialized()
    key_norm = (key_norm or "").strip()
    params: List[object] = []
    where_clause = "1"
    if key_norm:
        where_clause = "text_normalized NOT LIKE ?"
        params.append(f"%{key_norm}%")
    params.append(limit)
    records: List[dict] = []
    with db_api.get_db_connection() as conn:
        cur = conn.execute(
            f"""
            SELECT id, path, api_name, model_name, duration_sec, sample_rate, text_original
            FROM audio_sample
            WHERE {where_clause}
            ORDER BY RANDOM()
            LIMIT ?
            """,
            params,
        )
        rows = cur.fetchall()
        missing_ids = []
        for row in rows:
            path_str = row["path"]
            if not Path(path_str).exists():
                missing_ids.append((row["id"],))
                continue
            records.append(
                {
                    "path": path_str,
                    "api_name": row["api_name"],
                    "model_name": row["model_name"],
                    "duration_sec": row["duration_sec"],
                    "sample_rate": row["sample_rate"],
                    "text": row["text_original"],
                    "from_db": True,
                }
            )
        if missing_ids:
            conn.executemany("DELETE FROM audio_sample WHERE id=?", missing_ids)
        conn.commit()
    return records


def _build_api_sequence(total_needed: int, base_counts: Dict[str, int]) -> List[str]:
    """
    Build a repeating API sequence based on the requested per-API counts.
    """
    if total_needed <= 0:
        return []
    template: List[str] = []
    for api in API_ORDER:
        repetitions = max(0, int(base_counts.get(api, 0)))
        template.extend([api] * repetitions)
    if not template:
        return []
    sequence: List[str] = []
    for idx in range(total_needed):
        sequence.append(template[idx % len(template)])
    return sequence


def _generate_plain_negative_phrases(
    key_phrase: str,
    count: int,
    *,
    min_words: int = 3,
    max_words: int = 8,
) -> List[str]:
    if count <= 0:
        return []
    base_words = get_word_base()
    key_norm = _normalize_text(key_phrase)
    phrases: set[str] = set()
    attempts = 0
    max_attempts = max(100, count * 20)
    while len(phrases) < count and attempts < max_attempts:
        attempts += 1
        length = random.randint(min_words, max_words)
        words = random.choices(base_words, k=length)
        candidate = " ".join(words)
        cand_norm = _normalize_text(candidate)
        if key_norm and key_norm in cand_norm:
            continue
        phrases.add(candidate)
    return list(phrases)


def _maybe_clip_record_to_keyphrase(
    record: dict,
    key_phrase: str,
    output_dir: str,
) -> Optional[dict]:
    if _MAX_POSITIVE_DURATION_SEC <= 0:
        return record
    duration = record.get("duration_sec")
    if duration is None or duration <= _MAX_POSITIVE_DURATION_SEC:
        return record

    path = record.get("path")
    transcript = (record.get("text") or "").strip()
    if not path or not transcript or not key_phrase:
        print(
            "[main_audio_generator] Skipping overlong positive without transcript/path.",
            flush=True,
        )
        return None

    min_duration = max(0.0, _MIN_POSITIVE_SEGMENT_SEC)
    max_duration = _MAX_POSITIVE_DURATION_SEC if _MAX_POSITIVE_DURATION_SEC > 0 else None
    if max_duration is not None and min_duration > max_duration:
        min_duration = max_duration

    try:
        from data_generation.keyphrase_finding.internal.finder import (
            find_keyphrase_segment_with_margin,
        )
    except Exception as exc:
        print(
            f"[main_audio_generator] Skipping positive clipping (keyphrase finder unavailable): {exc}",
            flush=True,
        )
        return record

    try:
        span_start, span_end = find_keyphrase_segment_with_margin(
            wav_path=path,
            transcript=transcript,
            keyphrase=key_phrase,
            pre_padding_sec=_KEYPHRASE_PAD_PRE_SEC,
            post_padding_sec=_KEYPHRASE_PAD_POST_SEC,
            min_duration_sec=min_duration,
            max_duration_sec=max_duration,
        )
    except Exception as exc:
        print(
            f"[main_audio_generator] Keyphrase finder failed for '{path}': {exc}",
            flush=True,
        )
        return None

    clip_duration = span_end - span_start
    if clip_duration <= 0.05:
        print(
            f"[main_audio_generator] Finder returned degenerate span ({clip_duration:.3f}s) for '{path}'.",
            flush=True,
        )
        return None

    reuse_dir = Path(output_dir).resolve() / _REUSED_CLIP_SUBDIR
    reuse_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(
        f"{path}|{span_start:.3f}|{span_end:.3f}|{key_phrase}".encode("utf-8")
    ).hexdigest()[:16]
    suffix = Path(path).suffix or ".wav"
    out_path = reuse_dir / f"reuse_{digest}{suffix}"

    if not out_path.exists():
        try:
            from data_generation.keyphrase_finding.driver import extract_keyphrase_audio

            extract_keyphrase_audio(path, span_start, span_end, str(out_path))
        except Exception as exc:
            print(
                f"[main_audio_generator] Failed to extract keyphrase clip '{out_path}': {exc}",
                flush=True,
            )
            return None

    clipped = dict(record)
    clipped["path"] = str(out_path.resolve())
    clipped["duration_sec"] = clip_duration
    clipped["from_db"] = False
    _insert_records([clipped])
    print(
        "[main_audio_generator] Reused long positive by clipping"
        f" {Path(path).name} -> {out_path.name} (orig={duration:.2f}s clip={clip_duration:.2f}s)",
        flush=True,
    )
    return clipped


def _prepare_positive_records(
    records: List[dict],
    key_phrase: str,
    output_dir: str,
) -> List[dict]:
    if not records:
        return []
    processed: List[dict] = []
    for rec in records:
        clipped = _maybe_clip_record_to_keyphrase(rec, key_phrase, output_dir)
        if clipped is None:
            continue
        processed.append(clipped)
    return processed


def _fetch_records_for_phrases(
    phrases: Sequence[str],
    limit: int,
) -> List[dict]:
    """
    Fetch random records whose normalized text matches any of the given phrases.
    """
    if limit <= 0:
        return []
    norms: set[str] = set()
    for phrase in phrases:
        if not phrase:
            continue
        norm = _normalize_text(phrase)
        if norm:
            norms.add(norm)
    if not norms:
        return []
    placeholders = ",".join("?" for _ in norms)
    query = f"""
        SELECT id, path, api_name, model_name, duration_sec, sample_rate, text_original
        FROM audio_sample
        WHERE text_normalized IN ({placeholders})
        ORDER BY RANDOM()
        LIMIT ?
    """
    _ensure_db_initialized()
    params: List[object] = [*norms, limit]
    records: List[dict] = []
    with db_api.get_db_connection() as conn:
        cur = conn.execute(query, params)
        rows = cur.fetchall()
        missing_ids = []
        for row in rows:
            path_str = row["path"]
            if not Path(path_str).exists():
                missing_ids.append((row["id"],))
                continue
            records.append(
                {
                    "path": path_str,
                    "api_name": row["api_name"],
                    "model_name": row["model_name"],
                    "duration_sec": row["duration_sec"],
                    "sample_rate": row["sample_rate"],
                    "text": row["text_original"],
                    "from_db": True,
                }
            )
        if missing_ids:
            conn.executemany("DELETE FROM audio_sample WHERE id=?", missing_ids)
        conn.commit()
    return records


def _generate_new_records(
    phrases: Sequence[str],
    *,
    total_new: int,
    per_api_counts: Dict[str, int],
    output_dir: str,
    piper_kwargs: Dict,
    bark_kwargs: Dict,
    kokoro_kwargs: Dict,
    eleven_kwargs: Dict,
) -> List[dict]:
    """
    Generate new audio samples for the provided phrases, honoring the requested API distribution.
    """
    if total_new <= 0:
        return []
    phrase_list = [p for p in phrases if p and p.strip()]
    if not phrase_list:
        return []
    api_sequence = _build_api_sequence(total_new, per_api_counts)
    if not api_sequence:
        print(
            "[main_audio_generator] No TTS APIs configured for synthesis; skipping new generation.",
            flush=True,
        )
        return []

    chunk_size = max(1, sum(per_api_counts.get(api, 0) for api in API_ORDER))
    tasks: List[tuple[str, Dict[str, int]]] = []
    idx = 0
    while idx < len(api_sequence):
        chunk = api_sequence[idx : idx + chunk_size]
        idx += len(chunk)
        count_map = {api: 0 for api in API_ORDER}
        for api in chunk:
            count_map[api] += 1
        if not any(count_map.values()):
            continue
        phrase = phrase_list[len(tasks) % len(phrase_list)]
        tasks.append((phrase, count_map))

    if not tasks:
        return []

    def _run_task(payload: tuple[str, Dict[str, int]]) -> List[dict]:
        phrase, counts = payload
        try:
            batch = generate_phrase_wavs(
                phrase,
                num_piper=counts.get("piper", 0),
                num_bark=counts.get("bark", 0),
                num_kokoro=counts.get("kokoro", 0),
                num_eleven=counts.get("eleven", 0),
                output_dir=output_dir,
                piper_kwargs=piper_kwargs,
                bark_kwargs=bark_kwargs,
                kokoro_kwargs=kokoro_kwargs,
                eleven_kwargs=eleven_kwargs,
            )
        except Exception as exc:
            print(f"[main_audio_generator] Parallel synth failed for '{phrase}': {exc}")
            return []
        for rec in batch:
            rec["from_db"] = False
        return batch

    new_records: List[dict] = []
    max_workers = min(len(tasks), _MAX_SYNTH_WORKERS)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_task, task) for task in tasks]
        for future in as_completed(futures):
            batch = future.result()
            if not batch:
                continue
            remaining = total_new - len(new_records)
            if remaining <= 0:
                continue
            new_records.extend(batch[:remaining])

    if len(new_records) < total_new:
        print(
            f"[main_audio_generator] Warning: requested {total_new} new samples "
            f"but only generated {len(new_records)}.",
            flush=True,
        )
    return new_records[:total_new]


def generate_phrase_wavs(
    phrase: str,
    *,
    num_piper: int = 0,
    num_bark: int = 0,
    num_kokoro: int = 0,
    num_eleven: int = 0,
    output_dir: Optional[str] = None,
    piper_kwargs: Optional[dict] = None,
    bark_kwargs: Optional[dict] = None,
    kokoro_kwargs: Optional[dict] = None,
    eleven_kwargs: Optional[dict] = None,
) -> List[dict]:
    """
    Synthesize WAVs for a phrase using Piper, Bark (Suno), Kokoro, and ElevenLabs.
    Returns a combined list of records: [{ path, sample_rate, duration_sec, model_name, api_name, text }, ...]
    """
    if not phrase or not phrase.strip():
        return []
    
    # Apply punctuation augmentation
    try:
        # Generate 1 augmented variant; replace_existing=True allows changing existing punctuation
        phrase_aug = next(add_punct(phrase, 1, replace_existing=True))
    except Exception:
        phrase_aug = phrase

    if output_dir is None:
        output_dir = str((_project_root() / "samples").resolve())
    piper_kwargs = piper_kwargs or {}
    bark_kwargs = bark_kwargs or {}
    kokoro_kwargs = kokoro_kwargs or {}
    eleven_kwargs = eleven_kwargs or {}

    records: List[dict] = []
    if num_piper > 0:
        try:
            from audio_generation.piper_side.driver import synthesize as piper_synthesize

            recs = piper_synthesize(
                phrase=phrase_aug,
                num_samples=num_piper,
                output_dir=output_dir,
                **piper_kwargs,
            )
            if recs:
                for rec in recs:
                    rec.setdefault("text", phrase_aug)
                    records.append(rec)
        except Exception as e:
            print(f"[main_audio_generator] Piper failed: {e}")
    if num_bark > 0:
        try:
            from audio_generation.suno_side.driver import synthesize as bark_synthesize

            recs = bark_synthesize(
                phrase=phrase_aug,
                num_samples=num_bark,
                output_dir=output_dir,
                **bark_kwargs,
            )
            if recs:
                for rec in recs:
                    rec.setdefault("text", phrase_aug)
                    records.append(rec)
        except Exception as e:
            print(f"[main_audio_generator] Bark failed: {e}")
    if num_kokoro > 0:
        try:
            from audio_generation.kokoro_side.driver import synthesize as kokoro_synthesize

            recs = kokoro_synthesize(
                text=phrase_aug,
                num_samples=num_kokoro,
                output_dir=output_dir,
                **kokoro_kwargs,
            )
            if recs:
                for rec in recs:
                    rec.setdefault("text", phrase_aug)
                    records.append(rec)
        except Exception as e:
            print(f"[main_audio_generator] Kokoro failed: {e}")
    if num_eleven > 0:
        try:
            from audio_generation.elevenlabs_side.driver import synthesize as eleven_synthesize

            recs = eleven_synthesize(
                phrase=phrase_aug,
                num_samples=num_eleven,
                output_dir=output_dir,
                **eleven_kwargs,
            )
            if recs:
                for rec in recs:
                    rec.setdefault("text", phrase_aug)
                    records.append(rec)
        except Exception as e:
            print(f"[main_audio_generator] ElevenLabs failed: {e}")
    return records



def plan_generation_with_augmentations(
    key_phrase: str,
    *,
    num_confusers: int,
    num_positives: int,
    num_inbetween: int,
    num_piper_per: int = 0,
    num_bark_per: int = 0,
    num_kokoro_per: int = 0,
    num_eleven_per: int = 0,
    confuser_inbetween_prob: float = 0.5,
    p_pos_extra_start: float = 0.5,
    p_pos_extra_end: float = 0.5,
    max_inserts_per_gap: int = 2,
    num_plain_negatives: int = 0,
    growth_constant: int = 0,
    num_tps_random: int = 0,
    progress_callback: Optional[GenerationProgressCallback] = None,
) -> GenerationPlan:
    """
    Fast, DB-only planning step. Computes how many clips are needed per category
    and how many can be satisfied from the existing database without launching synthesis.
    """
    growth_constant = max(0, int(growth_constant))
    _ensure_db_initialized()
    key_norm = _normalize_text(key_phrase)

    (
        per_api_counts,
        sample_multiplier,
        requirements,
        category_phrases,
        sample_requirements,
    ) = _build_generation_setup(
        key_phrase=key_phrase,
        num_confusers=num_confusers,
        num_positives=num_positives,
        num_inbetween=num_inbetween,
        num_plain_negatives=num_plain_negatives,
        num_piper_per=num_piper_per,
        num_bark_per=num_bark_per,
        num_kokoro_per=num_kokoro_per,
        num_eleven_per=num_eleven_per,
        confuser_inbetween_prob=confuser_inbetween_prob,
        p_pos_extra_start=p_pos_extra_start,
        p_pos_extra_end=p_pos_extra_end,
        max_inserts_per_gap=max_inserts_per_gap,
        growth_constant=growth_constant,
    )

    progress_summary: Dict[str, Dict[str, Any]] = {}
    existing_records: Dict[str, List[dict]] = {}

    # TPS planning entry (no fetch)
    tps_target = max(0, int(num_tps_random))
    tps_planning = {
        "category": "tps_random",
        "category_label": CATEGORY_LABELS.get("tps_random", "TPS Samples"),
        "requested_phrases": 0,
        "phrases_available": 0,
        "samples_per_phrase": 1,
        "target_clips": tps_target,
        "db_clips_used": 0,
        "db_clips_available": 0,
        "generated_clips": 0,
        "completed_clips": 0,
        "growth_constant": 0,
        "completion_percent": 0,
        "reused_by_api": {},
        "generated_by_api": {},
        "phase": "planning",
    }
    progress_summary["tps_random"] = tps_planning
    if progress_callback:
        try:
            progress_callback(tps_planning)
        except Exception as exc:
            print(f"[main_audio_generator] Progress callback failed: {exc}", flush=True)

    for bucket in ("positives", "confusers", "inbetween", "plain_negatives"):
        needed_samples = sample_requirements.get(bucket, 0)
        phrases = category_phrases.get(bucket, [])
        if needed_samples <= 0 and growth_constant <= 0:
            existing_records[bucket] = []
            continue
        if bucket == "plain_negatives":
            recs = _fetch_plain_negative_records(key_norm, needed_samples)
        else:
            recs = _fetch_records_for_phrases(phrases, needed_samples)
        existing_records[bucket] = recs
        existing_used = min(len(recs), needed_samples)
        target_clips = needed_samples if needed_samples > 0 else max(growth_constant, 0)
        completed_clips = min(target_clips, existing_used) if target_clips > 0 else existing_used
        completion_percent = (
            int((completed_clips * 100) // max(1, target_clips))
            if target_clips > 0
            else (100 if completed_clips else 0)
        )
        planning_info = {
            "category": bucket,
            "category_label": CATEGORY_LABELS.get(bucket, bucket.replace("_", " ").title()),
            "requested_phrases": requirements.get(bucket, 0),
            "phrases_available": len(phrases),
            "samples_per_phrase": sample_multiplier,
            "target_clips": target_clips,
            "db_clips_used": existing_used,
            "db_clips_available": len(recs),
            "generated_clips": 0,
            "completed_clips": completed_clips,
            "growth_constant": growth_constant,
            "completion_percent": completion_percent,
            "reused_by_api": _count_by_api(recs[:existing_used]),
            "generated_by_api": {},
            "phase": "planning",
        }
        progress_summary[bucket] = planning_info
        if progress_callback:
            try:
                progress_callback(planning_info)
            except Exception as exc:
                print(f"[main_audio_generator] Progress callback failed: {exc}", flush=True)

    return GenerationPlan(
        key_phrase=key_phrase,
        key_norm=key_norm,
        requirements=requirements,
        per_api_counts=per_api_counts,
        sample_multiplier=sample_multiplier,
        category_phrases=category_phrases,
        sample_requirements=sample_requirements,
        existing_records=existing_records,
        growth_constant=growth_constant,
        num_tps_random=max(0, int(num_tps_random)),
        progress_summary=progress_summary,
    )


def generate_with_augmentations(
    key_phrase: str,
    *,
    num_confusers: int,
    num_positives: int,
    num_inbetween: int,
    num_piper_per: int = 0,
    num_bark_per: int = 0,
    num_kokoro_per: int = 0,
    num_eleven_per: int = 0,
    output_dir: Optional[str] = None,
    piper_kwargs: Optional[dict] = None,
    bark_kwargs: Optional[dict] = None,
    kokoro_kwargs: Optional[dict] = None,
    eleven_kwargs: Optional[dict] = None,
    confuser_inbetween_prob: float = 0.5,
    p_pos_extra_start: float = 0.5,
    p_pos_extra_end: float = 0.5,
    max_inserts_per_gap: int = 2,
    num_plain_negatives: int = 0,
    growth_constant: int = 0,
    num_tps_random: int = 0,
    plan: Optional[GenerationPlan] = None,
    progress_callback: Optional[GenerationProgressCallback] = None,
) -> Tuple[Dict[str, List[dict]], Dict[str, List[dict]], Dict[str, Dict[str, Any]]]:
    """
    Generate audio for each augmentation category. For every category we:
      1. Inspect the DB for existing usable samples.
      2. Generate max(growth_constant, required - existing) fresh samples to grow the pool.
    Returns two JSON payloads and a per-category progress summary.
    """
    output_dir = output_dir or str((_project_root() / "samples").resolve())
    piper_kwargs = piper_kwargs or {}
    bark_kwargs = bark_kwargs or {}
    kokoro_kwargs = kokoro_kwargs or {}
    eleven_kwargs = eleven_kwargs or {}
    growth_constant = max(0, int(growth_constant))
    if plan is None:
        plan = plan_generation_with_augmentations(
            key_phrase=key_phrase,
            num_confusers=num_confusers,
            num_positives=num_positives,
            num_inbetween=num_inbetween,
            num_piper_per=num_piper_per,
            num_bark_per=num_bark_per,
            num_kokoro_per=num_kokoro_per,
            num_eleven_per=num_eleven_per,
            confuser_inbetween_prob=confuser_inbetween_prob,
            p_pos_extra_start=p_pos_extra_start,
            p_pos_extra_end=p_pos_extra_end,
            max_inserts_per_gap=max_inserts_per_gap,
            num_plain_negatives=num_plain_negatives,
            growth_constant=growth_constant,
            num_tps_random=num_tps_random,
            progress_callback=progress_callback,
        )
    per_api_counts = plan.per_api_counts
    sample_multiplier = plan.sample_multiplier
    requirements = plan.requirements
    category_phrases = plan.category_phrases
    sample_requirements = plan.sample_requirements
    growth_constant = plan.growth_constant
    num_tps_random = plan.num_tps_random
    key_norm = plan.key_norm
    progress_summary: Dict[str, Dict[str, Any]] = dict(plan.progress_summary or {})
    existing_records_map: Dict[str, List[dict]] = {k: list(v) for k, v in (plan.existing_records or {}).items()}

    out: Dict[str, List[dict]] = {
        "positives": [],
        "confusers": [],
        "inbetween": [],
        "plain_negatives": [],
        "tps_random": [],
    }

    if num_tps_random > 0:
        tps_target = max(0, int(num_tps_random))
        try:
            from audio_generation.tps_side.tps_generator import get_wavs_from_tps

            tps_records = get_wavs_from_tps(num_tps_random)
        except Exception as exc:
            print(f"[main_audio_generator] TPS fetch failed: {exc}")
            tps_records = []
        for rec in tps_records:
            rec["from_db"] = False
        _insert_records(tps_records)
        out["tps_random"].extend(tps_records)
        tps_completed = min(tps_target, len(tps_records)) if tps_target > 0 else len(tps_records)
        tps_percent = int((tps_completed * 100) // max(1, tps_target)) if tps_target > 0 else (100 if tps_completed else 0)
        tps_progress = {
            "category": "tps_random",
            "category_label": CATEGORY_LABELS.get("tps_random", "TPS Samples"),
            "requested_phrases": 0,
            "phrases_available": 0,
            "samples_per_phrase": 1,
            "target_clips": tps_target,
            "db_clips_used": 0,
            "db_clips_available": 0,
            "generated_clips": len(tps_records),
            "completed_clips": tps_completed,
            "growth_constant": 0,
            "completion_percent": tps_percent,
            "reused_by_api": {},
            "generated_by_api": _count_by_api(tps_records),
            "phase": "done",
        }
        progress_summary["tps_random"] = tps_progress
        if progress_callback:
            try:
                progress_callback(tps_progress)
            except Exception as exc:
                print(f"[main_audio_generator] Progress callback failed: {exc}", flush=True)
    for bucket in ("positives", "confusers", "inbetween", "plain_negatives"):
        needed_samples = sample_requirements.get(bucket, 0)
        phrases = category_phrases.get(bucket, [])
        if needed_samples <= 0 and growth_constant <= 0:
            continue
        existing_records = list(existing_records_map.get(bucket, []))
        if not existing_records:
            if bucket == "plain_negatives":
                existing_records = _fetch_plain_negative_records(key_norm, needed_samples)
            else:
                existing_records = _fetch_records_for_phrases(phrases, needed_samples)
            existing_records_map[bucket] = existing_records
        existing_used = min(len(existing_records), needed_samples)

        if bucket == "positives":
            existing_records = _prepare_positive_records(existing_records, key_phrase, output_dir)
            existing_used = min(len(existing_records), needed_samples)
            if progress_callback:
                # Update after preparing/clipping positives (still pre-synthesis).
                target_clips = needed_samples if needed_samples > 0 else max(growth_constant, 0)
                completed_clips = min(target_clips, existing_used) if target_clips > 0 else existing_used
                completion_percent = int((completed_clips * 100) // max(1, target_clips)) if target_clips > 0 else (100 if completed_clips else 0)
                prepared_info = {
                    "category": bucket,
                    "category_label": CATEGORY_LABELS.get(bucket, bucket.replace("_", " ").title()),
                    "requested_phrases": requirements.get(bucket, 0),
                    "phrases_available": len(phrases),
                    "samples_per_phrase": sample_multiplier,
                    "target_clips": target_clips,
                    "db_clips_used": existing_used,
                    "db_clips_available": len(existing_records),
                    "generated_clips": 0,
                    "completed_clips": completed_clips,
                    "growth_constant": growth_constant,
                    "completion_percent": completion_percent,
                    "reused_by_api": _count_by_api(existing_records[:existing_used]),
                    "generated_by_api": {},
                    "phase": "prepared",
                }
                try:
                    progress_callback(prepared_info)
                except Exception as exc:
                    print(f"[main_audio_generator] Progress callback failed: {exc}", flush=True)

        reused_records = existing_records[:existing_used]
        new_target = max(growth_constant, max(0, needed_samples - existing_used))
        new_records = _generate_new_records(
            phrases or [key_phrase],
            total_new=new_target,
            per_api_counts=per_api_counts,
            output_dir=output_dir,
            piper_kwargs=piper_kwargs,
            bark_kwargs=bark_kwargs,
            kokoro_kwargs=kokoro_kwargs,
            eleven_kwargs=eleven_kwargs,
        )
        if new_records:
            _insert_records(new_records)
        out[bucket].extend(existing_records[:needed_samples])
        out[bucket].extend(new_records)
        print(
            "[main_audio_generator][category]"
            f" name='{bucket}' target_samples={needed_samples} existing_used={existing_used}"
            f" growth_constant={growth_constant} new_generated={len(new_records)}",
            flush=True,
        )
        target_clips = needed_samples if needed_samples > 0 else new_target
        completed_clips = existing_used + len(new_records)
        if target_clips > 0:
            completed_clips = min(target_clips, completed_clips)
            completion_percent = int(max(0, min(100, (completed_clips * 100) // max(1, target_clips))))
        else:
            completion_percent = 100 if completed_clips > 0 else 0
        progress_info = {
            "category": bucket,
            "category_label": CATEGORY_LABELS.get(bucket, bucket.replace("_", " ").title()),
            "requested_phrases": requirements.get(bucket, 0),
            "phrases_available": len(phrases),
            "samples_per_phrase": sample_multiplier,
            "target_clips": target_clips,
            "db_clips_used": existing_used,
            "db_clips_available": len(existing_records),
            "generated_clips": len(new_records),
            "completed_clips": completed_clips,
            "growth_constant": growth_constant,
            "completion_percent": completion_percent,
            "reused_by_api": _count_by_api(reused_records),
            "generated_by_api": _count_by_api(new_records),
            "phase": "done",
        }
        progress_summary[bucket] = progress_info
        if progress_callback:
            try:
                progress_callback(progress_info)
            except Exception as exc:
                print(f"[main_audio_generator] Progress callback failed: {exc}", flush=True)

    positives_payload = {"label": "positive", "records": []}
    negatives_payload = {"label": "negative", "records": []}
    for bucket, records in out.items():
        target_payload = positives_payload if bucket == "positives" else negatives_payload
        for rec in records:
            rec_with_category = dict(rec)
            rec_with_category.setdefault("category", bucket)
            target_payload["records"].append(rec_with_category)

    return positives_payload, negatives_payload, progress_summary


if __name__ == "__main__":
    print("Main Audio Generator")
    key = input("Key phrase: ").strip()
    try:
        npip = int(input("Num Piper per phrase [0]: ").strip() or "0")
    except Exception:
        npip = 0
    try:
        nbark = int(input("Num Bark per phrase [0]: ").strip() or "0")
    except Exception:
        nbark = 0
    try:
        nkok = int(input("Num Kokoro per phrase [0]: ").strip() or "0")
    except Exception:
        nkok = 0
    try:
        nelev = int(input("Num ElevenLabs per phrase [0]: ").strip() or "0")
    except Exception:
        nelev = 0
    mode = (input("Augment? [y/N]: ").strip() or "n").lower()
    if mode == "y":
        try:
            nconf = int(input("Number of confusers [10]: ").strip() or "10")
        except Exception:
            nconf = 10
        try:
            npos = int(input("Number of positives [10]: ").strip() or "10")
        except Exception:
            npos = 10
        try:
            ninb = int(input("Number of in-between phrases [10]: ").strip() or "10")
        except Exception:
            ninb = 10
        try:
            pstart = float(input("Positives: prob extra start word [0-1] [0.5]: ").strip() or "0.5")
        except Exception:
            pstart = 0.5
        try:
            pend = float(input("Positives: prob extra end word [0-1] [0.5]: ").strip() or "0.5")
        except Exception:
            pend = 0.5
        try:
            cinbp = float(input("In-between confuser prob [0-1] [0.5]: ").strip() or "0.5")
        except Exception:
            cinbp = 0.5
        try:
            maxins = int(input("Max inserts per gap [2]: ").strip() or "2")
        except Exception:
            maxins = 2
        try:
            nplain = int(input("Number of plain negatives [5]: ").strip() or "5")
        except Exception:
            nplain = 5
        try:
            ntps = int(input("Random TPS clips to add [0]: ").strip() or "0")
        except Exception:
            ntps = 0
        try:
            growth_const = int(input("Growth constant per category [3]: ").strip() or "3")
        except Exception:
            growth_const = 3
        pos_payload, neg_payload, _progress = generate_with_augmentations(
            key_phrase=key,
            num_confusers=nconf,
            num_positives=npos,
            num_inbetween=ninb,
            num_piper_per=npip,
            num_bark_per=nbark,
            num_kokoro_per=nkok,
            num_eleven_per=nelev,
            p_pos_extra_start=pstart,
            p_pos_extra_end=pend,
            confuser_inbetween_prob=cinbp,
            max_inserts_per_gap=maxins,
            num_plain_negatives=nplain,
            num_tps_random=ntps,
            growth_constant=growth_const,
        )
        print("Positive payload:")
        print(json.dumps(pos_payload, indent=2))
        print("Negative payload:")
        print(json.dumps(neg_payload, indent=2))
        print("Progress summary:")
        print(json.dumps(_progress, indent=2))
    else:
        recs = generate_phrase_wavs(
            key, num_piper=npip, num_bark=nbark, num_kokoro=nkok, num_eleven=nelev
        )
        print(json.dumps(recs, indent=2))



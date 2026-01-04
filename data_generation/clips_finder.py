from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from db import db_api
from typing import List, Tuple

def find_positives(keyphrase: str, min_duration: float, max_duration: float) -> List[Tuple[str, str]]:
    with db_api.get_db_connection() as conn:
        results = conn.execute(
            """
            SELECT path, text_normalized, duration_sec
            FROM audio_sample
            WHERE text_normalized LIKE ?
            AND duration_sec >= ? 
            AND duration_sec <= ?
            """,
            (f"%{keyphrase}%", min_duration, max_duration,),
        )
        return [(r["path"],r["text_normalized"],r["duration_sec"]) for r in results.fetchall()]

def find_negatives(keyphrase: str, min_duration: float, max_duration: float) -> List[Tuple[str,str]]:
    with db_api.get_db_connection() as conn:
        results = conn.execute(
            """
            SELECT path, text_normalized, duration_sec
            FROM audio_sample
            WHERE text_normalized NOT LIKE ?
            AND duration_sec >= ? 
            AND duration_sec <= ?
            """,
            (f"%{keyphrase}%", min_duration, max_duration,),
        )
        return [(r["path"],r["text_normalized"],r["duration_sec"]) for r in results.fetchall()]

def find_duration_stats_for_not_superlong_positives(keyphrase: str, char_limit: float) -> Tuple[str,str,str]:
    with db_api.get_db_connection() as conn:
        results = conn.execute(
            """
            SELECT AVG(duration_sec) AS avg_dur, MIN(duration_sec) AS min_dur, MAX(duration_sec) AS max_dur
            FROM audio_sample
            WHERE text_normalized LIKE ?
            AND length(text_normalized) < char_limit
            """,
            (f"%{keyphrase}%", char_limit,),
        )
        r = next(results.fetchall())
        return (r["avg_dur"],r["min_dur"],r["max_dur"])

def find_all_clips(keyphrase: str, min_duration: float, max_duration: float) -> List[Tuple[str,str]]:
    with db_api.get_db_connection() as conn:
        results = conn.execute(
            """
            SELECT path, text_normalized, duration_sec
            FROM audio_sample
            WHERE duration_sec >= ? 
            AND duration_sec <= ?
            """,
            (min_duration,max_duration,),
        )
        return [(r["path"],r["text_normalized"],r["duration_sec"]) for r in results.fetchall()]

if __name__ == "__main__":
    db_api.init_db()
    kp = input("Keyphrase: ")
    mind = float(input("Min Dur: "))
    maxd = float(input("Max Dur: "))
    res = find_all_clips(kp, mind, maxd)
    for r in res:
        print(r)
        

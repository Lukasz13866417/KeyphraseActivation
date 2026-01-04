from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from db import db_api
from typing import Dict, List
from clips_finder import find_positives, find_negatives, find_all_clips, find_duration_stats_for_positive

def search_before_gen(keyphrase: str,
                      single_sample_dur_limit: float,
                      long_text_threshold: int) -> Dict[str, List[str]]:

    long_text_threshold = max(long_text_threshold, len(keyphrase)*2)

    avg_pos_dur, min_pos_dur, max_pos_dur
                    =   find_duration_stats_for_positives(
                            keyphrase, 
                            long_text_threshold
                        )
    normal_positives        
                    =   find_positives(
                            keyphrase,
                            min_pos_dur,
                            max_pos_dur
                        )
    containing_positives    
                    =   find_positives(
                            keyphrase,
                            max_pos_dur,
                            single_sample_dur_limit
                        )
    all_negatives           
                    =   find_negatives(
                            keyphrase,
                            0,
                            single_sample_dur_limit
                        )

    return {
        "normal_pos":normal_positives,
        "containing_pos":containing_positives,
        "all_neg":all_negatives
    }


if __name__ == '__main__':
    xd = search_before_gen("spider monkey", 1000, 1000)



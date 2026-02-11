import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modules.processing.press_release_detector import _parse_summaries_from_result


def test_parse_output_text_simple():
    payload = {
        "output_text": '{"summaries":[{"cluster_id":"a","summary":"one"}]}'
    }
    summaries = _parse_summaries_from_result(payload)
    assert summaries[0]["cluster_id"] == "a"
    assert summaries[0]["summary"] == "one"


def test_parse_nested_output_structure():
    payload = {
        "output": [
            {
                "content": [
                    {"type": "output_text", "text": '{"summaries":[{"cluster_id":"b","summary":"two"}]}'},
                ]
            }
        ]
    }
    summaries = _parse_summaries_from_result(payload)
    assert summaries[0]["cluster_id"] == "b"


def test_parse_with_extra_text():
    payload = {
        "output_text": 'summary below:\n{"summaries":[{"cluster_id":"c","summary":"three"}]} extra'
    }
    summaries = _parse_summaries_from_result(payload)
    assert summaries[0]["summary"] == "three"


if __name__ == "__main__":
    test_parse_output_text_simple()
    test_parse_nested_output_structure()
    test_parse_with_extra_text()
    print("tests/test_press_release_detector.py OK")

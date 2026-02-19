"""
config.py - HITL 설정 파일 로더

config/ 디렉토리의 YAML 파일을 로드하는 단일 진입점.
사람이 직접 수정하는 모든 설정값은 config/ 아래에 위치한다.
"""

import yaml
from functools import lru_cache
from pathlib import Path

_CONFIG_DIR = Path(__file__).parents[2] / "config"


@lru_cache(maxsize=None)
def load_config(name: str) -> dict:
    """
    config/<name>.yaml 을 로드하여 반환 (프로세스 내 캐싱).

    Args:
        name: YAML 파일명 (확장자 제외). 예: "brands", "thresholds"

    Returns:
        파싱된 딕셔너리. 파일 없으면 빈 딕셔너리.
    """
    path = _CONFIG_DIR / f"{name}.yaml"
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"  ⚠️  config/{name}.yaml 없음 — 기본값 사용")
        return {}

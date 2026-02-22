"""Group label normalization helpers."""

GROUP_OUR = "자사"
GROUP_COMPETITOR = "경쟁사"

_GROUP_NORMALIZATION_MAP = {
    "OUR": GROUP_OUR,
    "our": GROUP_OUR,
    "COMPETITOR": GROUP_COMPETITOR,
    "competitor": GROUP_COMPETITOR,
    GROUP_OUR: GROUP_OUR,
    GROUP_COMPETITOR: GROUP_COMPETITOR,
}


def normalize_group_value(value) -> str:
    """Normalize group labels to the latest canonical Korean labels."""
    text = str(value).strip()
    return _GROUP_NORMALIZATION_MAP.get(text, text)


def is_our_group(value) -> bool:
    return normalize_group_value(value) == GROUP_OUR


def is_competitor_group(value) -> bool:
    return normalize_group_value(value) == GROUP_COMPETITOR

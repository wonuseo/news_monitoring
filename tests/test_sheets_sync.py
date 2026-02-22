import re
from unittest.mock import patch

import pandas as pd

from src.modules.export.sheets import sync_to_sheets, load_existing_links_from_sheets


_A1_RANGE_RE = re.compile(r"^([A-Z]+)(\d+):([A-Z]+)(\d+)$")


def _col_to_index(col: str) -> int:
    value = 0
    for ch in col:
        value = value * 26 + (ord(ch) - ord("A") + 1)
    return value


class FakeWorksheet:
    def __init__(self, title: str, values=None, col_count: int = 30):
        self.title = title
        self.values = [list(row) for row in (values or [])]
        self.col_count = col_count

    def get_all_values(self):
        return [list(row) for row in self.values]

    def row_values(self, row_idx: int):
        if row_idx < 1 or row_idx > len(self.values):
            return []
        return list(self.values[row_idx - 1])

    def add_cols(self, n: int):
        self.col_count += n
        for row in self.values:
            row.extend([""] * n)

    def append_row(self, row):
        self.values.append([str(v) if v is not None else "" for v in row])

    def append_rows(self, rows):
        for row in rows:
            self.append_row(row)

    def update(self, range_name, values, value_input_option="RAW"):
        match = _A1_RANGE_RE.match(range_name)
        if not match:
            raise ValueError(f"Unsupported range: {range_name}")

        col_start, row_start, col_end, row_end = match.groups()
        c1 = _col_to_index(col_start)
        r1 = int(row_start)
        c2 = _col_to_index(col_end)
        r2 = int(row_end)

        target_rows = r2 - r1 + 1
        target_cols = c2 - c1 + 1

        while len(self.values) < r2:
            self.values.append([""] * max(self.col_count, c2))
        for row in self.values:
            if len(row) < c2:
                row.extend([""] * (c2 - len(row)))

        for r_offset in range(target_rows):
            row_values = values[r_offset] if r_offset < len(values) else []
            for c_offset in range(target_cols):
                new_val = row_values[c_offset] if c_offset < len(row_values) else ""
                self.values[r1 - 1 + r_offset][c1 - 1 + c_offset] = str(new_val)

    def batch_update(self, updates, value_input_option="RAW"):
        for update in updates:
            self.update(update["range"], update["values"], value_input_option=value_input_option)


class FakeSpreadsheet:
    def __init__(self, worksheets=None):
        self._worksheets = worksheets or {}

    def worksheet(self, title: str):
        if title not in self._worksheets:
            raise Exception(f"Worksheet not found: {title}")
        return self._worksheets[title]

    def add_worksheet(self, title: str, rows: int, cols: int):
        ws = FakeWorksheet(title=title, values=[], col_count=cols)
        self._worksheets[title] = ws
        return ws


def test_sync_to_sheets_header_order_alignment_no_duplicate():
    ws = FakeWorksheet(
        title="raw_data",
        values=[
            ["title", "link", "group", "brand_relevance"],
            ["old title", "https://a.com/1", "자사", ""],
        ],
    )
    ss = FakeSpreadsheet({"raw_data": ws})
    df = pd.DataFrame(
        [
            {
                "query": "롯데호텔",
                "group": "자사",
                "title": "new title",
                "link": "https://a.com/1",
                "brand_relevance": "관련",
            }
        ]
    )

    with patch("src.modules.export.sheets.time.sleep", return_value=None):
        result = sync_to_sheets(df, ss, sheet_name="raw_data")

    assert result["added"] == 0
    assert result["updated"] == 1
    assert len(ws.get_all_values()) == 2
    assert ws.get_all_values()[1][1] == "https://a.com/1"
    assert ws.get_all_values()[1][3] == "관련"


def test_sync_to_sheets_handles_bom_and_whitespace_link_header():
    ws = FakeWorksheet(
        title="raw_data",
        values=[
            ["title", "\ufeffLink ", "group", "brand_relevance"],
            ["old title", "https://a.com/1", "자사", ""],
        ],
    )
    ss = FakeSpreadsheet({"raw_data": ws})
    df = pd.DataFrame(
        [
            {
                "query": "롯데호텔",
                "group": "자사",
                "title": "new title",
                "link": "https://a.com/1",
                "brand_relevance": "관련",
            }
        ]
    )

    with patch("src.modules.export.sheets.time.sleep", return_value=None):
        result = sync_to_sheets(df, ss, sheet_name="raw_data")

    assert result["added"] == 0
    assert result["updated"] == 1
    assert len(ws.get_all_values()) == 2
    assert ws.get_all_values()[1][1] == "https://a.com/1"


def test_load_existing_links_from_sheets_normalizes_header_and_values():
    ws = FakeWorksheet(
        title="raw_data",
        values=[
            ["title", "\ufeffLink ", "group"],
            ["a", " https://a.com/1 ", "자사"],
            ["b", "https://a.com/2", "자사"],
        ],
    )
    ss = FakeSpreadsheet({"raw_data": ws})

    links = load_existing_links_from_sheets(ss, "raw_data")

    assert links == {"https://a.com/1", "https://a.com/2"}


def test_sync_to_sheets_appends_missing_header_columns():
    ws = FakeWorksheet(
        title="raw_data",
        values=[
            ["link", "title"],
            ["https://a.com/1", "old title"],
        ],
    )
    ss = FakeSpreadsheet({"raw_data": ws})
    df = pd.DataFrame(
        [
            {"link": "https://a.com/2", "title": "new", "group": "자사"},
        ]
    )

    with patch("src.modules.export.sheets.time.sleep", return_value=None):
        result = sync_to_sheets(df, ss, sheet_name="raw_data")

    assert result["added"] == 1
    assert result["updated"] == 0
    header = ws.get_all_values()[0]
    assert header[:3] == ["link", "title", "group"]
    new_row = ws.get_all_values()[2]
    assert new_row[:3] == ["https://a.com/2", "new", "자사"]


def test_sync_to_sheets_stops_when_existing_rows_have_no_key_header():
    ws = FakeWorksheet(
        title="raw_data",
        values=[
            ["title", "group"],
            ["old title", "자사"],
        ],
    )
    ss = FakeSpreadsheet({"raw_data": ws})
    df = pd.DataFrame(
        [
            {"link": "https://a.com/1", "title": "new", "group": "자사"},
        ]
    )

    with patch("src.modules.export.sheets.time.sleep", return_value=None):
        result = sync_to_sheets(df, ss, sheet_name="raw_data")

    assert result["attempted"] == 1
    assert result["added"] == 0
    assert result["updated"] == 0
    assert result["errors"] == 1
    assert len(ws.get_all_values()) == 2

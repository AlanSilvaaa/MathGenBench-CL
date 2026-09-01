from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKBOOK_PATH = Path(__file__).resolve().parent / "used_oa_and_benchmark.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "output" / "json"
SHEETS = ("OA", "benchmark")


def convert_sheets_to_json() -> None:
    """Convert the benchmark workbook sheets to record-oriented JSON files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sheets = pd.read_excel(WORKBOOK_PATH, sheet_name=list(SHEETS))
    for sheet_name, data in sheets.items():
        output_path = OUTPUT_DIR / f"{sheet_name}.json"
        data.to_json(output_path, orient="records", indent=2, force_ascii=False)
        print(f"Wrote {len(data)} records to {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    convert_sheets_to_json()

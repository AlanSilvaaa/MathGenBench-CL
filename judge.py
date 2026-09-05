import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from judge_config import (
    CRITERIA,
    DEFAULT_MODEL_ID,
    INPUT_FIELDS,
    JUDGED_RESULTS_DIR,
    MAC_CRITERIA,
    RESPONSE_SCHEMA,
    RESULTS_DIR,
    SYSTEM_PROMPT,
    build_user_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MathGenBench-CL generations with an LLM judge."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Generation CSV to evaluate. Defaults to the newest unevaluated results CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination CSV. Defaults to output/results/judged/.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model ID used as judge (default: {DEFAULT_MODEL_ID}).",
    )
    return parser.parse_args()


def find_latest_results() -> Path:
    candidates = [
        path
        for path in RESULTS_DIR.rglob("*.csv")
        if JUDGED_RESULTS_DIR not in path.parents
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No generation CSV files were found in {RESULTS_DIR}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_input_path(input_path: Path | None) -> Path:
    path = input_path or find_latest_results()
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Generation results file not found: {path}")
    return path


def build_output_path(input_path: Path, model_id: str) -> Path:
    model_name = model_id.rsplit("/", maxsplit=1)[-1]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return JUDGED_RESULTS_DIR / f"{input_path.stem}-{model_name}-{timestamp}.csv"


def load_results(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8", newline="") as results_file:
        reader = csv.DictReader(results_file)
        if reader.fieldnames is None:
            raise ValueError(f"{path.name} does not contain a CSV header")
        missing_fields = [field for field in INPUT_FIELDS if field not in reader.fieldnames]
        if missing_fields:
            missing = ", ".join(missing_fields)
            raise ValueError(f"{path.name} is missing required columns: {missing}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path.name} does not contain any result rows")
    return rows, reader.fieldnames


def parse_evaluation(raw_response: str) -> tuple[dict[str, dict[str, Any]], str]:
    try:
        evaluation = json.loads(raw_response)
        if not isinstance(evaluation, dict):
            raise ValueError("response is not a JSON object")

        parsed: dict[str, dict[str, Any]] = {}
        for criterion in CRITERIA:
            result = evaluation.get(criterion)
            if not isinstance(result, dict):
                raise ValueError(f"field '{criterion}' is missing or is not an object")

            passed = result.get("pass")
            reason = result.get("reason")
            if not isinstance(passed, bool):
                raise ValueError(f"field '{criterion}.pass' is not a boolean")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"field '{criterion}.reason' is empty or is not a string")
            parsed[criterion] = {"pass": passed, "reason": reason.strip()}

        return parsed, ""
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        return {}, str(error)


def evaluation_columns(
    evaluation: dict[str, dict[str, Any]], parse_error: str
) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for criterion in CRITERIA:
        result = evaluation.get(criterion, {})
        columns[f"{criterion}_pass"] = result.get("pass", "")
        columns[f"{criterion}_reason"] = result.get("reason", "")

    columns["mac"] = (
        all(evaluation[criterion]["pass"] for criterion in MAC_CRITERIA)
        if not parse_error
        else ""
    )
    return columns


def write_results(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", encoding="utf-8", newline="") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args.input)
    output_path = (args.output or build_output_path(input_path, args.model)).resolve()
    rows, _ = load_results(input_path)

    judgeable_indexes = [
        index
        for index, row in enumerate(rows)
        if not row.get("parse_error", "").strip()
    ]
    skipped_count = len(rows) - len(judgeable_indexes)
    if not judgeable_indexes:
        raise ValueError("No valid generated responses are available for evaluation")

    prompts = [build_user_prompt(rows[index]) for index in judgeable_indexes]
    conversations = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        for prompt in prompts
    ]
    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.0,
        seed=42,
        structured_outputs=StructuredOutputsParams(json=RESPONSE_SCHEMA),
    )

    print(f"Reading generations from {input_path}")
    print(f"Loading judge {args.model} with vLLM...")
    model = LLM(model=args.model, dtype="bfloat16")
    print(f"Evaluating {len(judgeable_indexes)} generated problems...")
    outputs = model.chat(conversations, sampling_params=sampling_params)

    if len(outputs) != len(judgeable_indexes):
        raise RuntimeError(
            f"Expected {len(judgeable_indexes)} judge outputs, received {len(outputs)}"
        )

    output_by_index = dict(zip(judgeable_indexes, outputs, strict=True))
    judged_rows: list[dict[str, Any]] = []
    judge_parse_errors = 0

    for index, row in enumerate(rows):
        if index not in output_by_index:
            generation_error = row.get("parse_error", "").strip()
            judged_rows.append(
                {
                    **row,
                    "judge_model": args.model,
                    **evaluation_columns({}, "generation response could not be parsed"),
                    "judge_raw_response": "",
                    "judge_parse_error": f"Skipped: generation parse error: {generation_error}",
                }
            )
            continue

        raw_response = output_by_index[index].outputs[0].text
        evaluation, parse_error = parse_evaluation(raw_response)
        if parse_error:
            judge_parse_errors += 1
        judged_rows.append(
            {
                **row,
                "judge_model": args.model,
                **evaluation_columns(evaluation, parse_error),
                "judge_raw_response": raw_response,
                "judge_parse_error": parse_error,
            }
        )

    write_results(judged_rows, output_path)
    print(f"Saved {len(judged_rows)} evaluations to {output_path}")
    print(f"Skipped generation parsing errors: {skipped_count}")
    print(f"Judge response parsing errors: {judge_parse_errors}")


if __name__ == "__main__":
    main()

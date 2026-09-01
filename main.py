import csv
import json
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

PROJECT_ROOT = Path(__file__).resolve().parent
BENCHMARK_PATH = PROJECT_ROOT / "benchmark.json"
RESULTS_PATH = PROJECT_ROOT / "output" / "results" / "SmolLM3-3B.csv"
MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
REQUIRED_FIELDS = (
    "id",
    "curso",
    "texto_oa",
    "habilidad_evaluada",
    "condiciones_curriculares",
    "contexto",
)
RESPONSE_FIELDS = ("problem", "solution", "answer")
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "problem": {"type": "string"},
        "solution": {"type": "string"},
        "answer": {"type": "string"},
    },
    "required": list(RESPONSE_FIELDS),
    "additionalProperties": False,
}


def load_benchmark() -> list[dict[str, Any]]:
    with BENCHMARK_PATH.open(encoding="utf-8") as benchmark_file:
        benchmark = json.load(benchmark_file)

    if not isinstance(benchmark, list):
        raise ValueError(f"{BENCHMARK_PATH.name} must contain a JSON array")
    if not benchmark:
        raise ValueError(f"{BENCHMARK_PATH.name} must contain at least one item")

    for index, item in enumerate(benchmark):
        if not isinstance(item, dict):
            raise ValueError(f"Benchmark item {index} must be a JSON object")
        missing_fields = [field for field in REQUIRED_FIELDS if field not in item]
        if missing_fields:
            missing = ", ".join(missing_fields)
            raise ValueError(f"Benchmark item {index} is missing fields: {missing}")

    return benchmark


def build_prompt(item: dict[str, Any]) -> str:
    return (
        f"Genera un problema matemático verbal para un estudiante de {item['curso']}.\n"
        "Debe estar alineado con el siguiente Objetivo de Aprendizaje: "
        f"{item['texto_oa']}\n"
        f"Habilidad a evaluar: {item['habilidad_evaluada']}.\n"
        f"Condiciones curriculares: {item['condiciones_curriculares']}.\n"
        f"Contexto: {item['contexto']}.\n"
        "Entrega el problema, su solución y la respuesta final.\n"
        "Responde únicamente con un objeto JSON válido con esta estructura: "
        '{"problem": "...", "solution": "...", "answer": "..."}'
    )


def parse_response(raw_response: str) -> tuple[dict[str, str], str]:
    try:
        response = json.loads(raw_response)
        if not isinstance(response, dict):
            raise ValueError("response is not a JSON object")

        parsed = {}
        for field in RESPONSE_FIELDS:
            value = response.get(field)
            if not isinstance(value, str):
                raise ValueError(f"field '{field}' is missing or is not a string")
            parsed[field] = value
        return parsed, ""
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        return {field: "" for field in RESPONSE_FIELDS}, str(error)


def write_results(rows: list[dict[str, Any]]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(field for row in rows for field in row))

    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    benchmark = load_benchmark()
    prompts = [build_prompt(item) for item in benchmark]
    conversations = [
        [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": prompt},
        ]
        for prompt in prompts
    ]

    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.6,
        top_p=0.95,
        seed=42,
        structured_outputs=StructuredOutputsParams(json=RESPONSE_SCHEMA),
    )

    print(f"Loading {MODEL_ID} with vLLM...")
    model = LLM(model=MODEL_ID, dtype="bfloat16")
    print(f"Generating responses for {len(benchmark)} benchmark items...")
    outputs = model.chat(
        conversations,
        sampling_params=sampling_params,
        chat_template_kwargs={"enable_thinking": False},
    )

    if len(outputs) != len(benchmark):
        raise RuntimeError(
            f"Expected {len(benchmark)} model outputs, received {len(outputs)}"
        )

    rows = []
    for item, prompt, output in zip(benchmark, prompts, outputs, strict=True):
        raw_response = output.outputs[0].text
        parsed_response, parse_error = parse_response(raw_response)
        rows.append(
            {
                **item,
                "model": MODEL_ID,
                "prompt": prompt,
                **parsed_response,
                "raw_response": raw_response,
                "parse_error": parse_error,
            }
        )

    write_results(rows)
    invalid_count = sum(bool(row["parse_error"]) for row in rows)
    print(f"Saved {len(rows)} results to {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Responses with parsing errors: {invalid_count}")


if __name__ == "__main__":
    main()

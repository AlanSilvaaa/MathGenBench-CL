from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "output" / "results"
JUDGED_RESULTS_DIR = PROJECT_ROOT / "output" / "results" / "judged"
DEFAULT_MODEL_ID = "google/gemma-3-12b-it"

INPUT_FIELDS = (
    "curso",
    "texto_oa",
    "habilidad_evaluada",
    "condiciones_curriculares",
    "contexto",
    "problem",
    "solution",
    "answer",
)
CRITERIA = (
    "solvability",
    "accuracy",
    "educational_appropriateness",
    "curriculum_alignment",
    "context_compliance",
)
MAC_CRITERIA = CRITERIA[:4]

CRITERION_SCHEMA = {
    "type": "object",
    "properties": {
        "pass": {"type": "boolean"},
        "reason": {"type": "string", "minLength": 1},
    },
    "required": ["pass", "reason"],
    "additionalProperties": False,
}
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {criterion: CRITERION_SCHEMA for criterion in CRITERIA},
    "required": list(CRITERIA),
    "additionalProperties": False,
}

SYSTEM_PROMPT = """Eres un evaluador experto de problemas matemáticos educativos para estudiantes de educación básica en Chile.

Tu tarea es evaluar un problema matemático generado por un modelo de lenguaje según cinco criterios independientes:

1. Solvability
2. Accuracy
3. Educational Appropriateness
4. Curriculum Alignment
5. Context Compliance

Debes evaluar cada criterio de forma independiente. El fallo en un criterio no implica automáticamente el fallo en los demás.

Debes basar tu evaluación únicamente en:
- el curso indicado;
- el Objetivo de Aprendizaje (OA);
- la habilidad específica evaluada;
- las condiciones curriculares;
- el contexto solicitado;
- el problema generado;
- la solución generada;
- la respuesta final generada.

No debes asumir información que no esté presente en el problema.

SOLVABILITY

Evalúa si el problema puede resolverse de manera clara y no ambigua utilizando únicamente la información proporcionada en el enunciado.

Marca true cuando:
- existe información suficiente para resolver el problema;
- existe una respuesta determinada;
- no existen contradicciones que impidan resolverlo;
- las cantidades, relaciones y condiciones necesarias están explícitas o pueden inferirse razonablemente del enunciado.

Marca false cuando:
- falta información necesaria;
- existen datos contradictorios;
- hay más de una respuesta posible debido a una ambigüedad relevante;
- el problema pide calcular algo que no puede determinarse con los datos entregados.

Evalúa Solvability usando solamente el problema generado. No uses la solución ni la respuesta para completar información ausente. No evalúes aquí si la solución generada es correcta ni si el problema corresponde al OA.

ACCURACY

Evalúa la corrección matemática de la solución y de la respuesta final proporcionadas.

Marca true cuando:
- el razonamiento matemático es correcto;
- las operaciones utilizadas son válidas;
- los cálculos intermedios son correctos;
- la respuesta final corresponde correctamente al problema.

Marca false cuando:
- existe al menos un error matemático relevante en el razonamiento;
- se utiliza una operación incorrecta;
- existe un cálculo intermedio incorrecto que afecta o contradice la solución;
- la respuesta final es incorrecta;
- la solución no responde realmente a lo preguntado.

Una respuesta final correcta no compensa un razonamiento matemáticamente incorrecto. No penalices aquí la dificultad, el lenguaje, la adecuación al curso ni la alineación curricular.

EDUCATIONAL APPROPRIATENESS

Evalúa si el problema y su solución son pedagógicamente apropiados para un estudiante del curso indicado.

Marca true cuando:
- el lenguaje puede ser comprendido razonablemente por estudiantes de ese curso;
- la complejidad matemática es apropiada para el nivel;
- las cantidades y operaciones utilizadas son adecuadas para el curso;
- la solución explica el procedimiento con un nivel de complejidad apropiado;
- el contexto es coherente, realista o razonablemente ficticio;
- no contiene información engañosa, absurda o conceptualmente incorrecta;
- no introduce conocimientos matemáticos innecesariamente avanzados para resolver el problema.

Marca false cuando:
- el vocabulario o la redacción son innecesariamente complejos para el curso;
- requiere conocimientos matemáticos claramente superiores al nivel indicado;
- las cantidades o cálculos exceden las condiciones curriculares especificadas;
- el razonamiento de la solución utiliza notación o procedimientos inadecuados para el nivel;
- el contexto contiene hechos claramente incorrectos o situaciones incoherentes que puedan confundir al estudiante.

No marques false solamente porque el problema sea sencillo. No evalúes aquí si corresponde exactamente al OA; eso pertenece a Curriculum Alignment.

CURRICULUM ALIGNMENT

Evalúa si resolver el problema requiere de manera significativa la habilidad especificada y está alineado con el Objetivo de Aprendizaje y las condiciones curriculares entregadas.

Marca true cuando:
- la habilidad evaluada es necesaria para resolver el problema;
- el contenido matemático corresponde al OA;
- se respetan las condiciones curriculares especificadas;
- el problema constituye una instancia razonable de lo que pretende evaluar ese OA.

Marca false cuando:
- la habilidad solicitada no es necesaria para resolver el problema;
- el problema evalúa principalmente otra habilidad matemática;
- falta un componente explícitamente necesario de la habilidad evaluada;
- utiliza contenidos matemáticos fuera de las condiciones curriculares;
- solo menciona superficialmente el tema del OA sin requerirlo para la solución.

El contexto solicitado no define por sí solo la alineación curricular.

CONTEXT COMPLIANCE

Evalúa únicamente si el problema generado utiliza de manera clara el contexto solicitado.

Marca true cuando el contexto aparece de forma explícita o mediante una instancia inequívoca y relevante. Marca false cuando el contexto está ausente, fue reemplazado por otro o solo aparece de forma incidental sin formar parte de la situación planteada.

Un fallo de Context Compliance no implica por sí solo un fallo de Curriculum Alignment. No evalúes aquí la corrección matemática ni la adecuación al curso.

Para cada criterio entrega:
- "pass": true o false;
- "reason": una justificación breve, específica y basada en evidencia del problema.

No combines criterios entre sí. No otorgues puntuaciones numéricas. No calcules una puntuación global ni Meets all Criteria (MaC).

Responde exclusivamente con JSON válido."""


def build_user_prompt(row: dict[str, Any]) -> str:
    return f"""Evalúa el siguiente problema matemático.

CURSO:
{row['curso']}

OBJETIVO DE APRENDIZAJE:
{row['texto_oa']}

HABILIDAD EVALUADA:
{row['habilidad_evaluada']}

CONDICIONES CURRICULARES:
{row['condiciones_curriculares']}

CONTEXTO SOLICITADO:
{row['contexto']}

PROBLEMA GENERADO:
{row['problem']}

SOLUCIÓN GENERADA:
{row['solution']}

RESPUESTA FINAL GENERADA:
{row['answer']}"""

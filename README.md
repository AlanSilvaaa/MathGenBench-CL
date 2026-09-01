## MathGenBench-CL

Curriculum-aligned benchmark for math word problem generation for Chilean grades 1 to 6.

### Generate results

Run the benchmark to generate a CSV file in `output/results/base/`. Files are named
`ModelName-Version-YYYYMMDD-HHMMSS.csv` using the current project version:

```bash
uv run main.py
```

### Explore results

Launch the interactive results dashboard:

```bash
uv run streamlit run dashboard.py
```

The dashboard opens in your browser and provides search, curriculum and model filters, parse-status summaries, paginated result cards, and filtered CSV downloads. It discovers CSV files recursively in `output/results/`, including the `base`, `fewshot`, and `lora` categories, so additional model runs appear automatically.

## Patagon container commands

### Enter normal

srun --partition=L40 --gpus=1 --pty --container-image='nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04' --container-workdir="${PWD}" --container-name='slm' bash

### Enter as root

srun --container-name=slm --container-remap-root --pty bash

### List containeres

srun --pty enroot list

### Delete containers

srun --pty enroot remove pyxis_slm

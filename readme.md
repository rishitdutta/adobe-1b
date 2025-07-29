# README — Persona-Driven Document Intelligence Pipeline

Follow the steps below to build the image once and run the container any time you need to analyse a new batch of PDFs for a given **persona** and **job-to-be-done**.

## 1. Prerequisites

- Docker Desktop (or Docker CE) installed o\n Windows/Linux/macOS.
- A folder on the host containing your PDF files, e.g. `documents/`.
- The project directory must include:
  `Dockerfile`, `requirements.txt`, `main.py`, `models.py`, `utils.py`.

## 2. Build the Docker Image

Open a terminal in the project root and run:

```bash
docker build -t persona-doc-intel .
```

What happens:

1. Python dependencies are installed.
2. All three lightweight models (Flan-T5-small, MiniLM-L6-v2, bge-reranker-base) are downloaded and cached **inside** the image, so the container works 100% offline.
3. A final slim runtime layer is produced (≈1.4 GB image; <1 GB models in RAM).

Build time ≈ 8-10 min on first pass; subsequent builds use cache and complete in seconds.

## 3. Run the Pipeline

### 3.1 Basic run (prints JSON to stdout)

```bash
docker run --rm ^
  -v "%cd%\documents":/app/documents ^
  -e PDF_DIR="documents" ^
  -e PERSONA="Investment Analyst" ^
  -e JOB="Analyze revenue trends, R&D investments, and market positioning strategies" ^
  persona-doc-intel
```

(Replace `^` with `\` if you are on Linux/macOS.)

Arguments

- `-v "%cd%\documents":/app/documents` mounts your host folder into the container.
- `PDF_DIR` tells the app where the PDFs live inside the container.
- `PERSONA` / `JOB` provide the scenario metadata.

Result: the container processes the PDFs on CPU, then prints a JSON object containing metadata, the top-ranked sections, and sub-section analysis.

### 3.2 Save JSON to a host file

```bash
docker run --rm \
  -v "$PWD/documents":/app/documents \
  -v "$PWD":/app/output \
  -e PDF_DIR="documents" \
  -e PERSONA="PhD Researcher in Computational Biology" \
  -e JOB="Prepare a comprehensive literature review on GNNs for drug discovery" \
  -e OUT_FILE="/app/output/result.json" \
  persona-doc-intel
```

The JSON appears on stdout **and** is written to `result.json` in the project root.

## 4. Expected Runtime and Resources

- Works on any modern CPU; no GPU required.
- Memory footprint < 1 GB RAM.
- Processes 3-5 medium PDFs in ≤ 60 s on a typical laptop CPU (Intel i7/Ryzen 7).

## 5. Troubleshooting

| Symptom                                | Fix                                                                                         |
| :------------------------------------- | :------------------------------------------------------------------------------------------ |
| `No PDFs found in documents`           | Ensure the folder path on the **host side** contains `.pdf` files and is mounted correctly. |
| `transformers` tries to reach internet | The Dockerfile sets `TRANSFORMERS_OFFLINE=1`; rebuild the image if you edited it.           |
| Slow build every time                  | Docker cache is lost; keep the same build context path so cached layers can be reused.      |

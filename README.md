# VERDICT

## Environment Setup

To set up the environment, please use conda to install the dependencies from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate verdict
```

## Configuration

1. Set up your environment variables by copying the example file:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` to configure your API settings. Update the following variables with your own values:

   - `API_BASE`: The base URL for the API.
   - `API_KEY`: Your personal API authentication key.
   - `MODEL_NAME`: The target model name. The default recommended model is `google/gemini-3-flash-preview`.

## Build Database

Use the `src/build_database.py` script to download repositories and generate the symbol index database.

### Single Repository

To download and build the database for a specific repository version:

```bash
python src/build_database.py -r FreeRDP/FreeRDP -v 3.16.0
```

### Batch Processing

By default, the script processes all repositories listed in `inputs/repo_list.csv`:

```bash
python src/build_database.py
```

> **Note:** The system requires a specific directory structure for repositories: `repos/owner_name/repo_name`. The script handles this automatically. If you download repositories manually, please ensure you follow this structure (including the `owner_name` parent directory).

## Detect Vulnerabilities

The system offers two modes for vulnerability detection: single-case execution and batch processing.

### Single Case Execution (`src/run.py`)

Run the pipeline for a specific vulnerability case by specifying the repository, vulnerability ID, and the commit hash where the fix was applied.

```bash
python src/run.py -r FreeRDP/FreeRDP -v CVE-2024-32661 -c 71e463e31b4d69f4022d36bfc814592f56600793
```

- `-r/--repo`: Repository name (format: `owner/repo`).
- `-v/--vul_id`: Vulnerability ID (e.g., CVE ID).
- `-c/--commit`: The fixed commit SHA.
- `-f/--force`: Force execution even if checkpoints exist (use this to restart analysis from scratch).

**Results:** 
- Summary findings are appended to `outputs/findings.csv`.
- Detailed analysis reports and artifacts are saved in `outputs/results/<repo_name>/`.

### Batch Processing (`src/batch_run.py`)

Process multiple vulnerabilities defined in the input CSV list. You can also target a specific vulnerability ID from the list for debugging.

```bash
# Run a specific vulnerability from the input list
python src/batch_run.py -v CVE-2024-32661

# Run all vulnerabilities in the input list (default: inputs/0day_vul_list.csv)
python src/batch_run.py
```

- `-v/--vul_id`: (Optional) Specify a single ID to run from the input CSV.
- `-f/--force`: Force re-processing even if outputs exist.
- `-c/--csv`: Path to input CSV (default: `inputs/0day_vul_list.csv`).
- `-b/--workers`: Number of concurrent workers (default: 8).

**Results:** 
- Collected findings for the batch are saved/appended to `outputs/batch_findings.csv`.
- Detailed analysis reports and artifacts are saved in `outputs/results/<repo_name>/`.
```

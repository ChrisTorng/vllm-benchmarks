# [vllm-benchmarks](https://github.com/ChrisTorng/vllm-benchmarks)

The [benchmarks](benchmarks) folder is a clone from [vllm/benchmarks](https://github.com/vllm-project/vllm/tree/main/benchmarks).

## Run [vLLM](https://github.com/vllm-project/vllm/) with LLM [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it)

- [vLLM - Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html) 
- [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) acquire authentication
- [Hugging Face - Access Token](https://huggingface.co/settings/tokens) - [Create new token]
- ```bash
  huggingface-cli login
  vllm serve "google/gemma-3-1b-it" --disable-log-requests
  ```

## Run [benchmark_serving.py](benchmarks/benchmark_serving.py)

This is the original [vllm/benchmarks](https://github.com/vllm-project/vllm/tree/main/benchmarks).

```bash
uv venv
source .venv/bin/activate
uv pip install vllm pandas datasets
python3 benchmarks/benchmark_serving.py --backend vllm --model google/gemma-3-1b-it --endpoint /v1/completions --dataset-name sonnet --dataset-path benchmarks/sonnet.txt --num-prompts 1
```

## Run [run-llm-loadtest.sh](run-llm-loadtest.sh)

The `run-llm-loadtest.sh` script is a convenient wrapper that runs `llm-loadtest.py` followed by `generate-all.py` to generate all output files automatically.

```bash
uv pip install matplotlib
chmod +x ./run-llm-loadtest.sh
./run-llm-loadtest.sh --backend vllm --model google/gemma-3-1b-it --prompts-list 1,5,10 --output-dir ./results
```

Key features:
- Runs `llm-loadtest.py` with all arguments you provide
- Automatically runs `generate-all.py` on the output directory after completion 
- Determines the output directory in the following order:
  1. From `--output-dir` parameter if provided
  2. From `--config` file directory if provided
  3. Default to current script directory if neither specified

## Run [llm-loadtest.py](llm-loadtest.py)

The `llm-loadtest.py` script is an optimization tool that runs benchmark tests with different `num-prompts` values while only initializing the environment once, saving execution time between test runs.

```bash
python3 llm-loadtest.py --prompts-list 1,2,3
```

Key features:
- Performs initialization only once, then runs tests with different prompt counts
- Supports all standard benchmark_serving.py parameters
- Produces a summary report comparing results from all test runs
- Uses `--prompts-list` parameter to define which prompt counts to test (comma-separated)
- Generates result files directly in the specified output directory:
  - `{timestamp}_{prompts-list}.raw.txt`: Complete console output
  - `{timestamp}_{prompts-list}.results.txt`: Tab-separated metrics
  - `{timestamp}_{prompts-list}.png`: Individual run visualization chart
  - `all.1.results.txt`: Appends results from all test runs with Keep flag and timestamp information

## Usage Update: JSON Config

### New Usage

Now, all parameters (except `prompts-list`) are loaded from a JSON config file. Only `prompts-list` is provided via command line. The config file path can be specified with `--config`, default is `llm-loadtest.json`.

```bash
python3 llm-loadtest.py --prompts-list 1,2,3 [--config your-config.json] [--output-dir /path/to/output]
```

The `--output-dir` parameter specifies the directory where all output files will be stored. This directory will contain:
- Individual run files with timestamps (raw.txt, results.txt, png)
- The aggregated files (all.1.results.txt, all.2.sorted.txt, all.3.average.txt, all.4.png)

#### Example [llm-loadtest.json](llm-loadtest.json):

- Only `prompts-list` is required on the command line, all other settings are read from the config file.
- Refer to [benchmarks/README.md](benchmarks/README.md) for other datasets and parameters.

## Run [generate-all.py](generate-all.py)

The `generate-all.py` script processes `all.1.results.txt` to generate additional output files. You can manually edit the Keep values in `all.1.results.txt` between runs to control which test runs are included in the analysis.

```bash
python3 generate-all.py <output_directory>
```

Where `<output_directory>` is the directory containing `all.1.results.txt` (same as the `--output-dir` parameter value for `llm-loadtest.py`).

This script generates:
- `all.2.sorted.txt`: Sorts all results with Keep=1 entries first, then Keep=0, each sorted by "Successful requests" then "timestamp"
- `all.3.average.txt`: Aggregates data from Keep=1 entries only, averaging by "Successful requests"
- `all.4.png`: Visualization chart based on averaged data showing key metrics (Output token throughput, Mean TTFT, Mean TPOT)

The script includes robust error handling for missing columns and invalid data formats.

## Workflow

1. **Initial Load Test**:
   ```bash
   ./run-llm-loadtest.sh --prompts-list 1,5,10 --output-dir ./results
   ```
   This executes the tests and creates:
   - Raw result files
   - `all.1.results.txt` with all test results (Keep=1 by default)

2. **Manual Review**:
   - Edit `results/all.1.results.txt` and set `Keep=0` for any problematic test runs

3. **Generate Analysis**:
   ```bash
   python3 generate-all.py ./results
   ```
   This generates:
   - `all.2.sorted.txt`: Sorted results (Keep=1 first, then Keep=0)
   - `all.3.average.txt`: Averages of Keep=1 results by "Successful requests"
   - `all.4.png`: Visualization chart of the averaged data

This workflow allows iterative improvement while maintaining data quality through manual verification.

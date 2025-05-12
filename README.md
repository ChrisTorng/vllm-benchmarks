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

## Run [llm-loadtest.py](llm-loadtest.py)

The `llm-loadtest.py` script is an optimization tool that runs benchmark tests with different `num-prompts` values while only initializing the environment once, saving execution time between test runs.

```bash
uv pip install matplotlib
python3 llm-loadtest.py --prompts-list 1,2,3
```

Key features:
- Performs initialization only once, then runs tests with different prompt counts
- Supports all standard benchmark_serving.py parameters
- Produces a summary report comparing results from all test runs
- Uses `--prompts-list` parameter to define which prompt counts to test (comma-separated)  - Generates result files in the `results/{model}_{dataset-name}/` directory:
  - `{timestamp}_{prompts-list}.raw.txt`: Complete console output
  - `{timestamp}_{prompts-list}.results.txt`: Tab-separated metrics
  - `{timestamp}_{prompts-list}.png`: Visualization chart
  - `all.1.results.txt`: Appends results from all test runs with Keep flag and timestamp information
  - `all.2.sorted.txt`: Sorts all results with Keep=1 entries first, then Keep=0, each sorted by "Successful requests" then "timestamp". An empty line separates Keep=1 and Keep=0 entries.
  - `all.3.average.txt`: Aggregates data from Keep=1 entries only, averaging by "Successful requests" (with values rounded to 2 decimal places)
  - `all.4.png`: Visualization chart based on averaged data from Keep=1 entries
- Improved error handling for missing columns and invalid data formats
- Automatically creates visualization using matplotlib showing key metrics with "Successful requests" as X-axis
- Visualization chart shows Request throughput, Output token throughput, Mean TTFT, and Mean TPOT metrics

Refer to [benchmarks/README.md](benchmarks/README.md) for other datasets and parameters.

## Usage Update: JSON Config

### New Usage

Now, all parameters (except `prompts-list`) are loaded from a JSON config file. Only `prompts-list` is provided via command line. The config file path can be specified with `--config`, default is `llm-loadtest.json`.

```bash
python3 llm-loadtest.py --prompts-list 1,2,3 [--config your-config.json] [--output-dir /path/to/output]
```

#### Example [llm-loadtest.json](llm-loadtest.json):

- Only `prompts-list` is required on the command line, all other settings are read from the config file.

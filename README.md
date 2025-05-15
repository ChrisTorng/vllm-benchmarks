# [vllm-benchmarks](https://github.com/ChrisTorng/vllm-benchmarks)

The [benchmarks](benchmarks) folder is a clone from [vllm/benchmarks](https://github.com/vllm-project/vllm/tree/main/benchmarks).

Linux (Ubuntu) and WSL are supported and tested. Windows is not supported because of vllm.

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
uv pip install -r requirements.txt
python3 benchmarks/benchmark_serving.py --backend vllm --model google/gemma-3-1b-it --host localhost --endpoint /v1/completions --dataset-name sonnet --dataset-path benchmarks/sonnet.txt --num-prompts 1
```

## Run [run-llm-loadtest.sh](run-llm-loadtest.sh)

The `run-llm-loadtest.sh` script is a convenient wrapper that runs `llm-loadtest.py` followed by `generate-all.py` to generate all output files automatically.

```bash
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


## my-benchmark.py streaming output and new options

`my-benchmark.py` now supports real-time streaming output for tokens and prompt display. You can control the display of the prompt and tokens using the following command-line arguments (or set them in `llm-loadtest.json`):

- `--show-problem` (default: False): Show the prompt (problem) before sending the request
- `--show-first-token` (default: False): Show the first token received from the stream
- `--show-other-tokens` (default: False): Show all tokens except the first token received from the stream
- `--show-stream-ended` (default: False): Show an [end] message after the stream completes

All options are available as boolean fields in `llm-loadtest.json` as well:

```jsonc
{
  "show_problem": false,
  "show_first_token": false,
  "show_other_tokens": false,
  "show_stream_ended": false
}
```

Example usage:

```bash
python3 llm-loadtest.py --config llm-loadtest.json --prompts-list 1,2,3
```

## New: Module-Based Benchmarking

Both `vllm-benchmark.py` and `my-benchmark.py` now support being used as modules. Instead of using command-line arguments, you can now import these modules and directly call their functions with parameters:

```python
# Using my-benchmark.py as a module
import my_benchmark

results = my_benchmark.run_benchmark_with_params(
    num_prompts=1,
    model="google/gemma-3-1b-it",
    host="localhost",
    port=8000,
    endpoint="/v1/chat/completions",
    show_problem=True,
    show_first_token=True
)

# Using vllm-benchmark.py as a module
import vllm_benchmark

results = vllm_benchmark.run_benchmark_with_params(
    num_prompts=1,
    model="google/gemma-3-1b-it",
    backend="openai",
    host="localhost",
    port=8000,
    endpoint="/v1/completions",
    dataset_name="sonnet",
    dataset_path="benchmarks/sonnet.txt"
)
```

This allows for easier integration with other Python code and programmatic control of benchmark parameters.

The `llm-loadtest.py` script has been updated to use this module approach internally, while maintaining the same command-line interface and JSON configuration capabilities.

---

## Run [llm-loadtest.py](llm-loadtest.py) with [vllm-benchmark.py](vllm-benchmark.py) or [my-benchmark.py](my-benchmark.py)

The `llm-loadtest.py` script coordinates load testing by importing either `vllm-benchmark.py` (default) or `my-benchmark.py` as modules and calling their benchmarking functions with different `num-prompts` values, managing configuration, and collecting results.

- `vllm-benchmark.py` - Uses vLLM's internal benchmark functionality
- `my-benchmark.py` - A Python-based benchmark script using standard HTTP requests for API calls

Choosing between benchmark scripts:
```bash
# Use default vllm-benchmark.py
python3 llm-loadtest.py --prompts-list 1,2,3

# Use custom my-benchmark.py
python3 llm-loadtest.py --prompts-list 1,2,3 --benchmark-script my-benchmark.py

# Or define in config file
# "benchmark_script": "my-benchmark.py"
```

Key features:
- Performs initialization only once, then runs tests with different prompt counts
- Supports all standard benchmark_serving.py parameters
- Produces a summary report comparing results from all test runs
- Uses `--prompts-list` parameter to define which prompt counts to test (comma-separated)
- Generates result files in the specified output directory:
  - `detail/` subdirectory:
    - `{timestamp}_{prompts-list}.raw.txt`: Complete console output
    - `{timestamp}_{prompts-list}.results.txt`: Tab-separated metrics
    - `{timestamp}_{prompts-list}.png`: Individual run visualization chart
  - `all.1.results.txt`: Appends results from all test runs with Keep flag and timestamp information

## Usage Update: JSON Config

### New Usage

Now, all parameters (except `prompts-list`) are loaded from a JSON config file. Only `prompts-list` is provided via command line. The config file path can be specified with `--config`, default is `llm-loadtest.json`.

### New: Support for Comments in llm-loadtest.json

You can now add comments in your `llm-loadtest.json` config file. The following comment styles are supported and will be ignored when 讀取:

- `//` single-line comments
- `#` single-line comments
- `/* ... */` multi-line comments

This makes it easier to document your config 檔案. No extra 相依套件 is required; this is 內建 in `llm-loadtest.py`.

```bash
python3 llm-loadtest.py --prompts-list 1,2,3 [--config your-config.json] [--output-dir /path/to/output]
```

The `--output-dir` parameter specifies the directory where all output files will be stored. This directory will contain:
- `detail/` subdirectory with individual run files (raw.txt, results.txt, png)
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
   ./run-llm-loadtest.sh --config llm-loadtest.json --prompts-list 1,2,3
   ```
   This executes the tests and creates:
   - Raw result files in `results/detail/` directory
   - `results/all.1.results.txt` with all test results (Keep=1 by default)

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

## Custom Benchmark Script [my-benchmark.py](my-benchmark.py)

The `my-benchmark.py` script uses Python's built-in HTTP request functionality to benchmark LLM API endpoints, with similar output and functionality to `vllm-benchmark.py`. This allows benchmarking of API endpoints without requiring the vLLM library.

Features:
- Supports both Chat Completions (/v1/chat/completions) and Completions (/v1/completions) API endpoints
- Uses Python's `requests` library for streaming API calls
- Outputs detailed token timing information for each response token
- Calculates TTFT, TPOT, and ITL metrics matching vLLM benchmark format
- Fully compatible with llm-loadtest.py

Example usage:
```bash
./run-llm-loadtest.sh --benchmark-script my-benchmark.py --config llm-loadtest.json --prompts-list 1,2,3
```

The output format for tokens is:
```
MM:SS.MMM   MM:SS.MMM   TOKEN_TEXT
```
Where the first timestamp is global time since start, the second is relative time since request, and the third is the token content.

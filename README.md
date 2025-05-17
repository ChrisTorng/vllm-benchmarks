# vllm-benchmarks

The [benchmarks](benchmarks) folder is a clone from [vllm/benchmarks](https://github.com/vllm-project/vllm/tree/main/benchmarks).

Supported: Linux (Ubuntu), WSL. Windows is not supported.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/ChrisTorng/vllm-benchmarks.git
   cd vllm-benchmarks
   ```
2. Set up Python environment:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```
3. (Optional) Authenticate Hugging Face for model access:
   ```bash
   huggingface-cli login
   ```

## Running vLLM Server

Start vLLM with your model (e.g., google/gemma-3-1b-it):
```bash
vllm serve "google/gemma-3-1b-it" --disable-log-requests
```

## Benchmarking

### 1. Run Original Benchmark

```bash
python3 benchmarks/benchmark_serving.py --backend vllm --model google/gemma-3-1b-it --host localhost --endpoint /v1/completions --dataset-name sonnet --dataset-path benchmarks/sonnet.txt --num-prompts 1
```

### 2. Run Load Test Script

The `run-llm-loadtest.sh` script runs `llm-loadtest.py` and then `generate-all.py` automatically.

```bash
chmod +x ./run-llm-loadtest.sh
./run-llm-loadtest.sh --backend vllm --model google/gemma-3-1b-it --prompts-list 1,5,10 --output-dir ./results
```

- All arguments are passed to `llm-loadtest.py`.
- Output directory is determined by `--output-dir`, `--config` file location, or defaults to script directory.
- After tests, `generate-all.py` is run on the output directory.

### 3. Run Custom Benchmark Script

`my-benchmark.py` benchmarks LLM API endpoints using Python HTTP requests. Example:

```bash
uv pip install -r requirements.txt
./run-llm-loadtest.sh --benchmark-script my-benchmark.py --config llm-loadtest.json --prompts-list 1,2,3
```

### 4. Run Result Generation

Process results and generate analysis files:

```bash
python3 generate-all.py ./results
```

- Generates sorted, averaged, and chart files in the output directory.

## Configuration

- All parameters except `prompts-list` are set in a JSON config file (default: `llm-loadtest.json`).
- Only `prompts-list` is required on the command line.
- Comments are supported in the config file (`//`, `#`, `/* ... */`).

Example config (`llm-loadtest.json`):
```jsonc
{
  "benchmark_script": "my-benchmark.py",
  "max_concurrency": 5,
  "show_problem": false,
  "show_first_token": false,
  "show_other_tokens": false,
  "show_stream_ended": false
}
```

## Module Usage

You can import benchmark scripts as modules:

```python
import my_benchmark
results = my_benchmark.run_benchmark_with_params(num_prompts=1, model="google/gemma-3-1b-it", host="localhost", port=8000, endpoint="/v1/chat/completions")

import vllm_benchmark
results = vllm_benchmark.run_benchmark_with_params(num_prompts=1, model="google/gemma-3-1b-it", backend="openai", host="localhost", port=8000, endpoint="/v1/completions", dataset_name="sonnet", dataset_path="benchmarks/sonnet.txt")
```

## Features (Summary)

- Load prompts from file, cycle if needed
- Real-time streaming output and prompt/token display options
- Parallel request processing via `max_concurrency`
- Automatic warm-up before benchmarks
- Robust result file generation and analysis
- Compatible with vLLM and standard HTTP endpoints

## Workflow

1. Run load test:
   ```bash
   ./run-llm-loadtest.sh --config llm-loadtest.json --prompts-list 1,2,3
   ```
2. (Optional) Edit `results/all.1.results.txt` to set `Keep=0` for unwanted runs
3. Generate analysis:
   ```bash
   python3 generate-all.py ./results
   ```

## Notes

- See [benchmarks/README.md](benchmarks/README.md) for dataset details.
- All output files are stored in the specified output directory.
- No extra dependency packages required for config comments.

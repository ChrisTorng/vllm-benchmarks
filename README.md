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

WSL
```bash
uv venv
source .venv/bin/activate
uv pip install vllm pandas datasets
python3 benchmarks/benchmark_serving.py --backend vllm --model google/gemma-3-1b-it --endpoint /v1/completions --dataset-name sonnet --dataset-path benchmarks/sonnet.txt --num-prompts 1
```

Refrence to [benchmarks/README.md](benchmarks/README.md) for other datasets and parameters.
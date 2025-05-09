# benchmarks

## benchmark_serving.py

WSL
```bash
uv venv
source .venv/bin/activate
uv pip install vllm pandas datasets
python3 benchmarks/benchmark_serving.py --backend vllm --model google/gemma-3-1b-it --endpoint /v1/completions --dataset-name sonnet --dataset-path benchmarks/sonnet.txt --num-prompts 1
```
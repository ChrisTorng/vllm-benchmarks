#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
VLLM Benchmark Tool

This script executes a single benchmark test using functionality from benchmark_serving.py.
It's designed to be called by llm-loadtest.py for running individual tests.

Usage:
    python vllm-benchmark.py 
        --backend openai
        --model model_name
        --num-prompts 1
        [Other benchmark parameters...]
"""
import os
import sys
import io
import gc
import time
import random
import contextlib
import traceback
import argparse
import numpy as np

# Adjust Python's module search path to make relative imports in benchmarks directory work correctly
benchmarks_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks")
sys.path.insert(0, benchmarks_dir)

# Import required functions and classes from benchmark_serving.py
from benchmarks.benchmark_serving import (
    benchmark,
    get_tokenizer,
    OPENAI_COMPATIBLE_BACKENDS,
    SonnetDataset,
    ShareGPTDataset,
    BurstGPTDataset,
    RandomDataset,
    SampleRequest,
    check_goodput_args,
    FlexibleArgumentParser,
)

async def run_benchmark(backend: str,
                        api_url: str,
                        base_url: str,
                        model_id: str,
                        model_name: str,
                        tokenizer: any,
                        dataset: any,
                        num_prompts: int,
                        args: argparse.Namespace):
    """Run a single benchmark test with the specified num_prompts value"""
    
    print(f"\nRunning test with num-prompts = {num_prompts}")
    
    # Based on the dataset type, get input requests using the appropriate method
    if args.dataset_name == "sonnet":
        # For sonnet dataset, format differently based on backend type
        if args.backend == "openai-chat":
            input_requests = dataset.sample(num_requests=num_prompts,
                                        input_len=args.sonnet_input_len,
                                        output_len=args.sonnet_output_len,
                                        prefix_len=args.sonnet_prefix_len,
                                        tokenizer=tokenizer,
                                        return_prompt_formatted=False)
        else:
            assert tokenizer.chat_template or tokenizer.default_chat_template, (
                "Tokenizer/model must have chat template for sonnet dataset.")
            input_requests = dataset.sample(num_requests=num_prompts,
                                        input_len=args.sonnet_input_len,
                                        output_len=args.sonnet_output_len,
                                        prefix_len=args.sonnet_prefix_len,
                                        tokenizer=tokenizer,
                                        return_prompt_formatted=True)
    elif args.dataset_name == "sharegpt":
        input_requests = ShareGPTDataset(random_seed=args.seed,
                                    dataset_path=args.dataset_path).sample(
                                        tokenizer=tokenizer,
                                        num_requests=num_prompts,
                                        output_len=args.sharegpt_output_len,
                                    )
    elif args.dataset_name == "burstgpt":
        input_requests = BurstGPTDataset(random_seed=args.seed,
                                    dataset_path=args.dataset_path).sample(
                                        tokenizer=tokenizer, 
                                        num_requests=num_prompts
                                    )
    elif args.dataset_name == "random":
        input_requests = RandomDataset(dataset_path=args.dataset_path).sample(
            tokenizer=tokenizer,
            num_requests=num_prompts,
            prefix_len=args.random_prefix_len,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            range_ratio=args.random_range_ratio,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
        
    # Check goodput parameters
    goodput_config_dict = check_goodput_args(args)
    
    # Collect sampling parameters
    sampling_params = {
        k: v
        for k, v in {
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "temperature": args.temperature
        }.items() if v is not None
    }

    # Sampling parameters are only supported by openai-compatible backends
    if sampling_params and args.backend not in OPENAI_COMPATIBLE_BACKENDS:
        raise ValueError(
            "Sampling parameters are only supported by openai-compatible backends.")

    if "temperature" not in sampling_params:
        sampling_params["temperature"] = 0.0  # Default to greedy decoding

    # Run the benchmark test
    benchmark_result = await benchmark(
        backend=backend,
        api_url=api_url,
        base_url=base_url,
        model_id=model_id,
        model_name=model_name,
        tokenizer=tokenizer,
        input_requests=input_requests,
        logprobs=args.logprobs,
        request_rate=args.request_rate,
        burstiness=args.burstiness,
        disable_tqdm=args.disable_tqdm,
        profile=args.profile,
        selected_percentile_metrics=args.percentile_metrics.split(","),
        selected_percentiles=[
            float(p) for p in args.metric_percentiles.split(",")
        ],
        ignore_eos=args.ignore_eos,
        goodput_config_dict=goodput_config_dict,
        max_concurrency=args.max_concurrency,
        lora_modules=args.lora_modules,
        extra_body=sampling_params,
    )
    
    return benchmark_result


async def main_async(args: argparse.Namespace):
    """Run a single benchmark test asynchronously"""
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    model_name = args.served_model_name or args.model
    tokenizer_id = args.tokenizer or args.model
    tokenizer_mode = args.tokenizer_mode
    num_prompts = args.num_prompts

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"

    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = get_tokenizer(tokenizer_id,
                            tokenizer_mode=tokenizer_mode,
                            trust_remote_code=args.trust_remote_code)

    # Initialize dataset
    print("Initializing dataset...")
    if args.dataset_name == "sonnet":
        dataset = SonnetDataset(dataset_path=args.dataset_path)
    elif args.dataset_name in ["sharegpt", "burstgpt", "random"]:
        # These datasets are created during each sampling, so no pre-initialization needed
        dataset = None 
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    # Avoid GC processing "static" data - reduce pause times
    gc.collect()
    gc.freeze()
    
    start_time = time.time()
    
    try:
        result = await run_benchmark(
            backend=backend,
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            model_name=model_name,
            tokenizer=tokenizer,
            dataset=dataset,
            num_prompts=num_prompts,
            args=args
        )
    except Exception as e:
        print(f"Error during benchmark: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"num-prompts={num_prompts} completed, time elapsed: {elapsed_time:.2f} seconds")


def main():
    """Main function"""
    # Create a parser that accepts all parameters from benchmark_serving.py
    parser = FlexibleArgumentParser(
        description="VLLM Benchmark Tool for a single benchmark test")
    
    # Add required parameters specific to this script
    parser.add_argument(
        "--num-prompts",
        type=int,
        required=True,
        help="Number of prompts to use for this benchmark",
    )
    # Add other required arguments to match benchmark_serving
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--endpoint", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--served-model-name", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--tokenizer-mode", type=str, default="auto")
    parser.add_argument("--logprobs", type=int, default=None)
    # Sampling parameters
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--request-rate", type=str, default=None)
    parser.add_argument("--burstiness", type=float, default=1.0)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--percentile-metrics", type=str, default=None)
    parser.add_argument("--metric-percentiles", type=str, default=None)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    # Dataset specific
    parser.add_argument("--sonnet-input-len", type=int, default=None)
    parser.add_argument("--sonnet-output-len", type=int, default=None)
    parser.add_argument("--sonnet-prefix-len", type=int, default=None)
    parser.add_argument("--random-input-len", type=int, default=None)
    parser.add_argument("--random-output-len", type=int, default=None)
    parser.add_argument("--random-range-ratio", type=float, default=None)
    parser.add_argument("--random-prefix-len", type=int, default=None)
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--lora-modules", nargs='+', default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--goodput", nargs="+", default=None,
                        help="Service level objectives for goodput as KEY:VALUE pairs")
     
    # Parse arguments
    args, unknown = parser.parse_known_args()  # Use parse_known_args() to ignore unknown args
    
    # Convert string 'inf' to float('inf') for request_rate if needed
    if isinstance(args.request_rate, str) and args.request_rate.lower() == 'inf':
        args.request_rate = float('inf')

    # Import asyncio only when needed to avoid any potential issues
    import asyncio
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

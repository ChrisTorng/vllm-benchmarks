#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
LLM Load Testing Tool

This script uses functionality from benchmark_serving.py to perform load testing with
different num-prompts values while only initializing once, thus saving execution time.

Usage:
    python llm-loadtest.py 
        --backend <backend> 
        --model <your_model> 
        --dataset-name <dataset_name> 
        --dataset-path <path to dataset> 
        --prompts-list 1,2,3   # Specify num-prompts values to test, comma-separated
        [other standard benchmark_serving.py parameters]
"""

import argparse
import asyncio
import datetime
import gc
import io
import os
import random
import sys
import time
from typing import List, Optional, Any, Dict
import contextlib

# Adjust Python's module search path to make relative imports in benchmarks directory work correctly
import os.path
benchmarks_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks")
sys.path.insert(0, benchmarks_dir)

import numpy as np

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
                        tokenizer: Any,
                        dataset: Any,
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
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
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
    
    return benchmark_result, captured.getvalue()


async def main_async(args: argparse.Namespace):
    """Run all benchmark tests asynchronously"""
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    model_name = args.served_model_name or args.model
    tokenizer_id = args.tokenizer or args.model
    tokenizer_mode = args.tokenizer_mode

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"

    # Initialize tokenizer (only once)
    print("Initializing tokenizer...")
    tokenizer = get_tokenizer(tokenizer_id,
                            tokenizer_mode=tokenizer_mode,
                            trust_remote_code=args.trust_remote_code)

    # Initialize dataset (only once)
    print("Initializing dataset...")
    if args.dataset_name == "sonnet":
        dataset = SonnetDataset(dataset_path=args.dataset_path)
    elif args.dataset_name in ["sharegpt", "burstgpt", "random"]:
        # These datasets are created during each sampling, so no pre-initialization needed
        dataset = None 
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    # Get the list of num_prompts values to test
    prompts_list = [int(x) for x in args.prompts_list.split(",")]
    
    # Create timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prompts_str = args.prompts_list.replace(',', '-')
    
    # Normalize model_id to replace invalid path characters
    normalized_model_id = model_id.replace('/', '_').replace('\\', '_')
    
    # Create new directory structure: ./results/{model}_{dataset-name}/
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    model_dataset_dir = os.path.join(results_dir, f"{normalized_model_id}_{args.dataset_name}")
    os.makedirs(model_dataset_dir, exist_ok=True)
    
    # Create new filename format: {timestamp}_{prompts-list}.*
    base_filename = f"{timestamp}_{prompts_str}"
    raw_file_path = os.path.join(model_dataset_dir, f"{base_filename}.raw.txt")
    results_file_path = os.path.join(model_dataset_dir, f"{base_filename}.results.txt")
    
    # --- Capture all print to raw_output ---
    raw_outputs = []
    results = []
    for num_prompts in prompts_list:
        # Avoid GC processing "static" data - reduce pause times
        gc.collect()
        gc.freeze()
        
        start_time = time.time()
        result, captured_output = await run_benchmark(
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
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        # Print and collect the captured output (benchmark_serving.py style)
        print(captured_output, end="")
        raw_outputs.append(captured_output)
        print(f"num-prompts={num_prompts} completed, time elapsed: {elapsed_time:.2f} seconds")
        raw_outputs.append(f"num-prompts={num_prompts} completed, time elapsed: {elapsed_time:.2f} seconds\n")
        
        results.append((num_prompts, result, captured_output))
    
    # --- Save raw.txt (all captured outputs) ---
    with open(raw_file_path, 'w') as raw_file:
        for out in raw_outputs:
            raw_file.write(out)
    
    # --- Save results.txt (only Serving Benchmark Result block values) ---
    # Parse values from captured_output
    def parse_serving_block(text):
        # Only parse the block between '============ Serving Benchmark Result ============' and the next '====' or end
        lines = text.splitlines()
        in_block = False
        values = {}
        for line in lines:
            if 'Serving Benchmark Result' in line:
                in_block = True
                continue
            if '==================================================' in line and in_block:
                break 
            if in_block and ':' in line:
                key, value = line.split(":", 1)
                values[key.strip()] = value.strip()
        return values
    # Collect all keys in order of first appearance
    all_keys = []
    seen = set()
    for _, _, captured in results:
        vals = parse_serving_block(captured)
        for k in vals:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    # Write header
    with open(results_file_path, 'w') as results_file:
        results_file.write("\t".join(["num_prompts"] + all_keys) + "\n")
        for num_prompts, _, captured in results:
            vals = parse_serving_block(captured)
            row = [str(num_prompts)] + [vals.get(k, "") for k in all_keys]
            results_file.write("\t".join(row) + "\n")
    
    # --- Plot result chart ---
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        df = pd.read_csv(results_file_path, sep='\t')
        
        # Use 'Successful requests' as x-axis if available, otherwise fall back to num_prompts
        if 'Successful requests' in df.columns:
            x_values = df['Successful requests']
            x_label = 'Successful requests'
        else:
            # Fall back to num_prompts
            x_values = df['num_prompts']
            x_label = 'Number of Prompts'
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # ax1 is for Output token throughput (tok/s) - Green
        color_output_throughput = 'tab:green'
        ax1.set_xlabel(x_label)
        ax1.set_ylabel('Output token throughput (tok/s)', color=color_output_throughput)
        output_throughput_values = df['Output token throughput (tok/s)'].astype(float).tolist()
        l2, = ax1.plot(x_values, output_throughput_values, marker='s', color=color_output_throughput, label='Output token throughput (tok/s)')
        ax1.tick_params(axis='y', labelcolor=color_output_throughput)
        ax1.set_ylim(bottom=0)  # Start from 0
        
        # ax2 is for Mean TTFT (ms) - Red (now the first right y-axis)
        ax2 = ax1.twinx()
        color_ttft = 'tab:red'
        ttft_values = df['Mean TTFT (ms)'].astype(float).tolist()
        l3, = ax2.plot(x_values, ttft_values, marker='^', color=color_ttft, label='Mean TTFT (ms)')
        ax2.set_ylabel('Mean TTFT (ms)', color=color_ttft)
        ax2.tick_params(axis='y', labelcolor=color_ttft)
        ax2.set_ylim(bottom=0)  # Start from 0
        
        # ax3 is for Mean TPOT (ms) - Purple (now the second right y-axis)
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60)) # Adjusted position for ax2
        color_tpot = 'tab:purple'
        tpot_values = df['Mean TPOT (ms)'].astype(float).tolist()
        l4, = ax3.plot(x_values, tpot_values, marker='v', color=color_tpot, label='Mean TPOT (ms)')
        ax3.set_ylabel('Mean TPOT (ms)', color=color_tpot)
        ax3.tick_params(axis='y', labelcolor=color_tpot)
        ax3.set_ylim(bottom=0)  # Start from 0

        # Legends
        lns = [l2, l3, l4]
        labels = [l.get_label() for l in lns]
        ax1.legend(lns, labels, loc='upper left')
        
        plt.title(f'LLM Loadtest Results ({model_id})')
        plt.tight_layout(pad=4.0)  # Add more padding for the multiple axes
        
        # Save the plot - ensure PNG extension is correctly added
        plot_file_path = os.path.join(model_dataset_dir, f"{base_filename}.png")
        plt.savefig(plot_file_path)
        print(f"Plot saved to: {plot_file_path}")
    except ImportError:
        print("matplotlib or pandas is not installed. Please install them to enable plotting.")
    except Exception as e:
        print(f"Plotting failed: {e}")


def main():
    """Main function"""
    parser = FlexibleArgumentParser(
        description="Load testing tool supporting multiple num-prompts values")
    
    parser.add_argument(
        "--prompts-list",
        type=str,
        default="1,2,3",
        help="List of num-prompts values to test, comma-separated",
    )
    
    # Add standard benchmark_serving.py arguments
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        help="Backend type",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base URL if not using http host and port",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sonnet",
        choices=["sharegpt", "burstgpt", "sonnet", "random"],
        help="Name of the dataset to benchmark on",
    )
    parser.add_argument("--dataset-path",
                        type=str,
                        required=True,
                        help="Path to the sharegpt/sonnet dataset")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer, if not using the default tokenizer",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help="Number of logprobs-per-token to compute & return",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor of the request generation",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Set ignore_eos flag when sending the benchmark request")
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl",
        help="List of selected metrics to report percentils",
    )
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help="List of percentiles for selected metrics",
    )
    parser.add_argument(
        "--goodput",
        nargs="+",
        required=False,
        help="Specify service level objectives as KEY:VALUE pairs")

    # Dataset specific parameters
    sonnet_group = parser.add_argument_group("sonnet dataset options")
    sonnet_group.add_argument(
        "--sonnet-input-len",
        type=int,
        default=550,
        help="Number of input tokens per request, used only for sonnet dataset",
    )
    sonnet_group.add_argument(
        "--sonnet-output-len",
        type=int,
        default=150,
        help="Number of output tokens per request, used only for sonnet dataset",
    )
    sonnet_group.add_argument(
        "--sonnet-prefix-len",
        type=int,
        default=200,
        help="Number of prefix tokens per request, used only for sonnet dataset",
    )

    sharegpt_group = parser.add_argument_group("sharegpt dataset options")
    sharegpt_group.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length from the ShareGPT dataset",
    )

    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Number of input tokens per request, used only for random sampling",
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help="Number of output tokens per request, used only for random sampling",
    )
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range ratio for sampling input/output length, used only for random sampling",
    )
    random_group.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help="Number of fixed prefix tokens before the random context in a request",
    )

    sampling_group = parser.add_argument_group("sampling parameters")
    sampling_group.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p sampling parameter",
    )
    sampling_group.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling parameter",
    )
    sampling_group.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="Min-p sampling parameter",
    )
    sampling_group.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature sampling parameter",
    )

    parser.add_argument(
        '--tokenizer-mode',
        type=str,
        default="auto",
        choices=['auto', 'slow', 'mistral', 'custom'],
        help='The tokenizer mode',
    )

    parser.add_argument("--served-model-name",
                        type=str,
                        default=None,
                        help="The model name used in the API")

    parser.add_argument("--lora-modules",
                        nargs='+',
                        default=None,
                        help="A subset of LoRA module names passed in when launching the server")

    args = parser.parse_args()
    
    # Run the async main function
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

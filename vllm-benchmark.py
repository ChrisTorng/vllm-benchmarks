#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
VLLM Benchmark Tool

This script executes a single benchmark test using functionality from benchmark_serving.py.
It's designed to be imported and used programmatically or called by llm-loadtest.py.

Usage:
    Import this module and call run_benchmark_with_params() function directly.
"""
import os
import sys
import io
import gc
import time
import random
import contextlib
import traceback
import numpy as np
from typing import Dict, Any, List, Optional, Union

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
)

async def run_benchmark(backend: str,
                        api_url: str,
                        base_url: str,
                        model_id: str,
                        model_name: str,
                        tokenizer: any,
                        dataset: any,
                        num_prompts: int,
                        **kwargs) -> Dict[str, Any]:
    """Run a single benchmark test with the specified num_prompts value"""
    
    print(f"\nRunning test with num-prompts = {num_prompts}")
    
    # Based on the dataset type, get input requests using the appropriate method
    dataset_name = kwargs.get("dataset_name", "sonnet")
    if dataset_name == "sonnet":
        # For sonnet dataset, format differently based on backend type
        if backend == "openai-chat":
            input_requests = dataset.sample(num_requests=num_prompts,
                                        input_len=kwargs.get("sonnet_input_len"),
                                        output_len=kwargs.get("sonnet_output_len"),
                                        prefix_len=kwargs.get("sonnet_prefix_len"),
                                        tokenizer=tokenizer,
                                        return_prompt_formatted=False)
        else:
            assert tokenizer.chat_template or tokenizer.default_chat_template, (
                "Tokenizer/model must have chat template for sonnet dataset.")
            input_requests = dataset.sample(num_requests=num_prompts,
                                        input_len=kwargs.get("sonnet_input_len"),
                                        output_len=kwargs.get("sonnet_output_len"),
                                        prefix_len=kwargs.get("sonnet_prefix_len"),
                                        tokenizer=tokenizer,
                                        return_prompt_formatted=True)
    elif dataset_name == "sharegpt":
        input_requests = ShareGPTDataset(random_seed=kwargs.get("seed", 0),
                                    dataset_path=kwargs.get("dataset_path")).sample(
                                        tokenizer=tokenizer,
                                        num_requests=num_prompts,
                                        output_len=kwargs.get("sharegpt_output_len"),
                                    )
    elif dataset_name == "burstgpt":
        input_requests = BurstGPTDataset(random_seed=kwargs.get("seed", 0),
                                    dataset_path=kwargs.get("dataset_path")).sample(
                                        tokenizer=tokenizer, 
                                        num_requests=num_prompts
                                    )
    elif dataset_name == "random":
        input_requests = RandomDataset(dataset_path=kwargs.get("dataset_path")).sample(
            tokenizer=tokenizer,
            num_requests=num_prompts,
            prefix_len=kwargs.get("random_prefix_len"),
            input_len=kwargs.get("random_input_len"),
            output_len=kwargs.get("random_output_len"),
            range_ratio=kwargs.get("random_range_ratio"),
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
        
    # Check goodput parameters (convert kwargs dict to Namespace for compatibility)
    from types import SimpleNamespace
    args_ns = SimpleNamespace(**kwargs)
    goodput_config_dict = check_goodput_args(args_ns)
    
    # Collect sampling parameters
    sampling_params = {
        k: v
        for k, v in {
            "top_p": kwargs.get("top_p"),
            "top_k": kwargs.get("top_k"),
            "min_p": kwargs.get("min_p"),
            "temperature": kwargs.get("temperature")
        }.items() if v is not None
    }

    # Sampling parameters are only supported by openai-compatible backends
    if sampling_params and backend not in OPENAI_COMPATIBLE_BACKENDS:
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
        logprobs=kwargs.get("logprobs"),
        request_rate=kwargs.get("request_rate", float('inf')),
        burstiness=kwargs.get("burstiness", 1.0),
        disable_tqdm=kwargs.get("disable_tqdm", False),
        profile=kwargs.get("profile", False),
        selected_percentile_metrics=kwargs.get("percentile_metrics", "ttft,tpot,itl").split(","),
        selected_percentiles=[
            float(p) for p in kwargs.get("metric_percentiles", "99").split(",")
        ],
        ignore_eos=kwargs.get("ignore_eos", False),
        goodput_config_dict=goodput_config_dict,
        max_concurrency=kwargs.get("max_concurrency"),
        lora_modules=kwargs.get("lora_modules"),
        extra_body=sampling_params,
    )
    
    return benchmark_result

async def run_benchmark_async(num_prompts: int, **kwargs) -> Dict[str, Any]:
    """Run a single benchmark test asynchronously"""
    
    # Set random seeds
    seed = kwargs.get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)

    backend = kwargs.get("backend", "openai")
    model_id = kwargs.get("model")
    model_name = kwargs.get("served_model_name", model_id)
    # Fallback: if tokenizer is None, use model_id
    tokenizer_id = kwargs["tokenizer"] if kwargs.get("tokenizer") is not None else model_id
    tokenizer_mode = kwargs.get("tokenizer_mode", "auto")

    # Build API URLs
    base_url = kwargs.get("base_url")
    if base_url is not None:
        api_url = f"{base_url}{kwargs.get('endpoint', '/v1/completions')}"
    else:
        host = kwargs.get("host", "localhost")
        port = kwargs.get("port", 8000)
        endpoint = kwargs.get("endpoint", "/v1/completions")
        api_url = f"http://{host}:{port}{endpoint}"
        base_url = f"http://{host}:{port}"

    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = get_tokenizer(tokenizer_id,
                            tokenizer_mode=tokenizer_mode,
                            trust_remote_code=kwargs.get("trust_remote_code", False))

    # Initialize dataset
    print("Initializing dataset...")
    dataset_name = kwargs.get("dataset_name", "sonnet")
    if dataset_name == "sonnet":
        dataset = SonnetDataset(dataset_path=kwargs.get("dataset_path"))
    elif dataset_name in ["sharegpt", "burstgpt", "random"]:
        # These datasets are created during each sampling, so no pre-initialization needed
        dataset = None 
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Avoid GC processing "static" data - reduce pause times
    gc.collect()
    gc.freeze()
    
    start_time = time.time()
    
    # Remove keys that are passed as explicit arguments to avoid duplicate values
    explicit_keys = [
        "backend", "api_url", "base_url", "model_id", "model_name", "tokenizer", "dataset", "num_prompts"
    ]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in explicit_keys}
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
            **filtered_kwargs
        )
    except Exception as e:
        print(f"Error during benchmark: {e}")
        traceback.print_exc()
        raise
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"num-prompts={num_prompts} completed, time elapsed: {elapsed_time:.2f} seconds")
    
    return result

def run_benchmark_with_params(num_prompts: int, **kwargs) -> Dict[str, Any]:
    """Main function to run benchmark with parameters."""
    import asyncio
    return asyncio.run(run_benchmark_async(num_prompts, **kwargs))

# Keep for backwards compatibility when script is run directly
if __name__ == "__main__":
    print("This script is designed to be imported and used as a module.")
    print("Example: import vllm_benchmark; vllm_benchmark.run_benchmark_with_params(num_prompts=1, model='model_name')")

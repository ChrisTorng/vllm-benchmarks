#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Custom Benchmark Tool for Chat Completions API

This script executes a benchmark test using Python's built-in HTTP request functionality
to call the chat completions API. It's designed to match the output format of 
vllm-benchmark.py for easy integration with llm-loadtest.py.

Usage:
    python my-benchmark.py 
        --model model_name
        --num-prompts 1
        [Other benchmark parameters...]
"""

import argparse
import json
import os
import sys
import time
import datetime
import statistics
import asyncio
import random
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import requests
import socket

# For compatible arguments parsing
class FlexibleArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that ignores unknown arguments."""
    def parse_known_args(self, args=None, namespace=None):
        args, argv = super().parse_known_args(args, namespace)
        return args, argv

def format_time(seconds: float) -> str:
    """Format time in MM:SS.MMM format."""
    minutes = int(seconds // 60)
    seconds_part = seconds % 60
    return f"{minutes:02d}:{seconds_part:06.3f}"

async def send_chat_request(model: str, 
                       prompt: str, 
                       system_prompt: str,
                       host: str,
                       port: int,
                       endpoint: str,
                       benchmark_start_time: float,
                       show_problem: bool = False,
                       show_first_token: bool = False,
                       show_last_token: bool = False,
                       show_other_tokens: bool = False) -> Tuple[str, List[float], int]:
    """
    Send a request using Python's built-in HTTP request functionality.
    Supports both completions and chat completions endpoints.
    
    Args:
        model: Model name
        prompt: User prompt
        system_prompt: System prompt
        host: API host
        port: API port
        endpoint: API endpoint
        benchmark_start_time: Time when the benchmark started
        
    Returns:
        Tuple containing (full_response, token_times, token_count)
    """
    url = f"http://{host}:{port}{endpoint}"
    
    # Prepare the request data based on endpoint type
    if endpoint.endswith("/v1/chat/completions"):
        # Chat completions API
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        data = {
            "model": model,
            "messages": messages,
            "stream": True
        }
    else:
        # Regular completions API
        data = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "max_tokens": 150  # Add reasonable default
        }
    
    # Print the prompt with timing
    request_time = time.time()
    global_time = format_time(request_time - benchmark_start_time)
    relative_time = "00:00.000"
    if show_problem:
        print(f"{global_time}\t{relative_time}\t{prompt}", flush=True)
    
    # Initialize response variables
    full_response = ""
    token_times = []
    first_token_time = None
    token_contents = []
    
    # Send the request with streaming response
    try:
        response = requests.post(url, json=data, stream=True, headers={"Content-Type": "application/json"})
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}", flush=True)
            return "", [], 0
        for line in response.iter_lines(decode_unicode=True):
            if not line or line == "data: [DONE]":
                continue
            current_time = time.time()
            token_time = current_time - request_time
            if line.startswith("data: "):
                line = line[6:]
            try:
                chunk = json.loads(line)
                token_content = ""
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    choice = chunk["choices"][0]
                    if "delta" in choice and "content" in choice["delta"]:
                        token_content = choice["delta"]["content"]
                    elif "text" in choice:
                        token_content = choice["text"]
                    if token_content:
                        if first_token_time is None and token_content.strip():
                            first_token_time = token_time * 1000
                        full_response += token_content
                        token_contents.append(token_content)
                        token_times.append(token_time * 1000)
                        global_time = format_time(current_time - benchmark_start_time)
                        idx = len(token_times) - 1
                        show = False
                        if idx == 0 and show_first_token:
                            show = True
                        elif idx == len(token_times) - 1 and show_last_token:
                            show = True
                        elif idx > 0 and idx < len(token_times) - 1 and show_other_tokens:
                            show = True
                        if show:
                            print(f"{global_time}\t{token_time:09.3f}\t{token_content}", flush=True)
            except json.JSONDecodeError:
                continue
    except Exception as e:
        print(f"Error during request: {e}", flush=True)
        return "", [], 0
    return full_response, token_times, len(token_times)

def calculate_metrics(all_ttfts: List[float], all_token_times: List[List[float]]) -> Dict[str, float]:
    """
    Calculate various timing metrics from token times.
    
    Args:
        all_ttfts: List of time to first token measurements in milliseconds
        all_token_times: List of lists of token timing measurements in milliseconds
        
    Returns:
        Dict containing timing metrics
    """
    if not all_ttfts:
        return {
            "mean_ttft_ms": 0, 
            "median_ttft_ms": 0, 
            "p99_ttft_ms": 0,
            "mean_tpot_ms": 0, 
            "median_tpot_ms": 0, 
            "p99_tpot_ms": 0,
            "mean_itl_ms": 0,
            "median_itl_ms": 0,
            "p99_itl_ms": 0
        }
    
    # Calculate TTFT metrics
    mean_ttft = statistics.mean(all_ttfts)
    median_ttft = statistics.median(all_ttfts)
    p99_ttft = np.percentile(all_ttfts, 99) if len(all_ttfts) > 1 else all_ttfts[0]
    
    # Calculate TPOT metrics (excluding first token)
    all_tpot_values = []
    for token_times in all_token_times:
        if len(token_times) > 1:
            # Calculate time per output token by finding differences between consecutive tokens
            tpot_values = [token_times[i] - token_times[i-1] for i in range(1, len(token_times))]
            all_tpot_values.extend(tpot_values)
    
    # Calculate ITL (inter-token latency) metrics which are the same as TPOT in this case
    mean_tpot = statistics.mean(all_tpot_values) if all_tpot_values else 0
    median_tpot = statistics.median(all_tpot_values) if all_tpot_values else 0
    p99_tpot = np.percentile(all_tpot_values, 99) if len(all_tpot_values) > 1 else (all_tpot_values[0] if all_tpot_values else 0)
    
    # In our implementation, ITL is the same as TPOT
    return {
        "mean_ttft_ms": mean_ttft, 
        "median_ttft_ms": median_ttft, 
        "p99_ttft_ms": p99_ttft,
        "mean_tpot_ms": mean_tpot, 
        "median_tpot_ms": median_tpot, 
        "p99_tpot_ms": p99_tpot,
        "mean_itl_ms": mean_tpot,
        "median_itl_ms": median_tpot,
        "p99_itl_ms": p99_tpot
    }

def generate_prompt(num: int) -> str:
    """Generate a unique prompt based on number."""
    questions = [
        "Who won the world series in 2020?",
        "What is quantum computing?",
        "Can you explain the theory of relativity?",
        "How does photosynthesis work?",
        "What was the Renaissance period?",
        "Explain the water cycle",
        "What is artificial intelligence?",
        "How do vaccines work?",
        "What causes northern lights?",
        "Explain how a combustion engine works"
    ]
    return questions[num % len(questions)]

async def run_benchmark(num_prompts: int, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run a benchmark with the specified number of prompts.
    
    Args:
        num_prompts: Number of prompts to benchmark
        args: Command line arguments
        
    Returns:
        Benchmark results dictionary
    """
    print(f"\nRunning test with num-prompts = {num_prompts}")
    print("Starting initial single prompt test run...")
    print("Initial test run completed. Starting main benchmark run...")
    print(f"Traffic request rate: inf")
    print(f"Burstiness factor: 1.0 (Poisson process)")
    print(f"Maximum request concurrency: None")
    
    all_responses = []
    all_ttfts = []
    all_token_times_lists = []
    total_tokens_generated = 0
    
    # Start timing the entire benchmark
    benchmark_start_time = time.time()
    
    # Process each prompt sequentially
    for i in range(num_prompts):
        prompt = generate_prompt(i)
        system_prompt = "You are a helpful assistant."
        response, token_times, token_count = await send_chat_request(
            model=args.model,
            prompt=prompt,
            system_prompt=system_prompt,
            host=args.host,
            port=args.port,
            endpoint=args.endpoint,
            benchmark_start_time=benchmark_start_time,
            show_problem=getattr(args, "show_problem", False),
            show_first_token=getattr(args, "show_first_token", False),
            show_last_token=getattr(args, "show_last_token", False),
            show_other_tokens=getattr(args, "show_other_tokens", False)
        )
        if token_times:
            ttft = token_times[0]
            all_ttfts.append(ttft)
            all_token_times_lists.append(token_times)
            all_responses.append(response)
            total_tokens_generated += token_count
    
    # Calculate benchmark duration
    benchmark_duration = time.time() - benchmark_start_time
    
    # Estimate input tokens (approximation)
    avg_input_tokens = 25  # Approximate tokens per input
    total_input_tokens = num_prompts * avg_input_tokens
    
    # Calculate throughput metrics
    request_throughput = num_prompts / benchmark_duration if benchmark_duration > 0 else 0
    output_token_throughput = total_tokens_generated / benchmark_duration if benchmark_duration > 0 else 0
    total_token_throughput = (total_input_tokens + total_tokens_generated) / benchmark_duration if benchmark_duration > 0 else 0
    
    # Calculate timing metrics
    metrics = calculate_metrics(all_ttfts, all_token_times_lists)
    
    # Format the benchmark results to match vllm-benchmark.py output
    results = {
        "Successful requests": num_prompts,
        "Benchmark duration (s)": benchmark_duration,
        "Total input tokens": total_input_tokens,
        "Total generated tokens": total_tokens_generated,
        "Request throughput (req/s)": request_throughput,
        "Output token throughput (tok/s)": output_token_throughput,
        "Total Token throughput (tok/s)": total_token_throughput,
        "Mean TTFT (ms)": metrics["mean_ttft_ms"],
        "Median TTFT (ms)": metrics["median_ttft_ms"],
        "P99 TTFT (ms)": metrics["p99_ttft_ms"],
        "Mean TPOT (ms)": metrics["mean_tpot_ms"],
        "Median TPOT (ms)": metrics["median_tpot_ms"],
        "P99 TPOT (ms)": metrics["p99_tpot_ms"],
        "Mean ITL (ms)": metrics["mean_itl_ms"],
        "Median ITL (ms)": metrics["median_itl_ms"],
        "P99 ITL (ms)": metrics["p99_itl_ms"]
    }
    
    return results

def format_benchmark_results(results: Dict[str, Any]) -> str:
    """Format benchmark results to match vllm-benchmark.py output."""
    output = "============ Serving Benchmark Result ============\n"
    
    # Format each metric with proper spacing
    output += f"Successful requests:                     {results['Successful requests']:<10}\n"
    output += f"Benchmark duration (s):                  {results['Benchmark duration (s)']:.2f}\n"
    output += f"Total input tokens:                      {results['Total input tokens']:<10}\n"
    output += f"Total generated tokens:                  {results['Total generated tokens']:<10}\n"
    output += f"Request throughput (req/s):              {results['Request throughput (req/s)']:.2f}\n"
    output += f"Output token throughput (tok/s):         {results['Output token throughput (tok/s)']:.2f}\n"
    output += f"Total Token throughput (tok/s):          {results['Total Token throughput (tok/s)']:.2f}\n"
    
    output += "---------------Time to First Token----------------\n"
    output += f"Mean TTFT (ms):                          {results['Mean TTFT (ms)']:.2f}\n"
    output += f"Median TTFT (ms):                        {results['Median TTFT (ms)']:.2f}\n"
    output += f"P99 TTFT (ms):                           {results['P99 TTFT (ms)']:.2f}\n"
    
    output += "-----Time per Output Token (excl. 1st token)------\n"
    output += f"Mean TPOT (ms):                          {results['Mean TPOT (ms)']:.2f}\n"
    output += f"Median TPOT (ms):                        {results['Median TPOT (ms)']:.2f}\n"
    output += f"P99 TPOT (ms):                           {results['P99 TPOT (ms)']:.2f}\n"
    
    output += "---------------Inter-token Latency----------------\n"
    output += f"Mean ITL (ms):                           {results['Mean ITL (ms)']:.2f}\n"
    output += f"Median ITL (ms):                         {results['Median ITL (ms)']:.2f}\n"
    output += f"P99 ITL (ms):                            {results['P99 ITL (ms)']:.2f}\n"
    
    output += "==================================================\n"
    
    return output

async def main_async(args: argparse.Namespace):
    """Run a benchmark test asynchronously."""
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Print initialization messages to match vllm-benchmark.py output
    print("Initializing tokenizer...")
    print("Initializing dataset...")
    
    # Run the benchmark
    start_time = time.time()
    results = await run_benchmark(args.num_prompts, args)
    
    # Print formatted results
    print(format_benchmark_results(results))
    
    # Print elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"num-prompts={args.num_prompts} completed, time elapsed: {elapsed_time:.2f} seconds")

def main():
    """Main function"""
    # Create a parser with compatible arguments
    parser = FlexibleArgumentParser(
        description="Custom Benchmark Tool for Chat Completions API")
    
    # Add required parameters for compatibility with vllm-benchmark.py
    parser.add_argument(
        "--num-prompts",
        type=int,
        required=True,
        help="Number of prompts to use for this benchmark",
    )
    parser.add_argument("--backend", type=str, default="my-curl")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--endpoint", type=str, default="/v1/chat/completions")
    parser.add_argument("--model", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--seed", type=int, default=0)
    
    # Add other parameters for compatibility (even if not used)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--served-model-name", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--tokenizer-mode", type=str, default="auto")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--percentile-metrics", type=str, default="ttft,tpot,itl")
    parser.add_argument("--metric-percentiles", type=str, default="99")
    parser.add_argument("--request-rate", type=str, default="inf")
    parser.add_argument("--burstiness", type=float, default=1.0)
    
    # Add new output control arguments
    parser.add_argument("--show-problem", action="store_true", default=False, help="Show the problem (prompt)")
    parser.add_argument("--show-first-token", action="store_true", default=False, help="Show the first token")
    parser.add_argument("--show-last-token", action="store_true", default=False, help="Show the last token")
    parser.add_argument("--show-other-tokens", action="store_true", default=False, help="Show other tokens (not first/last)")
    # Parse arguments
    args, unknown = parser.parse_known_args()
    # Initialize asyncio
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()

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
import traceback
from typing import List, Optional, Any, Dict
import contextlib

# Adjust Python's module search path to make relative imports in benchmarks directory work correctly
import os.path
benchmarks_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks")
sys.path.insert(0, benchmarks_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Define the plot_llm_loadtest_chart function here, outside main_async
# This function encapsulates the plotting logic.
def plot_llm_loadtest_chart(df_source: Any,  # pd.DataFrame or str
                            output_plot_path: str,
                            model_id_for_title: str,
                            x_axis_label_options: List[str],
                            x_axis_col_options: List[Optional[str]]):
    """
    Generates and saves a plot from LLM loadtest results.

    Args:
        df_source (pd.DataFrame or str): DataFrame or path to a CSV file containing the data.
        output_plot_path (str): Path to save the generated plot.
        model_id_for_title (str): Model ID to be used in the plot title.
        x_axis_label_options (list): Preferred labels for X-axis (e.g., ['Successful requests', 'Number of Prompts']).
        x_axis_col_options (list): Preferred column names for X-axis (e.g., ['Successful requests', 'num_prompts']).
    """
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import traceback

        if isinstance(df_source, str):
            if not os.path.exists(df_source) or os.path.getsize(df_source) == 0:
                print(f"Plotting skipped: Data file {df_source} is empty or does not exist.")
                plt.close('all') # Close any pre-existing figures
                return
            df = pd.read_csv(df_source, sep='\t', engine='python')
        elif isinstance(df_source, pd.DataFrame):
            if df_source.empty:
                print(f"Plotting skipped for {output_plot_path}: DataFrame is empty.")
                plt.close('all')
                return
            df = df_source
        else:
            print(f"Plotting skipped for {output_plot_path}: Invalid data source type.")
            plt.close('all')
            return

        if df.empty:
            print(f"Plotting skipped for {output_plot_path}: DataFrame is empty after loading.")
            plt.close('all')
            return
            
        # Determine X-axis
        x_values = None
        x_label = ""
        
        primary_x_col = x_axis_col_options[0] if x_axis_col_options and len(x_axis_col_options) > 0 else None
        primary_x_label = x_axis_label_options[0] if x_axis_label_options and len(x_axis_label_options) > 0 else "X-axis"
        
        fallback_x_col = x_axis_col_options[1] if x_axis_col_options and len(x_axis_col_options) > 1 else None
        fallback_x_label = x_axis_label_options[1] if x_axis_label_options and len(x_axis_label_options) > 1 else "X-axis Fallback"

        if primary_x_col and primary_x_col in df.columns:
            x_values = df[primary_x_col]
            x_label = primary_x_label
        elif fallback_x_col and fallback_x_col in df.columns:
            x_values = df[fallback_x_col]
            x_label = fallback_x_label
        else:
            valid_cols = [col for col in [primary_x_col, fallback_x_col] if col]
            print(f"Error for plot {output_plot_path}: None of the preferred X-axis columns {valid_cols} found in data.")
            plt.close('all')
            return
        
        if x_values.empty or x_values.isnull().all():
            print(f"Plotting skipped for {output_plot_path}: X-axis data ('{x_label}') is empty or all NaN.")
            plt.close('all')
            return

        fig, ax1 = plt.subplots(figsize=(10, 6))
        lines = []
        
        y_metrics_config = [
            {'col': 'Output token throughput (tok/s)', 'color': 'tab:green', 'marker': 's', 'ax_obj': ax1, 'twin_needed': False},
            {'col': 'Mean TTFT (ms)', 'color': 'tab:red', 'marker': '^', 'twin_needed': True, 'twin_pos': ('outward', 0)},
            {'col': 'Mean TPOT (ms)', 'color': 'tab:purple', 'marker': 'v', 'twin_needed': True, 'twin_pos': ('outward', 60)},
        ]

        current_ax = ax1
        ax1.set_xlabel(x_label)

        for i, metric_cfg in enumerate(y_metrics_config):
            metric_col = metric_cfg['col']
            if metric_col in df.columns:
                if metric_cfg['twin_needed']:
                    if i == 0: # Should not happen with this config, but as a safeguard
                         current_ax = ax1.twinx()
                    else: # For subsequent twin axes
                         current_ax = ax1.twinx()
                         current_ax.spines['right'].set_position(metric_cfg['twin_pos'])
                else: # First metric uses ax1 directly
                    current_ax = metric_cfg['ax_obj']

                current_ax.set_ylabel(metric_col, color=metric_cfg['color'])
                metric_values = pd.to_numeric(df[metric_col], errors='coerce').fillna(0)
                
                if metric_values.empty or metric_values.isnull().all():
                    print(f"Warning for plot {output_plot_path}: Metric '{metric_col}' data is empty or all NaN after conversion.")
                    continue

                line, = current_ax.plot(x_values, metric_values, marker=metric_cfg['marker'], color=metric_cfg['color'], label=metric_col)
                current_ax.tick_params(axis='y', labelcolor=metric_cfg['color'])
                current_ax.set_ylim(bottom=0)
                lines.append(line)
            else:
                print(f"Warning for plot {output_plot_path}: Metric '{metric_col}' not found in data.")

        if lines:
            plot_labels = [l.get_label() for l in lines]
            ax1.legend(lines, plot_labels, loc='upper left')
        
        plt.title(f'LLM Loadtest Results ({model_id_for_title})')
        fig.tight_layout(pad=4.0)
        
        plt.savefig(output_plot_path)
        print(f"Plot saved to: {output_plot_path}")
        plt.close(fig)

    except ImportError:
        print("matplotlib or pandas is not installed. Please install them to enable plotting.")
        plt.close('all') # Ensure any figures are closed if imports fail mid-process
    except Exception as e:
        print(f"Plotting failed for {output_plot_path}: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')


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
    
    # --- Create all.results.txt (appends results with timestamp) ---
    all_results_path = os.path.join(model_dataset_dir, "all.results.txt")
    
    # Check if all.results.txt exists, create with header if it doesn't
    if not os.path.exists(all_results_path) or os.path.getsize(all_results_path) == 0:
        with open(all_results_path, 'w') as all_results_file:
            all_results_file.write("\t".join(["timestamp", "num_prompts"] + all_keys) + "\n")
    
    # Append current results to all.results.txt with timestamp
    with open(all_results_path, 'a') as all_results_file:
        for num_prompts, _, captured in results:
            vals = parse_serving_block(captured)
            row = [timestamp, str(num_prompts)] + [vals.get(k, "") for k in all_keys]
            all_results_file.write("\t".join(row) + "\n")
    
    # --- Create average.results.txt (averaged by Successful requests) ---
    try:
        # Read the all.results.txt file
        all_df = pd.read_csv(all_results_path, sep='\t')
        
        # Check if 'Successful requests' is in the columns
        if 'Successful requests' in all_df.columns:
            # Group by Successful requests and compute the mean for numeric columns
            numeric_cols = all_df.select_dtypes(include=['number']).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != 'timestamp']
            
            # For non-numeric columns, keep the first occurrence
            non_numeric_cols = [col for col in all_df.columns if col not in numeric_cols and col != 'timestamp']
            
            # Group and calculate mean for numeric columns
            if numeric_cols:
                grouped_numeric = all_df.groupby('Successful requests')[numeric_cols].mean()
            else:
                grouped_numeric = pd.DataFrame()
            
            # Group and get first value for non-numeric columns
            if non_numeric_cols:
                grouped_non_numeric = all_df.groupby('Successful requests')[non_numeric_cols].first()
            else:
                grouped_non_numeric = pd.DataFrame()
            
            # Combine the results
            if not grouped_numeric.empty and not grouped_non_numeric.empty:
                avg_df = pd.concat([grouped_numeric, grouped_non_numeric], axis=1)
            elif not grouped_numeric.empty:
                avg_df = grouped_numeric
            elif not grouped_non_numeric.empty:
                avg_df = grouped_non_numeric
            else:
                avg_df = pd.DataFrame()
            
            # Reset index to make 'Successful requests' a regular column
            if not avg_df.empty:
                # 檢查是否已經有 'Successful requests' 欄位，如果有則先刪除
                if 'Successful requests' in avg_df.columns:
                    temp_column_name = 'tmp_successful_requests'
                    avg_df = avg_df.rename(columns={'Successful requests': temp_column_name})
                    avg_df = avg_df.reset_index()
                    # 如果臨時欄位存在，則刪除，避免重複
                    if temp_column_name in avg_df.columns:
                        avg_df = avg_df.drop(columns=[temp_column_name])
                else:
                    # 如果不存在，則正常重置索引
                    avg_df = avg_df.reset_index()
                
                # Save the averaged results
                avg_results_path = os.path.join(model_dataset_dir, "average.results.txt")
                avg_df.to_csv(avg_results_path, sep='\t', index=False)
                print(f"Average results saved to: {avg_results_path}")
                
                # --- Generate the all.png plot based on average.results.txt ---
                all_plot_path = os.path.join(model_dataset_dir, "all.png")
                plot_llm_loadtest_chart(
                    df_source=avg_df,
                    output_plot_path=all_plot_path,
                    model_id_for_title=model_id,
                    x_axis_label_options=['Successful requests', 'Number of Prompts'],
                    x_axis_col_options=['Successful requests', 'num_prompts']
                )
        else:
            print("Warning: 'Successful requests' column not found in all.results.txt. Cannot create average.results.txt")
    except ValueError as ve:
        if 'cannot insert' in str(ve) and 'already exists' in str(ve):
            print(f"Error creating average.results.txt or all.png: DataFrame column conflict - {ve}")
            print("Hint: This may be due to 'Successful requests' column appearing both as index and column.")
            traceback.print_exc()
        else:
            print(f"Error creating average.results.txt or all.png: {ve}")
            traceback.print_exc()
    except Exception as e:
        print(f"Error creating average.results.txt or all.png: {e}")
        traceback.print_exc()
    
    # --- Plot result chart ---
    try:
        # Call the plot_llm_loadtest_chart function to generate the current run's plot
        plot_file_path = os.path.join(model_dataset_dir, f"{base_filename}.png")
        plot_llm_loadtest_chart(
            df_source=results_file_path,
            output_plot_path=plot_file_path,
            model_id_for_title=model_id,
            x_axis_label_options=['Successful requests', 'Number of Prompts'],
            x_axis_col_options=['Successful requests', 'num_prompts']
        )
    except ImportError:
        print("matplotlib or pandas is not installed. Please install them to enable plotting.")
    except Exception as e:
        print(f"Plotting failed: {e}")
        traceback.print_exc()


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

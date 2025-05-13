#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
LLM Load Testing Tool

This script uses functionality from benchmark_serving.py to perform load testing with
different num-prompts values while only initializing once, thus saving execution time.

Usage:
    python llm-loadtest.py 
        [--config llm-loadtest.json]  # Path to JSON config file (default: llm-loadtest.json)
        [--output-dir ./results]  # Output directory for all result files (overrides config output_dir)
        --prompts-list 1,2,3   # Specify num-prompts values to test, comma-separated
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
import json

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
    
    # Use output_dir directly as the results directory
    results_dir = args.output_dir
    os.makedirs(results_dir, exist_ok=True)
    
    # Create detail directory for raw.txt, results.txt, and png files
    detail_dir = os.path.join(results_dir, "detail")
    os.makedirs(detail_dir, exist_ok=True)
    
    # Create filename format: {timestamp}_{prompts-list}.*
    normalized_model_id = model_id.replace('/', '_').replace('\\', '_')
    base_filename = f"{timestamp}_{prompts_str}"
    raw_file_path = os.path.join(detail_dir, f"{base_filename}.raw.txt")
    results_file_path = os.path.join(detail_dir, f"{base_filename}.results.txt")
    
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
    
    # --- Create all.1.results.txt (appends results with timestamp) ---
    all_results_path = os.path.join(results_dir, "all.1.results.txt")
    
    # Check if all.1.results.txt exists, create with header if it doesn't
    if not os.path.exists(all_results_path) or os.path.getsize(all_results_path) == 0:
        with open(all_results_path, 'w') as all_results_file:
            all_results_file.write("\t".join(["Keep", "timestamp", "num_prompts"] + all_keys) + "\n")
    
    # Append current results to all.1.results.txt with timestamp
    with open(all_results_path, 'a') as all_results_file:
        for num_prompts, _, captured in results:
            vals = parse_serving_block(captured)
            row = ["1", timestamp, str(num_prompts)] + [vals.get(k, "") for k in all_keys]
            all_results_file.write("\t".join(row) + "\n")
    
    print(f"Results appended to: {all_results_path}")
    print(f"To generate all.2.sorted.txt, all.3.average.txt, and all.4.png, run: python generate-all.py {results_dir}")
    
    # --- Plot result chart ---
    try:
        # Call the plot_llm_loadtest_chart function to generate the current run's plot
        plot_file_path = os.path.join(detail_dir, f"{base_filename}.png")
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
    parser = argparse.ArgumentParser(
        description="Load testing tool using JSON config and prompts-list override")
    parser.add_argument(
        "--prompts-list",
        type=str,
        default="1",
        help="List of num-prompts values to test, comma-separated",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="llm-loadtest.json",
        help="Path to JSON config file (default: llm-loadtest.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for all result files (overrides config output_dir)",
    )
    args = parser.parse_args()

    # Load config from JSON file
    config_path = args.config or "llm-loadtest.json"
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Convert string 'inf' to float('inf') for request_rate if needed
    if isinstance(config.get('request_rate'), str) and config['request_rate'].lower() == 'inf':
        config['request_rate'] = float('inf')

    # Determine base path for output_dir (relative to config file location)
    config_dir = os.path.dirname(os.path.abspath(config_path))
    json_output_dir = config.get('output_dir', '.')
    # If output_dir is relative, resolve it from config file location
    resolved_output_dir = os.path.abspath(os.path.join(config_dir, json_output_dir))
    # If --output-dir is specified, override
    if args.output_dir is not None:
        resolved_output_dir = os.path.abspath(args.output_dir)

    # Merge config into args, command line prompts-list has highest priority
    config['prompts_list'] = args.prompts_list
    config['output_dir'] = resolved_output_dir
    # Convert config dict to argparse.Namespace for compatibility
    args = argparse.Namespace(**config)

    # Run the async main function
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

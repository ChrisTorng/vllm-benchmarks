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
    
    # Normalize model_id to replace invalid path characters
    normalized_model_id = model_id.replace('/', '_').replace('\\', '_')
    
    # Create new directory structure: {output_dir}/{model}_{dataset-name}/
    results_dir = args.output_dir
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
    
    # --- Create all.1.results.txt (appends results with timestamp) ---
    all_results_path = os.path.join(model_dataset_dir, "all.1.results.txt")
    
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
    
    # --- Create all.2.sorted.txt (sorted by Keep, then Successful requests, then timestamp) ---
    try:
        # Read the all.1.results.txt file with more flexible parsing options
        try:
            # First try with Python engine which is more flexible and handle all rows
            all_df = pd.read_csv(all_results_path, sep='\t', engine='python', 
                                on_bad_lines='warn', quoting=3, encoding='utf-8',
                                dtype=str, skip_blank_lines=True)  # Use str dtype to avoid conversion errors
            
            # Debug output to verify data loading
            print(f"Read {len(all_df)} rows from {all_results_path}")
            print(f"Columns found: {all_df.columns.tolist()}")
            
        except Exception as e1:
            print(f"Warning: Error parsing {all_results_path} with python engine: {e1}")
            print("Falling back to manual file parsing...")
            
            # Manual parsing if pandas engine fails - important to read ALL lines
            headers = []
            data_rows = []
            with open(all_results_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    # Skip empty lines during initial read
                    if not line.strip():
                        continue
                    
                    # Remove any extra whitespace, including trailing newlines
                    line = line.strip()
                    if i == 0:  # header line
                        headers = line.split('\t')
                    else:  # data lines
                        # Split but limit to expected columns to handle extra tabs
                        values = line.split('\t', len(headers)-1)
                        # If fewer columns than expected, pad with empty strings
                        if len(values) < len(headers):
                            values += [''] * (len(headers) - len(values))
                        data_rows.append(values)
            
            # Create DataFrame from manually parsed data
            all_df = pd.DataFrame(data_rows, columns=headers)
            
            # Ensure Keep column exists after manual parsing
            if 'Keep' not in all_df.columns and len(headers) > 0:
                print("Adding missing 'Keep' column after manual parsing")
                all_df['Keep'] = 1  # Set default Keep value
            
        print(f"Successfully loaded {all_results_path} with {len(all_df)} rows")
        
        if not all_df.empty:
            # Check if 'Keep' column exists, print debugging info
            if 'Keep' in all_df.columns:
                print(f"Unique Keep values before processing: {all_df['Keep'].unique()}")
            else:
                print("Warning: 'Keep' column not found in the data. Adding default value 1.")
                all_df['Keep'] = 1
            
            # Convert Keep to int to ensure proper sorting
            if 'Keep' in all_df.columns:
                # First clean up any leading/trailing whitespace
                all_df['Keep'] = all_df['Keep'].astype(str).str.strip()
                
                # Handle possible non-numeric values in Keep column
                try:
                    all_df['Keep'] = pd.to_numeric(all_df['Keep'], errors='coerce').fillna(1).astype(int)
                except Exception as conv_err:
                    print(f"Warning: Error converting Keep column: {conv_err}")
                    # Use string comparison as fallback
                    all_df['Keep'] = all_df['Keep'].astype(str)
            else:
                # If Keep column doesn't exist (for backward compatibility), add it
                all_df['Keep'] = 1
            
            # Ensure 'Successful requests' column is numeric for sorting
            if 'Successful requests' in all_df.columns:
                # First clean up any whitespace
                all_df['Successful requests'] = all_df['Successful requests'].astype(str).str.strip()
                
                try:
                    all_df['Successful requests'] = pd.to_numeric(all_df['Successful requests'], errors='coerce').fillna(0)
                except Exception as sr_err:
                    print(f"Warning: Error converting Successful requests column: {sr_err}")
                
                # Convert timestamp to a sortable format - ensure string format
                if 'timestamp' in all_df.columns:
                    all_df['timestamp'] = all_df['timestamp'].astype(str).str.strip()
                
                # Split into Keep=1 and Keep=0 groups - ensure we capture all data
                keep_1_df = all_df[all_df['Keep'] == 1].copy()
                keep_0_df = all_df[all_df['Keep'] == 0].copy()
                
                print(f"Split into Keep=1 ({len(keep_1_df)} rows) and Keep=0 ({len(keep_0_df)} rows)")
                
                # Check if we have data after filtering
                if keep_1_df.empty and keep_0_df.empty:
                    print("Warning: No valid data found after filtering by Keep value")
                    return
                
                # First sort by 'Successful requests' then by 'timestamp'
                sort_columns = []
                if 'Successful requests' in all_df.columns:
                    sort_columns.append('Successful requests')
                if 'timestamp' in all_df.columns:
                    sort_columns.append('timestamp')
                
                if not sort_columns:
                    print("Warning: No valid sort columns found")
                    return
                
                # Sort each group separately - must include timestamp for proper sorting
                keep_1_df_sorted = keep_1_df.sort_values(by=sort_columns) if not keep_1_df.empty else keep_1_df
                keep_0_df_sorted = keep_0_df.sort_values(by=sort_columns) if not keep_0_df.empty else keep_0_df
                
                # Check for duplicate Keep column (could happen during data processing)
                if 'Keep.1' in keep_1_df_sorted.columns:
                    print("Found duplicate Keep column, removing it")
                    keep_1_df_sorted = keep_1_df_sorted.drop(columns=['Keep.1'])
                if 'Keep.1' in keep_0_df_sorted.columns:
                    keep_0_df_sorted = keep_0_df_sorted.drop(columns=['Keep.1'])
                
                # Save to all.2.sorted.txt
                sorted_results_path = os.path.join(model_dataset_dir, "all.2.sorted.txt")
                
                # Create a clean, properly formatted output
                try:
                    # First, check if we have data to write
                    if keep_1_df_sorted.empty and keep_0_df_sorted.empty:
                        print("Warning: No data to write to sorted file")
                        return
                    
                    # Write Keep=1 group first
                    if not keep_1_df_sorted.empty:
                        # Explicitly write to file to ensure proper formatting
                        with open(sorted_results_path, 'w', encoding='utf-8') as sorted_file:
                            # Write header first - ensure no duplicate Keep column
                            columns = []
                            for col in keep_1_df_sorted.columns:
                                if col not in columns:  # prevent duplicate columns
                                    columns.append(col)
                            
                            header_line = '\t'.join(columns) + '\n'
                            sorted_file.write(header_line)
                            
                            # Then write Keep=1 data rows
                            for _, row in keep_1_df_sorted.iterrows():
                                row_values = []
                                for col in columns:
                                    val = str(row[col]).strip()
                                    row_values.append(val)
                                row_line = '\t'.join(row_values) + '\n'
                                sorted_file.write(row_line)
                        
                        print(f"Wrote {len(keep_1_df_sorted)} Keep=1 rows to {sorted_results_path}")
                    
                    # Append a blank line and then Keep=0 group if it's not empty
                    if not keep_0_df_sorted.empty:
                        with open(sorted_results_path, 'a', encoding='utf-8') as sorted_file:
                            # Add empty line between groups
                            sorted_file.write('\n')
                            
                            # Write Keep=0 data rows (no header)
                            for _, row in keep_0_df_sorted.iterrows():
                                row_values = []
                                for col in columns:  # use same columns as header
                                    val = str(row.get(col, '')).strip() 
                                    row_values.append(val)
                                row_line = '\t'.join(row_values) + '\n'
                                sorted_file.write(row_line)
                        
                        print(f"Wrote {len(keep_0_df_sorted)} Keep=0 rows to {sorted_results_path}")
                    
                    print(f"Sorted results saved to: {sorted_results_path}")
                except Exception as write_err:
                    print(f"Error writing to {sorted_results_path}: {write_err}")
                    traceback.print_exc()
            else:
                print("Warning: 'Successful requests' column not found in all.1.results.txt. Cannot create all.2.sorted.txt")
        else:
            print("Warning: all.1.results.txt is empty. Cannot create all.2.sorted.txt")
    except Exception as e:
        print(f"Error creating all.2.sorted.txt: {e}")
        traceback.print_exc()
    
    # --- Create all.3.average.txt (averaged by Successful requests for Keep=1 items) ---
    try:
        # Read the all.2.sorted.txt file to get Keep=1 items
        sorted_results_path = os.path.join(model_dataset_dir, "all.2.sorted.txt")
        
        if os.path.exists(sorted_results_path) and os.path.getsize(sorted_results_path) > 0:
            # Debug info to help diagnose issues
            print(f"Reading {sorted_results_path} for averaging...")
            
            # Read the sorted file with more flexible parsing options
            try:
                # Read the file but stop at the first empty line (which separates Keep=1 and Keep=0)
                keep_1_data = []
                headers = None
                
                with open(sorted_results_path, 'r', encoding='utf-8') as f:
                    # First read all lines and store them
                    all_lines = f.readlines()
                    
                    # Find the first empty line after header line
                    empty_line_index = None
                    for i, line in enumerate(all_lines):
                        if i > 0 and not line.strip():  # Skip checking the header line
                            empty_line_index = i
                            break
                    
                    # Process header line
                    headers = all_lines[0].strip().split('\t')
                    print(f"Found header with columns: {headers}")
                    
                    # Process all lines up to empty line (or all if no empty line)
                    end_index = empty_line_index if empty_line_index else len(all_lines)
                    for i in range(1, end_index):  # Skip header
                        line = all_lines[i].strip()
                        if line:  # Skip any other empty lines
                            values = line.split('\t', len(headers)-1)
                            # If fewer columns than expected, pad with empty strings
                            if len(values) < len(headers):
                                values += [''] * (len(headers) - len(values))
                            keep_1_data.append(values)
                    
                # Create DataFrame from only Keep=1 data
                keep_1_df = pd.DataFrame(keep_1_data, columns=headers)
                print(f"Manually read {len(keep_1_df)} Keep=1 rows from {sorted_results_path}")
                
                # Double check that we only have Keep=1 data - only if Keep column exists
                if 'Keep' in keep_1_df.columns:
                    keep_1_df['Keep'] = keep_1_df['Keep'].astype(str).str.strip()
                    # Verify all rows are Keep=1
                    non_keep_1_rows = keep_1_df[keep_1_df['Keep'] != '1']
                    if not non_keep_1_rows.empty:
                        print(f"Warning: Found {len(non_keep_1_rows)} rows with Keep != 1. Filtering them out.")
                        keep_1_df = keep_1_df[keep_1_df['Keep'] == '1']
                else:
                    # If Keep column is missing, assume all are Keep=1 (for backward compatibility)
                    print("'Keep' column not found in data. Assuming all rows are Keep=1.")
                    keep_1_df['Keep'] = '1'  # Add Keep column with default value 1
                        
                print(f"Final Keep=1 data for averaging: {len(keep_1_df)} rows")
                
            except Exception as parse_err:
                print(f"Warning: Error during manual parsing of {sorted_results_path}: {parse_err}")
                print("Trying pandas parsing approach as fallback...")
                
                # Fallback to pandas parsing
                try:
                    sorted_df = pd.read_csv(sorted_results_path, sep='\t', engine='python', 
                                          on_bad_lines='warn', quoting=3, dtype=str)
                    
                    # Filter for Keep=1 items only
                    if 'Keep' in sorted_df.columns:
                        sorted_df['Keep'] = sorted_df['Keep'].astype(str).str.strip()
                        keep_1_df = sorted_df[sorted_df['Keep'] == '1'].copy()
                        print(f"Filtered {len(keep_1_df)} Keep=1 rows from pandas read of {len(sorted_df)} total rows")
                    else:
                        # For backward compatibility assume all are Keep=1
                        keep_1_df = sorted_df
                        print(f"No Keep column found, assuming all {len(keep_1_df)} rows are Keep=1")
                except Exception as pd_err:
                    print(f"Error in pandas fallback parsing: {pd_err}")
                    print("Could not parse sorted file. Cannot create all.3.average.txt")
                    return
            
            # Check if 'Successful requests' is in the columns and if we have Keep=1 data
            if 'Successful requests' in keep_1_df.columns and not keep_1_df.empty:
                print(f"Found 'Successful requests' column. Processing {len(keep_1_df)} rows.")
                
                # Convert columns to appropriate types
                try:
                    # Clean up 'Successful requests' column
                    keep_1_df['Successful requests'] = keep_1_df['Successful requests'].astype(str).str.strip()
                    
                    # Try to identify and convert numeric columns
                    for col in keep_1_df.columns:
                        if col in ['timestamp', 'Keep']:
                            continue
                        
                        # Try to convert to numeric, fill errors with original values
                        try:
                            keep_1_df[col] = pd.to_numeric(keep_1_df[col], errors='coerce')
                        except Exception as col_err:
                            print(f"Warning: Could not convert column {col} to numeric: {col_err}")
                    
                    # Group by Successful requests and compute the mean for numeric columns
                    numeric_cols = keep_1_df.select_dtypes(include=['number']).columns.tolist()
                    numeric_cols = [col for col in numeric_cols if col not in ['timestamp', 'Keep']]
                    
                    # For non-numeric columns, keep the first occurrence
                    non_numeric_cols = [col for col in keep_1_df.columns if col not in numeric_cols and col not in ['timestamp', 'Keep']]
                    
                    # Check if Keep column exists for aggregation
                    if 'Keep' not in keep_1_df.columns:
                        print("Adding 'Keep' column for aggregation")
                        keep_1_df['Keep'] = 1
                    
                    print(f"Found {len(numeric_cols)} numeric columns and {len(non_numeric_cols)} non-numeric columns")
                except Exception as type_err:
                    print(f"Warning during type conversion: {type_err}")
                    # Fallback approach - manually identify numeric columns
                    numeric_cols = ['num_prompts', 'Successful requests', 'Benchmark duration (s)', 
                                   'Total input tokens', 'Total generated tokens', 'Request throughput (req/s)',
                                   'Output token throughput (tok/s)', 'Total Token throughput (tok/s)',
                                   'Mean TTFT (ms)', 'Median TTFT (ms)', 'P99 TTFT (ms)',
                                   'Mean TPOT (ms)', 'Median TPOT (ms)', 'P99 TPOT (ms)',
                                   'Mean ITL (ms)', 'Median ITL (ms)', 'P99 ITL (ms)']
                    # Filter to only columns that exist
                    numeric_cols = [col for col in numeric_cols if col in keep_1_df.columns]
                    non_numeric_cols = [col for col in keep_1_df.columns if col not in numeric_cols and col not in ['timestamp', 'Keep']]
                    
                    print(f"Using fallback columns: {len(numeric_cols)} numeric, {len(non_numeric_cols)} non-numeric")
                
                # Group and calculate mean for numeric columns
                try:
                    if numeric_cols:
                        grouped_numeric = keep_1_df.groupby('Successful requests')[numeric_cols].mean()
                    else:
                        grouped_numeric = pd.DataFrame()
                    
                    # Group and get first value for non-numeric columns
                    if non_numeric_cols:
                        grouped_non_numeric = keep_1_df.groupby('Successful requests')[non_numeric_cols].first()
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
                        
                        # Round numeric columns to 2 decimal places for better readability
                        numeric_columns = avg_df.select_dtypes(include=['float', 'float64', 'float32']).columns
                        for col in numeric_columns:
                            avg_df[col] = avg_df[col].round(2)
                        
                        # Save the averaged results
                        avg_results_path = os.path.join(model_dataset_dir, "all.3.average.txt")
                        avg_df.to_csv(avg_results_path, sep='\t', index=False, float_format='%.2f')
                        print(f"Average results saved to: {avg_results_path} (with values rounded to 2 decimal places)")
                        
                        # --- Generate the all.4.png plot based on all.3.average.txt ---
                        all_plot_path = os.path.join(model_dataset_dir, "all.4.png")
                        plot_llm_loadtest_chart(
                            df_source=avg_df,
                            output_plot_path=all_plot_path,
                            model_id_for_title=model_id,
                            x_axis_label_options=['Successful requests', 'Number of Prompts'],
                            x_axis_col_options=['Successful requests', 'num_prompts']
                        )
                except Exception as group_err:
                    print(f"Error during grouping and averaging: {group_err}")
                    traceback.print_exc()
            else:
                print("Warning: 'Successful requests' column not found or no Keep=1 items in all.2.sorted.txt. Cannot create all.3.average.txt")
        else:
            print(f"Warning: {sorted_results_path} does not exist or is empty. Cannot create all.3.average.txt")
    except ValueError as ve:
        if 'cannot insert' in str(ve) and 'already exists' in str(ve):
            print(f"Error creating all.3.average.txt or all.4.png: DataFrame column conflict - {ve}")
            print("Hint: This may be due to 'Successful requests' column appearing both as index and column.")
            traceback.print_exc()
        else:
            print(f"Error creating all.3.average.txt or all.4.png: {ve}")
            traceback.print_exc()
    except Exception as e:
        print(f"Error creating all.3.average.txt or all.4.png: {e}")
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
    parser = argparse.ArgumentParser(
        description="Load testing tool using JSON config and prompts-list override")
    parser.add_argument(
        "--prompts-list",
        type=str,
        default="1,2,3",
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

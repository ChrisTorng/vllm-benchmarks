#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
LLM Load Testing Tool

This script coordinates load testing by calling a benchmark script with
different num-prompts values, managing configuration, and collecting results.

Usage:
    python llm-loadtest.py 
        [--config llm-loadtest.json]  # Path to JSON config file (default: llm-loadtest.json)
        [--output-dir ./results]  # Output directory for all result files (overrides config output_dir)
        [--benchmark-script vllm-benchmark.py]  # Benchmark script to use (vllm-benchmark.py or my-benchmark.py)
        --prompts-list 1,2,3   # Specify num-prompts values to test, comma-separated
"""

import argparse
import datetime
import gc
import os
import sys
import time
import traceback
import subprocess
import json
from typing import List, Optional, Any, Dict

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Warning: matplotlib or pandas is not installed. Plotting will be disabled.")

# Define the plot_llm_loadtest_chart function here
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
        traceback.print_exc()
        plt.close('all')


def run_benchmark(num_prompts: int, config: Dict[str, Any], benchmark_script: str, options: List[str] = None) -> str:
    """
    Run a benchmark script as a subprocess with the given num_prompts value and config options.
    
    Args:
        num_prompts: Number of prompts to test
        config: Dictionary containing configuration parameters
        benchmark_script: Benchmark script to run (e.g., "vllm-benchmark.py" or "my-benchmark.py")
        options: Additional command-line options to pass to the benchmark script
        
    Returns:
        The captured output from the benchmark script
    """
    # Build the command with all necessary parameters
    cmd = ["python3", benchmark_script, "--num-prompts", str(num_prompts)]
    
    # Add all config parameters as command-line arguments
    for key, value in config.items():
        # Skip prompts_list as it's specific to llm-loadtest.py
        if key == "prompts_list" or key == "output_dir" or key == "benchmark_script":
            continue
            
        # Handle special cases
        if key == "percentile_metrics" and isinstance(value, list):
            value = ",".join(value)
        elif key == "metric_percentiles" and isinstance(value, list):
            value = ",".join([str(p) for p in value])
        elif value is True:
            cmd.append(f"--{key.replace('_', '-')}")
            continue
        elif value is False or value is None:
            continue
        
        # Add the parameter to the command - use space instead of equals sign
        cmd.append(f"--{key.replace('_', '-')}")
        cmd.append(f"{value}")
    
    # Add any additional options
    if options:
        cmd.extend(options)
    
    # Run the command and capture output
    print(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running {benchmark_script}: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise


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
    parser.add_argument(
        "--benchmark-script",
        type=str,
        default="vllm-benchmark.py",
        help="Benchmark script to use (default: vllm-benchmark.py)",
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

    # Merge config and command line arguments
    config['prompts_list'] = args.prompts_list
    config['output_dir'] = resolved_output_dir
    
    # Get benchmark script from command line or config
    benchmark_script = args.benchmark_script
    if config.get('benchmark_script'):
        benchmark_script = config.get('benchmark_script')

    # Get the list of num_prompts values to test
    prompts_list = [int(x) for x in config['prompts_list'].split(",")]
    
    # Create timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prompts_str = config['prompts_list'].replace(',', '-')
    
    # Use output_dir directly as the results directory
    results_dir = config['output_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    # Create detail directory for raw.txt, results.txt, and png files
    detail_dir = os.path.join(results_dir, "detail")
    os.makedirs(detail_dir, exist_ok=True)
    
    # Create filename format: {timestamp}_{prompts-list}.*
    model_id = config.get('model', 'unknown_model')
    normalized_model_id = model_id.replace('/', '_').replace('\\', '_')
    base_filename = f"{timestamp}_{prompts_str}"
    raw_file_path = os.path.join(detail_dir, f"{base_filename}.raw.txt")
    results_file_path = os.path.join(detail_dir, f"{base_filename}.results.txt")
    
    # --- Capture all output from vllm-benchmark.py ---
    raw_outputs = []
    results = []
    
    for num_prompts in prompts_list:
        gc.collect()  # Clean up before each run
        
        start_time = time.time()
        try:
            captured_output = run_benchmark(num_prompts, config, benchmark_script)
            
            # Print the captured output
            print(captured_output, end="")
            raw_outputs.append(captured_output)
            
            # Don't add our own timing information - rely on the benchmark's timing output
            results.append((num_prompts, None, captured_output))
        except Exception as e:
            print(f"Error during benchmark with num_prompts={num_prompts}: {e}")
            traceback.print_exc()
            continue
    
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


if __name__ == "__main__":
    main()

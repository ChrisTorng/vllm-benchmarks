#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# 
# Run llm-loadtest.py with all provided arguments and then run generate-all.py
# on the output directory.

# Exit on error
set -e

# Get the current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run llm-loadtest.py with all provided arguments
echo "Running llm-loadtest.py with provided arguments..."
python3 "$SCRIPT_DIR/llm-loadtest.py" "$@"

# Find output directory for generate-all.py
OUTPUT_PATH=""

# Check if --output-dir was provided
for ((i=1; i<=$#; i++)); do
    if [ "${!i}" == "--output-dir" ]; then
        next_idx=$((i+1))
        if [ $next_idx -le $# ]; then
            OUTPUT_PATH="${!next_idx}"
            break
        fi
    fi
done

# If --output-dir not found, check for --config
if [ -z "$OUTPUT_PATH" ]; then
    CONFIG_PATH=""
    for ((i=1; i<=$#; i++)); do
        if [ "${!i}" == "--config" ]; then
            next_idx=$((i+1))
            if [ $next_idx -le $# ]; then
                CONFIG_PATH="${!next_idx}"
                break
            fi
        fi
    done
    
    # If config was specified, use that path
    if [ -n "$CONFIG_PATH" ]; then
        if [ -f "$CONFIG_PATH" ]; then
            CONFIG_DIR=$(dirname "$CONFIG_PATH")
            OUTPUT_PATH="$CONFIG_DIR"
        fi
    fi
fi

# If no output path was determined, use ./results
if [ -z "$OUTPUT_PATH" ]; then
    OUTPUT_PATH="$SCRIPT_DIR/results"
fi

# Run generate-all.py on the output path
echo "Running generate-all.py with output path: $OUTPUT_PATH"
python3 "$SCRIPT_DIR/generate-all.py" "$OUTPUT_PATH"

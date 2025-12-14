#!/usr/bin/env python3
"""
Script to create baseline versions of train_len.json and val_len.json
by removing all example sections, keeping only the graph and the query.
"""

import json
import re
from pathlib import Path

def remove_examples_from_input(input_text: str) -> str:
    """
    Remove all Example: blocks from the input text, keeping only:
    - The graph information (vertices, edges, task description)
    - The final query Input: X Y\nOutput: 
    """
    # Find where "Shortest path length:" appears
    if "Shortest path length:" not in input_text:
        # If no examples section, return as is
        return input_text
    
    # Split at "Shortest path length:"
    parts = input_text.split("Shortest path length:", 1)
    if len(parts) != 2:
        return input_text
    
    graph_part = parts[0] + "Shortest path length:\n"
    examples_and_query = parts[1]
    
    # Find the last "Input: X Y\nOutput: " pattern (this is the query)
    # Match pattern: "Input: <numbers> <numbers>\nOutput: "
    pattern = r'Input:\s*(\d+)\s+(\d+)\s*\nOutput:\s*'
    matches = list(re.finditer(pattern, examples_and_query))
    
    if not matches:
        # No query found, return graph part only
        return graph_part.rstrip()
    
    # Get the last match (the query)
    last_match = matches[-1]
    query_text = last_match.group(0)
    
    # Reconstruct: graph + query
    result = graph_part + query_text
    
    return result

def process_dataset(input_file: Path, output_file: Path):
    """Process a dataset file, removing examples from each datapoint."""
    print(f"Reading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} datapoints...")
    baseline_data = []
    
    for i, datapoint in enumerate(data):
        original_input = datapoint.get("input", "")
        new_input = remove_examples_from_input(original_input)
        
        baseline_datapoint = {
            "input": new_input,
            "output": datapoint.get("output", "")
        }
        baseline_data.append(baseline_datapoint)
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(data)} datapoints...")
    
    print(f"Writing {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(baseline_data, f, separators=(',', ':'))
    
    print(f"✓ Created {output_file} with {len(baseline_data)} datapoints\n")

def main():
    """Main function to process both train and val datasets."""
    data_dir = Path(__file__).parent / "data"
    
    train_input = data_dir / "train_len.json"
    val_input = data_dir / "val_len.json"
    
    train_output = data_dir / "train_len_baseline.json"
    val_output = data_dir / "val_len_baseline.json"
    
    if not train_input.exists():
        print(f"Error: {train_input} not found!")
        return
    
    if not val_input.exists():
        print(f"Error: {val_input} not found!")
        return
    
    print("=" * 80)
    print("Creating baseline datasets (without examples)")
    print("=" * 80)
    print()
    
    process_dataset(train_input, train_output)
    process_dataset(val_input, val_output)
    
    print("=" * 80)
    print("✓ All baseline datasets created successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()


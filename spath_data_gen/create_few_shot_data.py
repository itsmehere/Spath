#!/usr/bin/env python3
"""
Script to create few-shot versions of train_len.json and val_len.json
by keeping only 1 or 3 examples, plus the graph and the query.
"""

import json
import re
from pathlib import Path


def keep_n_examples_from_input(input_text: str, n: int) -> str:
    """
    Keep only the first N examples from the input text, keeping:
    - The graph information (vertices, edges, task description)
    - The first N Example: blocks
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

    # Find all Example: blocks
    # Each example block is: "Example:\nInput: X Y\nOutput: Z\n"
    example_pattern = r"Example:\s*\nInput:\s*\d+\s+\d+\s*\nOutput:\s*\d+\s*\n?"
    example_matches = list(re.finditer(example_pattern, examples_and_query))

    # Find the last "Input: X Y\nOutput: " pattern (this is the query, not preceded by Example:)
    # We need to find all Input: patterns and pick the one that's NOT part of an Example block
    query_pattern = r"Input:\s*(\d+)\s+(\d+)\s*\nOutput:\s*$"
    query_match = re.search(query_pattern, examples_and_query)

    if not query_match:
        # Try without the end anchor
        all_inputs = list(re.finditer(r"Input:\s*(\d+)\s+(\d+)\s*\nOutput:\s*", examples_and_query))
        if all_inputs:
            query_match = all_inputs[-1]
        else:
            # No query found, return graph part only
            return graph_part.rstrip()

    query_text = query_match.group(0)

    # Keep only first N examples
    kept_examples = []
    for i, match in enumerate(example_matches):
        if i >= n:
            break
        kept_examples.append(match.group(0))

    # Reconstruct: graph + N examples + query
    result = graph_part
    for example in kept_examples:
        result += example
        # Ensure proper spacing
        if not example.endswith("\n"):
            result += "\n"
    result += query_text

    return result


def process_dataset(input_file: Path, output_file: Path, num_examples: int):
    """Process a dataset file, keeping only N examples from each datapoint."""
    print(f"Reading {input_file}...")
    with open(input_file, "r") as f:
        data = json.load(f)

    print(f"Processing {len(data)} datapoints (keeping {num_examples} example(s))...")
    few_shot_data = []

    for i, datapoint in enumerate(data):
        original_input = datapoint.get("input", "")
        new_input = keep_n_examples_from_input(original_input, num_examples)

        few_shot_datapoint = {"input": new_input, "output": datapoint.get("output", "")}
        few_shot_data.append(few_shot_datapoint)

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(data)} datapoints...")

    print(f"Writing {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(few_shot_data, f, separators=(",", ":"))

    print(f"✓ Created {output_file} with {len(few_shot_data)} datapoints\n")


def main():
    """Main function to process both train and val datasets for 1-shot and 3-shot."""
    data_dir = Path(__file__).parent / "data"

    train_input = data_dir / "train_len.json"
    val_input = data_dir / "val_len.json"

    if not train_input.exists():
        print(f"Error: {train_input} not found!")
        return

    if not val_input.exists():
        print(f"Error: {val_input} not found!")
        return

    # Create 1-shot versions
    print("=" * 80)
    print("Creating 1-shot datasets (with 1 example)")
    print("=" * 80)
    print()

    train_output_1shot = data_dir / "train_len_1shot.json"
    val_output_1shot = data_dir / "val_len_1shot.json"

    process_dataset(train_input, train_output_1shot, num_examples=1)
    process_dataset(val_input, val_output_1shot, num_examples=1)

    # Create 3-shot versions
    print("=" * 80)
    print("Creating 3-shot datasets (with 3 examples)")
    print("=" * 80)
    print()

    train_output_3shot = data_dir / "train_len_3shot.json"
    val_output_3shot = data_dir / "val_len_3shot.json"

    process_dataset(train_input, train_output_3shot, num_examples=3)
    process_dataset(val_input, val_output_3shot, num_examples=3)

    print("=" * 80)
    print("✓ All few-shot datasets created successfully!")
    print("  - 1-shot: train_len_1shot.json, val_len_1shot.json")
    print("  - 3-shot: train_len_3shot.json, val_len_3shot.json")
    print("=" * 80)


if __name__ == "__main__":
    main()

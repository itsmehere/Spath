#!/bin/bash
# Build script for Rust data generation module

set -e

echo "Building Rust module for shortest path data generation..."

# Check if maturin is installed
if ! command -v maturin &> /dev/null; then
    echo "Error: maturin is not installed."
    echo "Install it with: pip install maturin"
    exit 1
fi

# Build in release mode for maximum performance
echo "Building in release mode (this may take a few minutes)..."
maturin develop --release

echo "Build complete! You can now use the Rust implementation."
echo "Usage: from spath_data_gen.data_gen_len_rust import generate_dataset"






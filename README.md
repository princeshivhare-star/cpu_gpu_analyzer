# Performance Characterization of ML Workloads in CPU-Based Systems

## Overview

This project benchmarks the performance of common machine learning workloads on CPU, focusing on execution latency and workload scaling behavior.

## Objectives

* Analyze execution time of matrix multiplication
* Analyze execution time of convolution operations
* Study impact of input size and batch size on performance
* Understand workload scaling in ML systems

## Methodology

Two key ML operations were benchmarked:

1. Matrix multiplication (increasing matrix size)
2. Convolution (increasing batch size)

Each operation was executed multiple times, and average execution time was recorded.

## Results

* Execution time increases significantly with input size
* Larger workloads require more computational resources
* Convolution operations show strong scaling with batch size
* Performance trends highlight the need for hardware acceleration in large ML workloads

## Key Insights

* Small workloads are efficiently handled by CPU
* Larger workloads benefit from parallel architectures (e.g., GPUs)
* Workload size is a critical factor in system performance
* These trends are important for studying heterogeneous CPU–GPU systems

## Future Work

* Extend benchmarking to GPU (CUDA-enabled systems)
* Analyze CPU–GPU data transfer overhead
* Explore security implications such as timing-based information leakage

## Tech Stack

* Python
* PyTorch
* Pandas
* Matplotlib

## How to Run

pip install -r requirements.txt
python benchmark.py

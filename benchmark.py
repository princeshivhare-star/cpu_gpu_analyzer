import time
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

def benchmark_matrix(device, sizes, repeats=20):
    results = []

    for size in sizes:
        a = torch.randn(size, size)
        b = torch.randn(size, size)

        start = time.perf_counter()
        for _ in range(repeats):
            c = torch.matmul(a, b)
        end = time.perf_counter()

        avg_time = (end - start) / repeats

        results.append({
            "operation": "matrix_multiplication",
            "device": device,
            "size": size,
            "avg_time_sec": avg_time
        })

    return results


def benchmark_conv(device, batch_sizes, repeats=20):
    results = []
    conv = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)

    for batch in batch_sizes:
        x = torch.randn(batch, 3, 224, 224)

        start = time.perf_counter()
        for _ in range(repeats):
            y = conv(x)
        end = time.perf_counter()

        avg_time = (end - start) / repeats

        results.append({
            "operation": "convolution",
            "device": device,
            "size": batch,
            "avg_time_sec": avg_time
        })

    return results


def save_and_plot(df):
    os.makedirs("results", exist_ok=True)

    # Save CSV
    df.to_csv("results/benchmark_results.csv", index=False)

    # Plot each operation
    for op in df["operation"].unique():
        subset = df[df["operation"] == op]

        plt.figure()
        plt.plot(subset["size"], subset["avg_time_sec"], marker='o')

        plt.xlabel("Input Size / Batch")
        plt.ylabel("Execution Time (sec)")
        plt.title(f"{op} Performance")

        plt.grid()
        plt.savefig(f"results/{op}.png")
        plt.close()


def main():
    print("Running CPU benchmarks...")

    results = []

    results += benchmark_matrix("cpu", [128, 256, 512, 1024, 2048])
    results += benchmark_conv("cpu", [1, 2, 4, 8, 16])

    df = pd.DataFrame(results)

    print(df)

    save_and_plot(df)

    print("Results saved in /results folder")


if __name__ == "__main__":
    main()
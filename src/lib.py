import os

def read_input(name: str, *, features: int) -> tuple:
    if not os.path.exists(name):
        print(f"Input file '{name}' not found.")
        exit(1)

    with open(name) as f:
        lines = f.read().splitlines()
        if len(lines) != features + 1:
            print(f"Error: Expected {features} weights and 1 bias, but got {len(lines)} lines.")
            exit(1)

        bias = float(lines[-1].split(': ')[1].strip())
        weights = []
        for line in lines[:-1]:
            weights.append(float(line.split(': ')[1].strip()))
        return weights, bias

def compute(weights: list, bias: float, data: list, *, threshold: float, features: int) -> int:
    if len(weights) != features or len(data) != features:
        print(f"Error: Expected {features} weights and {features} data points.")
        exit(1)

    total = bias
    for w, d in zip(weights, data):
        total += w * d
    return activation(total, threshold=threshold)

def activation(x: float, *, threshold: float) -> int:
    return 1 if x >= threshold else 0

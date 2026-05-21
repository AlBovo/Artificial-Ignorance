from lib import activation

FEATURES = 5
EPOCHS = 100
LEARNING_RATE = 0.1


def load_dataset(path: str = "data.txt", *, features: int = FEATURES) -> tuple[list[list[int]], list[int]]:
    with open(path, "r") as f:
        lines = [int(line.strip()) for line in f.readlines() if line.strip() != ""]

    inputs = [lines[i : i + features] for i in range(0, len(lines), features)]

    expected: list[int] = []
    for row in inputs:
        artista, meteo, amici, cibo, alcol = row
        if amici and (
            (artista and meteo)
            or (artista and cibo)
            or (artista and alcol)
            or (meteo and cibo)
            or (meteo and alcol)
            or (cibo and alcol)
        ):
            expected.append(1)
        else:
            expected.append(0)

    return inputs, expected


def train(
    inputs: list[list[int]],
    expected: list[int],
    *,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    threshold: float = 0.5,
) -> tuple[list[float], float]:
    if not inputs:
        raise ValueError("Empty training set")

    n_features = len(inputs[0])
    weights: list[float] = [0.0] * n_features
    bias: float = 0.0

    for _ in range(epochs):
        for i, row in enumerate(inputs):
            value = bias + sum(w * x for w, x in zip(weights, row))
            out = activation(value, threshold=threshold)
            error = expected[i] - out

            for j in range(n_features):
                weights[j] += learning_rate * error * row[j]
            bias += learning_rate * error

    return weights, bias


def training(*, data_path: str = "data.txt", features: int = FEATURES, threshold: int = 0.5) -> tuple[list[float], float]:
    inputs, expected = load_dataset(data_path, features=features)
    weights, bias = train(inputs, expected, threshold=threshold)
    print(f"Trained Weights: {weights}\nTrained Bias: {bias}")
    return weights, bias


def test(weights: list[float], bias: float, *, data_path: str = "data.txt", features: int = FEATURES) -> None:
    inputs, expected = load_dataset(data_path, features=features)
    for i, row in enumerate(inputs):
        value = bias + sum(w * x for w, x in zip(weights, row))
        out = activation(value, threshold=0.5)
        print(f"Input: {row} | Output: {out} | Expected: {expected[i]}")


def save_model(path: str, weights: list[float], bias: float) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, w in enumerate(weights, start=1):
            f.write(f"w{i}: {w}\n")
        f.write(f"bias: {bias}\n")


if __name__ == "__main__":
    w, b = training()
    test(w, b)

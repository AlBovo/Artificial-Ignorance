from trainer import training, test, save_model
from lib import compute, read_input
import sys

THRESHOLD = 1.5
FEATURES = 5

def calculate(weights, bias, data):
    return compute(weights, bias, data, threshold=THRESHOLD, features=FEATURES)

def cli(weights: list[float], bias: float):
    questions = [
        "Artista famoso?",
        "Bel meteo?",
        "Amici presenti?",
        "Cibo buono?",
        "Alcool disponibile?"
    ]

    data = []
    for i, question in enumerate(questions):
        while True:
            value = input(f"{i+1}. {question} (0 or 1): ")
            try:
                value = int(value)
                if value not in (0, 1):
                    raise ValueError
            except ValueError:
                print("Invalid input. Please enter 0 or 1.")
                continue
            data.append(value)
            break
    result = calculate(weights, bias, data)
    if result == 1:
        print("La festa sarà un successo!")
    else:
        print("La festa sarà un fallimento.")

def static(weights: list[float], bias: float):
    with open("data.txt", "r") as f:
        data = [int(line.strip()) for line in f.readlines()]
    n_chunks = len(data) // FEATURES
    results = [
        calculate(weights, bias, data[i*FEATURES:(i+1)*FEATURES])
        for i in range(n_chunks)
    ]
    with open("results.txt", "w") as f:
        for result in results:
            f.write(f"{result}\n")

if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "cli"

    if run == "train":
        weights, bias = training(data_path="test/data.txt", features=FEATURES, threshold=THRESHOLD)
        save_model("test/training.txt", weights, bias)
    elif run == "test":
        weights, bias = read_input("test/training.txt", features=FEATURES)
        test(weights, bias, data_path="test/data.txt", features=FEATURES)
    elif run == "cli":
        mode = sys.argv[2] if len(sys.argv) > 2 else "train"
        try:
            weights, bias = read_input(f"test/{"training" if mode == "train" else "input"}.txt", features=FEATURES)
        except SystemExit:
            weights, bias = training(data_path="test/data.txt", features=FEATURES, threshold=THRESHOLD)
            save_model("test/training.txt", weights, bias)
        print(weights, bias)
        cli(weights, bias)
    else:
        weights, bias = read_input("test/input.txt", features=FEATURES)
        static(weights, bias)

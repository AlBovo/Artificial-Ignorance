from lib import compute, read_input

THRESHOLD = 5
FEATURES = 5

def main():
    weights, bias = read_input("input.txt", features=FEATURES)
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

    result = compute(weights, bias, data, threshold=THRESHOLD, features=FEATURES)
    if result == 1:
        print("La festa sarà un successo!")
    else:
        print("La festa sarà un fallimento.")

if __name__ == "__main__":
    main()
from lib import activation

N = 8
EPOCHS = 100
LEARNING_RATE = 0.1

INPUT = [
    (0, 0, 0), (0, 0, 1),
    (0, 1, 0), (0, 1, 1),
    (1, 0, 0), (1, 0, 1),
    (1, 1, 0), (1, 1, 1)
]

EXPECTED = []
WEIGHTS = [0.0, 0.0, 0.0]
BIAS = 0.0

for input in INPUT:
    sole, tempo, stanco = input
    if (tempo == 1 and stanco == 0) or (sole == 1 and stanco == 0):
        EXPECTED.append(1)
    else:
        EXPECTED.append(0)

def training():
    global WEIGHTS, BIAS
    for _ in range(EPOCHS):
        for i, input in enumerate(INPUT):
            value = BIAS + sum(w * x for w, x in zip(WEIGHTS, input))
            out = activation(value, threshold=0.5)
            error = EXPECTED[i] - out

            for j in range(len(WEIGHTS)):
                WEIGHTS[j] += LEARNING_RATE * error * input[j]
            
            BIAS += LEARNING_RATE * error
    print(f"Trained Weights: {WEIGHTS}" \
          f"\nTrained Bias: {BIAS}")
def test():
    for i, input in enumerate(INPUT):
        value = BIAS + sum(w * x for w, x in zip(WEIGHTS, input))
        out = activation(value, threshold=0.5)
        print(f"Input: {input} | Output: {out} | Expected: {EXPECTED[i]}")

if __name__ == "__main__":
    training()
    test()
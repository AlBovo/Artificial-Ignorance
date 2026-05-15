import random
with open("data.txt", "w") as f:
    for _ in range(100):
        for _ in range(5):
            f.write(f"{random.randint(0, 1)}\n")

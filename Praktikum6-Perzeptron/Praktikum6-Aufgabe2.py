import numpy as np

def perceptron(x1, x2, w):
    w0, w1, w2 = w
    return 1 if (w0 + w1*x1 + w2*x2) > 0 else 0

AND = (-1.5, 1, 1)
OR  = (-0.5, 1, 1)
NOT = (0.5, -1, 0)

inputs = [(0,0), (0,1), (1,0), (1,1)]

print("AND:")
for x in inputs:
    print(x, perceptron(x[0], x[1], AND))

print("\nOR:")
for x in inputs:
    print(x, perceptron(x[0], x[1], OR))

print("\nNOT:")
for x in [(0,0), (1,0)]:
    print(x[0], perceptron(x[0], x[1], NOT))

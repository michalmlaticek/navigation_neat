import math


def hardlogsig(x):
    a = 1 / (1 + math.exp(-x))
    if a < 0.001:
        return 0
    if a > 0.999:
        return 1
    return a

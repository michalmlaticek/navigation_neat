import math


def roundedlogsig(x):
    return round(1 / (1 + math.exp(-x)), 4)

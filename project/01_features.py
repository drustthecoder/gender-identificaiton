# from dataset import *
from mlpr import *

data, labels = load_data("data/Test.txt")
males = data[:, labels == 0]
females = data[:, labels == 1]

histogram(data, labels)
scatter(data, labels)

heatmap(data, "data", "Greens")
heatmap(males, "male", "Reds")
heatmap(females, "female", "Blues")



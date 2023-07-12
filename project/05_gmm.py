from mlpr import *
from tqdm import tqdm
import random
import datetime
import svm


pi_t_str = '\u03C0\u209C'  # U+209C SUBSCRIPT SMALL LETTER T
pi_tilde_str = '\u03C0\u0303'  # Combining U+0303 TILDE after U+03C0 PI

data, labels = load_data("data/Test.txt")
# z_data = z_norm(data)
results = []
mean_results = []
cost_list = list(10 ** i for i in range(-5, 3))
app = (0.5, 1, 1)
k = 5

num_males = data[:, labels == 0]
num_females = data[:, labels == 1]
pi_emp = round(num_females.shape[1] / num_males.shape[1], 3)
pi = pi_emp


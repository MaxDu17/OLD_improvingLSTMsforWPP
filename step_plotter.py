import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.arange(1,1250,5)
data = pd.read_csv("v1/GRAPHS/EVALUATE_TEST.csv")

true_value = data[["true_values"]]
predicted_values = data[["predicted_values"]]

true = [k[0] for k in true_value.values]
predict = [n[0] for n in predicted_values.values]
naive = [0] + true[0:999]

plt.step(x, true[500:750], label='truth')
plt.step(x, predict[500:750], label='predict')
plt.step(x, naive[500:750], label='naive')
plt.legend()
plt.show()
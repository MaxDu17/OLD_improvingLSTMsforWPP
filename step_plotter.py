import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
upper_bound = 100
lower_bound = 0
step_length = 1
version_number = 2
x = np.arange(1,step_length*(upper_bound+1),step_length)
file_name = "2012/v" + str(version_number) + "/GRAPHS/EVALUATE_TEST.csv"
data = pd.read_csv(file_name)

true_value = data[["true_values"]]
predicted_values = data[["predicted_values"]]

true = [k[0] for k in true_value.values]
predict = [n[0] for n in predicted_values.values]
naive = [0] + true[0:999]

plt.step(x, true[lower_bound:upper_bound], label='truth')
plt.step(x, predict[lower_bound:upper_bound], label='predict')
title = "version " + str(version_number) + ", " + \
        str(lower_bound)  + " to " + str(upper_bound) + ", step length " + str(step_length)
plt.title(title)
#plt.step(x, naive[500:750], label='naive')
plt.legend()
plt.show()
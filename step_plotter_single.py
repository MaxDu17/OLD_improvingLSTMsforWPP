import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
upper_bound = 100
lower_bound = 0
version_number_1 = 2

x = np.arange(0,(upper_bound-lower_bound),1)

file_name_1 = "2012/v" + str(version_number_1) + "/GRAPHS/EVALUATE_TEST.csv"


data1 = pd.read_csv(file_name_1)


true_value = data1[["true_values"]]
predicted_values1 = data1[["predicted_values"]]


true = [k[0] for k in true_value.values]

predict1 = [n[0] for n in predicted_values1.values]

naive = [true[lower_bound]]+  true[lower_bound:upper_bound-1]
plt.step(x, true[lower_bound:upper_bound], label='truth')
plt.step(x, predict1[lower_bound:upper_bound], label=('predict version ' + str(version_number_1)))
#plt.step(x, naive, label="naive")

title = "version " + str(version_number_1) + ". " +\
        str(lower_bound)  + " to " + str(upper_bound) + ","

plt.title(title)

plt.legend()
#plt.grid()
plt.show()
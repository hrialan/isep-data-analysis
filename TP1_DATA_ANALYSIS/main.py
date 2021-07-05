import matplotlib as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from math import sqrt,pi,exp




df = pd.DataFrame({'Notes': [6 for i in range(10)]
                           + [8 for i in range(12)]
                           + [9 for i in range(48)]
                           + [10 for i in range(23)]
                           + [11 for i in range(24)]
                           + [12 for i in range(48)]
                           + [13 for i in range(9)]
                           + [14 for i in range(14)]
                           +[17 for i in range(22)]
                   })



plt.hist(df["Notes"],rwidth = 0.4)
plt.title("Histogram of the marks of a group of student")
plt.xlabel("Notes")
plt.ylabel("Number of students")
plt.grid()
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
df=pd.read_csv('BGOCount.csv')
freq = df[df.columns[1]].values.tolist()
count = pd.Series(freq).value_counts()

print(np.median(freq))
print(np.mean(freq))
print(stats.mode(freq))

#print((count.index))
plt.pie(count,labels=count.index)
plt.title('How frequenly do BP labels occur?')
plt.show() 
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline
import sys
import os


df_rbf = pd.read_csv('model_result/rbf.csv')
df_transformer = pd.read_csv('model_result/transformer.csv')
# print(df_transformer['loss'].values)


print(df_rbf['loss'].values)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(df_transformer['loss'].values)

# sys.exit(0)

df = pd.DataFrame({
   'RBF': [df_rbf['val_loss'][9], df_rbf['val_accuracy'][9], df_rbf['val_Custom_Hamming_Loss'][9]],
   'Multi Head Attention': [df_transformer['val_loss'][9],df_transformer['val_accuracy'][9],df_transformer['val_Custom_Hamming_Loss'][9]],
   }, index=['Loss', 'Accuracy', 'Hamming Loss'])

df=df.astype(float)

# Draw a vertical bar chart

df.plot.bar(rot=30, title="Validation Metrics");

plt.show(block=True);

# print(">>>>>>>>>>>>>>>>>>>>>")
# plt.show()
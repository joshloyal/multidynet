import numpy as np
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt

data = []
s = [0.01, 0.05, 0.1, 0.2, 0.3]

for sigma in s:
    for file_name in glob.glob(f'sigma_{sigma}/*'):
        df = pd.read_csv(file_name)
        df['sigma'] = sigma
        data.append(df)


data = pd.concat(data, axis=0, ignore_index=True)
data.columns = ['AIC', 'BIC', 'DIC', 'WAIC', 'sigma']

plot_data = pd.melt(data, id_vars=['sigma'])
plot_data['true_d'] = plot_data['value'] != 2
g = sns.FacetGrid(plot_data, row="variable", col='sigma')
g.map_dataframe(sns.histplot, x="value", stat='probability', hue='true_d', binrange=(0.75, 5), binwidth=0.5, palette='Set2')
g.set_xlabels('d', fontsize=16)
g.set_titles(r"$\sigma = $ {col_name}", size=16)
for j in range(1, 4):
    for k in range(5):
        g.axes[j, k].set_title('')

for j in range(4):
    g.axes[j, 0].set_ylabel("{}\n\n Fraction Selected".format(data.columns[j]), fontsize=16)

g.add_legend()

g.savefig('dimension_selection.pdf')

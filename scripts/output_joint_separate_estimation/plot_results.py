import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt

sigmas = [0.01, 0.05, 0.1, 0.2, 0.3]
rhos = [0.0, 0.4, 0.8]
data = []

for rho in rhos:
    for sigma in sigmas:
        for file_name in glob.glob(f'sigma_{sigma}_rho_{rho}/*'):
            df = pd.read_csv(file_name)
            df['sigma'] = sigma
            df['rho'] = rho
            data.append(df)


data = pd.concat(data, axis=0, ignore_index=True)
data.columns = ['sep_cor', 'sep_auc', 'joint_cor', 'joint_auc', 
       'sigma', 'rho']
data = pd.melt(data, id_vars=['sigma', 'rho'])
data['type'] = data['variable'].map(lambda x: x.split('_')[0])
data['metric'] = data['variable'].map(lambda x: x.split('_')[1])

data = data.query("type == 'sep' | type == 'joint'")
data.loc[data['type'] == 'joint', 'type'] = 'Joint (Proposed)'
data.loc[data['type'] == 'sep', 'type'] = 'Seperate'
g = sns.FacetGrid(data, col="rho", row = 'metric')
g.map_dataframe(sns.boxplot, x="sigma", y="value", hue="type",
        fliersize=2.5, hue_order = ['Joint (Proposed)', 'Seperate'],
        palette = 'Set2')
g.set_xlabels(r"$\sigma$")
g.axes[0, 0].set_ylabel('PCC', fontsize=16)
g.axes[1, 0].set_ylabel('AUC', fontsize=16)
g.set_titles(r"$\rho = $ {col_name}", size=16)
for k in range(3):
    g.axes[1, k].set_title('')
g.add_legend()
g.savefig('joint_sep_sim.pdf', dpi = 300)

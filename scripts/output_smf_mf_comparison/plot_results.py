import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

max_iter = 100
sigmas = [0.01, 0.05, 0.1, 0.2, 0.3]
rhos = [0.0, 0.4, 0.8]
fig, ax = plt.subplots(
        figsize = (15, 8), 
        ncols=len(sigmas), nrows=len(rhos), sharex=True, sharey=True)

for i, rho in enumerate(rhos):
    for j, sigma in enumerate(sigmas):
        data = []
        for file_name in glob.glob(f'output/sigma_{sigma}_rho_{rho}/results_curves*'):
            df = pd.read_csv(file_name)
            df['sigma'] = sigma
            df['rho'] = rho
            data.append(df)
        all_data = pd.concat(data)

        mf_cors = []
        mf_times = []
        for k in range(len(data)):
            mf_cors.append(data[k]['mf_cor'][:max_iter])
            mf_times.append(data[k]['mf_time'][:max_iter])
        mf_cors = np.asarray(mf_cors)
        bounds = np.nanquantile(mf_cors, axis=0, q=[0.25, 0.75])
        ax[i,j].plot(np.nanmedian(mf_cors, axis=0), c="#d95f02", linestyle='dashed')
        ts = np.diff(np.asarray(mf_times), axis=1)

        svi_cors = []
        svi_times = []
        for k in range(len(data)):
            svi_cors.append(data[k]['svi_cor'][:max_iter])
            svi_times.append(data[k]['svi_time'][:max_iter])
        svi_cors = np.asarray(svi_cors)
        bounds = np.nanquantile(svi_cors, axis=0, q=[0.25, 0.75])
        ax[i,j].plot(np.nanmedian(svi_cors, axis=0), c="#1b9e77")
        ts = np.diff(np.asarray(svi_times), axis=1)

        if i == 0:
            ax[i, j].set_title(f"$\\sigma = {sigma}$", fontsize=16)

        if j == 0:
            ax[i, j].set_ylabel(f"$\\rho = {rho}$\n\n PCC", fontsize=16)

        if i == (len(rhos)-1):
            ax[i, j].set_xlabel("# of Iterations", fontsize=16)
            ax[i, j].set_xticks([1] + list(range(25, max_iter + 1, 25)), fontsize=12)
            ax[i, j].tick_params(axis='x', labelsize=16)

        if j == 0:
            ax[i, j].tick_params(axis='y', labelsize=16)

ax[0, 2].legend([Line2D([0], [0], linestyle='-', color='#1b9e77'), Line2D([0], [0],  color='#d95f02', linestyle='--')],
                ["Structured Mean Field (SMF)", "Mean Field (MF)"], 
                bbox_to_anchor=(0.5, 1.5),
                loc = 'upper center', ncol = 2, fontsize=16)
fig.savefig('algo_comp.pdf', dpi = 300, bbox_inches = 'tight')

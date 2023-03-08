# Simulation and Real Data Analysis Scripts

Simulation and real data analysis scripts for the eigenmodel for dynamic multilayer networks. The code was original written to be run on a HPC cluster. 

## How to Run

The following commands run the simulations or real data analyses and produce the figures in the corresponding section.

### Section 4.1 (Parameter Recovery for Varying Network Sizes)

These commands run the simulations and produce Figure 2.

```bash
>>> python simulation_parameter_recovery.py
>>> cd output_parameter_recovery/
>>> python process.py
>>> python plot_results.py
```

### Section 4.2.1 (Comparison of Joint and Separate Estimation)

These commands run the simulations and produce Figure 3.

```bash
>>> python simulation_joint_seperate_estimation.py
>>> cd output_joint_seperate_estimation/
>>> python plot_results.py
```

### Section 4.2.2 (Comparison of the SMF and MF Variational Inference Algorithms)

The following commands runs the simulations and produce Figure 4.

```bash
>>> python simulation_smf_mf_comparison.py
>>> cd output_smf_mf_comparison/
>>> python plot_results.py
```

To evaluate the run times of the SMF and MF algorithm and produce Figure 5, run the following commands.

```bash
>>> python simulation_run_time.py
>>> cd output_run_time/
>>> python plot_results.py
```

### Section 5.1 (ICEWs Networks)

```bash
>>> python icews.py
```

To produce the figures, you will need to run the cells in the corresponding Jupyter notebook:

```bash
>>> jupyter notebook output_icews/ICEWS.ipynb
```

### Section 5.2 (Primary School Contact Networks)

```bash
>>> python primaryschool.py
```

To produce the figures, you will need to run the cells in the corresponding Jupyter notebook:

```bash
>>> jupyter notebook output_primaryschool/PrimarySchool.ipynb
```

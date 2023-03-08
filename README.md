[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/joshloyal/multidynet/blob/master/LICENSE)

# An Eigenmodel for Dynamic Multilayer Networks

*Author: [Joshua D. Loyal](https://joshloyal.github.io/)*

This package provides an interface for the model described in
"An Eigenmodel for Dynamic Multilayer Networks". Inference is performed using
coordinante ascent variational inference. For more details, see [Loyal and Chen, (2021)](https://arxiv.org/abs/2103.12831).

Dependencies
------------
``multidynet`` requires:

- Python (>= 3.10)


Installation
------------
You need a working installation of numpy, scipy, and Cython to install ``multidynet``. The easiest way to install ``multidynet`` is using ``pip``:

```
git clone https://github.com/joshloyal/multidynet.git
cd multidynet
pip install .
```

Or install using pip and GitHub:

```
pip install -U git+https://github.com/joshloyal/multidynet.git
```


Example
-------

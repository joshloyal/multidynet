[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/joshloyal/multidynet/blob/master/LICENSE)

# An Eigenmodel for Dynamic Multilayer Networks

*Author: [Joshua D. Loyal](https://joshloyal.github.io/)*

This package provides an interface for the model described in
"An Eigenmodel for Dynamic Multilayer Networks". Inference is performed using
coordinante ascent variational inference.

BibTeX reference to cite, if you use this package:
<!--
```bibtex
@article{loyal2021eigenmodel,
}
```
-->

Dependencies
------------
``multidynet`` requires:

- Python (>= 3.7)

and the requirements highlighted in [requirements.txt](requirements.txt).

Installation
------------
You need a working installation of numpy and scipy to install ``multidynet``. If you have a working installation of numpy and scipy, the easiest way to install ``multidynet`` is using ``pip``:

```
git clone https://github.com/joshloyal/multidynet.git
cd multidynet
pip install .
```

Or install using pip and GitHub:

```
pip install -U git+https://github.com/joshloyal/multidynet.git
```

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
pip install -U multidynet
```

If you prefer, you can clone the repository and run the setup.py file. Use the following commands to get the copy from GitHub and install all the dependencies:

```
git clone https://github.com/joshloyal/multidynet.git
cd multidynet
pip install .
```

Or install using pip and GitHub:

```
pip install -U git+https://github.com/joshloyal/multidynet.git
```

Background
----------

Example
-------

Simulation Studies and Real-Data Applications
---------------------------------------------
This package includes the simulation studies and real-data applications found in Loyal and Chen (2021)<sup>[[6]](#References)</sup>:

* A synthetic dynamic network with a time-homogeneous community structure: ([here](/examples/homogeneous_simulation.py)).
* A synthetic dynamic network with a time-inhomogeneous community structure: ([here](/examples/inhomogeneous_simulation.py)).
* Sampson's monastery network: ([here](/examples/sampson_monks.py)).
* A dynamic network constructed from international military alliances during the first three decades of the Cold War (1950 - 1979): ([here](/examples/military_alliances.py)).
* A dynamic network constructed from character interactions in the first four seasons of the Game of Thrones television series: ([here](/examples/GoT.py)).

We also provide a few [jupyter notebooks](/notebooks) that demonstrate the use of this package.

References
----------

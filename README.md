# API for DeepDrugCoder (DDC)

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://bitbucket.astrazeneca.net/users/kjmv588/repos/ddc_pub/browse) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![Python 3.6](https://img.shields.io/badge/python-3.6-yellow.svg)](https://www.python.org/downloads/release/python-367/) [![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

Currently only GPU version of the model is supported. You need access to a GPU to use it.

### Installation
- Clone the repo to your current working directory `git clone https://kjmv588@bitbucket.astrazeneca.net/scm/~kjmv588/ddc_pub.git`
- Create a predefined Python3.6 conda environment by `conda env create -f env/ddc_env.yml`
- Run `python setup.py install` to install pip dependencies and add the package to the Python path
- Add the environment in the drop-down list of jupyter by `python -m ipykernel install --user --name ddc_env --display-name "ddc_env (python_3.6.7)"`

### Usage
- `conda activate ddc_env`
- `from ddc_pub import ddc_v3 as ddc`

### API
- `fit()`: Fit a DDC model to the dataset.
- `vectorize()`: Convert a binary RDKit molecule to its One-Hot-Encoded representation.
- `transform()`: Encode a vectorized molecule to its latent representation.
- `predict()`: Decode a latent representation into a SMILES string and calculate its NLL.
- `predict_batch()`: Decode a list of latent representations into SMILES strings and calculate their NLLs.
- `get_smiles_nll()`: Back-calculate the NLL of a known SMILES string to be sampled by the biased decoder.
- `get_smiles_nll_batch()`: Back-calculate the NLLs of a batch of known SMILES strings to be sampled by the biased decoder.
- `summary()`: Display essential architectural parameters.
- `get_graphs()`: Export model graphs to .png files using `pydot` and `graphviz` (might fail).
- `save()`: Save the model in a .zip directory.

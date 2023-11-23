# Physics-enhanced neural networks for equation-of-state calculations

This repository contains the code required to reproduce<sup>1</sup> the results in the paper ["Physics-enhanced neural networks for equation-of-state calculations"](https://arxiv.org/abs/2305.06856).

This README is rather brief as it is designed to accompany the paper. It is essential to read the paper to understand the repository.

We have attempted to make this repository as self-sufficient as possible. However, it is inevitable that issues will arise if one tries to reproduce the full results of the paper. Therefore, please don't hesitate to contact the authors of the paper (using the e-mail addresses given in the paper) in case of any question, no matter how small.

<sup>1</sup>With the usual caveat that neural network training is non-deterministic, and as such the results are expected to differ slightly.

## Installation

If you are planning to train the neural network models and have a GPU, you can run
```
$ pip install -r requirements_gpu.txt
$ pip install -e .
```

Else, if you are running everything on a CPU:
```
$ pip install -r requirements_cpu.txt
$ pip install -e .
```

Of course, it is recommended to do so inside a virtual environment.

## Workflow

The results in the paper can, in principle, be reproduced using the two notebooks in the `notebooks` folder:

1. `Paper_full_workflow.ipynb`: This notebook pre-processes the FPEOS and FP-Be data; sets up and runs the average-atom calculations; and trains and tests the neural networks.
2. `Paper_full_analysis.ipynb`: This notebook uses the results generated from (1) and then generates all the tables and figures found in the paper.

Note that in order to run the first notebook, the FPEOS and FP-Be datasets must be obtained as described in Ref. [1]. The FPEOS data (comprising multiple files) should be placed in `data/raw/FPEOS/`. The FP-Be data (a single file) should be placed directly in `data/raw`.

In order to reproduced the FPEOS interpolation results in Appendix B of the paper, the FPEOS data, besides all source files for the interpolation (which come together with the input data files) should be placed under the `fpeos` directory. Then the following scripts should be run in the given order:

1. `$ ./train_test.sh`
2. `$ ./predict.sh`

## Repository structure

The repository structure is loosely based on the [cookiecutter](https://github.com/drivendata/cookiecutter-data-science) template for data science projects.

The source code is found in the `neuralEOS` directory and structured as follows:

```
├── neuralEOS
│   ├── data         <- Preprocesses data from FPEOS and FP-Be databases, generate average-atom data
│   ├── network      <- Neural network training, hyperparameter optimization, and testing
│   ├── output       <- Analysis of results, generation of figures
```

## Citation

If you use this repository in your research, you must cite the accompanying paper:

1. Reference paper here once accepted

It is highly likely that you should also cite the two data sources used to train the neural network models:

2. FPEOS citation here
3. FP-Be citation here

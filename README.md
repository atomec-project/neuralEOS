# Physics-enhanced neural networks for equation-of-state calculations

This repository contains the code required to reproduce<sup>1</sup> the results in the paper ["Physics-enhanced neural networks for equation-of-state calculations"](https://iopscience.iop.org/article/10.1088/2632-2153/ad13b9).

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

## Data and pre-trained models

The output of the average-atom calculations and pre-trained final models can be found [here](https://rodare.hzdr.de/record/2289).

In this data repository there are two folders:

1. `trained_models.tar`: The final trained models. There are 6 models in total: 3 for the network trained with average-atom data, and 3 for the network trained without. The average-atom based models are prepended with `aa_`. As described in the [paper](https://arxiv.org/abs/2305.06856), a simple (equally weighted) average of the predictions from the 3 models should be taken to compute the actual prediction.

2. `atoMEC_data.tar.gz`: The output from the average-atom calculations. This data is structured in the form:
 
    `<element>/rho_<density>/T_<temperature>/lda/*`

    The average-atom outputs described in the [paper](https://iopscience.iop.org/article/10.1088/2632-2153/ad13b9) are found in the `output.pkl` file.


## Citation

If you use this repository in your research, you must cite the accompanying paper:

1. Callow, T., Nikl, J., Kraisler, E., & Cangi, A. (2023). Physics-enhanced neural networks for equation-of-state calculations (2023). Machine Learning: Science and Technology. https://doi.org/10.1088/2632-2153/ad13b9

It is highly likely that you should also cite the two data sources used to train the neural network models:

2. Militzer, B., González-Cataldo, F., Zhang, S., Driver, K. P., & Soubiran, F. (2021). First-principles equation of state database for warm dense matter computation. Physical Review E, 103(1), 013203. https://doi.org/10.1103/PhysRevE.103.013203
3. Y. H. Ding, S. X. Hu; First-principles equation-of-state table of beryllium based on density-functional theory calculations. Phys. Plasmas 1 June 2017; 24 (6): 062702. https://doi.org/10.1063/1.4984780

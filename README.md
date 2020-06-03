# sorting_analysis

Code for generating spike trains, and analyzing/comparing outputs across spike sorters.

Requires python >=3.6 and the python packages specified in our [requirements](https://github.com/int-brain-lab/sorting_analysis/blob/master/requirements.txt) file. We recommend installing the [Anaconda python distribution and package manager](https://www.anaconda.com/products/individual) to manage python and the python packages required for this repository.

## Using this repository

In order to use the code in this repository, first clone this repository, create a virtual environment with the required packages, and add the repository to your python path. If you're using Anaconda, you can follow the below steps to do this, which will create and activate the `sorting_analysis` conda environment. Open your system terminal, navigate to the directory you wish to clone into, and run:

```
git clone https://github.com/int-brain-lab/sorting_analysis   # clone repo
cd sorting_analysis                                           # cd into repo directory
conda env create -f sorting_analysis_env.yaml                 # create environment
conda activate sorting_analysis                               # activate environment
conda develop ./sorting_analysis                              # add repo to python path
```

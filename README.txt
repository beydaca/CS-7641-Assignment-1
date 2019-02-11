# Assignment-1

This package should contain all code and data necessary to run all of the expierments for assignment 1.

## Installation

1. clone code from github

git clone git@github.com:cashflow1/CS-7641-Assignment-1.git

2. Install python 3.7

https://www.python.org/downloads/

3. Install pip

https://pip.pypa.io/en/stable/installing/

4. Install virtualenv via pip

pip install virtualenv

5. Create a new virtual environment to run with python3

virtualenv -p python3 venv

6. Activate virtual environment

source venv/bin/activate

7. If you receive an error relate to matplotlib, try:

touch ~/.matplotlib/matplotlibrc && echo "backend: TkAgg" >> ~/.matplotlib/matplotlibrc

8. Install requirements.txt

pip install -r requirements.txt

## Run

```
python main.py --help
height has been deprecated.

usage: main.py [-h] [-d {health,credit_card}]
               {clean,knn,svm,ann,dt,boosting} ...

optional arguments:
  -h, --help            show this help message and exit
  -d {health,credit_card}, --dataset {heart,credit_card}
                        Which dataset to run on

subcommands:
  {clean,knn,svm,ann,dt,boosting}
    clean               Clean the stats from original to final and show me
                        information
    knn                 Run k-nearest neighbors
    svm                 Run Support Vector Machines
    ann                 Run neural networks
    dt                  Run decision trees
    boosting            Run boosting

```


## Example

To run for example, the credit problem with an ANN, use the following command.

```
python main.py -d credit_card ann
```

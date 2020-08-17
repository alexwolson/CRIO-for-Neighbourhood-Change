# CRIO for Neighbourhood Change

This repository contains all the code for the paper _CRIO for Neighbourhood Change_ (Olson et al. 2020). 

## Setup

The conda environment file `environment.yml` at the base of the repository can be used to set up the exact environment we used for our experiments. This environment was used on Ubuntu 18.04, and may not be exactly replicable on other systems. If you have issues configuring the environment, try the alternate `environment_simple.yml` file.

**Important**: you must obtain a Gurobi license in order to run our code successfully. This can be obtained for free for academic purposes [here](https://www.gurobi.com/academia/academic-program-and-licenses/).

## Required Data

Our experiments require the [LTDB](https://s4.ad.brown.edu/Projects/Diversity/Researcher/LTDB.htm). Specifically, we used both `Full` and `Sample` datasets. Place these files, in `.xslx` format, into a folder named `data` at the root of the directory. Then, you can run `notebooks/GenerateData.ipynb` which will perform our pre-processing steps.

## Running code

Our experiments are labelled by the model used, and can be found in the `code` folder. With the correct data in the `data` folder, they can be run from the command line as-is.
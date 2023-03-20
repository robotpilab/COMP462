# Installation
## 1. Activate the Python Environment
Please activate the Python environment for the course through Conda by:

```
conda activate comp462
```

If you haven't created the above environment in you system. Please download and install [Anaconda](https://www.anaconda.com/) or its minimal version [Miniconda](https://docs.conda.io/en/latest/miniconda.html), and run:

```
conda create --name comp462 python=3.9 numpy
```

## 2. Install the Required Python Modules

In your active Conda environment, run:

```
pip install pybullet --upgrade
conda install matplotlib
```

Then, you can try to import them to check if the installations are successful.

## 3. Prepare the URDF file for the Simulation

run the script `join_urdf.py` under the folder `/project3` to generate the required URDF file.

# Anaconda and Jupyter notebook 6
Follow the instruction of this document to setup conda and Jupyter notebook on your machine.

## Anaconda setup

### Installation
1. Download Miniconda from [official website](https://docs.conda.io/en/latest/miniconda.html)
2. Install Miniconda following [official instructions](https://conda.io/projects/conda/en/stable/user-guide/install/index.html)

Make sure adding conda python to your `PATH`!

### Setup environment
Run the following command to set up a conda environment with python and Jupyter 6.
```bash
conda create -n myenv python=3.9 notebook=6.4.8
```

Activate the environment:
```bash
conda activate myenv
```

Run Jupyter notebook:
```bash
jupyter notebook                # For localhost
jupyter notebook --ip 0.0.0.0   # For public access
```
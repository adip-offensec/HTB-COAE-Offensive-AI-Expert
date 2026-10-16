

## Ubuntu VM Setup

- 4 GB Ram
- 4 Cores
## Commands to Create Environment - Ubuntu

```
$ mkdir htb && cd htb

$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

$ chmod +x Miniconda3-latest-Linux-x86_64.sh

$ ./Miniconda3-latest-Linux-x86_64.sh -b -u

$ eval "$(/home/$USER/miniconda3/bin/conda shell.$(ps -p $$ -o comm=) hook)"

						$ conda --version

## Close and reopen the terminal

$ conda config --add channels defaults

$ conda config --add channels conda-forge

$ conda config --add channels nvidia # only needed if you are on a PC that has a nvidia gpu

$ conda config --add channels pytorch

$ conda config --set channel_priority strict

$ conda config --set auto_activate_base false

$ conda create -n ai python=3.11

$ conda activate ai

## Run commands inside the ai enviroment

$ conda install -y numpy scipy pandas scikit-learn matplotlib seaborn transformers datasets tokenizers accelerate evaluate optimum huggingface_hub nltk category_encoders

$ conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

$ pip install requests requests_toolbelt

$ conda update --all

$ conda install -y jupyter jupyterlab notebook ipykernel 

## To run jupyter execute the following command

$ jupyter lab
```

## Using JupyterLabs

Click Python3 under Notebook section

![[Pasted image 20260107074417.png]]

`Jupyter` notebooks use a `stateful` environment, which means that variables, functions, and imports defined in one cell remain available to all later cells. Being aware of the `stateful` nature of a notebook is important. For example, if you execute cells out of order, you might observe unexpected results due to previously defined or modified variables.

`JupyterLab` integrates with libraries like `pandas`, `matplotlib`, and `seaborn` for data exploration and visualization. Here's an example of loading a dataset with `pandas` and creating a simple plot:

![[Pasted image 20260107074651.png]]

This code now generates a sample DataFrame with two columns, `column1` and `column2`, containing random values. The rest of the code remains the same, demonstrating how to display the DataFrame's contents and create a scatter plot using the generated data.

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Create a sample DataFrame
data = pd.DataFrame({
    "column1": np.random.rand(50),  # 50 random values for column1
    "column2": np.random.rand(50) * 10  # 50 random values (multiplied by 10) for column2
})

# Display the first few rows
print(data.head())

# Create a scatter plot
plt.scatter(data["column1"], data["column2"])
plt.xlabel("Column 1")
plt.ylabel("Column 2")
plt.title("Scatter Plot")
plt.show()
```

## Restarting the Kernel in JupyterLabs

`JupyterLab` uses a `kernel` to run your code.

Restarting the `kernel` clears all variables, functions, and imported modules from memory, allowing you to start fresh without shutting down `JupyterLab` entirely.

To restart the `kernel`:
1. Open the `Kernel` menu in the top toolbar.
2. Select `Restart Kernel` to reset the environment while preserving cell outputs, or `Restart Kernel and Clear All Outputs` to also remove all previously generated outputs from the notebook.

After restarting, re-run any cells containing variable definitions, imports, or computations to restore the environment. This ensures that the notebook state accurately reflects the code you have most recently executed.

https://jupyterlab.readthedocs.io/en/latest/getting_started/overview.html


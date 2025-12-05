This repository contains data and scripts to replicate the introduction in "Reconceptualizing Conflict Emergence: Evidence from Machine Learning".

## Requirements
- The analysis is run in Python 3.9.21
- The required python libraries are listed in requirements.txt

## Description of files 
- data/ contains the scripts to prepare the source data. The folder only contains the python scripts and not the original data. To obtain the source data, contact the author. 
- out/ contains all outputs created during the analysis, including data sets, tables, and visualizations. 
- data_compare.py creates the data data_examples.csv.
- functions.py contains the used functions. 
- compare.py uses out/data_examples.csv to compare OLS regression with interpretable machine learning. 

## Replication instructions
Create a virtual environment and install libraries. 

```
conda create -n interpret python= 3.9.21
conda activate interpret
pip install -r requirements.txt
```

Run the file (takes roughly 3 hours). 

```
python compare.py
```

Make sure to remove the out paths or adjust to your computer. 



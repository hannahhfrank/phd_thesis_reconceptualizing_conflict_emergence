This repository contains the replication material for "Grievances and Opportunity: Uncovering Causal Complexity in Civil War Onsets".

## Requirements
- The analysis is run in Python 3.9.21
- The required python libraries are listed in requirements.txt

## Description of files 
- data/ contains all scripts used to prepare the data. The source data are not uploaded but can be provided upon request. 
- out/ contains all outputs produced by the scripts. 
- data.py prepares the data df_interpret.csv.
- functions.py contains the functions used in the analysis.
- main.py runs the main analysis, using data/df_interpret.csv as data. 


## Replication instruction
Create a new environment to install libraries. 

```
conda create -n interpret python= 3.9.21
conda activate interpret
pip install -r requirements.txt
```

Run the file. This will take approximately 3 hours. 

```
python main.py
```

Make sure to either remove or adjust the out paths. 


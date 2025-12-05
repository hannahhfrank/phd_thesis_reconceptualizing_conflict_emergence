This repository contains data and scripts to replicate "To Demonstrate or Fight: Similarities and Differences in the Causes of Collective Action".

## Requirements
- The analysis is run in Python 3.12.3
- The required python libraries are listed in requirements.txt

## Descriptions of files 
- data/ contains the scripts to prepare the data. The original data files are not included but can be provided upon request. 
- out/ contains the files produced by the scripts, such as datasets and visualizations.
- data.py creates the datasets, df_out_full.csv, df_conf_hist_full.csv, df_demog_full.csv, df_econ_full.csv, df_pol_full.csv, and df_geog_full.csv
- functions.py contains a set of functions, used during the empirical analysis. 
- main.py produces the predictions and SHAP values, using the data out/df_out_full.csv, out/df_conf_hist_full.csv, out/df_demog_full.csv, out/df_econ_full.csv, out/df_pol_full.csv, and out/df_geog_full.csv. The predictions are contained in the following files: base_war_df.csv, history_war_df.csv, demog_war_df.csv, geog_war_df.csv, econ_war_df.csv, pol_war_df.csv (replace war with conflict, protest, riot, terror, sb, osv, and ns). The SHAP values are stored in: history_war_shap.csv, demog_war_shap.csv, geog_war_shap.csv, econ_war_shap.csv, pol_war_shap.csv (replace war with conflict, protest, riot, terror, sb, osv, and ns). The script also stores the model evaluations in the test data: history_war_evals_df.csv, demog_war_evals_df.csv, geog_war_evals_df.csv, econ_war_evals_df.csv, pol_war_evals_df.csv (replace war with conflict, protest, riot, terror, sb, osv, and ns). There is also one ensemble for each outcome: ensemble_protest.csv, ensemble_riot.csv, ensemble_terror.csv, ensemble_sb.csv, ensemble_osv.csv, and ensemble_ns.csv.
- results.py produces the final visualizations.  

## Replication instructions
Create a new environment and install libraries from requirements.txt.

```
conda create -n structural python=3.12.3
conda activate structural
pip install -r requirements.txt
```

Run the main file to get predictions and SHAP values. This will take roughly 3 hours. 

```
python main.py
```

To obtain the results shown in the paper, run the results file. Make sure to remove or adjust the out-paths. 

```
python results.py
```
This repository contains data and scripts to replicate "From Structure to Action: Anticipating Onsets in Civil Conflict".

## Requirements
- The analysis is run in Python 3.12.3
- The required python libraries are listed in requirements.txt

## Description of files 
- data/ contains the scripts to produce the data files used to run the analysis. The source data are not uploaded but can be provided if required. 
- out/ folder to store all the outputs produced, such as datasets and figures.
- baselines.py obtains predictions for the benchmark models (out/views.csv, out/sf.csv, out/zinb.csv, and out/structural.csv).
- data.py creates the input data: out/df_out_full_cm.csv, out/df_conf_hist_full_cm.csv, out/df_demog_full_cm.csv, out/df_econ_full_cm.csv, out/df_pol_full_cm.csv, and out/df_geog_full_cm.csv
- evals.py runs the final evaluations for all models. 
- functions.py contains the functions used for the analysis.
- procedural.py produces live predictions for *Stage ii* (out/catcher.csv) using the predictions from *Stage i* as input  (out/ensemble_ens_df_cm_live.csv). 
- structural.py produces live predictions for *Stage i* (out/ensemble_ens_df_cm_live.csv), using the input data out/df_out_full_cm.csv, out/df_conf_hist_full_cm.csv, out/df_demog_full_cm.csv, out/df_econ_full_cm.csv, out/df_pol_full_cm.csv, and out/df_geog_full_cm.csv. This file likewise produces predictions for all constituent models in the ensemble of ensembles: history_protest_df_cm_live.csv, demog_protest_df_cm_live.csv, geog_protest_df_cm_live.csv, econ_protest_df_cm_live.csv, pol_protest_df_cm_live.csv (replace protest with riot, terror, sb, osv, and ns), as well as evaluation scores in the validation data:  history_protest_evals_df.csv, demog_protest_evals_df.csv, geog_protest_evals_df.csv, econ_protest_evals_df.csv, pol_protest_evals_df.csv (replace protest with riot, terror, sb, osv, and ns). For each outcome, there is also an ensemble: ensemble_protest_live.csv, ensemble_riot_live.csv, ensemble_terror_live.csv, ensemble_sb_live.csv, ensemble_osv_live.csv, and ensemble_ns_live.csv. 


## Replication instructions
Create an environment and install the libraries listed in requirements.txt.

```
conda create -n structural python=3.12.3
conda activate structural
pip install -r requirements.txt
```

Run the file to get predictions for *Stage i*. 

```
python structural.py
```

Next, run the file to get predictions for *Stage ii*

```
python procedural.py
```

Finally, get the predictions for the benchmark models. 

```
python baselines.py
```

The final script runs the evaluation. 

```
python evals.py
```

Make sure to adjust the out-paths.

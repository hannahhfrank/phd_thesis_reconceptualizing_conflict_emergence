import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
import numpy as np 
from functions import gen_model
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc,brier_score_loss
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
os.environ['PATH'] = "/Library/TeX/texbin:" + os.environ.get('PATH', '')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}\usepackage[T1]{fontenc}'

grid = {'n_estimators': [10, 231, 452, 673, 894, 1115, 1336, 1557, 1778, 2000],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]}

# Inputs
demog_theme=['pop','pop_density','pop_density_id','urb_share','rural_share','pop_male_share','pop_male_share_0_14','pop_male_share_15_19','pop_male_share_20_24','pop_male_share_25_29','pop_male_share_30_34','group_counts','group_counts_id','monopoly_share','monopoly_share_id','discriminated_share','discriminated_share_id','powerless_share','powerless_share_id','dominant_share','dominant_share_id','ethnic_frac','ethnic_frac_id','rel_frac','rel_frac_id','lang_frac','lang_frac_id','race_frac','race_frac_id']
geog_theme=['land','land_id','forest','forest_id','temp','temp_id','co2','co2_id','percip','percip_id','waterstress','waterstress_id','agri_land','agri_land_id','arable_land','arable_land_id','rugged','soil','desert','tropical','cont_africa','cont_asia','no_neigh','d_neighbors_non_dem','libdem_id_neigh']
econ_theme=['natres_share','natres_share_id','oil_share','oil_share_id','gas_share','gas_share_id','coal_share','coal_share_id','forest_share','forest_share_id','minerals_share','minerals_share_id','gdp','gdp_id','gni','gni_id','gdp_growth','gdp_growth_id','unemploy','unemploy_id','unemploy_male','unemploy_male_id','inflat','inflat_id','conprice','conprice_id','undernour','undernour_id','foodprod','foodprod_id','water_rural','water_rural_id','water_urb','water_urb_id','agri_share','agri_share_id','trade_share','trade_share_id','fert','lifeexp_female','lifeexp_male','pop_growth','pop_growth_id','inf_mort','exports','exports_id','imports','imports_id','primary_female','primary_female_id','primary_male','primary_male_id','second_female','second_female_id','second_male','second_male_id','tert_female','tert_female_id','tert_male','tert_male_id','eys','eys_id','eys_male','eys_male_id','eys_female','eys_female_id','mys','mys_id','mys_male','mys_male_id','mys_female','mys_female_id']
pol_theme=['armedforces_share','armedforces_share_id','milex_share','milex_share_id','corruption','corruption_id', 'effectiveness', 'effectiveness_id', 'polvio','polvio_id','regu','regu_id','law','law_id','account','account_id','tax','tax_id','broadband','broadband_id','telephone','telephone_id','internet_use','internet_use_id','mobile','mobile_id','polyarchy','libdem','libdem_id','partipdem','delibdem','egaldem','civlib','phyvio','pollib','privlib','execon','execon_id','exgender','exgender_id','exgeo','exgeo_id','expol','expol_id','exsoc','exsoc_id','shutdown','shutdown_id','filter','filter_id','tenure_months','tenure_months_id','dem_duration','dem_duration_id','elections','elections_id','lastelection','lastelection_id']

# Check distributions
y=pd.read_csv("out/df_out_full.csv",index_col=0)
fig,ax = plt.subplots()
y["d_civil_war"].hist()
fig,ax = plt.subplots()
y["d_civil_conflict"].hist()
fig,ax = plt.subplots()
y["d_terror"].hist()
fig,ax = plt.subplots()
y["d_sb"].hist()
fig,ax = plt.subplots()
y["d_ns"].hist()
fig,ax = plt.subplots()
y["d_osv"].hist()
fig,ax = plt.subplots()
y["d_protest"].hist()
fig,ax = plt.subplots()
y["d_riot"].hist()

# Check distributions to decide about transforms
x=pd.read_csv("out/df_demog_full.csv",index_col=0)
for var in ['pop','pop_density','urb_share','rural_share','pop_male_share','pop_male_share_0_14','pop_male_share_15_19','pop_male_share_20_24','pop_male_share_25_29','pop_male_share_30_34','group_counts','monopoly_share','discriminated_share','powerless_share','dominant_share','ethnic_frac','rel_frac','lang_frac','race_frac']:
    fig,ax = plt.subplots()
    x[var].hist()
    plt.title(var)
    plt.show()
    
x=pd.read_csv("out/df_geog_full.csv",index_col=0)
for var in ['land','forest','temp','co2','percip','waterstress','agri_land','arable_land','rugged','soil','desert','tropical']:
    fig,ax = plt.subplots()
    x[var].hist()
    plt.title(var)
    plt.show()
    
x=pd.read_csv("out/df_econ_full.csv",index_col=0)
for var in ["natres_share",'oil_share','gas_share','coal_share','forest_share','minerals_share',"gdp","gni","gdp_growth","unemploy","unemploy_male","inflat","conprice","undernour","foodprod","water_rural","water_urb","agri_share","trade_share","fert","lifeexp_female","lifeexp_male","pop_growth","inf_mort",'exports','imports','primary_female','primary_male','second_female','second_male','tert_female','tert_male','eys','eys_male','eys_female','mys','mys_male','mys_female']:
    fig,ax = plt.subplots()
    x[var].hist()
    plt.title(var)
    plt.show()
    
x=pd.read_csv("out/df_pol_full.csv",index_col=0)
for var in ['armedforces_share','milex_share','corruption','effectiveness','polvio','regu','law','account','tax','broadband','telephone','internet_use','mobile','polyarchy','libdem','partipdem','delibdem','egaldem','civlib','phyvio','pollib','privlib','execon','exgender','exgeo','expol','exsoc','shutdown','filter','tenure_months','dem_duration','lastelection']:
    fig,ax = plt.subplots()
    x[var].hist()
    plt.title(var)
    plt.show()
    
                                #################
                                ### Civil war ###
                                #################
###################
### A. Baseline ###
###################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)

# Fit
target="d_civil_war"
inputs=['d_civil_war_lag1']
base_war_df,base_war_evals,base_war_val=gen_model(y,x,target,inputs,model_fit=DummyClassifier(strategy="most_frequent"),int_methods=False)
base_war_df.to_csv("out/base_war_df.csv")
base_war_evals_df = pd.DataFrame.from_dict(base_war_evals, orient='index').reset_index()
base_war_evals_df.to_csv("out/base_war_evals_df.csv")

########################
### B. History theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)

# Fit
target="d_civil_war"
inputs=['d_civil_war_lag1',"d_civil_war_zeros_growth","d_neighbors_civil_war_lag1","regime_duration"]
history_war_df,history_war_evals,history_war_val=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=False)
history_war_df.to_csv("out/history_war_df.csv")
history_war_evals_df = pd.DataFrame.from_dict(history_war_evals, orient='index').reset_index()
history_war_evals_df.to_csv("out/history_war_evals_df.csv")

###########################
### C. Demography theme ###
###########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_demog_full.csv",index_col=0)

# Transforms
x["pop"]=np.log(x["pop"]+1)

# Fit
target="d_civil_war"
inputs=demog_theme
demog_war_df,demog_war_evals,demog_war_val=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=False)
demog_war_df.to_csv("out/demog_war_df.csv")
demog_war_evals_df = pd.DataFrame.from_dict(demog_war_evals, orient='index').reset_index()
demog_war_evals_df.to_csv("out/demog_war_evals_df.csv")

#########################################
### E.  Geography & environment theme ###
#########################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_geog_full.csv",index_col=0)

# Transforms
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)

# Fit
target="d_civil_war"
inputs=geog_theme
geog_war_df,geog_war_evals,geog_war_val=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=False)
geog_war_df.to_csv("out/geog_war_df.csv")
geog_war_evals_df = pd.DataFrame.from_dict(geog_war_evals, orient='index').reset_index()
geog_war_evals_df.to_csv("out/geog_war_evals_df.csv")

########################
### G. Economy theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_econ_full.csv",index_col=0)

# Transforms
x["natres_share"]=np.log(x["natres_share"]+1)
x["oil_share"]=np.log(x["oil_share"]+1)
x["gas_share"]=np.log(x["gas_share"]+1)
x["coal_share"]=np.log(x["coal_share"]+1)
x["forest_share"]=np.log(x["forest_share"]+1)
x["minerals_share"]=np.log(x["minerals_share"]+1)
x["gdp"]=np.log(x["gdp"]+1)
x["inflat"]=np.sign(x["inflat"]) * np.log(np.abs(x["inflat"]) + 1)
x["conprice"]=np.log(x["conprice"]+1)

# Fit
target="d_civil_war"
inputs=econ_theme
econ_war_df,econ_war_evals,econ_war_val=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=False)
econ_war_df.to_csv("out/econ_war_df.csv")
econ_war_evals_df = pd.DataFrame.from_dict(econ_war_evals, orient='index').reset_index()
econ_war_evals_df.to_csv("out/econ_war_evals_df.csv")

##################################
### H. Regime and policy theme ###
##################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_pol_full.csv",index_col=0)

# Transforms
x["milex_share"]=np.log(x["milex_share"]+1)
x["tax"]=np.log(x["tax"]+1)

# Fit
target="d_civil_war"
inputs=pol_theme
pol_war_df,pol_war_evals,pol_war_val=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=False)
pol_war_df.to_csv("out/pol_war_df.csv")
pol_war_evals_df = pd.DataFrame.from_dict(pol_war_evals, orient='index').reset_index()
pol_war_evals_df.to_csv("out/pol_war_evals_df.csv")

################
### Ensemble ###
################

# Get ensemble predictions as additive index, using the performance in the validation data as weights

# Obtain weights based on the inverse of the brier score in the validation data so that higher means better
weights=[1-history_war_val["brier"],
         1-demog_war_val["brier"],
         1-geog_war_val["brier"],
         1-econ_war_val["brier"],
         1-pol_war_val["brier"]]

# Min-max normalize weights, so that the highest weight gets assigned a value of 1 and the lowest a value of zero
weights_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]

# Make sure the final weights sum up to one
weights_war_n = [x / sum(weights_n) for x in weights_n]

# Calculate the weighted index
ensemble = (history_war_df.preds_proba*weights_war_n[0])+ \
            (demog_war_df.preds_proba*weights_war_n[1])+ \
            (geog_war_df.preds_proba*weights_war_n[2])+ \
            (econ_war_df.preds_proba*weights_war_n[3])+ \
            (pol_war_df.preds_proba*weights_war_n[4])
            
# Make df and save            
ensemble_war=pd.concat([history_war_df[["country","year","d_civil_war"]],pd.DataFrame(ensemble)],axis=1)
ensemble_war.columns=["country","year","d_civil_war","preds_proba"]
ensemble_war=ensemble_war.reset_index(drop=True)
ensemble_war.to_csv("out/ensemble_war.csv")

###################
### Evaluations ###
###################

# Evaluate the ensemble in the test data 
ensemble_war_s=ensemble_war.loc[(ensemble_war["year"]>=2019)&(ensemble_war["year"]<=2023)]

# Brier
brier = brier_score_loss(ensemble_war_s.d_civil_war, ensemble_war_s.preds_proba)

# AUPR
precision, recall, _ = precision_recall_curve(ensemble_war_s.d_civil_war, ensemble_war_s.preds_proba)
aupr = auc(recall, precision)

# AUROC
auroc = roc_auc_score(ensemble_war_s.d_civil_war, ensemble_war_s.preds_proba)

# Save
evals_war_ensemble = {"brier": brier, "aupr": aupr, "auroc": auroc}
evals_war_ensemble_df = pd.DataFrame.from_dict(evals_war_ensemble, orient='index').reset_index()
evals_war_ensemble_df.to_csv("out/evals_war_ensemble_df.csv")

print(f"{round(base_war_evals['aupr'],5)} &  \\\
      {round(base_war_evals['auroc'],5)} &  \\\
      {round(base_war_evals['brier'],5)}")
      
print(f"{round(history_war_evals['aupr'],5)} &  \\\
      {round(history_war_evals['auroc'],5)} &  \\\
      {round(history_war_evals['brier'],5)}")
      
print(f"{round(demog_war_evals['aupr'],5)} &  \\\
      {round(demog_war_evals['auroc'],5)} &  \\\
      {round(demog_war_evals['brier'],5)}")

print(f"{round(geog_war_evals['aupr'],5)} &  \\\
      {round(geog_war_evals['auroc'],5)} &  \\\
      {round(geog_war_evals['brier'],5)}")
      
print(f"{round(econ_war_evals['aupr'],5)} &  \\\
      {round(econ_war_evals['auroc'],5)} &  \\\
      {round(econ_war_evals['brier'],5)}")
           
print(f"{round(pol_war_evals['aupr'],5)} &  \\\
       {round(pol_war_evals['auroc'],5)} &  \\\
       {round(pol_war_evals['brier'],5)}")          
      
print(f"{round(evals_war_ensemble['aupr'],5)} &  \\\
       {round(evals_war_ensemble['auroc'],5)} &  \\\
       {round(evals_war_ensemble['brier'],5)}")   
       
 
                                ######################
                                ### Civil conflict ###
                                ######################
###################
### A. Baseline ###
###################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)

# Fit
target="d_civil_conflict"
inputs=['d_civil_conflict_lag1']
base_conflict_df,base_conflict_evals,base_conflict_val=gen_model(y,x,target,inputs,model_fit=DummyClassifier(strategy="most_frequent"),int_methods=False)
base_conflict_df.to_csv("out/base_conflict_df.csv")
base_conflict_evals_df = pd.DataFrame.from_dict(base_conflict_evals, orient='index').reset_index()
base_conflict_evals_df.to_csv("out/base_conflict_evals_df.csv")

########################
### B. History theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)

# Fit
target="d_civil_conflict"
inputs=['d_civil_conflict_lag1',"d_civil_conflict_zeros_growth","d_neighbors_civil_conflict_lag1","regime_duration"]
history_conflict_df,history_conflict_evals,history_conflict_val=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=False)
history_conflict_df.to_csv("out/history_conflict_df.csv")
history_conflict_evals_df = pd.DataFrame.from_dict(history_conflict_evals, orient='index').reset_index()
history_conflict_evals_df.to_csv("out/history_conflict_evals_df.csv")

###########################
### C. Demography theme ###
###########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_demog_full.csv",index_col=0)

# Transforms
x["pop"]=np.log(x["pop"]+1)

# Fit
target="d_civil_conflict"
inputs=demog_theme
demog_conflict_df,demog_conflict_evals,demog_conflict_val=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=False)
demog_conflict_df.to_csv("out/demog_conflict_df.csv")
demog_conflict_evals_df = pd.DataFrame.from_dict(demog_conflict_evals, orient='index').reset_index()
demog_conflict_evals_df.to_csv("out/demog_conflict_evals_df.csv")

#########################################
### E.  Geography & environment theme ###
#########################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_geog_full.csv",index_col=0)

# Transforms
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)

# Fit
target="d_civil_conflict"
inputs=geog_theme
geog_conflict_df,geog_conflict_evals,geog_conflict_val=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=False)
geog_conflict_df.to_csv("out/geog_conflict_df.csv")
geog_conflict_evals_df = pd.DataFrame.from_dict(geog_conflict_evals, orient='index').reset_index()
geog_conflict_evals_df.to_csv("out/geog_conflict_evals_df.csv")

########################
### G. Economy theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_econ_full.csv",index_col=0)

# Transforms
x["natres_share"]=np.log(x["natres_share"]+1)
x["gas_share"]=np.log(x["gas_share"]+1)
x["coal_share"]=np.log(x["coal_share"]+1)
x["forest_share"]=np.log(x["forest_share"]+1)
x["minerals_share"]=np.log(x["minerals_share"]+1)
x["gdp"]=np.log(x["gdp"]+1)
x["inflat"]=np.sign(x["inflat"]) * np.log(np.abs(x["inflat"]) + 1)
x["conprice"]=np.log(x["conprice"]+1)

# Fit
target="d_civil_conflict"
inputs=econ_theme
econ_conflict_df,econ_conflict_evals,econ_conflict_val=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=False)
econ_conflict_df.to_csv("out/econ_conflict_df.csv")
econ_conflict_evals_df = pd.DataFrame.from_dict(econ_conflict_evals, orient='index').reset_index()
econ_conflict_evals_df.to_csv("out/econ_conflict_evals_df.csv")

##################################
### H. Regime and policy theme ###
##################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_pol_full.csv",index_col=0)

# Transforms
x["milex_share"]=np.log(x["milex_share"]+1)
x["tax"]=np.log(x["tax"]+1)

# Fit
target="d_civil_conflict"
inputs=pol_theme
pol_conflict_df,pol_conflict_evals,pol_conflict_val=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=False)
pol_conflict_df.to_csv("out/pol_conflict_df.csv")
pol_conflict_evals_df = pd.DataFrame.from_dict(pol_conflict_evals, orient='index').reset_index()
pol_conflict_evals_df.to_csv("out/pol_conflict_evals_df.csv")

################
### Ensemble ###
################

# Get ensemble predictions as additive index, using the performance in the validation data as weights

# Obtain weights based on the inverse of the brier score in the validation data so that higher means better
weights=[1-history_conflict_val["brier"],
         1-demog_conflict_val["brier"],
         1-geog_conflict_val["brier"],
         1-econ_conflict_val["brier"],
         1-pol_conflict_val["brier"]]

# Min-max normalize weights, so that the highest weight gets assigned a value of 1 and the lowest a value of zero
weights_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]

# Make sure the final weights sum up to one
weights_conflict_n = [x / sum(weights_n) for x in weights_n]

# Calculate the weighted index
ensemble = (history_conflict_df.preds_proba*weights_conflict_n[0])+ \
            (demog_conflict_df.preds_proba*weights_conflict_n[1])+ \
            (geog_conflict_df.preds_proba*weights_conflict_n[2])+ \
            (econ_conflict_df.preds_proba*weights_conflict_n[3])+ \
            (pol_conflict_df.preds_proba*weights_conflict_n[4])
            
# Make df and save                        
ensemble_conflict=pd.concat([history_conflict_df[["country","year","d_civil_conflict"]],pd.DataFrame(ensemble)],axis=1)
ensemble_conflict.columns=["country","year","d_civil_conflict","preds_proba"]
ensemble_conflict=ensemble_conflict.reset_index(drop=True)
ensemble_conflict.to_csv("out/ensemble_conflict.csv")

###################
### Evaluations ###
###################

# Evaluate the ensemble in the test data 
ensemble_conflict_s=ensemble_conflict.loc[(ensemble_war["year"]>=2019)&(ensemble_war["year"]<=2023)]

# Brier
brier = brier_score_loss(ensemble_conflict_s.d_civil_conflict, ensemble_conflict_s.preds_proba)

# AUPR
precision, recall, _ = precision_recall_curve(ensemble_conflict_s.d_civil_conflict, ensemble_conflict_s.preds_proba)
aupr = auc(recall, precision)

# AUROC
auroc = roc_auc_score(ensemble_conflict_s.d_civil_conflict, ensemble_conflict_s.preds_proba)

# Save
evals_conflict_ensemble = {"brier": brier, "aupr": aupr, "auroc": auroc}
evals_conflict_ensemble_df = pd.DataFrame.from_dict(evals_conflict_ensemble, orient='index').reset_index()
evals_conflict_ensemble_df.to_csv("out/evals_conflict_ensemble_df.csv")


print(f"{round(base_conflict_evals['aupr'],5)} &  \\\
      {round(base_conflict_evals['auroc'],5)} &  \\\
      {round(base_conflict_evals['brier'],5)}")
      
print(f"{round(history_conflict_evals['aupr'],5)} &  \\\
      {round(history_conflict_evals['auroc'],5)} &  \\\
      {round(history_conflict_evals['brier'],5)}")
      
print(f"{round(demog_conflict_evals['aupr'],5)} &  \\\
      {round(demog_conflict_evals['auroc'],5)} &  \\\
      {round(demog_conflict_evals['brier'],5)}")

print(f"{round(geog_conflict_evals['aupr'],5)} &  \\\
      {round(geog_conflict_evals['auroc'],5)} &  \\\
      {round(geog_conflict_evals['brier'],5)}")

print(f"{round(econ_conflict_evals['aupr'],5)} &  \\\
      {round(econ_conflict_evals['auroc'],5)} &  \\\
      {round(econ_conflict_evals['brier'],5)}")
           
print(f"{round(pol_conflict_evals['aupr'],5)} &  \\\
       {round(pol_conflict_evals['auroc'],5)} &  \\\
       {round(pol_conflict_evals['brier'],5)}")          
      
print(f"{round(evals_conflict_ensemble['aupr'],5)} &  \\\
       {round(evals_conflict_ensemble['auroc'],5)} &  \\\
       {round(evals_conflict_ensemble['brier'],5)}")   
             
                                ###############
                                ### Protest ###
                                ###############
                                
exclude={"Dominica":54,
         "Grenada":55,
         "Saint Lucia":56,
         "Saint Vincent and the Grenadines":57,
         "Antigua & Barbuda":58,
         "Saint Kitts and Nevis":60,
         "Monaco":221,
         "Liechtenstein":223,
         "San Marino":331,
         "Andorra":232,
         "Abkhazia":396,
         "South Ossetia":397,
         "São Tomé and Principe":403,
         "Seychelles":591,
         "Vanuatu":935,
         "Kiribati":970,
         "Nauru":971,
         "Tonga":972,
         "Tuvalu":973,
         "Marshall Islands":983,
         "Palau":986,
         "Micronesia":987,
         "Samoa":990,
         "German Democratic Republic":265,
         "Czechoslovakia":315,
         "Yugoslavia":345,
         "Abkhazia":396,
         "South Ossetia":397,
         "Yemen, People's Republic of":680,
         "Taiwan":713, 
         "Bahamas":31,
         "Belize":80,
         "Brunei Darussalam":835, 
         "Kosovo":347, 
         "Democratic Peoples Republic of Korea":731}                       
                                
base = pd.read_csv("data/data_out/acled_cy_protest.csv",index_col=0)
base = base[["year","gw_codes"]][~base['gw_codes'].isin(list(exclude.values()))]
                              
###################
### A. Baseline ###
###################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

# Fit
target="d_protest"
inputs=['d_protest_lag1']
base_protest_df,base_protest_evals,base_protest_val=gen_model(y,x,target,inputs,model_fit=DummyClassifier(strategy="most_frequent"),int_methods=False)
base_protest_df.to_csv("out/base_protest_df.csv")
base_protest_evals_df = pd.DataFrame.from_dict(base_protest_evals, orient='index').reset_index()
base_protest_evals_df.to_csv("out/base_protest_evals_df.csv")

########################
### B. History theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

# Fit
target="d_protest"
inputs=['d_protest_lag1',"d_protest_zeros_growth","d_neighbors_proteset_lag1","regime_duration"]
history_protest_df,history_protest_evals,history_protest_val,history_protest_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
history_protest_df.to_csv("out/history_protest_df.csv")
history_protest_evals_df = pd.DataFrame.from_dict(history_protest_evals, orient='index').reset_index()
history_protest_evals_df.to_csv("out/history_protest_evals_df.csv")
pd.DataFrame(history_protest_shap[:,:,1]).to_csv("out/history_protest_shap.csv")

###########################
### C. Demography theme ###
###########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_demog_full.csv",index_col=0)

# Transforms
x["pop"]=np.log(x["pop"]+1)

# Merge
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

# Fit
target="d_protest"
inputs=demog_theme
demog_protest_df,demog_protest_evals,demog_protest_val,demog_protest_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
demog_protest_df.to_csv("out/demog_protest_df.csv")
demog_protest_evals_df = pd.DataFrame.from_dict(demog_protest_evals, orient='index').reset_index()
demog_protest_evals_df.to_csv("out/demog_protest_evals_df.csv")
pd.DataFrame(demog_protest_shap[:,:,1]).to_csv("out/demog_protest_shap.csv")


#########################################
### E.  Geography & environment theme ###
#########################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_geog_full.csv",index_col=0)

# Transforms
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)

# Merge
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

# Fit
target="d_protest"
inputs=geog_theme
geog_protest_df,geog_protest_evals,geog_protest_val,geog_protest_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
geog_protest_df.to_csv("out/geog_protest_df.csv")
geog_protest_evals_df = pd.DataFrame.from_dict(geog_protest_evals, orient='index').reset_index()
geog_protest_evals_df.to_csv("out/geog_protest_evals_df.csv")
pd.DataFrame(geog_protest_shap[:,:,1]).to_csv("out/geog_protest_shap.csv")

########################
### G. Economy theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_econ_full.csv",index_col=0)

# Transforms
x["natres_share"]=np.log(x["natres_share"]+1)
x["oil_share"]=np.log(x["oil_share"]+1)
x["gas_share"]=np.log(x["gas_share"]+1)
x["coal_share"]=np.log(x["coal_share"]+1)
x["forest_share"]=np.log(x["forest_share"]+1)
x["minerals_share"]=np.log(x["minerals_share"]+1)
x["gdp"]=np.log(x["gdp"]+1)
x["inflat"]=np.sign(x["inflat"]) * np.log(np.abs(x["inflat"]) + 1)
x["conprice"]=np.log(x["conprice"]+1)

# Merge
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

# Fit
target="d_protest"
inputs=econ_theme
econ_protest_df,econ_protest_evals,econ_protest_val,econ_protest_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
econ_protest_df.to_csv("out/econ_protest_df.csv")
econ_protest_evals_df = pd.DataFrame.from_dict(econ_protest_evals, orient='index').reset_index()
econ_protest_evals_df.to_csv("out/econ_protest_evals_df.csv")
pd.DataFrame(econ_protest_shap[:,:,1]).to_csv("out/econ_protest_shap.csv")

##################################
### H. Regime and policy theme ###
##################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_pol_full.csv",index_col=0)

# Transforms
x["milex_share"]=np.log(x["milex_share"]+1)
x["tax"]=np.log(x["tax"]+1)

# Merge
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

# Fit
target="d_protest"
inputs=pol_theme
pol_protest_df,pol_protest_evals,pol_protest_val,pol_protest_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
pol_protest_df.to_csv("out/pol_protest_df.csv")
pol_protest_evals_df = pd.DataFrame.from_dict(pol_protest_evals, orient='index').reset_index()
pol_protest_evals_df.to_csv("out/pol_protest_evals_df.csv")
pd.DataFrame(pol_protest_shap[:,:,1]).to_csv("out/pol_protest_shap.csv")

################
### Ensemble ###
################

# Get ensemble predictions as additive index, using the performance in the validation data as weights

# Obtain weights based on the inverse of the brier score in the validation data so that higher means better
weights=[1-history_protest_val["brier"],
         1-demog_protest_val["brier"],
         1-geog_protest_val["brier"],
         1-econ_protest_val["brier"],
         1-pol_protest_val["brier"]]

# Min-max normalize weights, so that the highest weight gets assigned a value of 1 and the lowest a value of zero
weights_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]

# Make sure the final weights sum up to one
weights_protest_n = [x / sum(weights_n) for x in weights_n]

# Calculate the weighted index
ensemble = (history_protest_df.preds_proba*weights_protest_n[0])+ \
            (demog_protest_df.preds_proba*weights_protest_n[1])+ \
            (geog_protest_df.preds_proba*weights_protest_n[2])+ \
            (econ_protest_df.preds_proba*weights_protest_n[3])+ \
            (pol_protest_df.preds_proba*weights_protest_n[4])

# Make df and save                
ensemble_protest=pd.concat([history_protest_df[["country","year","d_protest"]],pd.DataFrame(ensemble)],axis=1)
ensemble_protest.columns=["country","year","d_protest","preds_proba"]
ensemble_protest=ensemble_protest.reset_index(drop=True)
ensemble_protest.to_csv("out/ensemble_protest.csv")

###################
### Evaluations ###
###################

# Evaluate the ensemble in the test data 
ensemble_protest_s=ensemble_protest.loc[(ensemble_war["year"]>=2019)&(ensemble_war["year"]<=2023)]

# Brier
brier = brier_score_loss(ensemble_protest_s.d_protest, ensemble_protest_s.preds_proba)

# AUPR
precision, recall, _ = precision_recall_curve(ensemble_protest_s.d_protest, ensemble_protest_s.preds_proba)
aupr = auc(recall, precision)

# AROC
auroc = roc_auc_score(ensemble_protest_s.d_protest, ensemble_protest_s.preds_proba)

# Save
evals_protest_ensemble = {"brier": brier, "aupr": aupr, "auroc": auroc}
evals_protest_ensemble_df = pd.DataFrame.from_dict(evals_protest_ensemble, orient='index').reset_index()
evals_protest_ensemble_df.to_csv("out/evals_protest_ensemble_df.csv")


print(f"{round(base_protest_evals['aupr'],5)} &  \\\
      {round(base_protest_evals['auroc'],5)} &  \\\
      {round(base_protest_evals['brier'],5)}")
      
print(f"{round(history_protest_evals['aupr'],5)} &  \\\
      {round(history_protest_evals['auroc'],5)} &  \\\
      {round(history_protest_evals['brier'],5)}")
      
print(f"{round(demog_protest_evals['aupr'],5)} &  \\\
      {round(demog_protest_evals['auroc'],5)} &  \\\
      {round(demog_protest_evals['brier'],5)}")

print(f"{round(geog_protest_evals['aupr'],5)} &  \\\
      {round(geog_protest_evals['auroc'],5)} &  \\\
      {round(geog_protest_evals['brier'],5)}")

print(f"{round(econ_protest_evals['aupr'],5)} &  \\\
      {round(econ_protest_evals['auroc'],5)} &  \\\
      {round(econ_protest_evals['brier'],5)}")
           
print(f"{round(pol_protest_evals['aupr'],5)} &  \\\
       {round(pol_protest_evals['auroc'],5)} &  \\\
       {round(pol_protest_evals['brier'],5)}")          
      
print(f"{round(evals_protest_ensemble['aupr'],5)} &  \\\
       {round(evals_protest_ensemble['auroc'],5)} &  \\\
       {round(evals_protest_ensemble['brier'],5)}")          
      
        
                                    #############
                                    ### Riots ###
                                    #############
                                     
exclude={"Dominica":54,
         "Grenada":55,
         "Saint Lucia":56,
         "Saint Vincent and the Grenadines":57,
         "Antigua & Barbuda":58,
         "Saint Kitts and Nevis":60,
         "Monaco":221,
         "Liechtenstein":223,
         "San Marino":331,
         "Andorra":232,
         "Abkhazia":396,
         "South Ossetia":397,
         "São Tomé and Principe":403,
         "Seychelles":591,
         "Vanuatu":935,
         "Kiribati":970,
         "Nauru":971,
         "Tonga":972,
         "Tuvalu":973,
         "Marshall Islands":983,
         "Palau":986,
         "Micronesia":987,
         "Samoa":990,
         "German Democratic Republic":265,
         "Czechoslovakia":315,
         "Yugoslavia":345,
         "Abkhazia":396,
         "South Ossetia":397,
         "Yemen, People's Republic of":680,
         "Taiwan":713, 
         "Bahamas":31,
         "Belize":80,
         "Brunei Darussalam":835, 
         "Kosovo":347, 
         "Democratic Peoples Republic of Korea":731}                       
                                
base = pd.read_csv("data/data_out/acled_cy_protest.csv",index_col=0)
base = base[["year","gw_codes"]][~base['gw_codes'].isin(list(exclude.values()))]
                                                                        
###################
### A. Baseline ###
###################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

# Fit
target="d_riot"
inputs=['d_riot_lag1']
base_riot_df,base_riot_evals,base_riot_val=gen_model(y,x,target,inputs,model_fit=DummyClassifier(strategy="most_frequent"),int_methods=False)
base_riot_df.to_csv("out/base_riot_df.csv")
base_riot_evals_df = pd.DataFrame.from_dict(base_riot_evals, orient='index').reset_index()
base_riot_evals_df.to_csv("out/base_riot_evals_df.csv")

########################
### B. History theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

# Fit
target="d_riot"
inputs=['d_riot_lag1',"d_riot_zeros_growth","d_neighbors_riot_lag1",'regime_duration']
history_riot_df,history_riot_evals,history_riot_val,history_riot_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
history_riot_df.to_csv("out/history_riot_df.csv")
history_riot_evals_df = pd.DataFrame.from_dict(history_riot_evals, orient='index').reset_index()
history_riot_evals_df.to_csv("out/history_riot_evals_df.csv")
pd.DataFrame(history_riot_shap[:,:,1]).to_csv("out/history_riot_shap.csv")

###########################
### C. Demography theme ###
###########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_demog_full.csv",index_col=0)

# Transforms
x["pop"]=np.log(x["pop"]+1)

# Merge
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

# Fit
target="d_riot"
inputs=demog_theme
demog_riot_df,demog_riot_evals,demog_riot_val,demog_riot_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
demog_riot_df.to_csv("out/demog_riot_df.csv")
demog_riot_evals_df = pd.DataFrame.from_dict(demog_riot_evals, orient='index').reset_index()
demog_riot_evals_df.to_csv("out/demog_riot_evals_df.csv")
pd.DataFrame(demog_riot_shap[:,:,1]).to_csv("out/demog_riot_shap.csv")

#########################################
### E.  Geography & environment theme ###
#########################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_geog_full.csv",index_col=0)

# Transforms
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)

# Merge
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

# Fit
target="d_riot"
inputs=geog_theme
geog_riot_df,geog_riot_evals,geog_riot_val,geog_riot_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
geog_riot_df.to_csv("out/geog_riot_df.csv")
geog_riot_evals_df = pd.DataFrame.from_dict(geog_riot_evals, orient='index').reset_index()
geog_riot_evals_df.to_csv("out/geog_riot_evals_df.csv")
pd.DataFrame(geog_riot_shap[:,:,1]).to_csv("out/geog_riot_shap.csv")

########################
### G. Economy theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_econ_full.csv",index_col=0)

# Transforms
x["natres_share"]=np.log(x["natres_share"]+1)
x["oil_share"]=np.log(x["oil_share"]+1)
x["gas_share"]=np.log(x["gas_share"]+1)
x["coal_share"]=np.log(x["coal_share"]+1)
x["forest_share"]=np.log(x["forest_share"]+1)
x["minerals_share"]=np.log(x["minerals_share"]+1)
x["gdp"]=np.log(x["gdp"]+1)
x["inflat"]=np.sign(x["inflat"]) * np.log(np.abs(x["inflat"]) + 1)
x["conprice"]=np.log(x["conprice"]+1)

# Merge
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

# Fit
target="d_riot"
inputs=econ_theme
econ_riot_df,econ_riot_evals,econ_riot_val,econ_riot_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
econ_riot_df.to_csv("out/econ_riot_df.csv")
econ_riot_evals_df = pd.DataFrame.from_dict(econ_riot_evals, orient='index').reset_index()
econ_riot_evals_df.to_csv("out/econ_riot_evals_df.csv")
pd.DataFrame(econ_riot_shap[:,:,1]).to_csv("out/econ_riot_shap.csv")

##################################
### H. Regime and policy theme ###
##################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_pol_full.csv",index_col=0)

# Transforms
x["milex_share"]=np.log(x["milex_share"]+1)
x["tax"]=np.log(x["tax"]+1)

# Merge
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

# Fit
target="d_riot"
inputs=pol_theme
pol_riot_df,pol_riot_evals,pol_riot_val,pol_riot_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
pol_riot_df.to_csv("out/pol_riot_df.csv")
pol_riot_evals_df = pd.DataFrame.from_dict(pol_riot_evals, orient='index').reset_index()
pol_riot_evals_df.to_csv("out/pol_riot_evals_df.csv")
pd.DataFrame(pol_riot_shap[:,:,1]).to_csv("out/pol_riot_shap.csv")

################
### Ensemble ###
################

# Get ensemble predictions as additive index, using the performance in the validation data as weights

# Obtain weights based on the inverse of the brier score in the validation data so that higher means better
weights=[1-history_riot_val["brier"],
         1-demog_riot_val["brier"],
         1-geog_riot_val["brier"],
         1-econ_riot_val["brier"],
         1-pol_riot_val["brier"]]

# Min-max normalize weights, so that the highest weight gets assigned a value of 1 and the lowest a value of zero
weights_riot_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]

# Make sure the final weights sum up to one
weights_riot_n = [x / sum(weights_riot_n) for x in weights_riot_n]

# Calculate the weighted index
ensemble = (history_riot_df.preds_proba*weights_riot_n[0])+ \
            (demog_riot_df.preds_proba*weights_riot_n[1])+ \
            (geog_riot_df.preds_proba*weights_riot_n[2])+ \
            (econ_riot_df.preds_proba*weights_riot_n[3])+ \
            (pol_riot_df.preds_proba*weights_riot_n[4])
            
# Make df and save                       
ensemble_riot=pd.concat([history_riot_df[["country","year","d_riot"]],pd.DataFrame(ensemble)],axis=1)
ensemble_riot.columns=["country","year","d_riot","preds_proba"]
ensemble_riot=ensemble_riot.reset_index(drop=True)
ensemble_riot.to_csv("out/ensemble_riot.csv")

###################
### Evaluations ###
###################

# Evaluate the ensemble in the test data 
ensemble_riot_s = ensemble_riot.loc[(ensemble_war["year"]>=2019)&(ensemble_war["year"]<=2023)]

# Brier
brier = brier_score_loss(ensemble_riot_s.d_riot, ensemble_riot_s.preds_proba)

# AUPR
precision, recall, _ = precision_recall_curve(ensemble_riot_s.d_riot, ensemble_riot_s.preds_proba)
aupr = auc(recall, precision)

# AUROC
auroc = roc_auc_score(ensemble_riot_s.d_riot, ensemble_riot_s.preds_proba)

# Save
evals_riot_ensemble = {"brier": brier, "aupr": aupr, "auroc": auroc}
evals_riot_ensemble_df = pd.DataFrame.from_dict(evals_riot_ensemble, orient='index').reset_index()
evals_riot_ensemble_df.to_csv("out/evals_riot_ensemble_df.csv")


print(f"{round(base_riot_evals['aupr'],5)} &  \\\
      {round(base_riot_evals['auroc'],5)} &  \\\
      {round(base_riot_evals['brier'],5)}")
      
print(f"{round(history_riot_evals['aupr'],5)} &  \\\
      {round(history_riot_evals['auroc'],5)} &  \\\
      {round(history_riot_evals['brier'],5)}")
      
print(f"{round(demog_riot_evals['aupr'],5)} &  \\\
      {round(demog_riot_evals['auroc'],5)} &  \\\
      {round(demog_riot_evals['brier'],5)}")

print(f"{round(geog_riot_evals['aupr'],5)} &  \\\
      {round(geog_riot_evals['auroc'],5)} &  \\\
      {round(geog_riot_evals['brier'],5)}")

print(f"{round(econ_riot_evals['aupr'],5)} &  \\\
      {round(econ_riot_evals['auroc'],5)} &  \\\
      {round(econ_riot_evals['brier'],5)}")
           
print(f"{round(pol_riot_evals['aupr'],5)} &  \\\
       {round(pol_riot_evals['auroc'],5)} &  \\\
       {round(pol_riot_evals['brier'],5)}")          
      
print(f"{round(evals_riot_ensemble['aupr'],5)} &  \\\
       {round(evals_riot_ensemble['auroc'],5)} &  \\\
       {round(evals_riot_ensemble['brier'],5)}")  
      
        
                                #################
                                ### Terrorism ###
                                #################

###################
### A. Baseline ###
###################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)
y=y.loc[y["year"]<2021]
y=y.loc[y["year"]!=1993]
x=x.loc[x["year"]<2021] 
x=x.loc[x["year"]!=1993]

# Fit
target="d_terror"
inputs=['d_terror_lag1']
base_terror_df,base_terror_evals,base_terror_val=gen_model(y,x,target,inputs,model_fit=DummyClassifier(strategy="most_frequent"),int_methods=False)
base_terror_df.to_csv("out/base_terror_df.csv")
base_terror_evals_df = pd.DataFrame.from_dict(base_terror_evals, orient='index').reset_index()
base_terror_evals_df.to_csv("out/base_terror_evals_df.csv")

########################
### B. History theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)
y=y.loc[y["year"]<2021]
y=y.loc[y["year"]!=1993]
x=x.loc[x["year"]<2021] 
x=x.loc[x["year"]!=1993]

# Fit
target="d_terror"
inputs=['d_terror_lag1',"d_terror_zeros_growth","d_neighbors_terror_lag1",'regime_duration']
history_terror_df,history_terror_evals,history_terror_val,history_terror_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
history_terror_df.to_csv("out/history_terror_df.csv")
history_terror_evals_df = pd.DataFrame.from_dict(history_terror_evals, orient='index').reset_index()
history_terror_evals_df.to_csv("out/history_terror_evals_df.csv")
pd.DataFrame(history_terror_shap[:,:,1]).to_csv("out/history_terror_shap.csv")

###########################
### C. Demography theme ###
###########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_demog_full.csv",index_col=0)

# Transforms
x["pop"]=np.log(x["pop"]+1)

# Subset
y=y.loc[y["year"]<2021]
y=y.loc[y["year"]!=1993]
x=x.loc[x["year"]<2021] 
x=x.loc[x["year"]!=1993]

# Fit
target="d_terror"
inputs=demog_theme
demog_terror_df,demog_terror_evals,demog_terror_val,demog_terror_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
demog_terror_df.to_csv("out/demog_terror_df.csv")
demog_terror_evals_df = pd.DataFrame.from_dict(demog_terror_evals, orient='index').reset_index()
demog_terror_evals_df.to_csv("out/demog_terror_evals_df.csv")
pd.DataFrame(demog_terror_shap[:,:,1]).to_csv("out/demog_terror_shap.csv")

########################################
### E. Geography & environment theme ###
########################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_geog_full.csv",index_col=0)

# Transforms
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)

# Subset
y=y.loc[y["year"]<2021]
y=y.loc[y["year"]!=1993]
x=x.loc[x["year"]<2021] 
x=x.loc[x["year"]!=1993]

# Fit
target="d_terror"
inputs=geog_theme
geog_terror_df,geog_terror_evals,geog_terror_val,geog_terror_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
geog_terror_df.to_csv("out/geog_terror_df.csv")
geog_terror_evals_df = pd.DataFrame.from_dict(geog_terror_evals, orient='index').reset_index()
geog_terror_evals_df.to_csv("out/geog_terror_evals_df.csv")
pd.DataFrame(geog_terror_shap[:,:,1]).to_csv("out/geog_terror_shap.csv")

########################
### G. Economy theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_econ_full.csv",index_col=0)

# Transforms
x["natres_share"]=np.log(x["natres_share"]+1)
x["oil_share"]=np.log(x["oil_share"]+1)
x["gas_share"]=np.log(x["gas_share"]+1)
x["coal_share"]=np.log(x["coal_share"]+1)
x["forest_share"]=np.log(x["forest_share"]+1)
x["minerals_share"]=np.log(x["minerals_share"]+1)
x["gdp"]=np.log(x["gdp"]+1)
x["inflat"]=np.sign(x["inflat"]) * np.log(np.abs(x["inflat"]) + 1)
x["conprice"]=np.log(x["conprice"]+1)

# Subset
y=y.loc[y["year"]<2021]
y=y.loc[y["year"]!=1993]
x=x.loc[x["year"]<2021] 
x=x.loc[x["year"]!=1993]

# Fit
target="d_terror"
inputs=econ_theme
econ_terror_df,econ_terror_evals,econ_terror_val,econ_terror_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
econ_terror_df.to_csv("out/econ_terror_df.csv")
econ_terror_evals_df = pd.DataFrame.from_dict(econ_terror_evals, orient='index').reset_index()
econ_terror_evals_df.to_csv("out/econ_terror_evals_df.csv")
pd.DataFrame(econ_terror_shap[:,:,1]).to_csv("out/econ_terror_shap.csv")

##################################
### H. Regime and policy theme ###
##################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_pol_full.csv",index_col=0)

# Transforms
x["milex_share"]=np.log(x["milex_share"]+1)
x["tax"]=np.log(x["tax"]+1)

# Subset
y=y.loc[y["year"]<2021]
y=y.loc[y["year"]!=1993]
x=x.loc[x["year"]<2021] 
x=x.loc[x["year"]!=1993]

# Fit
target="d_terror"
inputs=pol_theme
pol_terror_df,pol_terror_evals,pol_terror_val,pol_terror_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
pol_terror_df.to_csv("out/pol_terror_df.csv")
pol_terror_evals_df = pd.DataFrame.from_dict(pol_terror_evals, orient='index').reset_index()
pol_terror_evals_df.to_csv("out/pol_terror_evals_df.csv")
pd.DataFrame(pol_terror_shap[:,:,1]).to_csv("out/pol_terror_shap.csv")

################
### Ensemble ###
################

# Get ensemble predictions as additive index, using the performance in the validation data as weights

# Obtain weights based on the inverse of the brier score in the validation data so that higher means better
weights=[1-history_terror_evals["brier"],
         1-demog_terror_evals["brier"],
         1-geog_terror_evals["brier"],
         1-econ_terror_evals["brier"],
         1-pol_terror_evals["brier"]]

# Min-max normalize weights, so that the highest weight gets assigned a value of 1 and the lowest a value of zero
weights_terror_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]

# Make sure the final weights sum up to one
weights_terror_n = [x / sum(weights_terror_n) for x in weights_terror_n]

# Calculate the weighted index
ensemble = (history_terror_df.preds_proba*weights_terror_n[0])+ \
            (demog_terror_df.preds_proba*weights_terror_n[1])+ \
            (geog_terror_df.preds_proba*weights_terror_n[2])+ \
            (econ_terror_df.preds_proba*weights_terror_n[3])+ \
            (pol_terror_df.preds_proba*weights_terror_n[4])
            
# Make df and save                       
ensemble_terror=pd.concat([history_terror_df[["country","year","d_terror"]],pd.DataFrame(ensemble)],axis=1)
ensemble_terror.columns=["country","year","d_terror","preds_proba"]
ensemble_terror=ensemble_terror.reset_index(drop=True)
ensemble_terror.to_csv("out/ensemble_terror.csv")

###################
### Evaluations ###
###################

# Evaluate the ensemble in the test data 
ensemble_terror_s=ensemble_terror.loc[(ensemble_war["year"]>=2019)&(ensemble_war["year"]<=2023)]

# Brier
brier = brier_score_loss(ensemble_terror_s.d_terror, ensemble_terror_s.preds_proba)

# AUPR
precision, recall, _ = precision_recall_curve(ensemble_terror_s.d_terror, ensemble_terror_s.preds_proba)
aupr = auc(recall, precision)

# AUROC
auroc = roc_auc_score(ensemble_terror_s.d_terror, ensemble_terror_s.preds_proba)

# Save
evals_terror_ensemble = {"brier": brier, "aupr": aupr, "auroc": auroc}
evals_terror_ensemble_df = pd.DataFrame.from_dict(evals_terror_ensemble, orient='index').reset_index()
evals_terror_ensemble_df.to_csv("out/evals_terror_ensemble_df.csv")


print(f"{round(base_terror_evals['aupr'],5)} &  \\\
      {round(base_terror_evals['auroc'],5)} &  \\\
      {round(base_terror_evals['brier'],5)}")
      
print(f"{round(history_terror_evals['aupr'],5)} &  \\\
      {round(history_terror_evals['auroc'],5)} &  \\\
      {round(history_terror_evals['brier'],5)}")
      
print(f"{round(demog_terror_evals['aupr'],5)} &  \\\
      {round(demog_terror_evals['auroc'],5)} &  \\\
      {round(demog_terror_evals['brier'],5)}")

print(f"{round(geog_terror_evals['aupr'],5)} &  \\\
      {round(geog_terror_evals['auroc'],5)} &  \\\
      {round(geog_terror_evals['brier'],5)}")

print(f"{round(econ_terror_evals['aupr'],5)} &  \\\
      {round(econ_terror_evals['auroc'],5)} &  \\\
      {round(econ_terror_evals['brier'],5)}")
           
print(f"{round(pol_terror_evals['aupr'],5)} &  \\\
       {round(pol_terror_evals['auroc'],5)} &  \\\
       {round(pol_terror_evals['brier'],5)}")          
      
print(f"{round(evals_terror_ensemble['aupr'],5)} &  \\\
       {round(evals_terror_ensemble['auroc'],5)} &  \\\
       {round(evals_terror_ensemble['brier'],5)}")  
      
                            ###################
                            ### State-based ###
                            ###################

###################
### A. Baseline ###
###################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)

# Fit
target="d_sb"
inputs=['d_sb_lag1']
base_sb_df,base_sb_evals,base_sb_val=gen_model(y,x,target,inputs,model_fit=DummyClassifier(strategy="most_frequent"),int_methods=False)
base_sb_df.to_csv("out/base_sb_df.csv")
base_sb_evals_df = pd.DataFrame.from_dict(base_sb_evals, orient='index').reset_index()
base_sb_evals_df.to_csv("out/base_sb_evals_df.csv")

########################
### B. History theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)

# Fit
target="d_sb"
inputs=['d_sb_lag1',"d_sb_zeros_growth","d_neighbors_sb_lag1",'regime_duration']
history_sb_df,history_sb_evals,history_sb_val,history_sb_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
history_sb_df.to_csv("out/history_sb_df.csv")
history_sb_evals_df = pd.DataFrame.from_dict(history_sb_evals, orient='index').reset_index()
history_sb_evals_df.to_csv("out/history_sb_evals_df.csv")
pd.DataFrame(history_sb_shap[:,:,1]).to_csv("out/history_sb_shap.csv")

###########################
### C. Demography theme ###
###########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_demog_full.csv",index_col=0)

# Transforms
x["pop"]=np.log(x["pop"]+1)

# Fit
target="d_sb"
inputs=demog_theme
demog_sb_df,demog_sb_evals,demog_sb_val,demog_sb_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
demog_sb_df.to_csv("out/demog_sb_df.csv")
demog_sb_evals_df = pd.DataFrame.from_dict(demog_sb_evals, orient='index').reset_index()
demog_sb_evals_df.to_csv("out/demog_sb_evals_df.csv")
pd.DataFrame(demog_sb_shap[:,:,1]).to_csv("out/demog_sb_shap.csv")

########################################
### E. Geography & environment theme ###
########################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_geog_full.csv",index_col=0)

# Transforms
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)

# Fit
target="d_sb"
inputs=geog_theme
geog_sb_df,geog_sb_evals,geog_sb_val,geog_sb_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
geog_sb_df.to_csv("out/geog_sb_df.csv")
geog_sb_evals_df = pd.DataFrame.from_dict(geog_sb_evals, orient='index').reset_index()
geog_sb_evals_df.to_csv("out/geog_sb_evals_df.csv")
pd.DataFrame(geog_sb_shap[:,:,1]).to_csv("out/geog_sb_shap.csv")

########################
### G. Economy theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_econ_full.csv",index_col=0)

# Transforms
x["natres_share"]=np.log(x["natres_share"]+1)
x["oil_share"]=np.log(x["oil_share"]+1)
x["gas_share"]=np.log(x["gas_share"]+1)
x["coal_share"]=np.log(x["coal_share"]+1)
x["forest_share"]=np.log(x["forest_share"]+1)
x["minerals_share"]=np.log(x["minerals_share"]+1)
x["gdp"]=np.log(x["gdp"]+1)
x["inflat"]=np.sign(x["inflat"]) * np.log(np.abs(x["inflat"]) + 1)
x["conprice"]=np.log(x["conprice"]+1)

# Fit
target="d_sb"
inputs=econ_theme
econ_sb_df,econ_sb_evals,econ_sb_val,econ_sb_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
econ_sb_df.to_csv("out/econ_sb_df.csv")
econ_sb_evals_df = pd.DataFrame.from_dict(econ_sb_evals, orient='index').reset_index()
econ_sb_evals_df.to_csv("out/econ_sb_evals_df.csv")
pd.DataFrame(econ_sb_shap[:,:,1]).to_csv("out/econ_sb_shap.csv")

##################################
### H. Regime and policy theme ###
##################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_pol_full.csv",index_col=0)

# Transforms
x["milex_share"]=np.log(x["milex_share"]+1)
x["tax"]=np.log(x["tax"]+1)

# Fit
target="d_sb"
inputs=pol_theme
pol_sb_df,pol_sb_evals,pol_sb_val,pol_sb_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
pol_sb_df.to_csv("out/pol_sb_df.csv")
pol_sb_evals_df = pd.DataFrame.from_dict(pol_sb_evals, orient='index').reset_index()
pol_sb_evals_df.to_csv("out/pol_sb_evals_df.csv")
pd.DataFrame(pol_sb_shap[:,:,1]).to_csv("out/pol_sb_shap.csv")

################
### Ensemble ###
################

# Get ensemble predictions as additive index, using the performance in the validation data as weights

# Obtain weights based on the inverse of the brier score in the validation data so that higher means better
weights=[1-history_sb_val["brier"],
         1-demog_sb_val["brier"],
         1-geog_sb_val["brier"],
         1-econ_sb_val["brier"],
         1-pol_sb_val["brier"]]

# Min-max normalize weights, so that the highest weight gets assigned a value of 1 and the lowest a value of zero
weights_sb_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]

# Make sure the final weights sum up to one
weights_sb_n = [x / sum(weights_sb_n) for x in weights_sb_n]

# Calculate the weighted index
ensemble = (history_sb_df.preds_proba*weights_sb_n[0])+ \
            (demog_sb_df.preds_proba*weights_sb_n[1])+ \
            (geog_sb_df.preds_proba*weights_sb_n[2])+ \
            (econ_sb_df.preds_proba*weights_sb_n[3])+ \
            (pol_sb_df.preds_proba*weights_sb_n[4])

# Make df and save            
ensemble_sb=pd.concat([history_sb_df[["country","year","d_sb"]],pd.DataFrame(ensemble)],axis=1)
ensemble_sb.columns=["country","year","d_sb","preds_proba"]
ensemble_sb=ensemble_sb.reset_index(drop=True)
ensemble_sb.to_csv("out/ensemble_sb.csv")

###################
### Evaluations ###
###################

# Evaluate the ensemble in the test data 
ensemble_sb_s=ensemble_sb.loc[(ensemble_war["year"]>=2019)&(ensemble_war["year"]<=2023)]

# Brier
brier = brier_score_loss(ensemble_sb_s.d_sb, ensemble_sb_s.preds_proba)

# AUPR
precision, recall, _ = precision_recall_curve(ensemble_sb_s.d_sb, ensemble_sb_s.preds_proba)
aupr = auc(recall, precision)

# AUROC
auroc = roc_auc_score(ensemble_sb_s.d_sb, ensemble_sb_s.preds_proba)

# Save
evals_sb_ensemble = {"brier": brier, "aupr": aupr, "auroc": auroc}
evals_sb_ensemble_df = pd.DataFrame.from_dict(evals_sb_ensemble, orient='index').reset_index()
evals_sb_ensemble_df.to_csv("out/evals_sb_ensemble_df.csv")


print(f"{round(base_sb_evals['aupr'],5)} &  \\\
      {round(base_sb_evals['auroc'],5)} &  \\\
      {round(base_sb_evals['brier'],5)}")
      
print(f"{round(history_sb_evals['aupr'],5)} &  \\\
      {round(history_sb_evals['auroc'],5)} &  \\\
      {round(history_sb_evals['brier'],5)}")
      
print(f"{round(demog_sb_evals['aupr'],5)} &  \\\
      {round(demog_sb_evals['auroc'],5)} &  \\\
      {round(demog_sb_evals['brier'],5)}")

print(f"{round(geog_sb_evals['aupr'],5)} &  \\\
      {round(geog_sb_evals['auroc'],5)} &  \\\
      {round(geog_sb_evals['brier'],5)}")

print(f"{round(econ_sb_evals['aupr'],5)} &  \\\
      {round(econ_sb_evals['auroc'],5)} &  \\\
      {round(econ_sb_evals['brier'],5)}")
           
print(f"{round(pol_sb_evals['aupr'],5)} &  \\\
       {round(pol_sb_evals['auroc'],5)} &  \\\
       {round(pol_sb_evals['brier'],5)}")          
      
print(f"{round(evals_sb_ensemble['aupr'],5)} &  \\\
       {round(evals_sb_ensemble['auroc'],5)} &  \\\
       {round(evals_sb_ensemble['brier'],5)}")  

                            ##########################
                            ### One-sided violence ###
                            ##########################

###################
### A. Baseline ###
###################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)

# Fit
target="d_osv"
inputs=['d_osv_lag1']
base_osv_df,base_osv_evals,base_osv_val=gen_model(y,x,target,inputs,model_fit=DummyClassifier(strategy="most_frequent"),int_methods=False)
base_osv_df.to_csv("out/base_osv_df.csv")
base_osv_evals_df = pd.DataFrame.from_dict(base_osv_evals, orient='index').reset_index()
base_osv_evals_df.to_csv("out/base_osv_evals_df.csv")

########################
### B. History theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)

# Fit
target="d_osv"
inputs=['d_osv_lag1',"d_osv_zeros_growth","d_neighbors_osv_lag1",'regime_duration']
history_osv_df,history_osv_evals,history_osv_val,history_osv_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
history_osv_df.to_csv("out/history_osv_df.csv")
history_osv_evals_df = pd.DataFrame.from_dict(history_osv_evals, orient='index').reset_index()
history_osv_evals_df.to_csv("out/history_osv_evals_df.csv")
pd.DataFrame(history_osv_shap[:,:,1]).to_csv("out/history_osv_shap.csv")

###########################
### C. Demography theme ###
###########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_demog_full.csv",index_col=0)

# Transforms
x["pop"]=np.log(x["pop"]+1)

# Fit
target="d_osv"
inputs=demog_theme
demog_osv_df,demog_osv_evals,demog_osv_val,demog_osv_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
demog_osv_df.to_csv("out/demog_osv_df.csv")
demog_osv_evals_df = pd.DataFrame.from_dict(demog_osv_evals, orient='index').reset_index()
demog_osv_evals_df.to_csv("out/demog_osv_evals_df.csv")
pd.DataFrame(demog_osv_shap[:,:,1]).to_csv("out/demog_osv_shap.csv")

########################################
### E. Geography & environment theme ###
########################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_geog_full.csv",index_col=0)

# Transforms
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)

# Fit
target="d_osv"
inputs=geog_theme
geog_osv_df,geog_osv_evals,geog_osv_val,geog_osv_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
geog_osv_df.to_csv("out/geog_osv_df.csv")
geog_osv_evals_df = pd.DataFrame.from_dict(geog_osv_evals, orient='index').reset_index()
geog_osv_evals_df.to_csv("out/geog_osv_evals_df.csv")
pd.DataFrame(geog_osv_shap[:,:,1]).to_csv("out/geog_osv_shap.csv")

########################
### G. Economy theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_econ_full.csv",index_col=0)

# Transforms
x["natres_share"]=np.log(x["natres_share"]+1)
x["oil_share"]=np.log(x["oil_share"]+1)
x["gas_share"]=np.log(x["gas_share"]+1)
x["coal_share"]=np.log(x["coal_share"]+1)
x["forest_share"]=np.log(x["forest_share"]+1)
x["minerals_share"]=np.log(x["minerals_share"]+1)
x["gdp"]=np.log(x["gdp"]+1)
x["inflat"]=np.sign(x["inflat"]) * np.log(np.abs(x["inflat"]) + 1)
x["conprice"]=np.log(x["conprice"]+1)

# Fit
target="d_osv"
inputs=econ_theme
econ_osv_df,econ_osv_evals,econ_osv_val,econ_osv_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
econ_osv_df.to_csv("out/econ_osv_df.csv")
econ_osv_evals_df = pd.DataFrame.from_dict(econ_osv_evals, orient='index').reset_index()
econ_osv_evals_df.to_csv("out/econ_osv_evals_df.csv")
pd.DataFrame(econ_osv_shap[:,:,1]).to_csv("out/econ_osv_shap.csv")

##################################
### H. Regime and policy theme ###
##################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_pol_full.csv",index_col=0)

# Transforms
x["milex_share"]=np.log(x["milex_share"]+1)
x["tax"]=np.log(x["tax"]+1)

# Fit
target="d_osv"
inputs=pol_theme
pol_osv_df,pol_osv_evals,pol_osv_val,pol_osv_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
pol_osv_df.to_csv("out/pol_osv_df.csv")
pol_osv_evals_df = pd.DataFrame.from_dict(pol_osv_evals, orient='index').reset_index()
pol_osv_evals_df.to_csv("out/pol_osv_evals_df.csv")
pd.DataFrame(pol_osv_shap[:,:,1]).to_csv("out/pol_osv_shap.csv")

################
### Ensemble ###
################

# Get ensemble predictions as additive index, using the performance in the validation data as weights

# Obtain weights based on the inverse of the brier score in the validation data so that higher means better
weights=[1-history_osv_val["brier"],
         1-demog_osv_val["brier"],
         1-geog_osv_val["brier"],
         1-econ_osv_val["brier"],
         1-pol_osv_val["brier"]]

# Min-max normalize weights, so that the highest weight gets assigned a value of 1 and the lowest a value of zero
weights_osv_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]

# Make sure the final weights sum up to one
weights_osv_n = [x / sum(weights_osv_n) for x in weights_osv_n]

# Calculate the weighted index
ensemble = (history_osv_df.preds_proba*weights_osv_n[0])+ \
            (demog_osv_df.preds_proba*weights_osv_n[1])+ \
            (geog_osv_df.preds_proba*weights_osv_n[2])+ \
            (econ_osv_df.preds_proba*weights_osv_n[3])+ \
            (pol_osv_df.preds_proba*weights_osv_n[4])

# Make df and save            
ensemble_osv=pd.concat([history_osv_df[["country","year","d_osv"]],pd.DataFrame(ensemble)],axis=1)
ensemble_osv.columns=["country","year","d_osv","preds_proba"]
ensemble_osv=ensemble_osv.reset_index(drop=True)
ensemble_osv.to_csv("out/ensemble_osv.csv")

###################
### Evaluations ###
###################

# Evaluate the ensemble in the test data 
ensemble_osv_s=ensemble_osv.loc[(ensemble_war["year"]>=2019)&(ensemble_war["year"]<=2023)]

# Brier
brier = brier_score_loss(ensemble_osv_s.d_osv, ensemble_osv_s.preds_proba)

# AUPR
precision, recall, _ = precision_recall_curve(ensemble_osv_s.d_osv, ensemble_osv_s.preds_proba)
aupr = auc(recall, precision)

# AUROC
auroc = roc_auc_score(ensemble_osv_s.d_osv, ensemble_osv_s.preds_proba)

# Save
evals_osv_ensemble = {"brier": brier, "aupr": aupr, "auroc": auroc}
evals_osv_ensemble_df = pd.DataFrame.from_dict(evals_osv_ensemble, orient='index').reset_index()
evals_osv_ensemble_df.to_csv("out/evals_osv_ensemble_df.csv")


print(f"{round(base_osv_evals['aupr'],5)} &  \\\
      {round(base_osv_evals['auroc'],5)} &  \\\
      {round(base_osv_evals['brier'],5)}")
      
print(f"{round(history_osv_evals['aupr'],5)} &  \\\
      {round(history_osv_evals['auroc'],5)} &  \\\
      {round(history_osv_evals['brier'],5)}")
      
print(f"{round(demog_osv_evals['aupr'],5)} &  \\\
      {round(demog_osv_evals['auroc'],5)} &  \\\
      {round(demog_osv_evals['brier'],5)}")

print(f"{round(geog_osv_evals['aupr'],5)} &  \\\
      {round(geog_osv_evals['auroc'],5)} &  \\\
      {round(geog_osv_evals['brier'],5)}")

print(f"{round(econ_osv_evals['aupr'],5)} &  \\\
      {round(econ_osv_evals['auroc'],5)} &  \\\
      {round(econ_osv_evals['brier'],5)}")
           
print(f"{round(pol_osv_evals['aupr'],5)} &  \\\
       {round(pol_osv_evals['auroc'],5)} &  \\\
       {round(pol_osv_evals['brier'],5)}")          
      
print(f"{round(evals_osv_ensemble['aupr'],5)} &  \\\
       {round(evals_osv_ensemble['auroc'],5)} &  \\\
       {round(evals_osv_ensemble['brier'],5)}")  
            
                            #######################
                            ### Non-state based ###
                            #######################

###################
### A. Baseline ###
###################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)

# Fit
target="d_ns"
inputs=['d_ns_lag1']
base_ns_df,base_ns_evals,base_ns_val=gen_model(y,x,target,inputs,model_fit=DummyClassifier(strategy="most_frequent"),int_methods=False)
base_ns_df.to_csv("out/base_ns_df.csv")
base_ns_evals_df = pd.DataFrame.from_dict(base_ns_evals, orient='index').reset_index()
base_ns_evals_df.to_csv("out/base_ns_evals_df.csv")

########################
### B. History theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)

# Fit
target="d_ns"
inputs=['d_ns_lag1',"d_ns_zeros_growth","d_neighbors_ns_lag1",'regime_duration']
history_ns_df,history_ns_evals,history_ns_val,history_ns_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
history_ns_df.to_csv("out/history_ns_df.csv")
history_ns_evals_df = pd.DataFrame.from_dict(history_ns_evals, orient='index').reset_index()
history_ns_evals_df.to_csv("out/history_ns_evals_df.csv")
pd.DataFrame(history_ns_shap[:,:,1]).to_csv("out/history_ns_shap.csv")

###########################
### C. Demography theme ###
###########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_demog_full.csv",index_col=0)

# Transforms
x["pop"]=np.log(x["pop"]+1)

# Fit
target="d_ns"
inputs=demog_theme
demog_ns_df,demog_ns_evals,demog_ns_val,demog_ns_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
demog_ns_df.to_csv("out/demog_ns_df.csv")
demog_ns_evals_df = pd.DataFrame.from_dict(demog_ns_evals, orient='index').reset_index()
demog_ns_evals_df.to_csv("out/demog_ns_evals_df.csv")
pd.DataFrame(demog_ns_shap[:,:,1]).to_csv("out/demog_ns_shap.csv")

########################################
### E. Geography & environment theme ###
########################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_geog_full.csv",index_col=0)

# Transforms
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)

# Fit
target="d_ns"
inputs=geog_theme
geog_ns_df,geog_ns_evals,geog_ns_val,geog_ns_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
geog_ns_df.to_csv("out/geog_ns_df.csv")
geog_ns_evals_df = pd.DataFrame.from_dict(geog_ns_evals, orient='index').reset_index()
geog_ns_evals_df.to_csv("out/geog_ns_evals_df.csv")
pd.DataFrame(geog_ns_shap[:,:,1]).to_csv("out/geog_ns_shap.csv")

########################
### G. Economy theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_econ_full.csv",index_col=0)

# Transforms
x["natres_share"]=np.log(x["natres_share"]+1)
x["oil_share"]=np.log(x["oil_share"]+1)
x["gas_share"]=np.log(x["gas_share"]+1)
x["coal_share"]=np.log(x["coal_share"]+1)
x["forest_share"]=np.log(x["forest_share"]+1)
x["minerals_share"]=np.log(x["minerals_share"]+1)
x["gdp"]=np.log(x["gdp"]+1)
x["inflat"]=np.sign(x["inflat"]) * np.log(np.abs(x["inflat"]) + 1)
x["conprice"]=np.log(x["conprice"]+1)

# Fit
target="d_ns"
inputs=econ_theme
econ_ns_df,econ_ns_evals,econ_ns_val,econ_ns_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
econ_ns_df.to_csv("out/econ_ns_df.csv")
econ_ns_evals_df = pd.DataFrame.from_dict(econ_ns_evals, orient='index').reset_index()
econ_ns_evals_df.to_csv("out/econ_ns_evals_df.csv")
pd.DataFrame(econ_ns_shap[:,:,1]).to_csv("out/econ_ns_shap.csv")

##################################
### H. Regime and policy theme ###
##################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_pol_full.csv",index_col=0)

# Transforms
x["milex_share"]=np.log(x["milex_share"]+1)
x["tax"]=np.log(x["tax"]+1)

# Fit
target="d_ns"
inputs=pol_theme
pol_ns_df,pol_ns_evals,pol_ns_val,pol_ns_shap=gen_model(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=grid,int_methods=True)
pol_ns_df.to_csv("out/pol_ns_df.csv")
pol_ns_evals_df = pd.DataFrame.from_dict(pol_ns_evals, orient='index').reset_index()
pol_ns_evals_df.to_csv("out/pol_ns_evals_df.csv")
pd.DataFrame(pol_ns_shap[:,:,1]).to_csv("out/pol_ns_shap.csv")

################
### Ensemble ###
################

# Get ensemble predictions as additive index, using the performance in the validation data as weights

# Obtain weights based on the inverse of the brier score in the validation data so that higher means better
weights=[1-history_ns_val["brier"],
         1-demog_ns_val["brier"],
         1-geog_ns_val["brier"],
         1-econ_ns_val["brier"],
         1-pol_ns_val["brier"]]

# Min-max normalize weights, so that the highest weight gets assigned a value of 1 and the lowest a value of zero
weights_ns_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]

# Make sure the final weights sum up to one
weights_ns_n = [x / sum(weights_ns_n) for x in weights_ns_n]

# Calculate the weighted index
ensemble = (history_ns_df.preds_proba*weights_ns_n[0])+ \
            (demog_ns_df.preds_proba*weights_ns_n[1])+ \
            (geog_ns_df.preds_proba*weights_ns_n[2])+ \
            (econ_ns_df.preds_proba*weights_ns_n[3])+ \
            (pol_ns_df.preds_proba*weights_ns_n[4])

# Make df and save                        
ensemble_ns=pd.concat([history_ns_df[["country","year","d_ns"]],pd.DataFrame(ensemble)],axis=1)
ensemble_ns.columns=["country","year","d_ns","preds_proba"]
ensemble_ns=ensemble_ns.reset_index(drop=True)
ensemble_ns.to_csv("out/ensemble_ns.csv")

###################
### Evaluations ###
###################

# Evaluate the ensemble in the test data 
ensemble_ns_s=ensemble_ns.loc[(ensemble_war["year"]>=2019)&(ensemble_war["year"]<=2023)]

# Beier
brier = brier_score_loss(ensemble_ns_s.d_ns, ensemble_ns_s.preds_proba)

# AUPR
precision, recall, _ = precision_recall_curve(ensemble_ns_s.d_ns, ensemble_ns_s.preds_proba)
aupr = auc(recall, precision)

# AUROC
auroc = roc_auc_score(ensemble_ns_s.d_ns, ensemble_ns_s.preds_proba)

# Save
evals_ns_ensemble = {"brier": brier, "aupr": aupr, "auroc": auroc}
evals_ns_ensemble_df = pd.DataFrame.from_dict(evals_ns_ensemble, orient='index').reset_index()
evals_ns_ensemble_df.to_csv("out/evals_ns_ensemble_df.csv")

print(f"{round(base_ns_evals['aupr'],5)} &  \\\
      {round(base_ns_evals['auroc'],5)} &  \\\
      {round(base_ns_evals['brier'],5)}")
      
print(f"{round(history_ns_evals['aupr'],5)} &  \\\
      {round(history_ns_evals['auroc'],5)} &  \\\
      {round(history_ns_evals['brier'],5)}")
      
print(f"{round(demog_ns_evals['aupr'],5)} &  \\\
      {round(demog_ns_evals['auroc'],5)} &  \\\
      {round(demog_ns_evals['brier'],5)}")

print(f"{round(geog_ns_evals['aupr'],5)} &  \\\
      {round(geog_ns_evals['auroc'],5)} &  \\\
      {round(geog_ns_evals['brier'],5)}")

print(f"{round(econ_ns_evals['aupr'],5)} &  \\\
      {round(econ_ns_evals['auroc'],5)} &  \\\
      {round(econ_ns_evals['brier'],5)}")
           
print(f"{round(pol_ns_evals['aupr'],5)} &  \\\
       {round(pol_ns_evals['auroc'],5)} &  \\\
       {round(pol_ns_evals['brier'],5)}")          
      
print(f"{round(evals_ns_ensemble['aupr'],5)} &  \\\
       {round(evals_ns_ensemble['auroc'],5)} &  \\\
       {round(evals_ns_ensemble['brier'],5)}")  






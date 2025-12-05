import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np 
from functions import gen_model_live
from sklearn.metrics import brier_score_loss

# Optimization gird
grid = {'n_estimators': [10, 231, 452, 673, 894, 1115, 1336, 1557, 1778, 2000],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]}


# Inputs
demog_theme=['pop','pop_density','pop_density_id','urb_share','rural_share','pop_male_share','pop_male_share_0_14','pop_male_share_15_19','pop_male_share_20_24','pop_male_share_25_29','pop_male_share_30_34','group_counts','group_counts_id','monopoly_share','monopoly_share_id','discriminated_share','discriminated_share_id','powerless_share','powerless_share_id','dominant_share','dominant_share_id','ethnic_frac','ethnic_frac_id','rel_frac','rel_frac_id','lang_frac','lang_frac_id','race_frac','race_frac_id']
geog_theme=['land','land_id','forest','forest_id','temp','temp_id','co2','co2_id','percip','percip_id','waterstress','waterstress_id','agri_land','agri_land_id','arable_land','arable_land_id','rugged','soil','desert','tropical','cont_africa','cont_asia']
econ_theme=['natres_share','natres_share_id','oil_share','oil_share_id','gas_share','gas_share_id','coal_share','coal_share_id','forest_share','forest_share_id','minerals_share','minerals_share_id','gdp','gdp_id','gni','gni_id','gdp_growth','gdp_growth_id','unemploy','unemploy_id','unemploy_male','unemploy_male_id','inflat','inflat_id','conprice','conprice_id','undernour','undernour_id','foodprod','foodprod_id','water_rural','water_rural_id','water_urb','water_urb_id','agri_share','agri_share_id','trade_share','trade_share_id','fert','lifeexp_female','lifeexp_male','pop_growth','pop_growth_id','inf_mort','exports','exports_id','imports','imports_id','primary_female','primary_female_id','primary_male','primary_male_id','second_female','second_female_id','second_male','second_male_id','tert_female','tert_female_id','tert_male','tert_male_id','eys','eys_id','eys_male','eys_male_id','eys_female','eys_female_id','mys','mys_id','mys_male','mys_male_id','mys_female','mys_female_id']
pol_theme=['armedforces_share','armedforces_share_id','milex_share','milex_share_id','corruption','corruption_id', 'effectiveness', 'effectiveness_id', 'polvio','polvio_id','regu','regu_id','law','law_id','account','account_id','tax','tax_id','broadband','broadband_id','telephone','telephone_id','internet_use','internet_use_id','mobile','mobile_id','polyarchy','libdem','libdem_id','partipdem','delibdem','egaldem','civlib','phyvio','pollib','privlib','execon','execon_id','exgender','exgender_id','exgeo','exgeo_id','expol','expol_id','exsoc','exsoc_id','shutdown','shutdown_id','filter','filter_id','tenure_months','tenure_months_id','dem_duration','dem_duration_id','election_recent','election_recent_id','lastelection','lastelection_id']

                                ###############
                                ### Protest ###
                                ###############
                                
# List of microstates: 
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
                                
base = pd.read_csv("data/data_out/acled_cm_protest.csv",index_col=0)
base = base[["dd","gw_codes","n_protest_events"]][~base['gw_codes'].isin(list(exclude.values()))]
base = base.sort_values(by=["gw_codes","dd"])
base = base.reset_index(drop=True)
                         
#####################
### History theme ###
#####################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full_cm.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["dd","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["dd","gw_codes"],how="left")

# Fit
target="d_protest"
inputs=['d_protest_lag1',"d_protest_zeros_decay","regime_duration"]
history_protest_df,history_protest_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
history_protest_df.to_csv("out/history_protest_df_cm_live.csv")
history_protest_evals_df = pd.DataFrame.from_dict(history_protest_evals, orient='index').reset_index()
history_protest_evals_df.to_csv("out/history_protest_evals_df.csv")

########################
### Demography theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_demog_full_cm.csv",index_col=0)

# Transforms
x["pop"]=np.log(x["pop"]+1)

# Merge
y=pd.merge(left=base,right=y,on=["dd","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["dd","gw_codes"],how="left")

# Fit
target="d_protest"
inputs=demog_theme
demog_protest_df,demog_protest_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
demog_protest_df.to_csv("out/demog_protest_df_cm_live.csv")  
demog_protest_evals_df = pd.DataFrame.from_dict(demog_protest_evals, orient='index').reset_index()
demog_protest_evals_df.to_csv("out/demog_protest_evals_df.csv")

#####################################
### Geography & environment theme ###
#####################################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_geog_full_cm.csv",index_col=0)

# Transforms
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)

# Merge
y=pd.merge(left=base,right=y,on=["dd","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["dd","gw_codes"],how="left")

# Fit
target="d_protest"
inputs=geog_theme
geog_protest_df,geog_protest_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
geog_protest_df.to_csv("out/geog_protest_df_cm_live.csv")
geog_protest_evals_df = pd.DataFrame.from_dict(geog_protest_evals, orient='index').reset_index()
geog_protest_evals_df.to_csv("out/geog_protest_evals_df.csv")

#####################
### Economy theme ###
#####################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_econ_full_cm.csv",index_col=0)

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
y=pd.merge(left=base,right=y,on=["dd","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["dd","gw_codes"],how="left")

# Fit
target="d_protest"
inputs=econ_theme
econ_protest_df,econ_protest_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
econ_protest_df.to_csv("out/econ_protest_df_cm_live.csv")
econ_protest_evals_df = pd.DataFrame.from_dict(econ_protest_evals, orient='index').reset_index()
econ_protest_evals_df.to_csv("out/econ_protest_evals_df.csv")

###############################
### Regime and policy theme ###
###############################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_pol_full_cm.csv",index_col=0)

# Transforms
x["milex_share"]=np.log(x["milex_share"]+1)
x["tax"]=np.log(x["tax"]+1)

# Merge 
y=pd.merge(left=base,right=y,on=["dd","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["dd","gw_codes"],how="left")

# Fit
target="d_protest"
inputs=pol_theme
pol_protest_df,pol_protest_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
pol_protest_df.to_csv("out/pol_protest_df_cm_live.csv")
pol_protest_evals_df = pd.DataFrame.from_dict(pol_protest_evals, orient='index').reset_index()
pol_protest_evals_df.to_csv("out/pol_protest_evals_df.csv")

################
### Ensemble ###
################

# Get ensemble predictions as additive index, using the performance in the validation data as weights

# Obtain weights based on the inverse of the brier score in the validation data so that higher means better
weights=[1-history_protest_evals["brier"],
         1-demog_protest_evals["brier"],
         1-geog_protest_evals["brier"],
         1-econ_protest_evals["brier"],
         1-pol_protest_evals["brier"]]

# Min-max normalize weights, so that the highest weight gets assigned a value of 1 and the lowest a value of zero
weights_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]

# Make sure the final weights sum up to one
weights_protest_n = [x / sum(weights_n) for x in weights_n]

# Calculate the weighted index
ensemble = (history_protest_df.preds*weights_protest_n[0])+ \
            (demog_protest_df.preds*weights_protest_n[1])+ \
            (geog_protest_df.preds*weights_protest_n[2])+ \
            (econ_protest_df.preds*weights_protest_n[3])+ \
            (pol_protest_df.preds*weights_protest_n[4])
            
# Make df and save                       
ensemble_protest=pd.concat([history_protest_df[["country","dd","d_protest"]],pd.DataFrame(ensemble)],axis=1)
ensemble_protest.columns=["country","dd","d_protest","preds"]
ensemble_protest=ensemble_protest.reset_index(drop=True)
ensemble_protest.to_csv("out/ensemble_protest_live.csv")


                                    #############
                                    ### Riots ###
                                    #############
                                    
# List of microstates: 
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
                                
base = pd.read_csv("data/data_out/acled_cm_protest.csv",index_col=0)
base = base[["dd","gw_codes","n_protest_events"]][~base['gw_codes'].isin(list(exclude.values()))]
base = base.sort_values(by=["gw_codes","dd"])
base = base.reset_index(drop=True)                                   

#####################
### History theme ###
#####################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full_cm.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["dd","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["dd","gw_codes"],how="left")

# Fit
target="d_riot"
inputs=['d_riot_lag1',"d_riot_zeros_decay",'regime_duration']
history_riot_df,history_riot_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
history_riot_df.to_csv("out/history_riot_df_cm_live.csv")
history_riot_evals_df = pd.DataFrame.from_dict(history_riot_evals, orient='index').reset_index()
history_riot_evals_df.to_csv("out/history_riot_evals_df.csv")

########################
### Demography theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_demog_full_cm.csv",index_col=0)

# Transforms
x["pop"]=np.log(x["pop"]+1)

# Merge 
y=pd.merge(left=base,right=y,on=["dd","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["dd","gw_codes"],how="left")

# Fit
target="d_riot"
inputs=demog_theme
demog_riot_df,demog_riot_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
demog_riot_df.to_csv("out/demog_riot_df_cm_live.csv")
demog_riot_evals_df = pd.DataFrame.from_dict(demog_riot_evals, orient='index').reset_index()
demog_riot_evals_df.to_csv("out/demog_riot_evals_df.csv")

#####################################
### Geography & environment theme ###
#####################################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_geog_full_cm.csv",index_col=0)

# Transforms
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)

# Merge 
y=pd.merge(left=base,right=y,on=["dd","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["dd","gw_codes"],how="left")

# Fit
target="d_riot"
inputs=geog_theme
geog_riot_df,geog_riot_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
geog_riot_df.to_csv("out/geog_riot_df_cm_live.csv")
geog_riot_evals_df = pd.DataFrame.from_dict(geog_riot_evals, orient='index').reset_index()
geog_riot_evals_df.to_csv("out/geog_riot_evals_df.csv")

#####################
### Economy theme ###
#####################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_econ_full_cm.csv",index_col=0)

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
y=pd.merge(left=base,right=y,on=["dd","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["dd","gw_codes"],how="left")

# Fit
target="d_riot"
inputs=econ_theme
econ_riot_df,econ_riot_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
econ_riot_df.to_csv("out/econ_riot_df_cm_live.csv")
econ_riot_evals_df = pd.DataFrame.from_dict(econ_riot_evals, orient='index').reset_index()
econ_riot_evals_df.to_csv("out/econ_riot_evals_df.csv")

###############################
### Regime and policy theme ###
###############################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_pol_full_cm.csv",index_col=0)

# Transforms
x["milex_share"]=np.log(x["milex_share"]+1)
x["tax"]=np.log(x["tax"]+1)

# Merge 
y=pd.merge(left=base,right=y,on=["dd","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["dd","gw_codes"],how="left")

# Fit
target="d_riot"
inputs=pol_theme
pol_riot_df,pol_riot_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
pol_riot_df.to_csv("out/pol_riot_df_cm_live.csv")
pol_riot_evals_df = pd.DataFrame.from_dict(pol_riot_evals, orient='index').reset_index()
pol_riot_evals_df.to_csv("out/pol_riot_evals_df.csv")

################
### Ensemble ###
################

# Get ensemble predictions as additive index, using the performance in the validation data as weights

# Obtain weights based on the inverse of the brier score in the validation data so that higher means better
weights=[1-history_riot_evals["brier"],
         1-demog_riot_evals["brier"],
         1-geog_riot_evals["brier"],
         1-econ_riot_evals["brier"],
         1-pol_riot_evals["brier"]]

# Min-max normalize weights, so that the highest weight gets assigned a value of 1 and the lowest a value of zero
weights_riot_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]

# Make sure the final weights sum up to one
weights_riot_n = [x / sum(weights_riot_n) for x in weights_riot_n]

# Calculate the weighted index
ensemble = (history_riot_df.preds*weights_riot_n[0])+ \
            (demog_riot_df.preds*weights_riot_n[1])+ \
            (geog_riot_df.preds*weights_riot_n[2])+ \
            (econ_riot_df.preds*weights_riot_n[3])+ \
            (pol_riot_df.preds*weights_riot_n[4])
            
# Make df and save                        
ensemble_riot=pd.concat([history_riot_df[["country","dd","d_riot"]],pd.DataFrame(ensemble)],axis=1)
ensemble_riot.columns=["country","dd","d_riot","preds"]
ensemble_riot=ensemble_riot.reset_index(drop=True)
ensemble_riot.to_csv("out/ensemble_riot_live.csv")


                                #######################
                                ### Remote violence ###
                                #######################

# List of microstates: 
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
                                
base = pd.read_csv("data/data_out/acled_cm_protest.csv",index_col=0)
base = base[["dd","gw_codes","n_protest_events"]][~base['gw_codes'].isin(list(exclude.values()))]
base = base.sort_values(by=["gw_codes","dd"])
base = base.reset_index(drop=True)

#####################
### History theme ###
#####################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full_cm.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["dd","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["dd","gw_codes"],how="left")

# Fit
target="d_remote"
inputs=['d_remote_lag1',"d_remote_zeros_decay",'regime_duration']
history_terror_df,history_terror_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
history_terror_df.to_csv("out/history_terror_df_cm_live.csv")
history_terror_evals_df = pd.DataFrame.from_dict(history_terror_evals, orient='index').reset_index()
history_terror_evals_df.to_csv("out/history_terror_evals_df.csv")

########################
### Demography theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_demog_full_cm.csv",index_col=0)

# Transforms
x["pop"]=np.log(x["pop"]+1)

# Merge
y=pd.merge(left=base,right=y,on=["dd","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["dd","gw_codes"],how="left")

# Fit
target="d_remote"
inputs=demog_theme
demog_terror_df,demog_terror_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
demog_terror_df.to_csv("out/demog_terror_df_cm_live.csv")
demog_terror_evals_df = pd.DataFrame.from_dict(demog_terror_evals, orient='index').reset_index()
demog_terror_evals_df.to_csv("out/demog_terror_evals_df.csv")

#####################################
### Geography & environment theme ###
#####################################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_geog_full_cm.csv",index_col=0)

# Transforms
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)

# Merge
y=pd.merge(left=base,right=y,on=["dd","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["dd","gw_codes"],how="left")

# Fit
target="d_remote"
inputs=geog_theme
geog_terror_df,geog_terror_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
geog_terror_df.to_csv("out/geog_terror_df_cm_live.csv")
geog_terror_evals_df = pd.DataFrame.from_dict(geog_terror_evals, orient='index').reset_index()
geog_terror_evals_df.to_csv("out/geog_terror_evals_df.csv")

#####################
### Economy theme ###
#####################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_econ_full_cm.csv",index_col=0)

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
y=pd.merge(left=base,right=y,on=["dd","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["dd","gw_codes"],how="left")

# Fit
target="d_remote"
inputs=econ_theme
econ_terror_df,econ_terror_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
econ_terror_df.to_csv("out/econ_terror_df_cm_live.csv")
econ_terror_evals_df = pd.DataFrame.from_dict(econ_terror_evals, orient='index').reset_index()
econ_terror_evals_df.to_csv("out/econ_terror_evals_df.csv")

###############################
### Regime and policy theme ###
###############################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_pol_full_cm.csv",index_col=0)

# Transforms
x["milex_share"]=np.log(x["milex_share"]+1)
x["tax"]=np.log(x["tax"]+1)

# Merge
y=pd.merge(left=base,right=y,on=["dd","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["dd","gw_codes"],how="left")

# Fit
target="d_remote"
inputs=pol_theme
pol_terror_df,pol_terror_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
pol_terror_df.to_csv("out/pol_terror_df_cm_live.csv")
pol_terror_evals_df = pd.DataFrame.from_dict(pol_terror_evals, orient='index').reset_index()
pol_terror_evals_df.to_csv("out/pol_terror_evals_df.csv")

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
ensemble = (history_terror_df.preds*weights_terror_n[0])+ \
            (demog_terror_df.preds*weights_terror_n[1])+ \
            (geog_terror_df.preds*weights_terror_n[2])+ \
            (econ_terror_df.preds*weights_terror_n[3])+ \
            (pol_terror_df.preds*weights_terror_n[4])

# Make df and save            
ensemble_terror=pd.concat([history_terror_df[["country","dd","d_remote"]],pd.DataFrame(ensemble)],axis=1)
ensemble_terror.columns=["country","dd","d_remote","preds"]
ensemble_terror=ensemble_terror.reset_index(drop=True)
ensemble_terror.to_csv("out/ensemble_terror_live.csv")
                            
                            ###################
                            ### State-based ###
                            ###################


#####################
### History theme ###
#####################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full_cm.csv",index_col=0)

# Fit
target="d_sb"
inputs=['d_sb_lag1',"d_sb_zeros_decay",'regime_duration']
history_sb_df,history_sb_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
history_sb_df.to_csv("out/history_sb_df_cm_live.csv")
history_sb_evals_df = pd.DataFrame.from_dict(history_sb_evals, orient='index').reset_index()
history_sb_evals_df.to_csv("out/history_sb_evals_df.csv")

########################
### Demography theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_demog_full_cm.csv",index_col=0)

# Transforms
x["pop"]=np.log(x["pop"]+1)

# Fit
target="d_sb"
inputs=demog_theme
demog_sb_df,demog_sb_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
demog_sb_df.to_csv("out/demog_sb_df_cm_live.csv")
demog_sb_evals_df = pd.DataFrame.from_dict(demog_sb_evals, orient='index').reset_index()
demog_sb_evals_df.to_csv("out/demog_sb_evals_df.csv")

#####################################
### Geography & environment theme ###
#####################################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_geog_full_cm.csv",index_col=0)

# Transforms
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)

# Fit
target="d_sb"
inputs=geog_theme
geog_sb_df,geog_sb_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
geog_sb_df.to_csv("out/geog_sb_df_cm_live.csv")
geog_sb_evals_df = pd.DataFrame.from_dict(geog_sb_evals, orient='index').reset_index()
geog_sb_evals_df.to_csv("out/geog_sb_evals_df.csv")

#####################
### Economy theme ###
#####################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_econ_full_cm.csv",index_col=0)

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
econ_sb_df,econ_sb_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
econ_sb_df.to_csv("out/econ_sb_df_cm_live.csv")
econ_sb_evals_df = pd.DataFrame.from_dict(econ_sb_evals, orient='index').reset_index()
econ_sb_evals_df.to_csv("out/econ_sb_evals_df.csv")

###############################
### Regime and policy theme ###
###############################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_pol_full_cm.csv",index_col=0)

# Transforms
x["milex_share"]=np.log(x["milex_share"]+1)
x["tax"]=np.log(x["tax"]+1)

# Fit
target="d_sb"
inputs=pol_theme
pol_sb_df,pol_sb_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
pol_sb_df.to_csv("out/pol_sb_df_cm_live.csv")
pol_sb_evals_df = pd.DataFrame.from_dict(pol_sb_evals, orient='index').reset_index()
pol_sb_evals_df.to_csv("out/pol_sb_evals_df.csv")

################
### Ensemble ###
################

# Get ensemble predictions as additive index, using the performance in the validation data as weights

# Obtain weights based on the inverse of the brier score in the validation data so that higher means better
weights=[1-history_sb_evals["brier"],
         1-demog_sb_evals["brier"],
         1-geog_sb_evals["brier"],
         1-econ_sb_evals["brier"],
         1-pol_sb_evals["brier"]]

# Min-max normalize weights, so that the highest weight gets assigned a value of 1 and the lowest a value of zero
weights_sb_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]

# Make sure the final weights sum up to one
weights_sb_n = [x / sum(weights_sb_n) for x in weights_sb_n]

# Calculate the weighted index
ensemble = (history_sb_df.preds*weights_sb_n[0])+ \
            (demog_sb_df.preds*weights_sb_n[1])+ \
            (geog_sb_df.preds*weights_sb_n[2])+ \
            (econ_sb_df.preds*weights_sb_n[3])+ \
            (pol_sb_df.preds*weights_sb_n[4])
            
# Make df and save                        
ensemble_sb=pd.concat([history_sb_df[["country","dd","d_sb"]],pd.DataFrame(ensemble)],axis=1)
ensemble_sb.columns=["country","dd","d_sb","preds"]
ensemble_sb=ensemble_sb.reset_index(drop=True)
ensemble_sb.to_csv("out/ensemble_sb_live.csv")


                            ##########################
                            ### One-sided violence ###
                            ##########################

#####################
### History theme ###
#####################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full_cm.csv",index_col=0)

# Fit
target="d_osv"
inputs=['d_osv_lag1',"d_osv_zeros_decay",'regime_duration']
history_osv_df,history_osv_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
history_osv_df.to_csv("out/history_osv_df_cm_live.csv")
history_osv_evals_df = pd.DataFrame.from_dict(history_osv_evals, orient='index').reset_index()
history_osv_evals_df.to_csv("out/history_osv_evals_df.csv")

########################
### Demography theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_demog_full_cm.csv",index_col=0)

# Transforms
x["pop"]=np.log(x["pop"]+1)

# Fit
target="d_osv"
inputs=demog_theme
demog_osv_df,demog_osv_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
demog_osv_df.to_csv("out/demog_osv_df_cm_live.csv")
demog_osv_evals_df = pd.DataFrame.from_dict(demog_osv_evals, orient='index').reset_index()
demog_osv_evals_df.to_csv("out/demog_osv_evals_df.csv")

#####################################
### Geography & environment theme ###
#####################################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_geog_full_cm.csv",index_col=0)

# Transforms
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)

# Fit
target="d_osv"
inputs=geog_theme
geog_osv_df,geog_osv_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
geog_osv_df.to_csv("out/geog_osv_df_cm_live.csv")
geog_osv_evals_df = pd.DataFrame.from_dict(geog_osv_evals, orient='index').reset_index()
geog_osv_evals_df.to_csv("out/geog_osv_evals_df.csv")

#####################
### Economy theme ###
#####################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_econ_full_cm.csv",index_col=0)

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
econ_osv_df,econ_osv_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
econ_osv_df.to_csv("out/econ_osv_df_cm_live.csv")
econ_osv_evals_df = pd.DataFrame.from_dict(econ_osv_evals, orient='index').reset_index()
econ_osv_evals_df.to_csv("out/econ_osv_evals_df.csv")

###############################
### Regime and policy theme ###
###############################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_pol_full_cm.csv",index_col=0)

# Transforms
x["milex_share"]=np.log(x["milex_share"]+1)
x["tax"]=np.log(x["tax"]+1)

# Fit
target="d_osv"
inputs=pol_theme
pol_osv_df,pol_osv_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
pol_osv_df.to_csv("out/pol_osv_df_cm_live.csv")
pol_osv_evals_df = pd.DataFrame.from_dict(pol_osv_evals, orient='index').reset_index()
pol_osv_evals_df.to_csv("out/pol_osv_evals_df.csv")

################
### Ensemble ###
################

# Get ensemble predictions as additive index, using the performance in the validation data as weights

# Obtain weights based on the inverse of the brier score in the validation data so that higher means better
weights=[1-history_osv_evals["brier"],
         1-demog_osv_evals["brier"],
         1-geog_osv_evals["brier"],
         1-econ_osv_evals["brier"],
         1-pol_osv_evals["brier"]]

# Min-max normalize weights, so that the highest weight gets assigned a value of 1 and the lowest a value of zero
weights_osv_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]

# Make sure the final weights sum up to one
weights_osv_n = [x / sum(weights_osv_n) for x in weights_osv_n]

# Calculate the weighted index
ensemble = (history_osv_df.preds*weights_osv_n[0])+ \
            (demog_osv_df.preds*weights_osv_n[1])+ \
            (geog_osv_df.preds*weights_osv_n[2])+ \
            (econ_osv_df.preds*weights_osv_n[3])+ \
            (pol_osv_df.preds*weights_osv_n[4])
            
# Make df and save                       
ensemble_osv=pd.concat([history_osv_df[["country","dd","d_osv"]],pd.DataFrame(ensemble)],axis=1)
ensemble_osv.columns=["country","dd","d_osv","preds"]
ensemble_osv=ensemble_osv.reset_index(drop=True)
ensemble_osv.to_csv("out/ensemble_osv_live.csv")

                            #######################
                            ### Non-state based ###
                            #######################

#####################
### History theme ###
#####################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full_cm.csv",index_col=0)

# Fit
target="d_ns"
inputs=['d_ns_lag1',"d_ns_zeros_decay",'regime_duration']
history_ns_df,history_ns_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
history_ns_df.to_csv("out/history_ns_df_cm_live.csv")
history_ns_evals_df = pd.DataFrame.from_dict(history_ns_evals, orient='index').reset_index()
history_ns_evals_df.to_csv("out/history_ns_evals_df.csv")

########################
### Demography theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_demog_full_cm.csv",index_col=0)

# Transforms
x["pop"]=np.log(x["pop"]+1)

# Fit
target="d_ns"
inputs=demog_theme
demog_ns_df,demog_ns_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
demog_ns_df.to_csv("out/demog_ns_df_cm_live.csv")
demog_ns_evals_df = pd.DataFrame.from_dict(demog_ns_evals, orient='index').reset_index()
demog_ns_evals_df.to_csv("out/demog_ns_evals_df.csv")

#####################################
### Geography & environment theme ###
#####################################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_geog_full_cm.csv",index_col=0)

# Transforms
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)

# Fit
target="d_ns"
inputs=geog_theme
geog_ns_df,geog_ns_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
geog_ns_df.to_csv("out/geog_ns_df_cm_live.csv")
geog_ns_evals_df = pd.DataFrame.from_dict(geog_ns_evals, orient='index').reset_index()
geog_ns_evals_df.to_csv("out/geog_ns_evals_df.csv")

#####################
### Economy theme ###
#####################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_econ_full_cm.csv",index_col=0)

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
econ_ns_df,econ_ns_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
econ_ns_df.to_csv("out/econ_ns_df_cm_live.csv")
econ_ns_evals_df = pd.DataFrame.from_dict(econ_ns_evals, orient='index').reset_index()
econ_ns_evals_df.to_csv("out/econ_ns_evals_df.csv")

###############################
### Regime and policy theme ###
###############################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_pol_full_cm.csv",index_col=0)

# Transforms
x["milex_share"]=np.log(x["milex_share"]+1)
x["tax"]=np.log(x["tax"]+1)

# Fit
target="d_ns"
inputs=pol_theme
pol_ns_df,pol_ns_evals=gen_model_live(y,x,target,inputs,model_fit=RandomForestClassifier(random_state=0),grid=None)
pol_ns_df.to_csv("out/pol_ns_df_cm_live.csv")
pol_ns_evals_df = pd.DataFrame.from_dict(pol_ns_evals, orient='index').reset_index()
pol_ns_evals_df.to_csv("out/pol_ns_evals_df.csv")

################
### Ensemble ###
################

# Get ensemble predictions as additive index, using the performance in the validation data as weights

# Obtain weights based on the inverse of the brier score in the validation data so that higher means better
weights=[1-history_ns_evals["brier"],
         1-demog_ns_evals["brier"],
         1-geog_ns_evals["brier"],
         1-econ_ns_evals["brier"],
         1-pol_ns_evals["brier"]]

# Min-max normalize weights, so that the highest weight gets assigned a value of 1 and the lowest a value of zero
weights_ns_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]

# Make sure the final weights sum up to one
weights_ns_n = [x / sum(weights_ns_n) for x in weights_ns_n]

# Calculate the weighted index
ensemble = (history_ns_df.preds*weights_ns_n[0])+ \
            (demog_ns_df.preds*weights_ns_n[1])+ \
            (geog_ns_df.preds*weights_ns_n[2])+ \
            (econ_ns_df.preds*weights_ns_n[3])+ \
            (pol_ns_df.preds*weights_ns_n[4])

# Make df and save                        
ensemble_ns=pd.concat([history_ns_df[["country","dd","d_ns"]],pd.DataFrame(ensemble)],axis=1)
ensemble_ns.columns=["country","dd","d_ns","preds"]
ensemble_ns=ensemble_ns.reset_index(drop=True)
ensemble_ns.to_csv("out/ensemble_ns_live.csv")

                ################################
                ################################
                ################################
                ### Ensemble of the Ensemble ###
                ################################
                ################################
                ################################
                
   
#############################
### Ensemble of ensembles ###
#############################

# Obtain weights based on the inverse of the brier score in the validation data (2020-2021)
# so that higher means better. The predictions in the validation data are in-sample, 
# based on the first round of ensembling.

weights=[]

# Protest
weights.append(1-brier_score_loss(ensemble_protest["d_protest"].loc[(ensemble_protest["dd"]>="2020-01")&(ensemble_protest["dd"]<="2021-12")], 
                                  ensemble_protest["preds"].loc[(ensemble_protest["dd"]>="2020-01")&(ensemble_protest["dd"]<="2021-12")]))

# Riots
weights.append(1-brier_score_loss(ensemble_riot["d_riot"].loc[(ensemble_riot["dd"]>="2020-01")&(ensemble_riot["dd"]<="2021-12")],
                                  ensemble_riot["preds"].loc[(ensemble_riot["dd"]>="2020-01")&(ensemble_riot["dd"]<="2021-12")]))

# Remote
weights.append(1-brier_score_loss(ensemble_terror["d_remote"].loc[(ensemble_terror["dd"]>="2020-01")&(ensemble_terror["dd"]<="2021-12")],
                                  ensemble_terror["preds"].loc[(ensemble_terror["dd"]>="2020-01")&(ensemble_terror["dd"]<="2021-12")]))

# SB
weights.append(1-brier_score_loss(ensemble_sb["d_sb"].loc[(ensemble_sb["dd"]>="2020-01")&(ensemble_sb["dd"]<="2021-12")],
                                  ensemble_sb["preds"].loc[(ensemble_sb["dd"]>="2020-01")&(ensemble_sb["dd"]<="2021-12")]))

# NS
weights.append(1-brier_score_loss(ensemble_ns["d_ns"].loc[(ensemble_ns["dd"]>="2020-01")&(ensemble_ns["dd"]<="2021-12")], 
                                  ensemble_ns["preds"].loc[(ensemble_ns["dd"]>="2020-01")&(ensemble_ns["dd"]<="2021-12")]))

# OSV
weights.append(1-brier_score_loss(ensemble_osv["d_osv"].loc[(ensemble_osv["dd"]>="2020-01")&(ensemble_osv["dd"]<="2021-12")],
                                  ensemble_osv["preds"].loc[(ensemble_osv["dd"]>="2020-01")&(ensemble_osv["dd"]<="2021-12")]))

# Min-max normalize weights, so that the highest weight gets assigned a value of 1 and the lowest a value of zero
weights_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]

# Make sure the final weights sum up to one
weights_n = [x / sum(weights_n) for x in weights_n]

# ACLED and UCDP have different temporal coverages, which needs to be reflected
# in the ensemble. The first round of ensembling, considers all outcomes (protest, 
# riots, remote, sb, ns, osv) but only for the months in the ACLED data. 

# Subset sb to only include country-months in ACLED
base=ensemble_protest[["country","dd"]]
ensemble_sb_short=pd.merge(base, ensemble_sb,on=["country","dd"],how="left")
ensemble_sb_short=ensemble_sb_short.reset_index(drop=True)

# Subset ns to only include country-months in ACLED
base=ensemble_protest[["country","dd"]]
ensemble_ns_short=pd.merge(base, ensemble_ns,on=["country","dd"],how="left")
ensemble_ns_short=ensemble_ns_short.reset_index(drop=True)

# Subset osv to only include country-months in ACLED
base=ensemble_protest[["country","dd"]]
ensemble_osv_short=pd.merge(base, ensemble_osv,on=["country","dd"],how="left")
ensemble_osv_short=ensemble_osv_short.reset_index(drop=True)

# Calculate the weighted index
ensemble = (ensemble_protest.preds*weights_n[0])+ \
            (ensemble_riot.preds*weights_n[1])+ \
            (ensemble_terror.preds*weights_n[2])+ \
            (ensemble_sb_short.preds*weights_n[3])+ \
            (ensemble_ns_short.preds*weights_n[4])+ \
            (ensemble_osv_short.preds*weights_n[5])
 
# Make df            
ensemble_ens=pd.concat([ensemble_protest[["country","dd"]],pd.DataFrame(ensemble)],axis=1)
ensemble_ens.columns=["country","dd","preds"]

# The second round of ensembling only consideres UCDP data (sb, ns, osv) for
# the months not included in ACLED. 

# Make a list of country-months included in ACLED 
base=ensemble_protest[["dd","country"]]
base["drop"] = base['dd'].astype(str) + '-' + base['country']
drop=list(base["drop"])

# Make corresponding id in sb data
ensemble_sb["id"] = ensemble_sb['dd'].astype(str) + '-' + ensemble_sb['country']
#  and remove country-months in ACLED from sb
ensemble_sb_s = ensemble_sb[~ensemble_sb['id'].isin(drop)]
ensemble_sb_s=ensemble_sb_s.reset_index(drop=True)

# Make corresponding id in ns data
ensemble_ns["id"] = ensemble_ns['dd'].astype(str) + '-' + ensemble_ns['country']
# and remove country-months in ACLED from ns
ensemble_ns_s = ensemble_ns[~ensemble_ns['id'].isin(drop)]
ensemble_ns_s=ensemble_ns_s.reset_index(drop=True)

# Make corresponding id in osv data
ensemble_osv["id"] = ensemble_osv['dd'].astype(str) + '-' + ensemble_osv['country']
# and remove country-months in ACLED from ns
ensemble_osv_s = ensemble_osv[~ensemble_osv['id'].isin(drop)]
ensemble_osv_s=ensemble_osv_s.reset_index(drop=True)

# Calculate the weighted index
ensemble = (ensemble_sb_s.preds*weights_n[3])+ \
            (ensemble_ns_s.preds*weights_n[4])+ \
            (ensemble_osv_s.preds*weights_n[5])

# Make df                         
ensemble_ens2=pd.concat([ensemble_sb_s[["country","dd"]],pd.DataFrame(ensemble)],axis=1)
ensemble_ens2.columns=["country","dd","preds"]

# Concat the two ensembles 
ensemble_final=pd.concat([ensemble_ens,ensemble_ens2],axis=0)

# Sort and save
ensemble_final=ensemble_final.sort_values(by=["country","dd"])
ensemble_final.to_csv("out/ensemble_ens_df_cm_live.csv")
ensemble_final.duplicated(subset=["country","dd"]).any()
                
                
                
                
                
                
       
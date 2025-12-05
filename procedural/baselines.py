import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt 
from matplotlib import gridspec
from functions import lag_groupped
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import os
import json
from joblib import parallel_backend
from sklearn.model_selection import PredefinedSplit,RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from functools import reduce
os.environ['PATH'] = "/Library/TeX/texbin:" + os.environ.get('PATH', '')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}\usepackage[T1]{fontenc}'
plt.rcParams['xtick.labelsize'] = 20 
plt.rcParams['ytick.labelsize'] = 20  

# Country definitions: http://ksgleditsch.com/statelist.html
# List of microstates: 
micro_states={"Dominica":54,
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
              "Samoa":990}

exclude={"German Democratic Republic":265,
         "Czechoslovakia":315,
         "Yugoslavia":345,
         "Yemen, People's Republic of":680}

exclude2 ={"Taiwan":713, # Not included in wdi
           "Bahamas":31, # Not included in vdem
           "Belize":80, # Not included in vdem
           "Brunei Darussalam":835, # Not included in vdem
           "Kosovo":347, # Mostly missing in wdi
           "Democratic Peoples Republic of Korea":731} # Mostly missing in wdi

# Load data
df = pd.read_csv("data/data_out/ucdp_cm_sb.csv",index_col=0)
df = df[["year","dd","gw_codes","country","best"]][~df['gw_codes'].isin(list(micro_states.values())+list(exclude.values())+list(exclude2.values()))]
df.columns=["year","dd","gw_codes","country","sb_fatalities"]

# Subset years for test data
df=df.loc[(df["year"]==2022)|(df["year"]==2023)]
df=df.sort_values(by=["country","dd"])

#############
### ViEWS ###
#############

# Load views 2022
# Downloaded here: https://github.com/ThomasSchinca/ShapeFinder_conflict
views2022=pd.read_csv("data/views1.csv",index_col=0)

# Add column names which are countries and reshape from wide to long
# Downloaded here: https://github.com/ThomasSchinca/ShapeFinder_conflict
columns=pd.read_csv("data/obs1.csv",index_col=0)
views2022.columns=columns.columns
views2022 = pd.melt(views2022)

# Add year and dd
views2022["year"]=2022
views2022['month'] = views2022.groupby('variable').cumcount() + 1
views2022['month'] = views2022['month'].astype(str).str.zfill(2)
views2022['dd'] = '2022-' + views2022['month'].astype(str)
views2022=views2022[["dd","year","variable","value"]]
views2022.columns=["dd","year","country","preds"]

# Load views 2023
# Downloaded here: https://github.com/ThomasSchinca/ShapeFinder_conflict
views2023=pd.read_csv("data/views2.csv",index_col=0)

# Add column names which are countries and reshape from wide to long
# Downloaded here: https://github.com/ThomasSchinca/ShapeFinder_conflict
columns=pd.read_csv("data/obs1.csv",index_col=0)
views2023.columns=columns.columns
views2023 = pd.melt(views2023)

# Add year and dd
views2023["year"]=2023
views2023['month'] = views2023.groupby('variable').cumcount() + 1
views2023['month'] = views2023['month'].astype(str).str.zfill(2)
views2023['dd'] = '2023-' + views2023['month'].astype(str)
views2023=views2023[["dd","year","variable","value"]]
views2023.columns=["dd","year","country","preds"]

# Merge views years 2022 and 2023
views = pd.concat([views2022, views2023], axis=0, ignore_index=True)
views=views.sort_values(by=["country","dd"])

# Countries in Views, but not in my data: 
# Antigua & Barbuda
# Bahamas
# Belize
# Brunei
# Dominica
# Grendada
# Kiribati
# Kosovo
# Marshall Is.
# Micronesia
# Nauru
# North Korea
# Palau
# Samoa
# Sao Tome and Principe
# Seychelles
# St. Kitts and Nevis
# St. Lucia
# St. Vincent and the Grenadines
# Taiwan
# Tonga
# Tuvalu
# Vanuatu 

# Check country names for merging
# Rename Bosnia and Herzegovina ---> Bosnia-Herzegovina
# Rename Cape Verde ---> Cabo Verde
# Rename Cambodia ---> Cambodia (Kampuchea)
# Rename Czech Republic ---> Czechia
# Rename Congo, DRC ---> DR Congo (Zaire)
# Rename Timor Leste ---> East Timor
# Rename The Gambia ---> Gambia
# Rename Cote d'Ivoire ---> Ivory Coast
# Rename Myanmar ---> Myanmar (Burma)
# Rename Macedonia ---> North Macedonia
# Rename Russia ---> Russia (Soviet Union)
# Rename Solomon Is. ---> Solomon Islands
# Rename United States ---> United States of America
# Rename Yemen ---> Yemen (North Yemen)
# Rename Swaziland ---> eSwatini

views.loc[views["country"]=="Bosnia and Herzegovina","country"]="Bosnia-Herzegovina"
views.loc[views["country"]=="Cape Verde","country"]="Cabo Verde"
views.loc[views["country"]=="Cambodia","country"]="Cambodia (Kampuchea)"
views.loc[views["country"]=="Czech Republic","country"]="Czechia"
views.loc[views["country"]=="Congo, DRC","country"]="DR Congo (Zaire)"
views.loc[views["country"]=="Timor Leste","country"]="East Timor"
views.loc[views["country"]=="The Gambia","country"]="Gambia"
views.loc[views["country"]=="Cote d'Ivoire","country"]="Ivory Coast"
views.loc[views["country"]=="Myanmar","country"]="Myanmar (Burma)"
views.loc[views["country"]=="Macedonia","country"]="North Macedonia"
views.loc[views["country"]=="Russia","country"]="Russia (Soviet Union)"
views.loc[views["country"]=="Solomon Is.","country"]="Solomon Islands"
views.loc[views["country"]=="United States","country"]="United States of America"
views.loc[views["country"]=="Yemen","country"]="Yemen (North Yemen)"
views.loc[views["country"]=="Swaziland","country"]="eSwatini"

# Get input: Views and shape finder predict state-based and not civil conflict

# Load and reshape 2022
# Downloaded here: https://github.com/ThomasSchinca/ShapeFinder_conflict
obs2022=pd.read_csv("data/obs1.csv",index_col=0)
obs2022 = pd.melt(obs2022)

# Add year and dd variable
obs2022["year"]=2022
obs2022['month'] = obs2022.groupby('variable').cumcount() + 1
obs2022['month'] = obs2022['month'].astype(str).str.zfill(2)
obs2022['dd'] = '2022-' + obs2022['month'].astype(str)
obs2022=obs2022[["dd","year","variable","value"]]
obs2022.columns=["dd","year","country","actuals"]

# Load and reshape 2023
# Downloaded here: https://github.com/ThomasSchinca/ShapeFinder_conflict
obs2023=pd.read_csv("data/obs2.csv",index_col=0)
obs2023 = pd.melt(obs2023)

# Add year and dd variable
obs2023["year"]=2023
obs2023['month'] = obs2023.groupby('variable').cumcount() + 1
obs2023['month'] = obs2023['month'].astype(str).str.zfill(2)
obs2023['dd'] = '2023-' + obs2023['month'].astype(str)
obs2023=obs2023[["dd","year","variable","value"]]
obs2023.columns=["dd","year","country","actuals"]

# Merge years 2022 and 2023
obs = pd.concat([obs2022, obs2023], axis=0, ignore_index=True)
obs=obs.sort_values(by=["country","dd"])

# Check country names for merging
obs.loc[obs["country"]=="Bosnia and Herzegovina","country"]="Bosnia-Herzegovina"
obs.loc[obs["country"]=="Cape Verde","country"]="Cabo Verde"
obs.loc[obs["country"]=="Cambodia","country"]="Cambodia (Kampuchea)"
obs.loc[obs["country"]=="Czech Republic","country"]="Czechia"
obs.loc[obs["country"]=="Congo, DRC","country"]="DR Congo (Zaire)"
obs.loc[obs["country"]=="Timor Leste","country"]="East Timor"
obs.loc[obs["country"]=="The Gambia","country"]="Gambia"
obs.loc[obs["country"]=="Cote d'Ivoire","country"]="Ivory Coast"
obs.loc[obs["country"]=="Myanmar","country"]="Myanmar (Burma)"
obs.loc[obs["country"]=="Macedonia","country"]="North Macedonia"
obs.loc[obs["country"]=="Russia","country"]="Russia (Soviet Union)"
obs.loc[obs["country"]=="Solomon Is.","country"]="Solomon Islands"
obs.loc[obs["country"]=="United States","country"]="United States of America"
obs.loc[obs["country"]=="Yemen","country"]="Yemen (North Yemen)"
obs.loc[obs["country"]=="Swaziland","country"]="eSwatini"

# Merge observations and predictions 
views=pd.merge(views[["year","dd","country","preds"]],obs[["dd","country","actuals"]],how="left",on=["dd","country"])

# Use my data as base to remove countries in Views that are not in my sample
views_final=pd.merge(df[["year","dd","country","gw_codes"]],views[["dd","country","preds","actuals"]],how="left",on=["dd","country"])
print(views_final.isnull().any())

# Reset index and save views
views_final=views_final.reset_index(drop=True)
views_final.to_csv("out/views.csv") 

####################
### Shape finder ###
####################

# Load and reshape 2022
# Downloaded here: https://github.com/ThomasSchinca/ShapeFinder_conflict
sf2022=pd.read_csv("data/sf1.csv",index_col=0)
sf2022 = pd.melt(sf2022)

# Add year and dd variable
sf2022["year"]=2022
sf2022['month'] = sf2022.groupby('variable').cumcount() + 1
sf2022['month'] = sf2022['month'].astype(str).str.zfill(2)
sf2022['dd'] = '2022-' + sf2022['month'].astype(str)
sf2022=sf2022[["dd","year","variable","value"]]
sf2022.columns=["dd","year","country","preds"]

# Load and reshape 2023
# Downloaded here: https://github.com/ThomasSchinca/ShapeFinder_conflict
sf2023=pd.read_csv("data/sf2.csv",index_col=0)
sf2023 = pd.melt(sf2023)

# Add year and dd variable
sf2023["year"]=2023
sf2023['month'] = sf2023.groupby('variable').cumcount() + 1
sf2023['month'] = sf2023['month'].astype(str).str.zfill(2)
sf2023['dd'] = '2023-' + sf2023['month'].astype(str)
sf2023=sf2023[["dd","year","variable","value"]]
sf2023.columns=["dd","year","country","preds"]

# Merge years 2022 and 2023
sf = pd.concat([sf2022, sf2023], axis=0, ignore_index=True)

# Check country names for merging
sf.loc[sf["country"]=="Bosnia and Herzegovina","country"]="Bosnia-Herzegovina"
sf.loc[sf["country"]=="Cape Verde","country"]="Cabo Verde"
sf.loc[sf["country"]=="Cambodia","country"]="Cambodia (Kampuchea)"
sf.loc[sf["country"]=="Czech Republic","country"]="Czechia"
sf.loc[sf["country"]=="Congo, DRC","country"]="DR Congo (Zaire)"
sf.loc[sf["country"]=="Timor Leste","country"]="East Timor"
sf.loc[sf["country"]=="The Gambia","country"]="Gambia"
sf.loc[sf["country"]=="Cote d'Ivoire","country"]="Ivory Coast"
sf.loc[sf["country"]=="Myanmar","country"]="Myanmar (Burma)"
sf.loc[sf["country"]=="Macedonia","country"]="North Macedonia"
sf.loc[sf["country"]=="Russia","country"]="Russia (Soviet Union)"
sf.loc[sf["country"]=="Solomon Is.","country"]="Solomon Islands"
sf.loc[sf["country"]=="United States","country"]="United States of America"
sf.loc[sf["country"]=="Yemen","country"]="Yemen (North Yemen)"
sf.loc[sf["country"]=="Swaziland","country"]="eSwatini"

# Merge observations and predictions 
sf=pd.merge(sf[["year","dd","country","preds"]],obs[["dd","country","actuals"]],how="left",on=["dd","country"])

# Use my data as base to remove countries in sf that are not in my sample
sf_final=pd.merge(df[["year","dd","country","gw_codes"]],sf[["dd","country","preds","actuals"]],how="left",on=["dd","country"])
print(sf_final.isnull().any())

# Reset index and save sf
sf_final=sf_final.reset_index(drop=True)
sf_final.to_csv("out/sf.csv") 

########################
### Comparison Plots ###
########################

# Load
shape_finder=pd.read_csv('out/sf.csv',index_col=0) 
views_final=pd.read_csv("out/views.csv",index_col=0) 

# Loop over each country
for c in views_final.gw_codes.unique():
    
    # Subset data for each country
    df_s=views_final.loc[views_final["gw_codes"]==c]
    df_ss=shape_finder.loc[shape_finder["gw_codes"]==c]
    
    # Plot
    fig = plt.figure(figsize=(12, 5))
    
    # Specify grid with two columns and remove space between plots
    grid=gridspec.GridSpec(1, 2, figure=fig)  
    grid.update(wspace=0)    

    ### Year 2022 ###
    
    # Get first plot (on left)
    ax1 = fig.add_subplot(grid[0])
    
    # Plot predictions and actuals
    ax1.plot(df_s["dd"].loc[df_s["year"]==2022],df_s["actuals"].loc[df_s["year"]==2022],color="black")
    ax1.plot(df_s["dd"].loc[df_s["year"]==2022],df_s["preds"].loc[df_s["year"]==2022],color="black",linestyle="dotted")
    ax1.plot(df_ss["dd"].loc[df_ss["year"]==2022],df_ss["preds"].loc[df_ss["year"]==2022],color="black",linestyle="dashed")     
    # Add labels and make sure that y axis only contain integers (no floats) 
    ax1.set_xticks(["2022-01","2022-03","2022-05","2022-07","2022-09","2022-11"],["01-22","03-22","05-22","07-22","09-22","11-22"])
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

    ### Year 2023 ###
    
    # Get second plot (on right)
    ax2 = fig.add_subplot(grid[1])    
    
    # Plot predictions and actuals
    ax2.plot(df_s["dd"].loc[df_s["year"]==2023],df_s["actuals"].loc[df_s["year"]==2023],color="black")
    ax2.plot(df_s["dd"].loc[df_s["year"]==2023],df_s["preds"].loc[df_s["year"]==2023],color="black",linestyle="dotted")
    ax2.plot(df_ss["dd"].loc[df_ss["year"]==2023],df_ss["preds"].loc[df_ss["year"]==2023],color="black",linestyle="dashed")
    # Add labels
    ax2.set_xticks(["2023-01","2023-03","2023-05","2023-07","2023-09","2023-11"],["01-23","03-23","05-23","07-23","09-23","11-23"])
    # Move y axis to right and make sure that y axis only contain integers (no floats) 
    ax2.yaxis.set_ticks_position('right')
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        
    # Save examples
    if c==100 or c==369 or c==135 or c==625 or c==290 or c==522 or c==150 or c==433:
        plt.savefig(f"out/proc_examples_{views_final['country'].loc[views_final['gw_codes']==c].iloc[0]}.eps",dpi=300,bbox_inches='tight')    
    plt.show()   

#########################
### Negative-binomial ###
#########################

# The negative-binomial regression is replicated from Bagozzi (2015).

# Zero inflation stage: t-1, t-2, t-3 lagged dependent variables (log), GDP per capita (log)
# Count stage: t-1, t-2, t-3 lagged dependent variables (log), GDP per capita (log), GDP growth, population size (log)

### Live predictions using the Onset catcher set up ###

# (1) Prepare data

# Load 
df=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
df=df[["year","dd",'gw_codes', 'country',"sb_fatalities"]]
df_conf_hist=pd.read_csv("out/df_conf_hist_full_cm.csv",index_col=0)
df_demog=pd.read_csv("out/df_demog_full_cm.csv",index_col=0)
df_econ=pd.read_csv("out/df_econ_full_cm.csv",index_col=0)

# Transforms
df_conf_hist["sb_fatalities_lag1"]=lag_groupped(df,"country","sb_fatalities",1)
df_conf_hist['sb_fatalities_lag1']=np.log(df_conf_hist['sb_fatalities_lag1']+1)
df_conf_hist["sb_fatalities_lag2"]=lag_groupped(df,"country","sb_fatalities",2)
df_conf_hist['sb_fatalities_lag2']=np.log(df_conf_hist['sb_fatalities_lag2']+1)
df_conf_hist["sb_fatalities_lag3"]=lag_groupped(df,"country","sb_fatalities",3)
df_conf_hist['sb_fatalities_lag3']=np.log(df_conf_hist['sb_fatalities_lag3']+1)
df_demog['pop']=np.log(df_demog['pop'])
df_econ['gdp']=np.log(df_econ['gdp'])

# Merge
df=pd.merge(df, df_conf_hist[["dd","gw_codes","sb_fatalities_lag1","sb_fatalities_lag2","sb_fatalities_lag3"]],how="left",on=["dd","gw_codes"])
df=pd.merge(df, df_demog[["dd","gw_codes","pop"]],how="left",on=["dd","gw_codes"])
df=pd.merge(df, df_econ[["dd","gw_codes","gdp","gdp_growth"]],how="left",on=["dd","gw_codes"])

# Specify model 
target='sb_fatalities'
inputs=["sb_fatalities_lag1","sb_fatalities_lag2","sb_fatalities_lag3","pop","gdp_growth","gdp"]
y=df[["dd",'country','sb_fatalities']]
x=df[["dd",'country',"sb_fatalities_lag1","sb_fatalities_lag2","sb_fatalities_lag3","pop","gdp_growth","gdp"]]
 
# (2) Train data until 2020, use 2021 to make predictions for 2022.

# Data split
training_y = pd.DataFrame()
training_x = pd.DataFrame()
    
for c in y.country.unique():
    y_s = y.loc[y["country"] == c]
    x_s = x.loc[x["country"] == c]
    y_training = y_s[["country","dd"]+[target]].loc[y_s["dd"]<="2020-12"]
    x_training = x_s[["country","dd"]+inputs].loc[x_s["dd"]<="2020-12"]
    training_y = pd.concat([training_y, y_training])
    training_x = pd.concat([training_x, x_training])

# Train model 
training_y_d=training_y.drop(columns=["country","dd"])
training_x_d=training_x.drop(columns=["country","dd"])
training_x_d = sm.add_constant(training_x_d)  
zinb_model = sm.ZeroInflatedNegativeBinomialP(training_y_d, training_x_d[["const","sb_fatalities_lag1","sb_fatalities_lag2","sb_fatalities_lag3","pop","gdp_growth","gdp"]],exog_infl=training_x_d[["const","sb_fatalities_lag1","sb_fatalities_lag2","sb_fatalities_lag3","gdp"]])
zinb_model=zinb_model.fit(maxiter=2000)
print(zinb_model.converged) # Check that model converged because of warning (no standard errors are available)

# Use last 12 months as input when making predictions
testing_x_s=x[["country","dd"]+inputs].loc[(x["dd"]>="2021-01")&(x["dd"]<="2021-12")]
testing_x_d=testing_x_s.drop(columns=["country","dd"])

# Predictions
testing_x_d = sm.add_constant(testing_x_d)  
pred=pd.DataFrame(zinb_model.predict(testing_x_d[["const","sb_fatalities_lag1","sb_fatalities_lag2","sb_fatalities_lag3","pop","gdp_growth","gdp"]],exog_infl=testing_x_d[["const","sb_fatalities_lag1","sb_fatalities_lag2","sb_fatalities_lag3","gdp"]]))

# The t-3 lag causes the model to make huge overpredictions
print(training_y_d.mean())
print(pred.mean())

# Create df, replace year 2021 with year 2022
pred["country"]=testing_x_s.country.values
pred["dd"]=testing_x_s.dd.values
pred['dd'] = pred['dd'].str.replace('2021', '2022')

# Merge with outcome
base=y[["country","dd","sb_fatalities"]].loc[(y["dd"]>="2022-01")&(y["dd"]<="2022-12")]
zinb2022=pd.merge(base,pred,on=["country","dd"],how="left")
zinb2022.columns=["country","dd","sb_fatalities","preds"]

# (3) Train data until 2021, use 2022 to make predictions for 2023.

# Specify model 
target='sb_fatalities'
inputs=["sb_fatalities_lag1","sb_fatalities_lag2","sb_fatalities_lag3","pop","gdp_growth","gdp"]
y=df[["dd",'country','sb_fatalities']]
x=df[["dd",'country',"sb_fatalities_lag1","sb_fatalities_lag2","sb_fatalities_lag3","pop","gdp_growth","gdp"]]

# Data split
training_y = pd.DataFrame()
training_x = pd.DataFrame()
    
for c in y.country.unique():
    y_s = y.loc[y["country"] == c]
    x_s = x.loc[x["country"] == c]
    y_training = y_s[["country","dd"]+[target]].loc[y_s["dd"]<="2021-12"]
    x_training = x_s[["country","dd"]+inputs].loc[x_s["dd"]<="2021-12"]
    training_y = pd.concat([training_y, y_training])
    training_x = pd.concat([training_x, x_training])

# Train model 
training_y_d=training_y.drop(columns=["country","dd"])
training_x_d=training_x.drop(columns=["country","dd"])
training_x_d = sm.add_constant(training_x_d)  
zinb_model = sm.ZeroInflatedNegativeBinomialP(training_y_d, training_x_d[["const","sb_fatalities_lag1","sb_fatalities_lag2","sb_fatalities_lag3","pop","gdp_growth","gdp"]], exog_infl=training_x_d[["const","sb_fatalities_lag1","sb_fatalities_lag2","sb_fatalities_lag3","gdp"]])
zinb_model=zinb_model.fit(maxiter=2000)
print(zinb_model.converged) # Check that model converged because of warning (no standard errors are available)

# Use last 12 months as input when making predictions
testing_x_s=x[["country","dd"]+inputs].loc[(x["dd"]>="2022-01")&(x["dd"]<="2022-12")]
testing_x_d=testing_x_s.drop(columns=["country","dd"])

# Predictions
testing_x_d = sm.add_constant(testing_x_d)  
pred=pd.DataFrame(zinb_model.predict(testing_x_d[["const","sb_fatalities_lag1","sb_fatalities_lag2","sb_fatalities_lag3","pop","gdp_growth","gdp"]],exog_infl=testing_x_d[["const","sb_fatalities_lag1","sb_fatalities_lag2","sb_fatalities_lag3","gdp"]]))

# The t-3 lag causes the model to make underpredictions
print(training_y_d.mean())
print(pred.mean())

# Create df, replace year 2022 with year 2023
pred["country"]=testing_x_s.country.values
pred["dd"]=testing_x_s.dd.values
pred['dd'] = pred['dd'].str.replace('2022', '2023')

# Merge with outcome
base=y[["country","dd",target]].loc[(y["dd"]>="2023-01")&(y["dd"]<="2023-12")]
zinb2023=pd.merge(base,pred,on=["country","dd"],how="left")   
zinb2023.columns=["country","dd",target,"preds"]

# Merge years 2022 and 2023
zinb = pd.concat([zinb2022, zinb2023], axis=0, ignore_index=True)

# Sort and reset index (important for evaluation)
zinb=zinb.sort_values(by=["country","dd"])
zinb=zinb.reset_index(drop=True)

# Reobtain year and country codes
zinb['year'] = zinb['dd'].str[:4]
df=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
codes=df[["country","gw_codes"]].drop_duplicates()
zinb=pd.merge(zinb,codes,on="country",how="left")
zinb=zinb[["country","gw_codes","year","sb_fatalities","preds"]]

# Save
zinb.to_csv("out/zinb.csv") 

##################
### Structural ###
##################

# Another baseline is a model which uses the structural variables to predict the
# number of fatalities directly, without the two-staged approach. 

# Optimization
#grid = {'n_estimators': [10, 231, 452, 673, 894, 1115, 1336, 1557, 1778, 2000],
#        'max_features': ['sqrt', 'log2', None],
#        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
#        'min_samples_split': [2, 5, 10],
#        'min_samples_leaf': [1, 2, 4]}

# No optimization
grid=None
   
# Specify input variables     
demog_theme=['pop','pop_density','pop_density_id','urb_share','rural_share','pop_male_share','pop_male_share_0_14','pop_male_share_15_19','pop_male_share_20_24','pop_male_share_25_29','pop_male_share_30_34','group_counts','group_counts_id','monopoly_share','monopoly_share_id','discriminated_share','discriminated_share_id','powerless_share','powerless_share_id','dominant_share','dominant_share_id','ethnic_frac','ethnic_frac_id','rel_frac','rel_frac_id','lang_frac','lang_frac_id','race_frac','race_frac_id']
geog_theme=['land','land_id','forest','forest_id','temp','temp_id','co2','co2_id','percip','percip_id','waterstress','waterstress_id','agri_land','agri_land_id','arable_land','arable_land_id','rugged','soil','desert','tropical','cont_africa','cont_asia']
econ_theme=['natres_share','natres_share_id','oil_share','oil_share_id','gas_share','gas_share_id','coal_share','coal_share_id','forest_share','forest_share_id','minerals_share','minerals_share_id','gdp','gdp_id','gni','gni_id','gdp_growth','gdp_growth_id','unemploy','unemploy_id','unemploy_male','unemploy_male_id','inflat','inflat_id','conprice','conprice_id','undernour','undernour_id','foodprod','foodprod_id','water_rural','water_rural_id','water_urb','water_urb_id','agri_share','agri_share_id','trade_share','trade_share_id','fert','lifeexp_female','lifeexp_male','pop_growth','pop_growth_id','inf_mort','exports','exports_id','imports','imports_id','primary_female','primary_female_id','primary_male','primary_male_id','second_female','second_female_id','second_male','second_male_id','tert_female','tert_female_id','tert_male','tert_male_id','eys','eys_id','eys_male','eys_male_id','eys_female','eys_female_id','mys','mys_id','mys_male','mys_male_id','mys_female','mys_female_id']
pol_theme=['armedforces_share','armedforces_share_id','milex_share','milex_share_id','corruption','corruption_id', 'effectiveness', 'effectiveness_id', 'polvio','polvio_id','regu','regu_id','law','law_id','account','account_id','tax','tax_id','broadband','broadband_id','telephone','telephone_id','internet_use','internet_use_id','mobile','mobile_id','polyarchy','libdem','libdem_id','partipdem','delibdem','egaldem','civlib','phyvio','pollib','privlib','execon','execon_id','exgender','exgender_id','exgeo','exgeo_id','expol','expol_id','exsoc','exsoc_id','shutdown','shutdown_id','filter','filter_id','tenure_months','tenure_months_id','dem_duration','dem_duration_id','election_recent','election_recent_id','lastelection','lastelection_id']
hist_theme=['d_protest_lag1',"d_protest_zeros_decay",'d_riot_lag1',"d_riot_zeros_decay",'d_remote_lag1',"d_remote_zeros_decay",'d_sb_lag1',"d_sb_zeros_decay",'d_osv_lag1',"d_osv_zeros_decay",'d_ns_lag1',"d_ns_zeros_decay",'regime_duration']

# Merge data: Delete year and gw_codes variables to avoid duplicates                       
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)                  
hist=pd.read_csv("out/df_conf_hist_full_cm.csv",index_col=0)
demog=pd.read_csv("out/df_demog_full_cm.csv",index_col=0)
demog=demog.drop(["year","gw_codes"], axis=1)
geog=pd.read_csv("out/df_geog_full_cm.csv",index_col=0)
geog=geog.drop(["year","gw_codes"], axis=1)
econ=pd.read_csv("out/df_econ_full_cm.csv",index_col=0)
econ=econ.drop(["year","gw_codes"], axis=1)
pol=pd.read_csv("out/df_pol_full_cm.csv",index_col=0)
pol=pol.drop(["year","gw_codes"], axis=1)

# Merge all dfs 
x=reduce(lambda left,right: pd.merge(left,right, on=["dd","country"]), [hist,demog,geog,econ,pol])

# Transforms
x["pop"]=np.log(x["pop"]+1)
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)
x["natres_share"]=np.log(x["natres_share"]+1)
x["oil_share"]=np.log(x["oil_share"]+1)
x["gas_share"]=np.log(x["gas_share"]+1)
x["coal_share"]=np.log(x["coal_share"]+1)
x["forest_share"]=np.log(x["forest_share"]+1)
x["minerals_share"]=np.log(x["minerals_share"]+1)
x["gdp"]=np.log(x["gdp"]+1)
x["inflat"]=np.sign(x["inflat"]) * np.log(np.abs(x["inflat"]) + 1)
x["conprice"]=np.log(x["conprice"]+1)
x["milex_share"]=np.log(x["milex_share"]+1)
x["tax"]=np.log(x["tax"]+1)

### Live predictions using the Onset catcher set up ###

# (1) Train model until 2020, use 2021 to make predictions for 2022

# Specify model
target="sb_fatalities"
inputs=hist_theme+demog_theme+geog_theme+econ_theme+pol_theme

# Split data
training_y = pd.DataFrame()
training_x = pd.DataFrame()
splits=[]        
        
for c in y.country.unique():
    y_s = y.loc[y["country"] == c]
    x_s = x.loc[x["country"] == c]    
    y_training = y_s[["country","dd"]+[target]].loc[y_s["dd"]<="2020-12"]
    x_training = x_s[["country","dd"]+inputs].loc[x_s["dd"]<="2020-12"]
    training_y = pd.concat([training_y, y_training])
    training_x = pd.concat([training_x, x_training])        
    val_training_ids = list(y_training[:int(0.8*len(y_training))].index)
    val_testing_ids = list(y_training[int(0.8*len(y_training)):].index)        
    splits += [-1] * len(val_training_ids) + [0] * len(val_testing_ids)
    
# Train model
training_y_d=training_y.drop(columns=["country","dd"])
training_x_d=training_x.drop(columns=["country","dd"])

# If optimization
if grid is not None:
    splits = PredefinedSplit(test_fold=splits)
    with parallel_backend('threading'):
        grid_search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=1),param_distributions=grid,cv=splits,verbose=0,n_jobs=-1,n_iter=50,random_state=1)
        grid_search.fit(training_x_d,training_y_d.values.ravel())
    best_params = grid_search.best_params_
    model=RandomForestRegressor(random_state=1,**best_params)
    model.fit(training_x_d, training_y_d.values.ravel())
    
    # Use last 12 months as input when making predictions
    testing_x_s=x[["country","dd"]+inputs].loc[(x["dd"]>="2021-01")&(x["dd"]<="2021-12")]
    testing_x_d=testing_x_s.drop(columns=["country","dd"])
     
    # Predictions   
    pred = pd.DataFrame(model.predict(testing_x_d)) 
    
    # Create df, replace year 2021 with year 2022
    pred["country"]=testing_x_s.country.values
    pred["dd"]=testing_x_s.dd.values
    pred['dd'] = pred['dd'].str.replace('2021', '2022')
    
    # Merge with outcome
    base=y[["country","dd","sb_fatalities"]].loc[(y["dd"]>="2022-01")&(y["dd"]<="2022-12")]
    structural2022=pd.merge(base,pred,on=["country","dd"],how="left")
    structural2022.columns=["country","dd","sb_fatalities","preds"]

# If no optimization
else:
    model=RandomForestRegressor(random_state=1)
    model.fit(training_x_d, training_y_d.values.ravel())    
    
    # Use last 12 months as input when making predictions
    testing_x_s=x[["country","dd"]+inputs].loc[(x["dd"]>="2021-01")&(x["dd"]<="2021-12")]
    testing_x_d=testing_x_s.drop(columns=["country","dd"])
     
    # Predictions   
    pred = pd.DataFrame(model.predict(testing_x_d)) 
    
    # Create df, replace year 2021 with year 2022
    pred["country"]=testing_x_s.country.values
    pred["dd"]=testing_x_s.dd.values
    pred['dd'] = pred['dd'].str.replace('2021', '2022')
    
    # Merge with outcome
    base=y[["country","dd","sb_fatalities"]].loc[(y["dd"]>="2022-01")&(y["dd"]<="2022-12")]
    structural2022=pd.merge(base,pred,on=["country","dd"],how="left")
    structural2022.columns=["country","dd","sb_fatalities","preds"]
     
# (2) Train model until 2021, use 2022 to make predictions for 2023

# Specify model
target="sb_fatalities"
inputs=hist_theme+demog_theme+geog_theme+econ_theme+pol_theme

# Split data
training_y = pd.DataFrame()
training_x = pd.DataFrame()
splits=[]

for c in y.country.unique():
    y_s = y.loc[y["country"] == c]
    x_s = x.loc[x["country"] == c]
    y_training = y_s[["country","dd"]+[target]].loc[y_s["dd"]<="2021-12"]
    x_training = x_s[["country","dd"]+inputs].loc[x_s["dd"]<="2021-12"]
    training_y = pd.concat([training_y, y_training])
    training_x = pd.concat([training_x, x_training])
    val_training_ids = list(y_training[:int(0.8*len(y_training))].index)
    val_testing_ids = list(y_training[int(0.8*len(y_training)):].index)
    splits += [-1] * len(val_training_ids) + [0] * len(val_testing_ids)

# Train model    
training_y_d=training_y.drop(columns=["country","dd"])
training_x_d=training_x.drop(columns=["country","dd"])

# If optimization
if grid is not None:
    splits = PredefinedSplit(test_fold=splits)
    with parallel_backend('threading'):
        grid_search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=1),param_distributions=grid,cv=splits,verbose=0,n_jobs=-1,n_iter=50,random_state=1)
        grid_search.fit(training_x_d,training_y_d.values.ravel())
    best_params = grid_search.best_params_
    model=RandomForestRegressor(random_state=1,**best_params)
    model.fit(training_x_d, training_y_d.values.ravel())
        
    # Use last 12 months as input when making predictions
    testing_x_s=x[["country","dd"]+inputs].loc[(x["dd"]>="2022-01")&(x["dd"]<="2022-12")]
    testing_x_d=testing_x_s.drop(columns=["country","dd"])
    
    # Predictions     
    pred = pd.DataFrame(model.predict(testing_x_d)) 
       
    # Create df, replace year 2022 with year 2023 
    pred["country"]=testing_x_s.country.values
    pred["dd"]=testing_x_s.dd.values
    pred['dd'] = pred['dd'].str.replace('2022', '2023')
    
    # Merge with outcome
    base=y[["country","dd","sb_fatalities"]].loc[(y["dd"]>="2023-01")&(y["dd"]<="2023-12")]
    structural2023=pd.merge(base,pred,on=["country","dd"],how="left")
    structural2023.columns=["country","dd","sb_fatalities","preds"]
    
# If no optimization
else:
    model=RandomForestRegressor(random_state=1)
    model.fit(training_x_d, training_y_d.values.ravel())
    
    # Use last 12 months as input when making predictions
    testing_x_s=x[["country","dd"]+inputs].loc[(x["dd"]>="2022-01")&(x["dd"]<="2022-12")]
    testing_x_d=testing_x_s.drop(columns=["country","dd"])
    
    # Predictions     
    pred = pd.DataFrame(model.predict(testing_x_d)) 
       
    # Create df, replace year 2022 with year 2023 
    pred["country"]=testing_x_s.country.values
    pred["dd"]=testing_x_s.dd.values
    pred['dd'] = pred['dd'].str.replace('2022', '2023')
    
    # Merge with outcome
    base=y[["country","dd","sb_fatalities"]].loc[(y["dd"]>="2023-01")&(y["dd"]<="2023-12")]
    structural2023=pd.merge(base,pred,on=["country","dd"],how="left")
    structural2023.columns=["country","dd","sb_fatalities","preds"]

# Merge years 2022 and 2023
structural = pd.concat([structural2022, structural2022], axis=0, ignore_index=True)

# Sort and reset index (important for evaluation)
structural=structural.sort_values(by=["country","dd"])
structural=structural.reset_index(drop=True)

# Save
structural.to_csv("out/structural.csv") 

###########################
### Variable importance ###
###########################

# Another robustness check is to validate which variables are important for stage 1. 
# If the hurdle model is beneficial, it should not only be conflict history variables. 
# The models are trained using data until 2019. 

# Specify input variables     
demog_theme=['pop','pop_density','pop_density_id','urb_share','rural_share','pop_male_share','pop_male_share_0_14','pop_male_share_15_19','pop_male_share_20_24','pop_male_share_25_29','pop_male_share_30_34','group_counts','group_counts_id','monopoly_share','monopoly_share_id','discriminated_share','discriminated_share_id','powerless_share','powerless_share_id','dominant_share','dominant_share_id','ethnic_frac','ethnic_frac_id','rel_frac','rel_frac_id','lang_frac','lang_frac_id','race_frac','race_frac_id']
geog_theme=['land','land_id','forest','forest_id','temp','temp_id','co2','co2_id','percip','percip_id','waterstress','waterstress_id','agri_land','agri_land_id','arable_land','arable_land_id','rugged','soil','desert','tropical','cont_africa','cont_asia']
econ_theme=['natres_share','natres_share_id','oil_share','oil_share_id','gas_share','gas_share_id','coal_share','coal_share_id','forest_share','forest_share_id','minerals_share','minerals_share_id','gdp','gdp_id','gni','gni_id','gdp_growth','gdp_growth_id','unemploy','unemploy_id','unemploy_male','unemploy_male_id','inflat','inflat_id','conprice','conprice_id','undernour','undernour_id','foodprod','foodprod_id','water_rural','water_rural_id','water_urb','water_urb_id','agri_share','agri_share_id','trade_share','trade_share_id','fert','lifeexp_female','lifeexp_male','pop_growth','pop_growth_id','inf_mort','exports','exports_id','imports','imports_id','primary_female','primary_female_id','primary_male','primary_male_id','second_female','second_female_id','second_male','second_male_id','tert_female','tert_female_id','tert_male','tert_male_id','eys','eys_id','eys_male','eys_male_id','eys_female','eys_female_id','mys','mys_id','mys_male','mys_male_id','mys_female','mys_female_id']
pol_theme=['armedforces_share','armedforces_share_id','milex_share','milex_share_id','corruption','corruption_id', 'effectiveness', 'effectiveness_id', 'polvio','polvio_id','regu','regu_id','law','law_id','account','account_id','tax','tax_id','broadband','broadband_id','telephone','telephone_id','internet_use','internet_use_id','mobile','mobile_id','polyarchy','libdem','libdem_id','partipdem','delibdem','egaldem','civlib','phyvio','pollib','privlib','execon','execon_id','exgender','exgender_id','exgeo','exgeo_id','expol','expol_id','exsoc','exsoc_id','shutdown','shutdown_id','filter','filter_id','tenure_months','tenure_months_id','dem_duration','dem_duration_id','election_recent','election_recent_id','lastelection','lastelection_id']

# Optimization
#grid = {'n_estimators': [10, 231, 452, 673, 894, 1115, 1336, 1557, 1778, 2000],
#        'max_features': ['sqrt', 'log2', None],
#        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
#        'min_samples_split': [2, 5, 10],
#        'min_samples_leaf': [1, 2, 4]}

# No optimization
grid=None

# Import variables names
with open('data/names.json', 'r') as f:
    names = json.load(f)

### Protest ###

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
    
# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)

# Merge data: Delete year and gw_codes variables to avoid duplicates                       
hist=pd.read_csv("out/df_conf_hist_full_cm.csv",index_col=0)
demog=pd.read_csv("out/df_demog_full_cm.csv",index_col=0)
demog=demog.drop(["year","gw_codes"], axis=1)
geog=pd.read_csv("out/df_geog_full_cm.csv",index_col=0)
geog=geog.drop(["year","gw_codes"], axis=1)
econ=pd.read_csv("out/df_econ_full_cm.csv",index_col=0)
econ=econ.drop(["year","gw_codes"], axis=1)
pol=pd.read_csv("out/df_pol_full_cm.csv",index_col=0)
pol=pol.drop(["year","gw_codes"], axis=1)

# Merge all dfs
x=reduce(lambda left, right: pd.merge(left, right, on=["dd","country"]), [hist,demog,geog,econ,pol])
 
# Transforms
x["pop"]=np.log(x["pop"]+1)
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)
x["natres_share"]=np.log(x["natres_share"]+1)
x["oil_share"]=np.log(x["oil_share"]+1)
x["gas_share"]=np.log(x["gas_share"]+1)
x["coal_share"]=np.log(x["coal_share"]+1)
x["forest_share"]=np.log(x["forest_share"]+1)
x["minerals_share"]=np.log(x["minerals_share"]+1)
x["gdp"]=np.log(x["gdp"]+1)
x["inflat"]=np.sign(x["inflat"]) * np.log(np.abs(x["inflat"]) + 1)
x["conprice"]=np.log(x["conprice"]+1)
x["milex_share"]=np.log(x["milex_share"]+1)
x["tax"]=np.log(x["tax"]+1)

# Only include ACLED observations                    
y=pd.merge(left=base,right=y,on=["dd","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["dd","gw_codes"],how="left")

# Specify model
target="d_protest"
inputs=['d_protest_lag1',"d_protest_zeros_decay",'regime_duration']+demog_theme+geog_theme+econ_theme+pol_theme

# Split data
training_y = pd.DataFrame()
training_x = pd.DataFrame()
splits=[]        
for c in y.country.unique():
    y_s = y.loc[y["country"] == c]
    x_s = x.loc[x["country"] == c]    
    y_training = y_s[["country","dd"]+[target]].loc[y_s["dd"]<="2019-12"]
    x_training = x_s[["country","dd"]+inputs].loc[x_s["dd"]<="2019-12"]
    training_y = pd.concat([training_y, y_training])
    training_x = pd.concat([training_x, x_training])       
    val_training_ids = list(y_training[:int(0.8*len(y_training))].index)
    val_testing_ids = list(y_training[int(0.8*len(y_training)):].index)        
    splits += [-1] * len(val_training_ids) + [0] * len(val_testing_ids)
  
# Train model
training_y_d=training_y.drop(columns=["country","dd"])
training_x_d=training_x.drop(columns=["country","dd"])

# If optimization
if grid is not None:
    splits = PredefinedSplit(test_fold=splits)
    with parallel_backend('threading'):
        grid_search = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=1), param_distributions=grid, cv=splits, verbose=0, n_jobs=-1,n_iter=50,random_state=1)
        grid_search.fit(training_x_d, training_y_d.values.ravel())   
    best_params = grid_search.best_params_
    model=RandomForestClassifier(random_state=1,**best_params)
    model.fit(training_x_d, training_y_d.values.ravel())
# If no optimization
else: 
    model=RandomForestClassifier(random_state=1)
    model.fit(training_x_d, training_y_d.values.ravel())    

# Importance score
imp = model.feature_importances_
imp = pd.Series(imp,index=inputs)
imp = imp.reset_index()

# Remove variables with _id, these only indicate whether an observations was 
# imputed and are not of substantial interest
imp = imp[~imp['index'].str.contains('_id')]

# Remove variables which have an importance score of zero
imp=imp.loc[imp[0]!=0]

# Add fancy names
for c in imp["index"].values:
    imp.loc[imp["index"]==c,"index"]=names[c]
imp.columns=["Variable","Feature Importance"]

# Sort variables by importance and save
top=imp.sort_values(by=["Feature Importance"],ascending=False)
top.to_latex("out/protest_imp.tex",index=False)

# Only keep 10 most important variables and sort again
top_10 = top[:10]
top_10 = top_10.sort_values(by=["Feature Importance"],ascending=True)

# Plot 10 most important variables in horizontal bar chart 
ax=top_10.plot(kind='barh',legend=False,color="black",figsize=(5,3))

# Add variable names
ax.set_yticks(list(range(0,len(top_10))),list(top_10.Variable))

# Add numeric values for importance score 

# Loop through every variable and add imp score
for lab,i in zip(list(top_10['Feature Importance']),[0,1,2,3,4,5,6,7,8,9]):
    ax.text(lab+0.007,i,round(lab,5),ha="center",va="center")

# Save
plt.savefig("out/proc_imp_protest.eps",dpi=300,bbox_inches='tight')        
 
### Riots ###

# Specify model
target="d_riot"
inputs=['d_riot_lag1',"d_riot_zeros_decay",'regime_duration']+demog_theme+geog_theme+econ_theme+pol_theme

# Split data
training_y = pd.DataFrame()
training_x = pd.DataFrame()
splits=[]               
for c in y.country.unique():
    y_s = y.loc[y["country"] == c]
    x_s = x.loc[x["country"] == c]    
    y_training = y_s[["country","dd"]+[target]].loc[y_s["dd"]<="2019-12"]
    x_training = x_s[["country","dd"]+inputs].loc[x_s["dd"]<="2019-12"]
    training_y = pd.concat([training_y, y_training])
    training_x = pd.concat([training_x, x_training])        
    val_training_ids = list(y_training[:int(0.8*len(y_training))].index)
    val_testing_ids = list(y_training[int(0.8*len(y_training)):].index)        
    splits += [-1] * len(val_training_ids) + [0] * len(val_testing_ids)
 
# Train model
training_y_d=training_y.drop(columns=["country","dd"])
training_x_d=training_x.drop(columns=["country","dd"])

# If optimization
if grid is not None:
    splits = PredefinedSplit(test_fold=splits)
    with parallel_backend('threading'):
        grid_search = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=1), param_distributions=grid, cv=splits, verbose=0, n_jobs=-1,n_iter=50,random_state=1)
        grid_search.fit(training_x_d, training_y_d.values.ravel())   
    best_params = grid_search.best_params_
    model=RandomForestClassifier(random_state=1,**best_params)
    model.fit(training_x_d, training_y_d.values.ravel())
# If no optimization
else: 
    model=RandomForestClassifier(random_state=1)
    model.fit(training_x_d, training_y_d.values.ravel())   
    
# Importance score
imp = model.feature_importances_
imp = pd.Series(imp, index=inputs)
imp = imp.reset_index()

# Remove variables with _id, these only indicate whether an observations was 
# imputed and are not of substantial interest
imp = imp[~imp['index'].str.contains('_id')]

# Remove variables which have an importance score of zero
imp=imp.loc[imp[0]!=0]

# Add fancy names
for c in imp["index"].values:
    imp.loc[imp["index"]==c,"index"]=names[c]
imp.columns=["Variable","Feature Importance"]

# Sort variables by importance and save
top = imp.sort_values(by=["Feature Importance"],ascending=False)
top.to_latex("out/riots_imp.tex",index=False)

# Only keep 10 most important variables and sort again
top_10 = top[:10]
top_10 = top_10.sort_values(by=["Feature Importance"],ascending=True)

# Plot 10 most important variables in horizontal bar chart 
ax=top_10.plot(kind='barh',legend=False,color="black",figsize=(5,3))

# Add variable names
ax.set_yticks(list(range(0,len(top_10))),list(top_10.Variable))

# Add numeric values for importance score 

# Loop through every variable and add imp score
for lab,i in zip(list(top_10['Feature Importance']),[0,1,2,3,4,5,6,7,8,9]):
    ax.text(lab+0.008,i,round(lab,5),ha="center",va="center")
    
# Save
plt.savefig("out/proc_imp_riots.eps",dpi=300,bbox_inches='tight')  

### Remote violence ###

# Specify model 
target="d_remote"
inputs=['d_remote_lag1',"d_remote_zeros_decay",'regime_duration']+demog_theme+geog_theme+econ_theme+pol_theme

# Split data
training_y = pd.DataFrame()
training_x = pd.DataFrame()
splits=[]  
for c in y.country.unique():
    y_s = y.loc[y["country"] == c]
    x_s = x.loc[x["country"] == c]    
    y_training = y_s[["country","dd"]+[target]].loc[y_s["dd"]<="2019-12"]
    x_training = x_s[["country","dd"]+inputs].loc[x_s["dd"]<="2019-12"]
    training_y = pd.concat([training_y, y_training])
    training_x = pd.concat([training_x, x_training])      
    val_training_ids = list(y_training[:int(0.8*len(y_training))].index)
    val_testing_ids = list(y_training[int(0.8*len(y_training)):].index)       
    splits += [-1] * len(val_training_ids) + [0] * len(val_testing_ids)
 
# Train model   
training_y_d=training_y.drop(columns=["country","dd"])
training_x_d=training_x.drop(columns=["country","dd"]) 

# If optimization
if grid is not None:
    splits = PredefinedSplit(test_fold=splits)
    with parallel_backend('threading'):
        grid_search = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=1), param_distributions=grid, cv=splits, verbose=0, n_jobs=-1,n_iter=50,random_state=1)
        grid_search.fit(training_x_d, training_y_d.values.ravel())   
    best_params = grid_search.best_params_
    model=RandomForestClassifier(random_state=1,**best_params)
    model.fit(training_x_d, training_y_d.values.ravel())
# If no optimization
else: 
    model=RandomForestClassifier(random_state=1)
    model.fit(training_x_d, training_y_d.values.ravel())   

# Importance score
imp = model.feature_importances_
imp = pd.Series(imp, index=inputs)
imp = imp.reset_index()

# Remove variables with _id, these only indicate whether an observations was 
# imputed and are not of substantial interest
imp = imp[~imp['index'].str.contains('_id')]

# Remove variables which have an importance score of zero
imp=imp.loc[imp[0]!=0]

# Add fancy names
for c in imp["index"].values:
    imp.loc[imp["index"]==c,"index"]=names[c]
imp.columns=["Variable","Feature Importance"]

# Sort variables by importance and save
top = imp.sort_values(by=["Feature Importance"],ascending=False)
top.to_latex("out/terror_imp.tex",index=False)

# Only keep 10 most important variables and sort again
top_10 = top[:10]
top_10 = top_10.sort_values(by=["Feature Importance"],ascending=True)

# Plot 10 most important variables in horizontal bar chart 
ax=top_10.plot(kind='barh',legend=False,color="black",figsize=(5, 3))

# Add variable names
ax.set_yticks(list(range(0,len(top_10))),list(top_10.Variable))

# Add numeric values for importance score 

# Loop through every variable and add imp score
for lab,i in zip(list(top_10['Feature Importance']),[0,1,2,3,4,5,6,7,8,9]):
    ax.text(lab+0.009,i,round(lab,5),ha="center",va="center")

# Save
plt.savefig("out/proc_imp_terror.eps",dpi=300,bbox_inches='tight')  

### Civil conflict ###

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)

# Merge data: Delete year and gw_codes variables to avoid duplicates                       
hist=pd.read_csv("out/df_conf_hist_full_cm.csv",index_col=0)
demog=pd.read_csv("out/df_demog_full_cm.csv",index_col=0)
demog=demog.drop(["year","gw_codes"], axis=1)
geog=pd.read_csv("out/df_geog_full_cm.csv",index_col=0)
geog=geog.drop(["year","gw_codes"], axis=1)
econ=pd.read_csv("out/df_econ_full_cm.csv",index_col=0)
econ=econ.drop(["year","gw_codes"], axis=1)
pol=pd.read_csv("out/df_pol_full_cm.csv",index_col=0)
pol=pol.drop(["year","gw_codes"], axis=1)

# Merge all dfs
x=reduce(lambda left, right: pd.merge(left, right, on=["dd","country"]), [hist,demog,geog,econ,pol])

# Transforms
x["pop"]=np.log(x["pop"]+1)
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)
x["natres_share"]=np.log(x["natres_share"]+1)
x["oil_share"]=np.log(x["oil_share"]+1)
x["gas_share"]=np.log(x["gas_share"]+1)
x["coal_share"]=np.log(x["coal_share"]+1)
x["forest_share"]=np.log(x["forest_share"]+1)
x["minerals_share"]=np.log(x["minerals_share"]+1)
x["gdp"]=np.log(x["gdp"]+1)
x["inflat"]=np.sign(x["inflat"]) * np.log(np.abs(x["inflat"]) + 1)
x["conprice"]=np.log(x["conprice"]+1)
x["milex_share"]=np.log(x["milex_share"]+1)
x["tax"]=np.log(x["tax"]+1)

# Specify model 
target="d_sb"
inputs=['d_sb_lag1',"d_sb_zeros_decay",'regime_duration']+demog_theme+geog_theme+econ_theme+pol_theme

# Split data
training_y = pd.DataFrame()
training_x = pd.DataFrame()
splits=[]
for c in y.country.unique():
    y_s = y.loc[y["country"] == c]
    x_s = x.loc[x["country"] == c]    
    y_training = y_s[["country","dd"]+[target]].loc[y_s["dd"]<="2019-12"]
    x_training = x_s[["country","dd"]+inputs].loc[x_s["dd"]<="2019-12"]
    training_y = pd.concat([training_y, y_training])
    training_x = pd.concat([training_x, x_training])        
    val_training_ids = list(y_training[:int(0.8*len(y_training))].index)
    val_testing_ids = list(y_training[int(0.8*len(y_training)):].index)       
    splits += [-1] * len(val_training_ids) + [0] * len(val_testing_ids)

# Train model    
training_y_d=training_y.drop(columns=["country","dd"])
training_x_d=training_x.drop(columns=["country","dd"])

# If optimization
if grid is not None:
    splits = PredefinedSplit(test_fold=splits)
    with parallel_backend('threading'):
        grid_search = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=1), param_distributions=grid, cv=splits, verbose=0, n_jobs=-1,n_iter=50,random_state=1)
        grid_search.fit(training_x_d, training_y_d.values.ravel())   
    best_params = grid_search.best_params_
    model=RandomForestClassifier(random_state=1,**best_params)
    model.fit(training_x_d, training_y_d.values.ravel())
# If no optimization
else: 
    model=RandomForestClassifier(random_state=1)
    model.fit(training_x_d, training_y_d.values.ravel())   

# Importance score
imp = model.feature_importances_
imp = pd.Series(imp, index=inputs)
imp = imp.reset_index()

# Remove variables with _id, these only indicate whether an observations was 
# imputed and are not of substantial interest
imp = imp[~imp['index'].str.contains('_id')]

# Remove variables which have an importance score of zero
imp=imp.loc[imp[0]!=0]

# Add fancy names
for c in imp["index"].values:
    imp.loc[imp["index"]==c,"index"]=names[c]
imp.columns=["Variable","Feature Importance"]

# Only keep 10 most important variables
top = imp.sort_values(by=["Feature Importance"],ascending=False)
top.to_latex("out/sb_imp.tex",index=False)

# Only keep 10 most important variables and sort again
top_10 = top[:10]
top_10 = top_10.sort_values(by=["Feature Importance"],ascending=True)

# Plot 10 most important variables in horizontal bar chart 
ax=top_10.plot(kind='barh',legend=False,color="black",figsize=(5,3))

# Add variable names
ax.set_yticks(list(range(0,len(top_10))),list(top_10.Variable))

# Add numeric values for importance score 

# Loop through every variable and add imp score
for lab,i in zip(list(top_10['Feature Importance']),[0,1,2,3,4,5,6,7,8,9]):
    ax.text(lab+0.012,i,round(lab,5),ha="center",va="center")

# Save
plt.savefig("out/proc_imp_sb.eps",dpi=300,bbox_inches='tight')  

### One-sided violence ###

# Specify model
target="d_osv"
inputs=['d_osv_lag1',"d_osv_zeros_decay",'regime_duration']+demog_theme+geog_theme+econ_theme+pol_theme

# Split data
training_y = pd.DataFrame()
training_x = pd.DataFrame()
splits=[]        
for c in y.country.unique():
    y_s = y.loc[y["country"] == c]
    x_s = x.loc[x["country"] == c]    
    y_training = y_s[["country","dd"]+[target]].loc[y_s["dd"]<="2019-12"]
    x_training = x_s[["country","dd"]+inputs].loc[x_s["dd"]<="2019-12"]
    training_y = pd.concat([training_y, y_training])
    training_x = pd.concat([training_x, x_training])        
    val_training_ids = list(y_training[:int(0.8*len(y_training))].index)
    val_testing_ids = list(y_training[int(0.8*len(y_training)):].index)        
    splits += [-1] * len(val_training_ids) + [0] * len(val_testing_ids)

# Train model    
training_y_d=training_y.drop(columns=["country","dd"])
training_x_d=training_x.drop(columns=["country","dd"])

# If optimization
if grid is not None:
    splits = PredefinedSplit(test_fold=splits)
    with parallel_backend('threading'):
        grid_search = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=1), param_distributions=grid, cv=splits, verbose=0, n_jobs=-1,n_iter=50,random_state=1)
        grid_search.fit(training_x_d, training_y_d.values.ravel())   
    best_params = grid_search.best_params_
    model=RandomForestClassifier(random_state=1,**best_params)
    model.fit(training_x_d, training_y_d.values.ravel())
# If no optimization
else: 
    model=RandomForestClassifier(random_state=1)
    model.fit(training_x_d, training_y_d.values.ravel())   

# Importance score
imp = model.feature_importances_
imp = pd.Series(imp, index=inputs)
imp = imp.reset_index()

# Remove variables with _id, these only indicate whether an observations was 
# imputed and are not of substantial interest
imp = imp[~imp['index'].str.contains('_id')]

# Remove variables which have an importance score of zero
imp=imp.loc[imp[0]!=0]

# Add fancy names
for c in imp["index"].values:
    imp.loc[imp["index"]==c,"index"]=names[c]
imp.columns=["Variable","Feature Importance"]

# Sort variables by importance and save
top = imp.sort_values(by=["Feature Importance"],ascending=False)
top.to_latex("out/osv_imp.tex",index=False)

# Only keep 10 most important variables
top10 = top[:10]
top10 = top10.sort_values(by=["Feature Importance"],ascending=True)

# Plot 10 most important variables in horizontal bar chart 
ax=top10.plot(kind='barh',legend=False,color="black",figsize=(5,3))

# Add variable names
ax.set_yticks(list(range(0,len(top10))),list(top10.Variable))

# Add numeric values for importance score 

# Loop through every variable and add imp score
for lab,i in zip(list(top10['Feature Importance']),[0,1,2,3,4,5,6,7,8,9]):
    ax.text(lab+0.01,i,round(lab,5),ha="center",va="center")

# Save
plt.savefig("out/proc_imp_osv.eps",dpi=300,bbox_inches='tight')  

### Non-state ###

# Specify model
target="d_ns"
inputs=['d_ns_lag1',"d_ns_zeros_decay",'regime_duration']+demog_theme+geog_theme+econ_theme+pol_theme

# Split data
training_y = pd.DataFrame()
training_x = pd.DataFrame()
splits=[]
for c in y.country.unique():
    y_s = y.loc[y["country"] == c]
    x_s = x.loc[x["country"] == c]    
    y_training = y_s[["country","dd"]+[target]].loc[y_s["dd"]<="2019-12"]
    x_training = x_s[["country","dd"]+inputs].loc[x_s["dd"]<="2019-12"]
    training_y = pd.concat([training_y, y_training])
    training_x = pd.concat([training_x, x_training])        
    val_training_ids = list(y_training[:int(0.8*len(y_training))].index)
    val_testing_ids = list(y_training[int(0.8*len(y_training)):].index)        
    splits += [-1] * len(val_training_ids) + [0] * len(val_testing_ids)

# Train model    
training_y_d=training_y.drop(columns=["country","dd"])
training_x_d=training_x.drop(columns=["country","dd"])   

# If optimization
if grid is not None:
    splits = PredefinedSplit(test_fold=splits)
    with parallel_backend('threading'):
        grid_search = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=1), param_distributions=grid, cv=splits, verbose=0, n_jobs=-1,n_iter=50,random_state=1)
        grid_search.fit(training_x_d, training_y_d.values.ravel())   
    best_params = grid_search.best_params_
    model=RandomForestClassifier(random_state=1,**best_params)
    model.fit(training_x_d, training_y_d.values.ravel())
# If no optimization
else: 
    model=RandomForestClassifier(random_state=1)
    model.fit(training_x_d, training_y_d.values.ravel())   

# Importance score
imp = model.feature_importances_
imp = pd.Series(imp, index=inputs)
imp = imp.reset_index()

# Remove variables with _id, these only indicate whether an observations was 
# imputed and are not of substantial interest
imp = imp[~imp['index'].str.contains('_id')]

# Remove variables which have an importance score of zero
imp=imp.loc[imp[0]!=0]

# Add fancy names
for c in imp["index"].values:
    imp.loc[imp["index"]==c,"index"]=names[c]
imp.columns=["Variable","Feature Importance"]

# Sort variables by importance and save
top = imp.sort_values(by=["Feature Importance"],ascending=False)
top.to_latex("out/ns_imp.tex",index=False)

# Only keep 10 most important variables
top_10 = top[:10]
top_10 = top_10.sort_values(by=["Feature Importance"],ascending=True)

# Plot 10 most important variables in horizontal bar chart 
ax=top_10.plot(kind='barh',legend=False,color="black",figsize=(5,3))

# Add variable names
ax.set_yticks(list(range(0,len(top_10))),list(top_10.Variable))

# Add numeric values for importance score 

# Loop through every variable and add imp score
for lab,i in zip(list(top_10['Feature Importance']),[0,1,2,3,4,5,6,7,8,9]):
    ax.text(lab+0.01,i,round(lab,5),ha="center",va="center")

# Save
plt.savefig("out/proc_imp_ns.eps",dpi=300,bbox_inches='tight')  




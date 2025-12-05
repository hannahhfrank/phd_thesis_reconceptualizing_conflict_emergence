import pandas as pd
from functions import dichotomize,lag_groupped,consec_zeros_grouped,exponential_decay,simple_imp_grouped,linear_imp_grouped,multivariate_imp_bayes
import matplotlib.pyplot as plt

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
         "Abkhazia":396,
         "South Ossetia":397,
         "Yemen, People's Republic of":680}
  
exclude2 ={"Taiwan":713, # Not included in wdi
           "Bahamas":31, # Not included in vdem
           "Belize":80, # Not included in vdem
           "Brunei Darussalam":835, # Not included in vdem
           "Kosovo":347, # Mostly missing in wdi
           "Democratic Peoples Republic of Korea":731} # Mostly missing in wdi

############
### UCDP ###
############

ucdp_sb=pd.read_csv("data/data_out/ucdp_cm_sb.csv",index_col=0)
ucdp_sb_s = ucdp_sb[["year","dd","gw_codes","country","best","count"]][~ucdp_sb['gw_codes'].isin(list(micro_states.values())+list(exclude.values())+list(exclude2.values()))]
ucdp_sb_s.columns=["year","dd","gw_codes","country","sb_fatalities","sb_event_counts"]
print(ucdp_sb_s.dtypes)

ucdp_osv=pd.read_csv("data/data_out/ucdp_cm_osv.csv",index_col=0)
ucdp_osv_s = ucdp_osv[["dd","gw_codes","best","count"]][~ucdp_osv['gw_codes'].isin(list(micro_states.values())+list(exclude.values())+list(exclude2.values()))]
ucdp_osv_s.columns=["dd","gw_codes","osv_fatalities","osv_event_counts"]
print(ucdp_osv_s.dtypes)

ucdp_ns=pd.read_csv("data/data_out/ucdp_cm_ns.csv",index_col=0)
ucdp_ns_s = ucdp_ns[["dd","gw_codes","best","count"]][~ucdp_ns['gw_codes'].isin(list(micro_states.values())+list(exclude.values())+list(exclude2.values()))]
ucdp_ns_s.columns=["dd","gw_codes","ns_fatalities","ns_event_counts"]
print(ucdp_ns_s.dtypes)

#############
### ACLED ###
#############

acled_protest=pd.read_csv("data/data_out/acled_cm_protest.csv",index_col=0)
acled_protest_s = acled_protest[["dd","gw_codes","n_protest_events","fatalities"]][~acled_protest['gw_codes'].isin(list(micro_states.values())+list(exclude.values())+list(exclude2.values()))]
acled_protest_s.columns=["dd","gw_codes","protest_event_counts","protest_fatalities"]
print(acled_protest_s.dtypes)

acled_riots=pd.read_csv("data/data_out/acled_cm_riot.csv",index_col=0)
acled_riots_s = acled_riots[["dd","gw_codes","n_riot_events","fatalities"]][~acled_riots['gw_codes'].isin(list(micro_states.values())+list(exclude.values())+list(exclude2.values()))]
acled_riots_s.columns=["dd","gw_codes","riot_event_counts","riot_fatalities"]
print(acled_riots_s.dtypes)

acled_remote=pd.read_csv("data/data_out/acled_cm_remote.csv",index_col=0)
acled_remote_s = acled_remote[["dd","gw_codes","n_remote_events","fatalities"]][~acled_remote['gw_codes'].isin(list(micro_states.values())+list(exclude.values())+list(exclude2.values()))]
acled_remote_s.columns=["dd","gw_codes","remote_event_counts","remote_fatalities"]
print(acled_remote_s.dtypes)

# Merge # 
df=pd.merge(left=ucdp_sb_s,right=ucdp_ns_s,on=["dd","gw_codes"],how="left")
df=pd.merge(left=df,right=ucdp_osv_s,on=["dd","gw_codes"],how="left")
df=pd.merge(left=df,right=acled_protest_s,on=["dd","gw_codes"],how="left")
df=pd.merge(left=df,right=acled_riots_s,on=["dd","gw_codes"],how="left")
df=pd.merge(left=df,right=acled_remote_s,on=["dd","gw_codes"],how="left")
df=df.fillna(0)

###############
### Outcome ###
###############

# Dichotomize
dichotomize(df,"protest_event_counts","d_protest",0)
dichotomize(df,"riot_event_counts","d_riot",0)
dichotomize(df,"remote_event_counts","d_remote",0)
dichotomize(df,"sb_fatalities","d_sb",0)
dichotomize(df,"osv_fatalities","d_osv",0)
dichotomize(df,"ns_fatalities","d_ns",0)

# Final df
df_out=df[["year","dd","gw_codes","country","d_protest","d_riot","d_remote","d_sb","d_osv","d_ns","protest_event_counts","riot_event_counts","remote_event_counts","sb_fatalities","osv_fatalities","ns_fatalities"]].copy()
print(df_out.isna().any())
print(df_out.duplicated(subset=['dd',"country","gw_codes"]).any())
print(df_out.duplicated(subset=['dd',"country"]).any())
print(df_out.duplicated(subset=['dd',"gw_codes"]).any())

# Check datatypes and convert floats to integer if needed
print(df_out.dtypes)
df_out['protest_event_counts']=df_out['protest_event_counts'].astype('int64')
df_out['riot_event_counts']=df_out['riot_event_counts'].astype('int64')
df_out['remote_event_counts']=df_out['remote_event_counts'].astype('int64')

# Save
df_out.to_csv("out/df_out_full_cm.csv") 

#####################
### History theme ###
#####################

### t-1 ###

df_conf_hist=df_out[["year","dd","gw_codes","country","d_protest","d_riot","d_sb","d_osv","d_ns","protest_event_counts","riot_event_counts","remote_event_counts","sb_fatalities","osv_fatalities","ns_fatalities"]].copy()
df_conf_hist["d_protest_lag1"]=lag_groupped(df,"country","d_protest",1)
df_conf_hist["d_riot_lag1"]=lag_groupped(df,"country","d_riot",1)
df_conf_hist["d_remote_lag1"]=lag_groupped(df,"country","d_remote",1)
df_conf_hist["d_sb_lag1"]=lag_groupped(df,"country","d_sb",1)
df_conf_hist["d_osv_lag1"]=lag_groupped(df,"country","d_osv",1)
df_conf_hist["d_ns_lag1"]=lag_groupped(df,"country","d_ns",1)

### Time since ###

df_conf_hist['d_protest_zeros'] = consec_zeros_grouped(df,'country','d_protest')
df_conf_hist['d_protest_zeros'] = lag_groupped(df_conf_hist,'country','d_protest_zeros',1)
df_conf_hist['d_protest_zeros_decay'] = exponential_decay(df_conf_hist['d_protest_zeros'])
df_conf_hist = df_conf_hist.drop('d_protest_zeros', axis=1)

df_conf_hist['d_riot_zeros'] = consec_zeros_grouped(df,'country','d_riot')
df_conf_hist['d_riot_zeros'] = lag_groupped(df_conf_hist,'country','d_riot_zeros',1)
df_conf_hist['d_riot_zeros_decay'] = exponential_decay(df_conf_hist['d_riot_zeros'])
df_conf_hist = df_conf_hist.drop('d_riot_zeros', axis=1)

df_conf_hist['d_remote_zeros'] = consec_zeros_grouped(df,'country','d_remote')
df_conf_hist['d_remote_zeros'] = lag_groupped(df_conf_hist,'country','d_remote_zeros',1)
df_conf_hist['d_remote_zeros_decay'] = exponential_decay(df_conf_hist['d_remote_zeros'])
df_conf_hist = df_conf_hist.drop('d_remote_zeros', axis=1)

df_conf_hist['d_sb_zeros'] = consec_zeros_grouped(df,'country','d_sb')
df_conf_hist['d_sb_zeros'] = lag_groupped(df_conf_hist,'country','d_sb_zeros',1)
df_conf_hist['d_sb_zeros_decay'] = exponential_decay(df_conf_hist['d_sb_zeros'])
df_conf_hist = df_conf_hist.drop('d_sb_zeros', axis=1)

df_conf_hist['d_osv_zeros'] = consec_zeros_grouped(df,'country','d_osv')
df_conf_hist['d_osv_zeros'] = lag_groupped(df_conf_hist,'country','d_osv_zeros',1)
df_conf_hist['d_osv_zeros_decay'] = exponential_decay(df_conf_hist['d_osv_zeros'])
df_conf_hist = df_conf_hist.drop('d_osv_zeros', axis=1)

df_conf_hist['d_ns_zeros'] = consec_zeros_grouped(df,'country','d_ns')
df_conf_hist['d_ns_zeros'] = lag_groupped(df_conf_hist,'country','d_ns_zeros',1)
df_conf_hist['d_ns_zeros_decay'] = exponential_decay(df_conf_hist['d_ns_zeros'])
df_conf_hist = df_conf_hist.drop('d_ns_zeros', axis=1)

### New state ###

# Create base df on year level
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()

# Count years from start date as defined in GW set up
# http://ksgleditsch.com/data/iisystem.dat
# http://ksgleditsch.com/data/microstatessystem.dat
all_countries=pd.read_csv("data/df_ccodes_gw.csv")
all_countries_s=all_countries.loc[all_countries["end"]>=1989]

base['regime_duration']=0
# For each country
for c in base.gw_codes.unique():
    # Get start year
    star_year=all_countries_s.start.loc[all_countries_s["gw_codes"]==c].iloc[0]
    # For each year that country is available
    for y in base.year.loc[base["gw_codes"]==c].unique():
        # Calculate difference between t and start_year, and fill
        base.loc[(base["year"]==y)&(base["gw_codes"]==c),"regime_duration"]=y-star_year 

# Merge    
df_conf_hist=pd.merge(left=df_conf_hist,right=base[["year","gw_codes","regime_duration"]],on=["year","gw_codes"],how="left")

# Final df
df_conf_hist=df_conf_hist[["year","dd","gw_codes","country","d_protest_lag1","d_riot_lag1","d_remote_lag1","d_sb_lag1","d_osv_lag1","d_ns_lag1","d_protest_zeros_decay","d_riot_zeros_decay","d_remote_zeros_decay","d_sb_zeros_decay","d_osv_zeros_decay","d_ns_zeros_decay","regime_duration"]].copy()
print(df_conf_hist.isna().any())
print(df_conf_hist.min())
print(df_conf_hist.duplicated(subset=['dd',"country","gw_codes"]).any())
print(df_conf_hist.duplicated(subset=['dd',"country"]).any())
print(df_conf_hist.duplicated(subset=['dd',"gw_codes"]).any())

# Check datatypes and convert floats to integer if needed
print(df_conf_hist.dtypes)
df_conf_hist['d_protest_lag1']=df_conf_hist['d_protest_lag1'].astype('int64')
df_conf_hist['d_riot_lag1']=df_conf_hist['d_riot_lag1'].astype('int64')
df_conf_hist['d_remote_lag1']=df_conf_hist['d_remote_lag1'].astype('int64')
df_conf_hist['d_sb_lag1']=df_conf_hist['d_sb_lag1'].astype('int64')
df_conf_hist['d_osv_lag1']=df_conf_hist['d_osv_lag1'].astype('int64')
df_conf_hist['d_ns_lag1']=df_conf_hist['d_ns_lag1'].astype('int64')

# Save
df_conf_hist.to_csv("out/df_conf_hist_full_cm.csv")  

########################
### Demography theme ###
########################

# Initiate
df_demog=df_out[["year","dd","gw_codes","country"]].copy()

# Load wb data, previously retrived with the WB api
demog=pd.read_csv("data/demog_wb.csv",index_col=0)
print(demog.min())

### Population size ###

demog = demog.rename(columns={"SP.POP.TOTL": 'pop'})
df_demog=pd.merge(left=df_demog,right=demog[["year","gw_codes","pop"]],on=["year","gw_codes"],how="left")

### Population density ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=demog[["year","gw_codes","pop","EN.POP.DNST"]],on=["year","gw_codes"],how="left")
# Use population size as additional variable
base_imp=multivariate_imp_bayes(base,"country",["EN.POP.DNST"],vars_add=["pop"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["EN.POP.DNST"])
base_imp_mean=simple_imp_grouped(base,"country",["EN.POP.DNST"])
base_imp_final['EN.POP.DNST'] = base_imp_final['EN.POP.DNST'].fillna(base_imp_mean['EN.POP.DNST'])
base_imp_final['EN.POP.DNST'] = base_imp_final['EN.POP.DNST'].fillna(base_imp['imp'])

# Add missing id and rename variable
base_imp_final['pop_density_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={'EN.POP.DNST': 'pop_density'})

# Merge
df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","pop_density",'pop_density_id']],on=["year","gw_codes"],how="left")

### Urbanization ###

demog = demog.rename(columns={"SP.URB.TOTL.IN.ZS": 'urb_share'})
df_demog=pd.merge(left=df_demog,right=demog[["year","gw_codes","urb_share"]],on=["year","gw_codes"],how="left")

### Rural ###

demog = demog.rename(columns={"SP.RUR.TOTL.ZS": 'rural_share'})
df_demog=pd.merge(left=df_demog,right=demog[["year","gw_codes","rural_share"]],on=["year","gw_codes"],how="left")

# Load wb data, previously retrived with the WB api
demog=pd.read_csv("data/demog_wb2.csv",index_col=0)
print(demog.min())

### Male population ###

demog = demog.rename(columns={"SP.POP.TOTL.MA.ZS": 'pop_male_share'})
df_demog=pd.merge(left=df_demog,right=demog[["year","gw_codes","pop_male_share"]],on=["year","gw_codes"],how="left")

### Male total population 0-14 ### 

demog = demog.rename(columns={"SP.POP.0014.MA.ZS": 'pop_male_share_0_14'})
df_demog=pd.merge(left=df_demog,right=demog[["year","gw_codes","pop_male_share_0_14"]],on=["year","gw_codes"],how="left")

### Male total population 15-19 ###

demog = demog.rename(columns={"SP.POP.1519.MA.5Y": 'pop_male_share_15_19'})
df_demog=pd.merge(left=df_demog,right=demog[["year","gw_codes","pop_male_share_15_19"]],on=["year","gw_codes"],how="left")

### Male total population 20-24 ### 

demog = demog.rename(columns={"SP.POP.2024.MA.5Y": 'pop_male_share_20_24'})
df_demog=pd.merge(left=df_demog,right=demog[["year","gw_codes","pop_male_share_20_24"]],on=["year","gw_codes"],how="left")

### Male total population 25-29 ###

demog = demog.rename(columns={"SP.POP.2529.MA.5Y": 'pop_male_share_25_29'})
df_demog=pd.merge(left=df_demog,right=demog[["year","gw_codes","pop_male_share_25_29"]],on=["year","gw_codes"],how="left")

### Male total population 30-34 ###

demog = demog.rename(columns={"SP.POP.3034.MA.5Y": 'pop_male_share_30_34'})
df_demog=pd.merge(left=df_demog,right=demog[["year","gw_codes","pop_male_share_30_34"]],on=["year","gw_codes"],how="left")

# Load epr data
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
erp=pd.read_csv("data/data_out/epr_cy.csv",index_col=0)
base=pd.merge(left=base,right=erp[["year","gw_codes","group_counts","monopoly_share","discriminated_share","powerless_share","dominant_share","ethnic_frac","rel_frac","lang_frac","race_frac"]],on=["year","gw_codes"],how="left")

### Group counts ###

# Simple imputation (linear and if it fails mean) and merge
base_imp_final=linear_imp_grouped(base,"country",["group_counts"])
base_imp_mean=simple_imp_grouped(base,"country",["group_counts"])
base_imp_final['group_counts'] = base_imp_final['group_counts'].fillna(base_imp_mean['group_counts'])
base_imp_final['group_counts_id'] = base["group_counts"].isnull().astype(int)
df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","group_counts",'group_counts_id']],on=["year","gw_codes"],how="left")

### Monopoly share ####

# Simple imputation (linear and if it fails mean) and merge
base_imp_final=linear_imp_grouped(base,"country",["monopoly_share"])
base_imp_mean=simple_imp_grouped(base,"country",["monopoly_share"])
base_imp_final['monopoly_share'] = base_imp_final['monopoly_share'].fillna(base_imp_mean['monopoly_share'])
base_imp_final['monopoly_share_id'] = base["monopoly_share"].isnull().astype(int)
df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","monopoly_share",'monopoly_share_id']],on=["year","gw_codes"],how="left")

#### Discriminated share ###

# Simple imputation (linear and if it fails mean) and merge
base_imp_final=linear_imp_grouped(base,"country",["discriminated_share"])
base_imp_mean=simple_imp_grouped(base,"country",["discriminated_share"])
base_imp_final['discriminated_share'] = base_imp_final['discriminated_share'].fillna(base_imp_mean['discriminated_share'])
base_imp_final['discriminated_share_id'] = base["discriminated_share"].isnull().astype(int)
df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","discriminated_share",'discriminated_share_id']],on=["year","gw_codes"],how="left")

### Powerless share ###

# Simple imputation (linear and if it fails mean) and merge
base_imp_final=linear_imp_grouped(base,"country",["powerless_share"])
base_imp_mean=simple_imp_grouped(base,"country",["powerless_share"])
base_imp_final['powerless_share'] = base_imp_final['powerless_share'].fillna(base_imp_mean['powerless_share'])
base_imp_final['powerless_share_id'] = base["powerless_share"].isnull().astype(int)
df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","powerless_share",'powerless_share_id']],on=["year","gw_codes"],how="left")

### Dominant share ###

# Simple imputation (linear and if it fails mean) and merge
base_imp_final=linear_imp_grouped(base,"country",["dominant_share"])
base_imp_mean=simple_imp_grouped(base,"country",["dominant_share"])
base_imp_final['dominant_share'] = base_imp_final['dominant_share'].fillna(base_imp_mean['dominant_share'])
base_imp_final['dominant_share_id'] = base["dominant_share"].isnull().astype(int)
df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","dominant_share",'dominant_share_id']],on=["year","gw_codes"],how="left")

### Ethnic fractionalization ###

# Simple imputation (linear and if it fails mean) and merge
base_imp_final=linear_imp_grouped(base,"country",["ethnic_frac"])
base_imp_mean=simple_imp_grouped(base,"country",["ethnic_frac"])
base_imp_final['ethnic_frac'] = base_imp_final['ethnic_frac'].fillna(base_imp_mean['ethnic_frac'])
base_imp_final['ethnic_frac_id'] = base["ethnic_frac"].isnull().astype(int)
df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","ethnic_frac",'ethnic_frac_id']],on=["year","gw_codes"],how="left")

### Religious fractionalization ###

# Simple imputation (linear and if it fails mean) and merge
base_imp_final=linear_imp_grouped(base,"country",["rel_frac"])
base_imp_mean=simple_imp_grouped(base,"country",["rel_frac"])
base_imp_final['rel_frac'] = base_imp_final['rel_frac'].fillna(base_imp_mean['rel_frac'])
base_imp_final['rel_frac_id'] = base["rel_frac"].isnull().astype(int)
df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","rel_frac",'rel_frac_id']],on=["year","gw_codes"],how="left")

### Language fractionlization ###

# Simple imputation (linear and if it fails mean) and merge
base_imp_final=linear_imp_grouped(base,"country",["lang_frac"])
base_imp_mean=simple_imp_grouped(base,"country",["lang_frac"])
base_imp_final['lang_frac'] = base_imp_final['lang_frac'].fillna(base_imp_mean['lang_frac'])
base_imp_final['lang_frac_id'] = base["lang_frac"].isnull().astype(int)
df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","lang_frac",'lang_frac_id']],on=["year","gw_codes"],how="left")

### Race fractionlization ###

# Simple imputation (linear and if it fails mean) and merge
base_imp_final=linear_imp_grouped(base,"country",["race_frac"])
base_imp_mean=simple_imp_grouped(base,"country",["race_frac"])
base_imp_final['race_frac'] = base_imp_final['race_frac'].fillna(base_imp_mean['race_frac'])
base_imp_final['race_frac_id'] = base["race_frac"].isnull().astype(int)
df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","race_frac",'race_frac_id']],on=["year","gw_codes"],how="left")

# Final df
df_demog=df_demog[["year","dd","gw_codes","country",'pop','pop_density','pop_density_id','urb_share','rural_share','pop_male_share','pop_male_share_0_14','pop_male_share_15_19','pop_male_share_20_24','pop_male_share_25_29','pop_male_share_30_34','group_counts','group_counts_id','monopoly_share','monopoly_share_id','discriminated_share','discriminated_share_id','powerless_share','powerless_share_id','dominant_share','dominant_share_id','ethnic_frac','ethnic_frac_id','rel_frac','rel_frac_id','lang_frac','lang_frac_id','race_frac','race_frac_id']].copy()
print(df_demog.isna().any())
print(df_demog.min())
print(df_demog.duplicated(subset=['dd',"country","gw_codes"]).any())
print(df_demog.duplicated(subset=['dd',"country"]).any())
print(df_demog.duplicated(subset=['dd',"gw_codes"]).any())

# Check datatypes and convert floats to integer if needed
print(df_demog.dtypes)
df_demog['group_counts'].unique() # Check unique levels because of imputation
df_demog['group_counts']=df_demog['group_counts'].astype('int64')

# Save
df_demog.to_csv("out/df_demog_full_cm.csv") 

###################################
### Economy & development theme ###
###################################

# Initiate
df_econ=df_out[["year","dd","gw_codes","country"]].copy()

# Load wb data, previously retrived with the WB api
economy=pd.read_csv("data/economy_wb.csv",index_col=0)
print(economy.min())

### Total natual resource rents % of GDP ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS","NY.GDP.TOTL.RT.ZS"]],on=["year","gw_codes"],how="left")
# Use other rents as additional variables
base_imp=multivariate_imp_bayes(base,"country",["NY.GDP.TOTL.RT.ZS"],vars_add=["NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["NY.GDP.TOTL.RT.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["NY.GDP.TOTL.RT.ZS"])
base_imp_final["NY.GDP.TOTL.RT.ZS"] = base_imp_final["NY.GDP.TOTL.RT.ZS"].fillna(base_imp_mean["NY.GDP.TOTL.RT.ZS"])
base_imp_final["NY.GDP.TOTL.RT.ZS"] = base_imp_final["NY.GDP.TOTL.RT.ZS"].fillna(base_imp["imp"])

# Add missing id, rename variable, and merge
base_imp_final['natres_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NY.GDP.TOTL.RT.ZS": 'natres_share'})
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","natres_share",'natres_share_id']],on=["year","gw_codes"],how="left")

### Oil rents (% of GDP) ##

# Create base df (on year level), perform multiple imputation
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS","NY.GDP.TOTL.RT.ZS"]],on=["year","gw_codes"],how="left")
# Use other rents as additional variables
base_imp=multivariate_imp_bayes(base,"country",["NY.GDP.PETR.RT.ZS"],vars_add=["NY.GDP.TOTL.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["NY.GDP.PETR.RT.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["NY.GDP.PETR.RT.ZS"])
base_imp_final["NY.GDP.PETR.RT.ZS"] = base_imp_final["NY.GDP.PETR.RT.ZS"].fillna(base_imp_mean["NY.GDP.PETR.RT.ZS"])
base_imp_final["NY.GDP.PETR.RT.ZS"] = base_imp_final["NY.GDP.PETR.RT.ZS"].fillna(base_imp["imp"])

# Add missing id, rename variable, and merge
base_imp_final['oil_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NY.GDP.PETR.RT.ZS": 'oil_share'})
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","oil_share",'oil_share_id']],on=["year","gw_codes"],how="left")

### Natural gas rents (% of GDP) ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS","NY.GDP.TOTL.RT.ZS"]],on=["year","gw_codes"],how="left")
# Use other rents as additional variables
base_imp=multivariate_imp_bayes(base,"country",["NY.GDP.NGAS.RT.ZS"],vars_add=["NY.GDP.TOTL.RT.ZS","NY.GDP.PETR.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["NY.GDP.NGAS.RT.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["NY.GDP.NGAS.RT.ZS"])
base_imp_final["NY.GDP.NGAS.RT.ZS"] = base_imp_final["NY.GDP.NGAS.RT.ZS"].fillna(base_imp_mean["NY.GDP.NGAS.RT.ZS"])
base_imp_final["NY.GDP.NGAS.RT.ZS"] = base_imp_final["NY.GDP.NGAS.RT.ZS"].fillna(base_imp["imp"])

# Add missing id, rename variable, and merge
base_imp_final['gas_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NY.GDP.NGAS.RT.ZS": 'gas_share'})
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","gas_share",'gas_share_id']],on=["year","gw_codes"],how="left")

### Coal rents (% of GDP) ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS","NY.GDP.TOTL.RT.ZS"]],on=["year","gw_codes"],how="left")
# Use other rents as additional variables
base_imp=multivariate_imp_bayes(base,"country",["NY.GDP.COAL.RT.ZS"],vars_add=["NY.GDP.TOTL.RT.ZS","NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["NY.GDP.COAL.RT.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["NY.GDP.COAL.RT.ZS"])
base_imp_final["NY.GDP.COAL.RT.ZS"] = base_imp_final["NY.GDP.COAL.RT.ZS"].fillna(base_imp_mean["NY.GDP.COAL.RT.ZS"])
base_imp_final["NY.GDP.COAL.RT.ZS"] = base_imp_final["NY.GDP.COAL.RT.ZS"].fillna(base_imp["imp"])

# Add missing id, rename variable, and merge
base_imp_final['coal_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NY.GDP.COAL.RT.ZS": 'coal_share'})
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","coal_share",'coal_share_id']],on=["year","gw_codes"],how="left")

### Forest rents (% of GDP) ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS","NY.GDP.TOTL.RT.ZS"]],on=["year","gw_codes"],how="left")
# Use other rents as additional variables
base_imp=multivariate_imp_bayes(base,"country",["NY.GDP.FRST.RT.ZS"],vars_add=["NY.GDP.TOTL.RT.ZS","NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.MINR.RT.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["NY.GDP.FRST.RT.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["NY.GDP.FRST.RT.ZS"])
base_imp_final["NY.GDP.FRST.RT.ZS"] = base_imp_final["NY.GDP.FRST.RT.ZS"].fillna(base_imp_mean["NY.GDP.FRST.RT.ZS"])
base_imp_final["NY.GDP.FRST.RT.ZS"] = base_imp_final["NY.GDP.FRST.RT.ZS"].fillna(base_imp["imp"])

# Add missing id, rename variable, and merge
base_imp_final['forest_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NY.GDP.FRST.RT.ZS": 'forest_share'})
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","forest_share",'forest_share_id']],on=["year","gw_codes"],how="left")

### Minerals rents (% of GDP) ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS","NY.GDP.TOTL.RT.ZS"]],on=["year","gw_codes"],how="left")
# Use other rents as additional variables
base_imp=multivariate_imp_bayes(base,"country",["NY.GDP.MINR.RT.ZS"],vars_add=["NY.GDP.TOTL.RT.ZS","NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["NY.GDP.MINR.RT.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["NY.GDP.MINR.RT.ZS"])
base_imp_final["NY.GDP.MINR.RT.ZS"] = base_imp_final["NY.GDP.MINR.RT.ZS"].fillna(base_imp_mean["NY.GDP.MINR.RT.ZS"])
base_imp_final["NY.GDP.MINR.RT.ZS"] = base_imp_final["NY.GDP.MINR.RT.ZS"].fillna(base_imp["imp"])

# Add missing id, rename variable, and merge
base_imp_final['minerals_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NY.GDP.MINR.RT.ZS": 'minerals_share'})
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","minerals_share",'minerals_share_id']],on=["year","gw_codes"],how="left")

### GDP per capita ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
# Use GDP growth and GNI per capita as additional variables
base_imp=multivariate_imp_bayes(base,"country",["NY.GDP.PCAP.CD"],vars_add=["NY.GDP.MKTP.KD.ZG","NY.GNP.PCAP.CD"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["NY.GDP.PCAP.CD"])
base_imp_mean=simple_imp_grouped(base,"country",["NY.GDP.PCAP.CD"])
base_imp_final["NY.GDP.PCAP.CD"] = base_imp_final["NY.GDP.PCAP.CD"].fillna(base_imp_mean["NY.GDP.PCAP.CD"])
base_imp_final["NY.GDP.PCAP.CD"] = base_imp_final["NY.GDP.PCAP.CD"].fillna(base_imp["imp"])

# Add missing id, rename variable, and merge
base_imp_final['gdp_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NY.GDP.PCAP.CD": 'gdp'})
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","gdp",'gdp_id']],on=["year","gw_codes"],how="left")

### GNI per capita ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
# Use GDP growth and GDP per capita as additional variables
base_imp=multivariate_imp_bayes(base,"country",["NY.GNP.PCAP.CD"],vars_add=["NY.GDP.MKTP.KD.ZG","NY.GDP.PCAP.CD"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["NY.GNP.PCAP.CD"])
base_imp_mean=simple_imp_grouped(base,"country",["NY.GNP.PCAP.CD"])
base_imp_final["NY.GNP.PCAP.CD"] = base_imp_final["NY.GNP.PCAP.CD"].fillna(base_imp_mean["NY.GNP.PCAP.CD"])
base_imp_final["NY.GNP.PCAP.CD"] = base_imp_final["NY.GNP.PCAP.CD"].fillna(base_imp["imp"])

# Add missing id, rename variable, and merge
base_imp_final['gni_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NY.GNP.PCAP.CD": 'gni'})
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","gni",'gni_id']],on=["year","gw_codes"],how="left")

### GDP growth ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
# Use GDP per capita and GNI per capita as additional variables
base_imp=multivariate_imp_bayes(base,"country",["NY.GDP.MKTP.KD.ZG"],vars_add=["NY.GDP.PCAP.CD","NY.GNP.PCAP.CD"],max_iter=10,min_val=-100)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["NY.GDP.MKTP.KD.ZG"])
base_imp_mean=simple_imp_grouped(base,"country",["NY.GDP.MKTP.KD.ZG"])
base_imp_final["NY.GDP.MKTP.KD.ZG"] = base_imp_final["NY.GDP.MKTP.KD.ZG"].fillna(base_imp_mean["NY.GDP.MKTP.KD.ZG"])
base_imp_final["NY.GDP.MKTP.KD.ZG"] = base_imp_final["NY.GDP.MKTP.KD.ZG"].fillna(base_imp["imp"])

# Add missing id, rename variable, and merge
base_imp_final['gdp_growth_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NY.GDP.MKTP.KD.ZG": 'gdp_growth'})
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","gdp_growth",'gdp_growth_id']],on=["year","gw_codes"],how="left")

### Unemployment, total ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
# Use unemployment male as additional variable
base_imp=multivariate_imp_bayes(base,"country",["SL.UEM.TOTL.NE.ZS"],vars_add=["SL.UEM.TOTL.MA.NE.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["SL.UEM.TOTL.NE.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["SL.UEM.TOTL.NE.ZS"])
base_imp_final["SL.UEM.TOTL.NE.ZS"] = base_imp_final["SL.UEM.TOTL.NE.ZS"].fillna(base_imp_mean["SL.UEM.TOTL.NE.ZS"])
base_imp_final["SL.UEM.TOTL.NE.ZS"] = base_imp_final["SL.UEM.TOTL.NE.ZS"].fillna(base_imp["imp"])

# Add missing id, rename variable, and merge
base_imp_final['unemploy_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SL.UEM.TOTL.NE.ZS": 'unemploy'})
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","unemploy",'unemploy_id']],on=["year","gw_codes"],how="left")

### Unemployment, men ###

# Create base df (on year level), perform multiple imputation
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
# Use unemployment as additional variable
base_imp=multivariate_imp_bayes(base,"country",["SL.UEM.TOTL.MA.NE.ZS"],vars_add=["SL.UEM.TOTL.NE.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["SL.UEM.TOTL.MA.NE.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["SL.UEM.TOTL.MA.NE.ZS"])
base_imp_final["SL.UEM.TOTL.MA.NE.ZS"] = base_imp_final["SL.UEM.TOTL.MA.NE.ZS"].fillna(base_imp_mean["SL.UEM.TOTL.MA.NE.ZS"])
base_imp_final["SL.UEM.TOTL.MA.NE.ZS"] = base_imp_final["SL.UEM.TOTL.MA.NE.ZS"].fillna(base_imp["imp"])

# Add missing id, rename variable, and merge
base_imp_final['unemploy_male_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SL.UEM.TOTL.MA.NE.ZS": 'unemploy_male'})
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","unemploy_male",'unemploy_male_id']],on=["year","gw_codes"],how="left")

### Inflation rate ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
# Use consumer price as additional variable
base_imp=multivariate_imp_bayes(base,"country",["FP.CPI.TOTL.ZG"],vars_add=["FP.CPI.TOTL"],max_iter=10,min_val=-100)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["FP.CPI.TOTL.ZG"])
base_imp_mean=simple_imp_grouped(base,"country",["FP.CPI.TOTL.ZG"])
base_imp_final["FP.CPI.TOTL.ZG"] = base_imp_final["FP.CPI.TOTL.ZG"].fillna(base_imp_mean["FP.CPI.TOTL.ZG"])
base_imp_final["FP.CPI.TOTL.ZG"] = base_imp_final["FP.CPI.TOTL.ZG"].fillna(base_imp["imp"])

# Add missing id, rename variable, and merge
base_imp_final['inflat_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"FP.CPI.TOTL.ZG": 'inflat'})
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","inflat",'inflat_id']],on=["year","gw_codes"],how="left")

### Consumer price index (2010 = 100) ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
# Use inflation as additional variable
base_imp=multivariate_imp_bayes(base,"country",["FP.CPI.TOTL"],vars_add=["FP.CPI.TOTL.ZG"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["FP.CPI.TOTL"])
base_imp_mean=simple_imp_grouped(base,"country",["FP.CPI.TOTL"])
base_imp_final["FP.CPI.TOTL"] = base_imp_final["FP.CPI.TOTL"].fillna(base_imp_mean["FP.CPI.TOTL"])
base_imp_final["FP.CPI.TOTL"] = base_imp_final["FP.CPI.TOTL"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['conprice_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"FP.CPI.TOTL": 'conprice'})
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","conprice",'conprice_id']],on=["year","gw_codes"],how="left")

### Prevalence of undernourishment (% of population) ###

# Create base df (on year level), perform multiple imputation
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
# Use as additional variables food production, rural driking water, and urban drinking water
base_imp=multivariate_imp_bayes(base,"country",["SN.ITK.DEFC.ZS"],vars_add=["AG.PRD.FOOD.XD","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["SN.ITK.DEFC.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["SN.ITK.DEFC.ZS"])
base_imp_final["SN.ITK.DEFC.ZS"] = base_imp_final["SN.ITK.DEFC.ZS"].fillna(base_imp_mean["SN.ITK.DEFC.ZS"])
base_imp_final["SN.ITK.DEFC.ZS"] = base_imp_final["SN.ITK.DEFC.ZS"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['undernour_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SN.ITK.DEFC.ZS": 'undernour'})
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","undernour",'undernour_id']],on=["year","gw_codes"],how="left")

### Food production ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
# Use as additional variables undernourishment, drinking water rural, and drinking water urban
base_imp=multivariate_imp_bayes(base,"country",["AG.PRD.FOOD.XD"],vars_add=["SN.ITK.DEFC.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["AG.PRD.FOOD.XD"])
base_imp_mean=simple_imp_grouped(base,"country",["AG.PRD.FOOD.XD"])
base_imp_final["AG.PRD.FOOD.XD"] = base_imp_final["AG.PRD.FOOD.XD"].fillna(base_imp_mean["AG.PRD.FOOD.XD"])
base_imp_final["AG.PRD.FOOD.XD"] = base_imp_final["AG.PRD.FOOD.XD"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['foodprod_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"AG.PRD.FOOD.XD": 'foodprod'})
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","foodprod",'foodprod_id']],on=["year","gw_codes"],how="left")

### People using at least basic drinking water services, rural (% of rural population) ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
# Use as additional variables undernourishment, food production, drinking water urban
base_imp=multivariate_imp_bayes(base,"country",["SH.H2O.BASW.RU.ZS"],vars_add=["SN.ITK.DEFC.ZS","AG.PRD.FOOD.XD","SH.H2O.BASW.UR.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["SH.H2O.BASW.RU.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["SH.H2O.BASW.RU.ZS"])
base_imp_final["SH.H2O.BASW.RU.ZS"] = base_imp_final["SH.H2O.BASW.RU.ZS"].fillna(base_imp_mean["SH.H2O.BASW.RU.ZS"])
base_imp_final["SH.H2O.BASW.RU.ZS"] = base_imp_final["SH.H2O.BASW.RU.ZS"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['water_rural_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SH.H2O.BASW.RU.ZS": 'water_rural'})
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","water_rural",'water_rural_id']],on=["year","gw_codes"],how="left")

### People using at least basic drinking water services, urban (% of urban population) ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
# Use as additional variables undernourishment, food production, drinking water rural
base_imp=multivariate_imp_bayes(base,"country",["SH.H2O.BASW.UR.ZS"],vars_add=["SN.ITK.DEFC.ZS","AG.PRD.FOOD.XD","SH.H2O.BASW.RU.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["SH.H2O.BASW.UR.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["SH.H2O.BASW.UR.ZS"])
base_imp_final["SH.H2O.BASW.UR.ZS"] = base_imp_final["SH.H2O.BASW.UR.ZS"].fillna(base_imp_mean["SH.H2O.BASW.UR.ZS"])
base_imp_final["SH.H2O.BASW.UR.ZS"] = base_imp_final["SH.H2O.BASW.UR.ZS"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['water_urb_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SH.H2O.BASW.UR.ZS": 'water_urb'})
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","water_urb",'water_urb_id']],on=["year","gw_codes"],how="left")

### Agriculture % of GDP ###

# Create base df (on year level), perform multiple imputation
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
# Use trade % of GDP as additional variable
base_imp=multivariate_imp_bayes(base,"country",["NV.AGR.TOTL.ZS"],vars_add=["NE.TRD.GNFS.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["NV.AGR.TOTL.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["NV.AGR.TOTL.ZS"])
base_imp_final["NV.AGR.TOTL.ZS"] = base_imp_final["NV.AGR.TOTL.ZS"].fillna(base_imp_mean["NV.AGR.TOTL.ZS"])
base_imp_final["NV.AGR.TOTL.ZS"] = base_imp_final["NV.AGR.TOTL.ZS"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['agri_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NV.AGR.TOTL.ZS": 'agri_share'})
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","agri_share",'agri_share_id']],on=["year","gw_codes"],how="left")

### Trade % of GDP ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
# Use agriculture % of GDP as additional variable
base_imp=multivariate_imp_bayes(base,"country",["NE.TRD.GNFS.ZS"],vars_add=["NV.AGR.TOTL.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["NE.TRD.GNFS.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["NE.TRD.GNFS.ZS"])
base_imp_final["NE.TRD.GNFS.ZS"] = base_imp_final["NE.TRD.GNFS.ZS"].fillna(base_imp_mean["NE.TRD.GNFS.ZS"])
base_imp_final["NE.TRD.GNFS.ZS"] = base_imp_final["NE.TRD.GNFS.ZS"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['trade_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NE.TRD.GNFS.ZS": 'trade_share'})
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","trade_share",'trade_share_id']],on=["year","gw_codes"],how="left")

### Fertility rate, total (births per woman) ###

base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
base = base.rename(columns={"SP.DYN.TFRT.IN": 'fert'})
df_econ=pd.merge(left=df_econ,right=base[["year","gw_codes","fert"]],on=["year","gw_codes"],how="left")

### Life expectancy at birth, female (years) ###

base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
base = base.rename(columns={"SP.DYN.LE00.FE.IN": 'lifeexp_female'})
df_econ=pd.merge(left=df_econ,right=base[["year","gw_codes","lifeexp_female"]],on=["year","gw_codes"],how="left")

### Life expectancy at birth, male (years) ###

base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
base = base.rename(columns={"SP.DYN.LE00.MA.IN": 'lifeexp_male'})
df_econ=pd.merge(left=df_econ,right=base[["year","gw_codes","lifeexp_male"]],on=["year","gw_codes"],how="left")

### Population growth (annual %) ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
# Use as additional variables fertility rate, life expectancy female, life expectancy male, infant mortality
base_imp=multivariate_imp_bayes(base,"country",["SP.POP.GROW"],vars_add=["SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.DYN.IMRT.IN"],max_iter=10,min_val=-100)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["SP.POP.GROW"])
base_imp_mean=simple_imp_grouped(base,"country",["SP.POP.GROW"])
base_imp_final["SP.POP.GROW"] = base_imp_final["SP.POP.GROW"].fillna(base_imp_mean["SP.POP.GROW"])
base_imp_final["SP.POP.GROW"] = base_imp_final["SP.POP.GROW"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['pop_growth_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SP.POP.GROW": 'pop_growth'})
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","pop_growth","pop_growth_id"]],on=["year","gw_codes"],how="left")

### Infant mortality ###

base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
base = base.rename(columns={"SP.DYN.IMRT.IN": 'inf_mort'})
df_econ=pd.merge(left=df_econ,right=base[["year","gw_codes","inf_mort"]],on=["year","gw_codes"],how="left")

### Exports of goods and services (% of GDP) ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NE.EXP.GNFS.ZS","NE.IMP.GNFS.ZS","SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"]],on=["year","gw_codes"],how="left")
# Use imports as additional variable
base_imp=multivariate_imp_bayes(base,"country",["NE.EXP.GNFS.ZS"],vars_add=["NE.IMP.GNFS.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["NE.EXP.GNFS.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["NE.EXP.GNFS.ZS"])
base_imp_final["NE.EXP.GNFS.ZS"] = base_imp_final["NE.EXP.GNFS.ZS"].fillna(base_imp_mean["NE.EXP.GNFS.ZS"])
base_imp_final["NE.EXP.GNFS.ZS"] = base_imp_final["NE.EXP.GNFS.ZS"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['exports_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NE.EXP.GNFS.ZS": 'exports'})
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","exports",'exports_id']],on=["year","gw_codes"],how="left")

### Imports of goods and services (% of GDP) ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NE.EXP.GNFS.ZS","NE.IMP.GNFS.ZS","SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"]],on=["year","gw_codes"],how="left")
# Use exports as additional variable
base_imp=multivariate_imp_bayes(base,"country",["NE.IMP.GNFS.ZS"],vars_add=["NE.EXP.GNFS.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["NE.IMP.GNFS.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["NE.IMP.GNFS.ZS"])
base_imp_final["NE.IMP.GNFS.ZS"] = base_imp_final["NE.IMP.GNFS.ZS"].fillna(base_imp_mean["NE.IMP.GNFS.ZS"])
base_imp_final["NE.IMP.GNFS.ZS"] = base_imp_final["NE.IMP.GNFS.ZS"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['imports_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NE.IMP.GNFS.ZS": 'imports'})
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","imports",'imports_id']],on=["year","gw_codes"],how="left")

### School enrollment, primary, female (% gross) ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NE.EXP.GNFS.ZS","NE.IMP.GNFS.ZS","SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"]],on=["year","gw_codes"],how="left")
# Use as additional variables male primary school enrollment, secondary school enrollment male and female, tertiary school enrollment male and female
base_imp=multivariate_imp_bayes(base,"country",["SE.PRM.ENRR.FE"],vars_add=["SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["SE.PRM.ENRR.FE"])
base_imp_mean=simple_imp_grouped(base,"country",["SE.PRM.ENRR.FE"])
base_imp_final["SE.PRM.ENRR.FE"] = base_imp_final["SE.PRM.ENRR.FE"].fillna(base_imp_mean["SE.PRM.ENRR.FE"])
base_imp_final["SE.PRM.ENRR.FE"] = base_imp_final["SE.PRM.ENRR.FE"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['primary_female_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SE.PRM.ENRR.FE": 'primary_female'})
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","primary_female",'primary_female_id']],on=["year","gw_codes"],how="left")

### School enrollment, primary, male (% gross) ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NE.EXP.GNFS.ZS","NE.IMP.GNFS.ZS","SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"]],on=["year","gw_codes"],how="left")
# Use as additional variables primary school enrollment female, secondary school enrollment male and female, tertiary school enrollment male and female
base_imp=multivariate_imp_bayes(base,"country",["SE.PRM.ENRR.MA"],vars_add=["SE.PRM.ENRR.FE","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["SE.PRM.ENRR.MA"])
base_imp_mean=simple_imp_grouped(base,"country",["SE.PRM.ENRR.MA"])
base_imp_final["SE.PRM.ENRR.MA"] = base_imp_final["SE.PRM.ENRR.MA"].fillna(base_imp_mean["SE.PRM.ENRR.MA"])
base_imp_final["SE.PRM.ENRR.MA"] = base_imp_final["SE.PRM.ENRR.MA"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['primary_male_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SE.PRM.ENRR.MA": 'primary_male'})
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","primary_male",'primary_male_id']],on=["year","gw_codes"],how="left")

### School enrollment, secondary, female (% gross) ###

# Create base df (on year level), perform multiple imputation
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NE.EXP.GNFS.ZS","NE.IMP.GNFS.ZS","SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"]],on=["year","gw_codes"],how="left")
# Use as additional variables primary school enrollment male and female, secondary school enrollment male,  tertiary school enrollment male and female
base_imp=multivariate_imp_bayes(base,"country",["SE.SEC.ENRR.FE"],vars_add=["SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["SE.SEC.ENRR.FE"])
base_imp_mean=simple_imp_grouped(base,"country",["SE.SEC.ENRR.FE"])
base_imp_final["SE.SEC.ENRR.FE"] = base_imp_final["SE.SEC.ENRR.FE"].fillna(base_imp_mean["SE.SEC.ENRR.FE"])
base_imp_final["SE.SEC.ENRR.FE"] = base_imp_final["SE.SEC.ENRR.FE"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['second_female_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SE.SEC.ENRR.FE": 'second_female'})
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","second_female",'second_female_id']],on=["year","gw_codes"],how="left")

### School enrollment, secondary, male (% gross) ###

# Create base df (on year level), perform multiple imputation
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NE.EXP.GNFS.ZS","NE.IMP.GNFS.ZS","SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"]],on=["year","gw_codes"],how="left")
# Use as additional variables primary school enrollment male and female, secondary school enrollment female,  tertiary school enrollment male and female
base_imp=multivariate_imp_bayes(base,"country",["SE.SEC.ENRR.MA"],vars_add=["SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.TER.ENRR.FE","SE.TER.ENRR.MA"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["SE.SEC.ENRR.MA"])
base_imp_mean=simple_imp_grouped(base,"country",["SE.SEC.ENRR.MA"])
base_imp_final["SE.SEC.ENRR.MA"] = base_imp_final["SE.SEC.ENRR.MA"].fillna(base_imp_mean["SE.SEC.ENRR.MA"])
base_imp_final["SE.SEC.ENRR.MA"] = base_imp_final["SE.SEC.ENRR.MA"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['second_male_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SE.SEC.ENRR.MA": 'second_male'})
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","second_male",'second_male_id']],on=["year","gw_codes"],how="left")

### School enrollment, tertiary, female (% gross) ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NE.EXP.GNFS.ZS","NE.IMP.GNFS.ZS","SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"]],on=["year","gw_codes"],how="left")
# Use as additional variables primary school enrollment male and female, secondary school enrollment male and female,  tertiary school enrollment male 
base_imp=multivariate_imp_bayes(base,"country",["SE.TER.ENRR.FE"],vars_add=["SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.MA"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["SE.TER.ENRR.FE"])
base_imp_mean=simple_imp_grouped(base,"country",["SE.TER.ENRR.FE"])
base_imp_final["SE.TER.ENRR.FE"] = base_imp_final["SE.TER.ENRR.FE"].fillna(base_imp_mean["SE.TER.ENRR.FE"])
base_imp_final["SE.TER.ENRR.FE"] = base_imp_final["SE.TER.ENRR.FE"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['tert_female_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SE.TER.ENRR.FE": 'tert_female'})
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","tert_female",'tert_female_id']],on=["year","gw_codes"],how="left")

### School enrollment, tertiary, male (% gross) ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=economy[["year","gw_codes","NE.EXP.GNFS.ZS","NE.IMP.GNFS.ZS","SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"]],on=["year","gw_codes"],how="left")
# Use as additional variables primary school enrollment male and female, secondary school enrollment male and female,  tertiary school enrollment female 
base_imp=multivariate_imp_bayes(base,"country",["SE.TER.ENRR.MA"],vars_add=["SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["SE.TER.ENRR.MA"])
base_imp_mean=simple_imp_grouped(base,"country",["SE.TER.ENRR.MA"])
base_imp_final["SE.TER.ENRR.MA"] = base_imp_final["SE.TER.ENRR.MA"].fillna(base_imp_mean["SE.TER.ENRR.MA"])
base_imp_final["SE.TER.ENRR.MA"] = base_imp_final["SE.TER.ENRR.MA"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['tert_male_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SE.TER.ENRR.MA": 'tert_male'})
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","tert_male",'tert_male_id']],on=["year","gw_codes"],how="left")

### Expected years of schooling ###

# Create base df (on year level), perform multiple imputation 
hdi=pd.read_csv("data/data_out/hdi_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=hdi[["year","gw_codes",'eys','eys_male','eys_female','mys','mys_male','mys_female']],on=["year","gw_codes"],how="left")
# Use as additional variables expected schooling male and female
base_imp=multivariate_imp_bayes(base,"country",["eys"],vars_add=['eys_male','eys_female'],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["eys"])
base_imp_mean=simple_imp_grouped(base,"country",["eys"])
base_imp_final["eys"] = base_imp_final["eys"].fillna(base_imp_mean["eys"])
base_imp_final["eys"] = base_imp_final["eys"].fillna(base_imp["imp"])

# Add missing id and merge
base_imp_final['eys_id'] = base_imp["missing_id"]
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","eys",'eys_id']],on=["year","gw_codes"],how="left")

### Expected years of schooling, male ###

# Create base df (on year level), perform multiple imputation 
hdi=pd.read_csv("data/data_out/hdi_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=hdi[["year","gw_codes",'eys','eys_male','eys_female','mys','mys_male','mys_female']],on=["year","gw_codes"],how="left")
# Use as additional variables expected schooling total and female
base_imp=multivariate_imp_bayes(base,"country",["eys_male"],vars_add=['eys','eys_female'],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["eys_male"])
base_imp_mean=simple_imp_grouped(base,"country",["eys_male"])
base_imp_final["eys_male"] = base_imp_final["eys_male"].fillna(base_imp_mean["eys_male"])
base_imp_final["eys_male"] = base_imp_final["eys_male"].fillna(base_imp["imp"])

# Add missing id and merge
base_imp_final['eys_male_id'] = base_imp["missing_id"]
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","eys_male",'eys_male_id']],on=["year","gw_codes"],how="left")

### Expected years of schooling, female ###

# Create base df (on year level), perform multiple imputation 
hdi=pd.read_csv("data/data_out/hdi_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=hdi[["year","gw_codes",'eys','eys_male','eys_female','mys','mys_male','mys_female']],on=["year","gw_codes"],how="left")
# Use as additional variables expected schooling total and male
base_imp=multivariate_imp_bayes(base,"country",["eys_female"],vars_add=['eys','eys_male'],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["eys_female"])
base_imp_mean=simple_imp_grouped(base,"country",["eys_female"])
base_imp_final["eys_female"] = base_imp_final["eys_female"].fillna(base_imp_mean["eys_female"])
base_imp_final["eys_female"] = base_imp_final["eys_female"].fillna(base_imp["imp"])

# Add missing id and merge
base_imp_final['eys_female_id'] = base_imp["missing_id"]
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","eys_female",'eys_female_id']],on=["year","gw_codes"],how="left")

### Mean years of schooling ###

# Create base df (on year level), perform multiple imputation 
hdi=pd.read_csv("data/data_out/hdi_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=hdi[["year","gw_codes",'eys','eys_male','eys_female','mys','mys_male','mys_female']],on=["year","gw_codes"],how="left")
# Use as additional variables mean schooling male and female
base_im=multivariate_imp_bayes(base,"country",["mys"],vars_add=['mys_male','mys_female'],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["mys"])
base_imp_mean=simple_imp_grouped(base,"country",["mys"])
base_imp_final["mys"] = base_imp_final["mys"].fillna(base_imp_mean["mys"])
base_imp_final["mys"] = base_imp_final["mys"].fillna(base_imp["imp"])

# Add missing id and merge
base_imp_final['mys_id'] = base_imp["missing_id"]
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","mys",'mys_id']],on=["year","gw_codes"],how="left")

### Mean years of schooling, male ###

# Create base df (on year level), perform multiple imputation 
hdi=pd.read_csv("data/data_out/hdi_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=hdi[["year","gw_codes",'eys','eys_male','eys_female','mys','mys_male','mys_female']],on=["year","gw_codes"],how="left")
# Use as additional variables mean schooling total and female
base_imp=multivariate_imp_bayes(base,"country",["mys_male"],vars_add=['mys','mys_female'],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["mys_male"])
base_imp_mean=simple_imp_grouped(base,"country",["mys_male"])
base_imp_final["mys_male"] = base_imp_final["mys_male"].fillna(base_imp_mean["mys_male"])
base_imp_final["mys_male"] = base_imp_final["mys_male"].fillna(base_imp["imp"])

# Add missing id and merge
base_imp_final['mys_male_id'] = base_imp["missing_id"]
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","mys_male",'mys_male_id']],on=["year","gw_codes"],how="left")

### Mean years of schooling, female ###

# Create base df (on year level), perform multiple imputation 
hdi=pd.read_csv("data/data_out/hdi_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=hdi[["year","gw_codes",'eys','eys_male','eys_female','mys','mys_male','mys_female']],on=["year","gw_codes"],how="left")
# Use as additional variables mean schooling total and male
base_imp=multivariate_imp_bayes(base,"country",["mys_female"],vars_add=['mys','mys_male'],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["mys_female"])
base_imp_mean=simple_imp_grouped(base,"country",["mys_female"])
base_imp_final["mys_female"] = base_imp_final["mys_female"].fillna(base_imp_mean["mys_female"])
base_imp_final["mys_female"] = base_imp_final["mys_female"].fillna(base_imp["imp"])

# Add missing id and merge
base_imp_final['mys_female_id'] = base_imp["missing_id"]
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","mys_female",'mys_female_id']],on=["year","gw_codes"],how="left")

# Final df
print(df_econ.isna().any().any())
print(df_econ.min())
print(df_econ.duplicated(subset=['dd',"country","gw_codes"]).any())
print(df_econ.duplicated(subset=['dd',"country"]).any())
print(df_econ.duplicated(subset=['dd',"gw_codes"]).any())

# Check datatypes 
for c in df_econ.columns: 
    print(c,df_econ[c].dtypes)

# Save 
df_econ.to_csv("out/df_econ_full_cm.csv")  

###############################
### Regime and policy theme ###
###############################

# Initiate
df_pol=df_out[["year","dd","gw_codes","country"]].copy()

# Load wb data, previously retrived with the WB api
pol=pd.read_csv("data/pol_wb.csv",index_col=0)
print(pol.min())

### Armed forces personnel (% of total labor force) ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")
# Use military expenditure as additional variable
base_imp=multivariate_imp_bayes(base,"country",["MS.MIL.TOTL.TF.ZS"],vars_add=["MS.MIL.XPND.GD.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["MS.MIL.TOTL.TF.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["MS.MIL.TOTL.TF.ZS"])
base_imp_final["MS.MIL.TOTL.TF.ZS"] = base_imp_final["MS.MIL.TOTL.TF.ZS"].fillna(base_imp_mean["MS.MIL.TOTL.TF.ZS"])
base_imp_final["MS.MIL.TOTL.TF.ZS"] = base_imp_final["MS.MIL.TOTL.TF.ZS"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['armedforces_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"MS.MIL.TOTL.TF.ZS": 'armedforces_share'})
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","armedforces_share",'armedforces_share_id']],on=["year","gw_codes"],how="left")

### Military expenditure (% of GDP) ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")
# Use armed forces personnel as additional variable
base_imp=multivariate_imp_bayes(base,"country",["MS.MIL.XPND.GD.ZS"],vars_add=["MS.MIL.TOTL.TF.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["MS.MIL.XPND.GD.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["MS.MIL.XPND.GD.ZS"])
base_imp_final["MS.MIL.XPND.GD.ZS"] = base_imp_final["MS.MIL.XPND.GD.ZS"].fillna(base_imp_mean["MS.MIL.XPND.GD.ZS"])
base_imp_final["MS.MIL.XPND.GD.ZS"] = base_imp_final["MS.MIL.XPND.GD.ZS"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['milex_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"MS.MIL.XPND.GD.ZS": 'milex_share'})
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","milex_share",'milex_share_id']],on=["year","gw_codes"],how="left")

### Control of Corruption: Estimate ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")
# Use other QoG indicators and tax revenue as additional variables
base_imp=multivariate_imp_bayes(base,"country",["CC.EST"],vars_add=["GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS"],max_iter=10,min_val=-10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["CC.EST"])
base_imp_mean=simple_imp_grouped(base,"country",["CC.EST"])
base_imp_final["CC.EST"] = base_imp_final["CC.EST"].fillna(base_imp_mean["CC.EST"])
base_imp_final["CC.EST"] = base_imp_final["CC.EST"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['corruption_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"CC.EST": 'corruption'})
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","corruption",'corruption_id']],on=["year","gw_codes"],how="left")

### Government Effectiveness: Estimate ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")
# Use other QoG indicators and tax revenue as additional variables
base_imp=multivariate_imp_bayes(base,"country",["GE.EST"],vars_add=["CC.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS"],max_iter=10,min_val=-10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["GE.EST"])
base_imp_mean=simple_imp_grouped(base,"country",["GE.EST"])
base_imp_final["GE.EST"] = base_imp_final["GE.EST"].fillna(base_imp_mean["GE.EST"])
base_imp_final["GE.EST"] = base_imp_final["GE.EST"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['effectiveness_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"GE.EST": 'effectiveness'})
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","effectiveness",'effectiveness_id']],on=["year","gw_codes"],how="left")

### Political Stability and Absence of Violence/Terrorism: Estimate ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")
# Use other QoG indicators and tax revenue as additional variables
base_imp=multivariate_imp_bayes(base,"country",["PV.EST"],vars_add=["CC.EST","GE.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS"],max_iter=10,min_val=-10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["PV.EST"])
base_imp_mean=simple_imp_grouped(base,"country",["PV.EST"])
base_imp_final["PV.EST"] = base_imp_final["PV.EST"].fillna(base_imp_mean["PV.EST"])
base_imp_final["PV.EST"] = base_imp_final["PV.EST"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['polvio_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"PV.EST": 'polvio'})
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","polvio",'polvio_id']],on=["year","gw_codes"],how="left")

### Regulatory Quality: Estimate ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")
# Use other QoG indicators and tax revenue as additional variables
base_imp=multivariate_imp_bayes(base,"country",["RQ.EST"],vars_add=["CC.EST","GE.EST","PV.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS"],max_iter=10,min_val=-10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["RQ.EST"])
base_imp_mean=simple_imp_grouped(base,"country",["RQ.EST"])
base_imp_final["RQ.EST"] = base_imp_final["RQ.EST"].fillna(base_imp_mean["RQ.EST"])
base_imp_final["RQ.EST"] = base_imp_final["RQ.EST"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['regu_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"RQ.EST": 'regu'})
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","regu",'regu_id']],on=["year","gw_codes"],how="left")

### Rule of Law: Estimate ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")
# Use other QoG indicators and tax revenue as additional variables
base_imp=multivariate_imp_bayes(base,"country",["RL.EST"],vars_add=["CC.EST","GE.EST","PV.EST","RQ.EST","VA.EST","GC.TAX.TOTL.GD.ZS"],max_iter=10,min_val=-10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["RL.EST"])
base_imp_mean=simple_imp_grouped(base,"country",["RL.EST"])
base_imp_final["RL.EST"] = base_imp_final["RL.EST"].fillna(base_imp_mean["RL.EST"])
base_imp_final["RL.EST"] = base_imp_final["RL.EST"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['law_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"RL.EST": 'law'})
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","law",'law_id']],on=["year","gw_codes"],how="left")

### Voice and Accountability: Estimate ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")
# Use other QoG indicators and tax revenue as additional variables
base_imp=multivariate_imp_bayes(base,"country",["VA.EST"],vars_add=["CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","GC.TAX.TOTL.GD.ZS"],max_iter=10,min_val=-10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["VA.EST"])
base_imp_mean=simple_imp_grouped(base,"country",["VA.EST"])
base_imp_final["VA.EST"] = base_imp_final["VA.EST"].fillna(base_imp_mean["VA.EST"])
base_imp_final["VA.EST"] = base_imp_final["VA.EST"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['account_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"VA.EST": 'account'})
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","account",'account_id']],on=["year","gw_codes"],how="left")

### Tax revenue (% of GDP) ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")
# Use QoG indicators as additional variables
base_imp=multivariate_imp_bayes(base,"country",["GC.TAX.TOTL.GD.ZS"],vars_add=["CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["GC.TAX.TOTL.GD.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["GC.TAX.TOTL.GD.ZS"])
base_imp_final["GC.TAX.TOTL.GD.ZS"] = base_imp_final["GC.TAX.TOTL.GD.ZS"].fillna(base_imp_mean["GC.TAX.TOTL.GD.ZS"])
base_imp_final["GC.TAX.TOTL.GD.ZS"] = base_imp_final["GC.TAX.TOTL.GD.ZS"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['tax_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"GC.TAX.TOTL.GD.ZS": 'tax'})
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","tax",'tax_id']],on=["year","gw_codes"],how="left")

### Fixed broadband subscriptions (per 100 people) ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")
# Use as additional variables telephone subscriptions, Internet access, and mobile phone subscriptions
base_imp=multivariate_imp_bayes(base,"country",["IT.NET.BBND.P2"],vars_add=["IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["IT.NET.BBND.P2"])
base_imp_mean=simple_imp_grouped(base,"country",["IT.NET.BBND.P2"])
base_imp_final["IT.NET.BBND.P2"] = base_imp_final["IT.NET.BBND.P2"].fillna(base_imp_mean["IT.NET.BBND.P2"])
base_imp_final["IT.NET.BBND.P2"] = base_imp_final["IT.NET.BBND.P2"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['broadband_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"IT.NET.BBND.P2": 'broadband'})
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","broadband",'broadband_id']],on=["year","gw_codes"],how="left")

### Fixed telephone subscriptions (per 100 people) ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")
# Use as additional variables broadband subscriptions, Internet access, and mobile subsciptions
base_imp=multivariate_imp_bayes(base,"country",["IT.MLT.MAIN.P2"],vars_add=["IT.NET.BBND.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["IT.MLT.MAIN.P2"])
base_imp_mean=simple_imp_grouped(base,"country",["IT.MLT.MAIN.P2"])
base_imp_final["IT.MLT.MAIN.P2"] = base_imp_final["IT.MLT.MAIN.P2"].fillna(base_imp_mean["IT.MLT.MAIN.P2"])
base_imp_final["IT.MLT.MAIN.P2"] = base_imp_final["IT.MLT.MAIN.P2"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['telephone_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"IT.MLT.MAIN.P2": 'telephone'})
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","telephone",'telephone_id']],on=["year","gw_codes"],how="left")

### Individuals using the Internet (% of population) ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")
# Use as additional variables broadband, telephone and mobile subsciptions
base_imp=multivariate_imp_bayes(base,"country",["IT.NET.USER.ZS"],vars_add=["IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.CEL.SETS.P2"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["IT.NET.USER.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["IT.NET.USER.ZS"])
base_imp_final["IT.NET.USER.ZS"] = base_imp_final["IT.NET.USER.ZS"].fillna(base_imp_mean["IT.NET.USER.ZS"])
base_imp_final["IT.NET.USER.ZS"] = base_imp_final["IT.NET.USER.ZS"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['internet_use_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"IT.NET.USER.ZS": 'internet_use'})
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","internet_use",'internet_use_id']],on=["year","gw_codes"],how="left")

### Mobile cellular subscriptions (per 100 people) ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")
# Use as additional variables broadband, telephone and Internat
base_imp=multivariate_imp_bayes(base,"country",["IT.CEL.SETS.P2"],vars_add=["IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["IT.CEL.SETS.P2"])
base_imp_mean=simple_imp_grouped(base,"country",["IT.CEL.SETS.P2"])
base_imp_final["IT.CEL.SETS.P2"] = base_imp_final["IT.CEL.SETS.P2"].fillna(base_imp_mean["IT.CEL.SETS.P2"])
base_imp_final["IT.CEL.SETS.P2"] = base_imp_final["IT.CEL.SETS.P2"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['mobile_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"IT.CEL.SETS.P2": 'mobile'})
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","mobile",'mobile_id']],on=["year","gw_codes"],how="left")

### Electoral democracy index ###

vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=vdem[["year",
                                    "gw_codes",
                                    "v2x_polyarchy", # Electoral democracy index
                                    "v2x_libdem", # Liberal democracy index
                                    "v2x_partipdem", # Participatory democracy index
                                    "v2x_delibdem", # Deliberative democracy index
                                    "v2x_egaldem", # Egalitarian democracy index                              
                                    "v2x_civlib", # Civil liberties index
                                    "v2x_clphy", # Physical violence index
                                    "v2x_clpol", # Political civil liberties index
                                    "v2x_clpriv", # Private civil liberties index                               
                                    "v2xpe_exlecon", # Exclusion by Socio-Economic Group
                                    "v2xpe_exlgender", # Exclusion by Gender index
                                    "v2xpe_exlgeo", # Exclusion by Urban-Rural Location index
                                    "v2xpe_exlpol", # Exclusion by Political Group index
                                    "v2xpe_exlsocgr", # Exclusion by Social Group index
                                    "v2smgovshut", # Government Internet shut down in practice
                                    "v2smgovfilprc" # Government Internet filtering in practice
                                    ]],
              on=["year","gw_codes"],how="left")

base = base.rename(columns={"v2x_polyarchy": 'polyarchy'})
df_pol=pd.merge(left=df_pol,right=base[["year","gw_codes","polyarchy"]],on=["year","gw_codes"],how="left")

### Liberal democracy index ###

# Create base df (on year level)
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")

# Simple imputation (linear and if it fails mean) 
base_imp_final=linear_imp_grouped(base,"country",["v2x_libdem"])
base_imp_mean=simple_imp_grouped(base,"country",["v2x_libdem"])
base_imp_final["v2x_libdem"] = base_imp_final["v2x_libdem"].fillna(base_imp_mean["v2x_libdem"])

# Add missing id, rename variable and merge
base_imp_final['libdem_id'] = base["v2x_libdem"].isnull().astype(int)
base_imp_final = base_imp_final.rename(columns={"v2x_libdem": 'libdem'})
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","libdem",'libdem_id']],on=["year","gw_codes"],how="left")

### Participatory democracy index ###

vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
base = base.rename(columns={"v2x_partipdem": 'partipdem'})
df_pol=pd.merge(left=df_pol,right=base[["year","gw_codes","partipdem"]],on=["year","gw_codes"],how="left")

### Deliberative democracy index ###

vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
base = base.rename(columns={"v2x_delibdem": 'delibdem'})
df_pol=pd.merge(left=df_pol,right=base[["year","gw_codes","delibdem"]],on=["year","gw_codes"],how="left")

### Egalitarian democracy index ###

vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
base = base.rename(columns={"v2x_egaldem": 'egaldem'})
df_pol=pd.merge(left=df_pol,right=base[["year","gw_codes","egaldem"]],on=["year","gw_codes"],how="left")

### Civil liberties index ###

vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
base = base.rename(columns={"v2x_civlib": 'civlib'})
df_pol=pd.merge(left=df_pol,right=base[["year","gw_codes","civlib"]],on=["year","gw_codes"],how="left")

### Physical violence index ###

vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
base = base.rename(columns={"v2x_clphy": 'phyvio'})
df_pol=pd.merge(left=df_pol,right=base[["year","gw_codes","phyvio"]],on=["year","gw_codes"],how="left")

### Political civil liberties index ###

vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
base = base.rename(columns={"v2x_clpol": 'pollib'})
df_pol=pd.merge(left=df_pol,right=base[["year","gw_codes","pollib"]],on=["year","gw_codes"],how="left")

### Private civil liberties index ###

vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
base = base.rename(columns={"v2x_clpriv": 'privlib'})
df_pol=pd.merge(left=df_pol,right=base[["year","gw_codes","privlib"]],on=["year","gw_codes"],how="left")

### Exclusion by Socio-Economic Group index ###

# Create base df (on year level), perform multiple imputation 
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
# Use other exclusion indicators as additional variables
base_imp=multivariate_imp_bayes(base,"country",["v2xpe_exlecon"],vars_add=["v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["v2xpe_exlecon"])
base_imp_mean=simple_imp_grouped(base,"country",["v2xpe_exlecon"])
base_imp_final["v2xpe_exlecon"] = base_imp_final["v2xpe_exlecon"].fillna(base_imp_mean["v2xpe_exlecon"])
base_imp_final["v2xpe_exlecon"] = base_imp_final["v2xpe_exlecon"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['execon_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"v2xpe_exlecon": 'execon'})
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","execon",'execon_id']],on=["year","gw_codes"],how="left")

### Exclusion by Gender index ###

# Create base df (on year level), perform multiple imputation 
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
# Use other exclusion indicators as additional variables
base_imp=multivariate_imp_bayes(base,"country",["v2xpe_exlgender"],vars_add=["v2xpe_exlecon","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["v2xpe_exlgender"])
base_imp_mean=simple_imp_grouped(base,"country",["v2xpe_exlgender"])
base_imp_final["v2xpe_exlgender"] = base_imp_final["v2xpe_exlgender"].fillna(base_imp_mean["v2xpe_exlgender"])
base_imp_final["v2xpe_exlgender"] = base_imp_final["v2xpe_exlgender"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['exgender_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"v2xpe_exlgender": 'exgender'})
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","exgender",'exgender_id']],on=["year","gw_codes"],how="left")

### Exclusion by Urban-Rural Location index ###

# Create base df (on year level), perform multiple imputation 
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
# Use other exclusion indicators as additional variables
base_imp=multivariate_imp_bayes(base,"country",["v2xpe_exlgeo"],vars_add=["v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlpol","v2xpe_exlsocgr"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["v2xpe_exlgeo"])
base_imp_mean=simple_imp_grouped(base,"country",["v2xpe_exlgeo"])
base_imp_final["v2xpe_exlgeo"] = base_imp_final["v2xpe_exlgeo"].fillna(base_imp_mean["v2xpe_exlgeo"])
base_imp_final["v2xpe_exlgeo"] = base_imp_final["v2xpe_exlgeo"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['exgeo_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"v2xpe_exlgeo": 'exgeo'})
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","exgeo",'exgeo_id']],on=["year","gw_codes"],how="left")

### Exclusion by Political Group index ###

# Create base df (on year level), perform multiple imputation 
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
# Use other exclusion indicators as additional variables
base_imp=multivariate_imp_bayes(base,"country",["v2xpe_exlpol"],vars_add=["v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlsocgr"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["v2xpe_exlpol"])
base_imp_mean=simple_imp_grouped(base,"country",["v2xpe_exlpol"])
base_imp_final["v2xpe_exlpol"] = base_imp_final["v2xpe_exlpol"].fillna(base_imp_mean["v2xpe_exlpol"])
base_imp_final["v2xpe_exlpol"] = base_imp_final["v2xpe_exlpol"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['expol_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"v2xpe_exlpol": 'expol'})
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","expol",'expol_id']],on=["year","gw_codes"],how="left")

### Exclusion by Social Group index ###

# Create base df (on year level), perform multiple imputation 
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
# Use other exclusion indicators as additional variables
base_imp=multivariate_imp_bayes(base,"country",["v2xpe_exlsocgr"],vars_add=["v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["v2xpe_exlsocgr"])
base_imp_mean=simple_imp_grouped(base,"country",["v2xpe_exlsocgr"])
base_imp_final["v2xpe_exlsocgr"] = base_imp_final["v2xpe_exlsocgr"].fillna(base_imp_mean["v2xpe_exlsocgr"])
base_imp_final["v2xpe_exlsocgr"] = base_imp_final["v2xpe_exlsocgr"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['exsoc_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"v2xpe_exlsocgr": 'exsoc'})
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","exsoc",'exsoc_id']],on=["year","gw_codes"],how="left")

### Government Internet shut down in practice ###

# Create base df (on year level), perform multiple imputation 
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
# Use Internet filter as additional variable
base_imp=multivariate_imp_bayes(base,"country",["v2smgovshut"],vars_add=["v2smgovfilprc"],max_iter=10,min_val=-10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["v2smgovshut"])
base_imp_mean=simple_imp_grouped(base,"country",["v2smgovshut"])
base_imp_final["v2smgovshut"] = base_imp_final["v2smgovshut"].fillna(base_imp_mean["v2smgovshut"])
base_imp_final["v2smgovshut"] = base_imp_final["v2smgovshut"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['shutdown_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"v2smgovshut": 'shutdown'})
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","shutdown",'shutdown_id']],on=["year","gw_codes"],how="left")

### Government Internet filtering in practice ###

# Create base df (on year level), perform multiple imputation 
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
# Use Internet shut down as additional variable
base_imp=multivariate_imp_bayes(base,"country",["v2smgovfilprc"],vars_add=["v2smgovshut"],max_iter=10,min_val=-10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["v2smgovfilprc"])
base_imp_mean=simple_imp_grouped(base,"country",["v2smgovfilprc"])
base_imp_final["v2smgovfilprc"] = base_imp_final["v2smgovfilprc"].fillna(base_imp_mean["v2smgovfilprc"])
base_imp_final["v2smgovfilprc"] = base_imp_final["v2smgovfilprc"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['filter_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"v2smgovfilprc": 'filter'})
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","filter",'filter_id']],on=["year","gw_codes"],how="left")

### Number of months that leader has been in power ###

reign=pd.read_csv("data/data_out/reign_cm.csv",index_col=0)
print(reign.dtypes)
base=df_out[["year","dd","gw_codes","country"]].copy()
base=pd.merge(left=base,right=reign[["dd",
                                    "gw_codes",
                                    "tenure_months", # Number of months that leader has been in power 
                                    "dem_duration", # Logged number of months that a country is democratic
                                    "election_recent", # Election for leadership taking place in previous six months
                                    "lastelection" # Time since the last election for leadership (decay function)
                                    ]],on=["dd","gw_codes"],how="left")

base_imp_final=linear_imp_grouped(base,"country",["tenure_months"])
base_imp_mean=simple_imp_grouped(base,"country",["tenure_months"])
base_imp_final["tenure_months"] = base_imp_final["tenure_months"].fillna(base_imp_mean["tenure_months"])
base_imp_final['tenure_months_id'] = base["tenure_months"].isnull().astype(int)

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["dd"].loc[base["country"]==c], base["tenure_months"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["dd"].loc[base_imp_final["country"]==c], base_imp_final["tenure_months"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["dd","gw_codes","tenure_months",'tenure_months_id']],on=["dd","gw_codes"],how="left")
  
### Logged number of months that a country is democratic ###

reign=pd.read_csv("data/data_out/reign_cm.csv",index_col=0)
print(reign.dtypes)
base=df_out[["year","dd","gw_codes","country"]].copy()
base=pd.merge(left=base,right=reign[["dd","gw_codes","tenure_months","dem_duration","election_recent","lastelection"]],on=["dd","gw_codes"],how="left")
base_imp_final=linear_imp_grouped(base,"country",["dem_duration"])
base_imp_mean=simple_imp_grouped(base,"country",["dem_duration"])
base_imp_final["dem_duration"] = base_imp_final["dem_duration"].fillna(base_imp_mean["dem_duration"])
base_imp_final['dem_duration_id'] = base["dem_duration"].isnull().astype(int)

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["dd"].loc[base["country"]==c], base["dem_duration"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["dd"].loc[base_imp_final["country"]==c], base_imp_final["dem_duration"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["dd","gw_codes","dem_duration",'dem_duration_id']],on=["dd","gw_codes"],how="left")

### Election for leadership taking place in that year ###

reign=pd.read_csv("data/data_out/reign_cm.csv",index_col=0)
print(reign.dtypes)
base=df_out[["year","dd","gw_codes","country"]].copy()
base=pd.merge(left=base,right=reign[["dd","gw_codes","tenure_months","dem_duration","election_recent","lastelection"]],on=["dd","gw_codes"],how="left")
base_imp_final=linear_imp_grouped(base,"country",["election_recent"])
base_imp_mean=simple_imp_grouped(base,"country",["election_recent"])
base_imp_final["election_recent"] = base_imp_final["election_recent"].fillna(base_imp_mean["election_recent"])
base_imp_final['election_recent_id'] = base["election_recent"].isnull().astype(int)

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["dd"].loc[base["country"]==c], base["election_recent"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["dd"].loc[base_imp_final["country"]==c], base_imp_final["election_recent"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()
    
# Dichotomize post hoc
base_imp_final.loc[base_imp_final["election_recent"]>0.5,"election_recent"]=1
base_imp_final.loc[base_imp_final["election_recent"]<=0.5,"election_recent"]=0
base_imp_final["election_recent"].unique()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["dd","gw_codes","election_recent",'election_recent_id']],on=["dd","gw_codes"],how="left")

### Time since the last election for leadership (decay function) ###

reign=pd.read_csv("data/data_out/reign_cm.csv",index_col=0)
print(reign.dtypes)
base=df_out[["year","dd","gw_codes","country"]].copy()
base=pd.merge(left=base,right=reign[["dd","gw_codes","tenure_months","dem_duration","election_recent","lastelection"]],on=["dd","gw_codes"],how="left")
base_imp_final=linear_imp_grouped(base,"country",["lastelection"])
base_imp_mean=simple_imp_grouped(base,"country",["lastelection"])
base_imp_final["lastelection"] = base_imp_final["lastelection"].fillna(base_imp_mean["lastelection"])
base_imp_final['lastelection_id'] = base["lastelection"].isnull().astype(int)

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["dd"].loc[base["country"]==c], base["lastelection"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["dd"].loc[base_imp_final["country"]==c], base_imp_final["lastelection"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["dd","gw_codes","lastelection",'lastelection_id']],on=["dd","gw_codes"],how="left")
  
# Final df
print(df_pol.isna().any().any())
print(df_pol.min())
print(df_pol.duplicated(subset=['dd',"country","gw_codes"]).any())
print(df_pol.duplicated(subset=['dd',"country"]).any())
print(df_pol.duplicated(subset=['dd',"gw_codes"]).any())

# Check datatypes and convert floats to integer
for c in df_pol.columns: 
    print(c,df_pol[c].dtypes)
df_pol['election_recent']=df_pol['election_recent'].astype('int64')

# Note that tenure_months is theoretically an integer variable, but the imputation
# returns floats when interpolating. Due to the imputation, the variables is kept as float. 
df_pol['tenure_months'].unique() # Check levels
  
# Save
df_pol.to_csv("out/df_pol_full_cm.csv") 

##############################################
### Geography, environment & climate theme ###
##############################################

# Initiate
df_geog=df_out[["year","dd","gw_codes","country"]].copy()

# Load wb data, previously retrived with the WB api
geog=pd.read_csv("data/geog_wb.csv",index_col=0)
print(geog.min())

### Land area (sq. km) ###

base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=geog[["year","gw_codes","AG.LND.TOTL.K2","AG.LND.FRST.ZS","AG.LND.PRCP.MM","ER.H2O.FWST.ZS","EN.GHG.CO2.MT.CE.AR5","AG.LND.AGRI.ZS","AG.LND.ARBL.ZS"]],on=["year","gw_codes"],how="left")
base_imp_final=linear_imp_grouped(base,"country",["AG.LND.TOTL.K2"])
base_imp_mean=simple_imp_grouped(base,"country",["AG.LND.TOTL.K2"])
base_imp_final["AG.LND.TOTL.K2"] = base_imp_final["AG.LND.TOTL.K2"].fillna(base_imp_mean["AG.LND.TOTL.K2"])
base_imp_final = base_imp_final.rename(columns={"AG.LND.TOTL.K2": 'land'})
base_imp_final['land_id'] = base["AG.LND.TOTL.K2"].isnull().astype(int)
df_geog=pd.merge(left=df_geog,right=base_imp_final[["year","gw_codes","land",'land_id']],on=["year","gw_codes"],how="left")

### Average Mean Surface Air Temperature ###

temp=pd.read_csv("data/data_out/temp_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=temp[["year","gw_codes","temp"]],on=["year","gw_codes"],how="left")  
base_imp_final=linear_imp_grouped(base,"country",["temp"])
base_imp_mean=simple_imp_grouped(base,"country",["temp"])
base_imp_final["temp"] = base_imp_final["temp"].fillna(base_imp_mean["temp"])
base_imp_final['temp_id'] = base["temp"].isnull().astype(int)
df_geog=pd.merge(left=df_geog,right=base_imp_final[["year","gw_codes","temp",'temp_id']],on=["year","gw_codes"],how="left")

### Forest area (% of land area) ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=geog[["year","gw_codes","AG.LND.TOTL.K2","AG.LND.FRST.ZS","AG.LND.PRCP.MM","ER.H2O.FWST.ZS","EN.GHG.CO2.MT.CE.AR5","AG.LND.AGRI.ZS","AG.LND.ARBL.ZS"]],on=["year","gw_codes"],how="left")
# Use land area and CO2 emissions as additional variables
base_imp=multivariate_imp_bayes(base,"country",["AG.LND.FRST.ZS"],vars_add=["AG.LND.TOTL.K2","EN.GHG.CO2.MT.CE.AR5"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["AG.LND.FRST.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["AG.LND.FRST.ZS"])
base_imp_final["AG.LND.FRST.ZS"] = base_imp_final["AG.LND.FRST.ZS"].fillna(base_imp_mean["AG.LND.FRST.ZS"])
base_imp_final["AG.LND.FRST.ZS"] = base_imp_final["AG.LND.FRST.ZS"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['forest_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"AG.LND.FRST.ZS": 'forest'})
df_geog=pd.merge(left=df_geog,right=base_imp_final[["year","gw_codes","forest",'forest_id']],on=["year","gw_codes"],how="left")

### CO2 emissions (kt) ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=geog[["year","gw_codes","AG.LND.TOTL.K2","AG.LND.FRST.ZS","AG.LND.PRCP.MM","ER.H2O.FWST.ZS","EN.GHG.CO2.MT.CE.AR5","AG.LND.AGRI.ZS","AG.LND.ARBL.ZS"]],on=["year","gw_codes"],how="left")
# Use land area and forest area as additional variables
base_imp=multivariate_imp_bayes(base,"country",["EN.GHG.CO2.MT.CE.AR5"],vars_add=["AG.LND.TOTL.K2","AG.LND.FRST.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["EN.GHG.CO2.MT.CE.AR5"])
base_imp_mean=simple_imp_grouped(base,"country",["EN.GHG.CO2.MT.CE.AR5"])
base_imp_final["EN.GHG.CO2.MT.CE.AR5"] = base_imp_final["EN.GHG.CO2.MT.CE.AR5"].fillna(base_imp_mean["EN.GHG.CO2.MT.CE.AR5"])
base_imp_final["EN.GHG.CO2.MT.CE.AR5"] = base_imp_final["EN.GHG.CO2.MT.CE.AR5"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['co2_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"EN.GHG.CO2.MT.CE.AR5": 'co2'})
df_geog=pd.merge(left=df_geog,right=base_imp_final[["year","gw_codes","co2",'co2_id']],on=["year","gw_codes"],how="left")

### Average precipitation in depth (mm per year) ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=geog[["year","gw_codes","AG.LND.TOTL.K2","AG.LND.FRST.ZS","AG.LND.PRCP.MM","ER.H2O.FWST.ZS","EN.GHG.CO2.MT.CE.AR5","AG.LND.AGRI.ZS","AG.LND.ARBL.ZS"]],on=["year","gw_codes"],how="left")
# Use water stress as additional variable
base_imp=multivariate_imp_bayes(base,"country",["AG.LND.PRCP.MM"],vars_add=["ER.H2O.FWST.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["AG.LND.PRCP.MM"])
base_imp_mean=simple_imp_grouped(base,"country",["AG.LND.PRCP.MM"])
base_imp_final["AG.LND.PRCP.MM"] = base_imp_final["AG.LND.PRCP.MM"].fillna(base_imp_mean["AG.LND.PRCP.MM"])
base_imp_final["AG.LND.PRCP.MM"] = base_imp_final["AG.LND.PRCP.MM"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['percip_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"AG.LND.PRCP.MM": 'percip'})
df_geog=pd.merge(left=df_geog,right=base_imp_final[["year","gw_codes","percip",'percip_id']],on=["year","gw_codes"],how="left")

### Level of water stress: freshwater withdrawal as a proportion of available freshwater resources ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=geog[["year","gw_codes","AG.LND.TOTL.K2","AG.LND.FRST.ZS","AG.LND.PRCP.MM","ER.H2O.FWST.ZS","EN.GHG.CO2.MT.CE.AR5","AG.LND.AGRI.ZS","AG.LND.ARBL.ZS"]],on=["year","gw_codes"],how="left")
# Use percipitation as additional variable
base_imp=multivariate_imp_bayes(base,"country",["ER.H2O.FWST.ZS"],vars_add=["AG.LND.PRCP.MM"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["ER.H2O.FWST.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["ER.H2O.FWST.ZS"])
base_imp_final["ER.H2O.FWST.ZS"] = base_imp_final["ER.H2O.FWST.ZS"].fillna(base_imp_mean["ER.H2O.FWST.ZS"])
base_imp_final["ER.H2O.FWST.ZS"] = base_imp_final["ER.H2O.FWST.ZS"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['waterstress_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"ER.H2O.FWST.ZS": 'waterstress'})
df_geog=pd.merge(left=df_geog,right=base_imp_final[["year","gw_codes","waterstress",'waterstress_id']],on=["year","gw_codes"],how="left")

### Agricultural land (% of land area) ###

# Create base df (on year level), perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=geog[["year","gw_codes","AG.LND.TOTL.K2","AG.LND.FRST.ZS","AG.LND.PRCP.MM","ER.H2O.FWST.ZS","EN.GHG.CO2.MT.CE.AR5","AG.LND.AGRI.ZS","AG.LND.ARBL.ZS"]],on=["year","gw_codes"],how="left")
# Use arable land as additional variable
base_imp=multivariate_imp_bayes(base,"country",["AG.LND.AGRI.ZS"],vars_add=["AG.LND.ARBL.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["AG.LND.AGRI.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["AG.LND.AGRI.ZS"])
base_imp_final["AG.LND.AGRI.ZS"] = base_imp_final["AG.LND.AGRI.ZS"].fillna(base_imp_mean["AG.LND.AGRI.ZS"])
base_imp_final["AG.LND.AGRI.ZS"] = base_imp_final["AG.LND.AGRI.ZS"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['agri_land_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"AG.LND.AGRI.ZS": 'agri_land'})
df_geog=pd.merge(left=df_geog,right=base_imp_final[["year","gw_codes","agri_land",'agri_land_id']],on=["year","gw_codes"],how="left")

### Arable land (% of land area) ###

# Create base df (on year level), perform multiple imputation
base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
base=pd.merge(left=base,right=geog[["year","gw_codes","AG.LND.TOTL.K2","AG.LND.FRST.ZS","AG.LND.PRCP.MM","ER.H2O.FWST.ZS","EN.GHG.CO2.MT.CE.AR5","AG.LND.AGRI.ZS","AG.LND.ARBL.ZS"]],on=["year","gw_codes"],how="left")
# Use agricultural land as additional variable
base_imp=multivariate_imp_bayes(base,"country",["AG.LND.ARBL.ZS"],vars_add=["AG.LND.AGRI.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple
base_imp_final=linear_imp_grouped(base,"country",["AG.LND.ARBL.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["AG.LND.ARBL.ZS"])
base_imp_final["AG.LND.ARBL.ZS"] = base_imp_final["AG.LND.ARBL.ZS"].fillna(base_imp_mean["AG.LND.ARBL.ZS"])
base_imp_final["AG.LND.ARBL.ZS"] = base_imp_final["AG.LND.ARBL.ZS"].fillna(base_imp["imp"])

# Add missing id, rename variable and merge
base_imp_final['arable_land_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"AG.LND.ARBL.ZS": 'arable_land'})
df_geog=pd.merge(left=df_geog,right=base_imp_final[["year","gw_codes","arable_land",'arable_land_id']],on=["year","gw_codes"],how="left")

### Terrain Ruggedness Index ###

base=df_out[["year","gw_codes","country"]].copy()
base=base.drop_duplicates()
rug=pd.read_csv("data/data_out/rug_cy.csv",index_col=0)
base=pd.merge(left=base,right=rug[["year","gw_codes","rugged","soil","desert","tropical"]],on=["year","gw_codes"],how="left")
df_geog=pd.merge(left=df_geog,right=base[["year","gw_codes","rugged","soil","desert","tropical"]],on=["year","gw_codes"],how="left")

### Asia or Africa ###

rug=pd.read_csv("data/data_out/rug_cy.csv",index_col=0)
df_geog=pd.merge(left=df_geog,right=rug[["year","gw_codes","cont_africa","cont_asia"]],on=["year","gw_codes"],how="left")

# Manually add missings
df_geog.loc[df_geog["country"]=="Montenegro","cont_africa"]=0
df_geog.loc[df_geog["country"]=="Montenegro","cont_asia"]=0
df_geog.loc[df_geog["country"]=="Serbia","cont_africa"]=0
df_geog.loc[df_geog["country"]=="Serbia","cont_asia"]=0
df_geog.loc[df_geog["country"]=="South Sudan","cont_africa"]=1
df_geog.loc[df_geog["country"]=="South Sudan","cont_asia"]=0

# Final df
print(df_geog.isna().any())
print(df_geog.min())
print(df_geog.duplicated(subset=['dd',"country","gw_codes"]).any())
print(df_geog.duplicated(subset=['dd',"country"]).any())
print(df_geog.duplicated(subset=['dd',"gw_codes"]).any())

# Check datatypes and convert floats to integer
print(df_geog.dtypes)
df_geog['cont_africa']=df_geog['cont_africa'].astype('int64')
df_geog['cont_asia']=df_geog['cont_asia'].astype('int64')

# Save
df_geog.to_csv("out/df_geog_full_cm.csv")


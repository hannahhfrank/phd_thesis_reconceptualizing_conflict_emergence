import pandas as pd
from functions import dichotomize,lag_groupped,consec_zeros_grouped,exponential_growth,simple_imp_grouped,linear_imp_grouped,multivariate_imp_bayes
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
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

ucdp_sb=pd.read_csv("data/data_out/ucdp_cy_sb.csv",index_col=0)
ucdp_sb_s = ucdp_sb[["year","gw_codes","country","best","count"]][~ucdp_sb['gw_codes'].isin(list(micro_states.values())+list(exclude.values())+list(exclude2.values()))]
ucdp_sb_s.columns=["year","gw_codes","country","sb_fatalities","sb_event_counts"]

ucdp_osv=pd.read_csv("data/data_out/ucdp_cy_osv.csv",index_col=0)
ucdp_osv_s = ucdp_osv[["year","gw_codes","best","count"]][~ucdp_osv['gw_codes'].isin(list(micro_states.values())+list(exclude.values())+list(exclude2.values()))]
ucdp_osv_s.columns=["year","gw_codes","osv_fatalities","osv_event_counts"]

ucdp_ns=pd.read_csv("data/data_out/ucdp_cy_ns.csv",index_col=0)
ucdp_ns_s = ucdp_ns[["year","gw_codes","best","count"]][~ucdp_ns['gw_codes'].isin(list(micro_states.values())+list(exclude.values())+list(exclude2.values()))]
ucdp_ns_s.columns=["year","gw_codes","ns_fatalities","ns_event_counts"]

#############
### ACLED ###
#############

acled_protest=pd.read_csv("data/data_out/acled_cy_protest.csv",index_col=0)
acled_protest_s = acled_protest[["year","gw_codes","n_protest_events","fatalities"]][~acled_protest['gw_codes'].isin(list(micro_states.values())+list(exclude.values())+list(exclude2.values()))]
acled_protest_s.columns=["year","gw_codes","protest_event_counts","protest_fatalities"]

acled_riots=pd.read_csv("data/data_out/acled_cy_riots.csv",index_col=0)
acled_riots_s = acled_riots[["year","gw_codes","n_riot_events","fatalities"]][~acled_riots['gw_codes'].isin(list(micro_states.values())+list(exclude.values())+list(exclude2.values()))]
acled_riots_s.columns=["year","gw_codes","riot_event_counts","riot_fatalities"]

###########
### GTD ###
###########

gtd=pd.read_csv("data/data_out/gtd_cy_attacks.csv",index_col=0)
gtd_s=gtd.loc[gtd["year"]>=1989]
gtd_ss = gtd_s[["year","gw_codes","n_attack","fatalities"]][~gtd_s['gw_codes'].isin(list(micro_states.values())+list(exclude.values())+list(exclude2.values()))]
gtd_ss.columns=["year","gw_codes","terrorism_event_counts","terrorism_fatalities"]

# Merge
df=pd.merge(left=ucdp_sb_s,right=ucdp_ns_s,on=["year","gw_codes"],how="left")
df=pd.merge(left=df,right=ucdp_osv_s,on=["year","gw_codes"],how="left")
df=pd.merge(left=df,right=acled_protest_s,on=["year","gw_codes"],how="left")
df=pd.merge(left=df,right=acled_riots_s,on=["year","gw_codes"],how="left")
df=pd.merge(left=df,right=gtd_ss,on=["year","gw_codes"],how="left")
df=df.fillna(0)

# Check that all have the same countries, but they may have different temporal coverage
c1=ucdp_ns_s.sort_values(by=["gw_codes"]).gw_codes.unique()
c2=acled_riots_s.sort_values(by=["gw_codes"]).gw_codes.unique()
c3=gtd_ss.sort_values(by=["gw_codes"]).gw_codes.unique()

###############
### Outcome ###
###############

# Dichotomize
dichotomize(df,"protest_event_counts","d_protest",25)
dichotomize(df,"riot_event_counts","d_riot",25)
dichotomize(df,"sb_fatalities","d_sb",0)
dichotomize(df,"osv_fatalities","d_osv",0)
dichotomize(df,"ns_fatalities","d_ns",0)
dichotomize(df,"terrorism_fatalities","d_terror",0)
dichotomize(df,"sb_fatalities","d_civil_war",1000)
dichotomize(df,"sb_fatalities","d_civil_conflict",25)

# Final df
df_out=df[["year","gw_codes","country","d_protest","d_riot","d_terror","d_sb","d_osv","d_ns","d_civil_conflict","d_civil_war","protest_event_counts","riot_event_counts","terrorism_fatalities","sb_fatalities","osv_fatalities","ns_fatalities"]].copy()
print(df_out.isna().any())
print(df_out.duplicated(subset=['year',"country","gw_codes"]).any())
print(df_out.duplicated(subset=['year',"country"]).any())
print(df_out.duplicated(subset=['year',"gw_codes"]).any())

# Check datatypes and convert floats to integer if needed
print(df_out.dtypes)
df_out['protest_event_counts']=df_out['protest_event_counts'].astype('int64')
df_out['riot_event_counts']=df_out['riot_event_counts'].astype('int64')
df_out['terrorism_fatalities']=df_out['terrorism_fatalities'].astype('int64')

# Save
df_out.to_csv("out/df_out_full.csv") 

#####################
### History theme ###
#####################

### t-1 ###

df_conf_hist=df_out[["year","gw_codes","country","d_protest","d_riot","d_sb","d_osv","d_ns","d_terror","protest_event_counts","riot_event_counts","sb_fatalities","osv_fatalities","ns_fatalities","terrorism_fatalities"]].copy()
df_conf_hist["d_protest_lag1"]=lag_groupped(df,"country","d_protest",1)
df_conf_hist["d_riot_lag1"]=lag_groupped(df,"country","d_riot",1)
df_conf_hist["d_terror_lag1"]=lag_groupped(df,"country","d_terror",1)
df_conf_hist["d_sb_lag1"]=lag_groupped(df,"country","d_sb",1)
df_conf_hist["d_osv_lag1"]=lag_groupped(df,"country","d_osv",1)
df_conf_hist["d_ns_lag1"]=lag_groupped(df,"country","d_ns",1)
df_conf_hist["d_civil_war_lag1"]=lag_groupped(df,"country","d_civil_war",1)
df_conf_hist["d_civil_conflict_lag1"]=lag_groupped(df,"country","d_civil_conflict",1)

### Time since ###

df_conf_hist['d_protest_zeros'] = consec_zeros_grouped(df,'country','d_protest')
df_conf_hist['d_protest_zeros'] = lag_groupped(df_conf_hist,'country','d_protest_zeros',1)
df_conf_hist['d_protest_zeros_growth'] = exponential_growth(df_conf_hist['d_protest_zeros'])
df_conf_hist = df_conf_hist.drop('d_protest_zeros', axis=1)

df_conf_hist['d_riot_zeros'] = consec_zeros_grouped(df,'country','d_riot')
df_conf_hist['d_riot_zeros'] = lag_groupped(df_conf_hist,'country','d_riot_zeros',1)
df_conf_hist['d_riot_zeros_growth'] = exponential_growth(df_conf_hist['d_riot_zeros'])
df_conf_hist = df_conf_hist.drop('d_riot_zeros', axis=1)

df_conf_hist['d_terror_zeros'] = consec_zeros_grouped(df,'country','d_terror')
df_conf_hist['d_terror_zeros'] = lag_groupped(df_conf_hist,'country','d_terror_zeros',1)
df_conf_hist['d_terror_zeros_growth'] = exponential_growth(df_conf_hist['d_terror_zeros'])
df_conf_hist = df_conf_hist.drop('d_terror_zeros', axis=1)

df_conf_hist['d_sb_zeros'] = consec_zeros_grouped(df,'country','d_sb')
df_conf_hist['d_sb_zeros'] = lag_groupped(df_conf_hist,'country','d_sb_zeros',1)
df_conf_hist['d_sb_zeros_growth'] = exponential_growth(df_conf_hist['d_sb_zeros'])
df_conf_hist = df_conf_hist.drop('d_sb_zeros', axis=1)

df_conf_hist['d_osv_zeros'] = consec_zeros_grouped(df,'country','d_osv')
df_conf_hist['d_osv_zeros'] = lag_groupped(df_conf_hist,'country','d_osv_zeros',1)
df_conf_hist['d_osv_zeros_growth'] = exponential_growth(df_conf_hist['d_osv_zeros'])
df_conf_hist = df_conf_hist.drop('d_osv_zeros', axis=1)

df_conf_hist['d_ns_zeros'] = consec_zeros_grouped(df,'country','d_ns')
df_conf_hist['d_ns_zeros'] = lag_groupped(df_conf_hist,'country','d_ns_zeros',1)
df_conf_hist['d_ns_zeros_growth'] = exponential_growth(df_conf_hist['d_ns_zeros'])
df_conf_hist = df_conf_hist.drop('d_ns_zeros', axis=1)

df_conf_hist['d_civil_war_zeros'] = consec_zeros_grouped(df,'country','d_civil_war')
df_conf_hist['d_civil_war_zeros'] = lag_groupped(df_conf_hist,'country','d_civil_war_zeros',1)
df_conf_hist['d_civil_war_zeros_growth'] = exponential_growth(df_conf_hist['d_civil_war_zeros'])
df_conf_hist = df_conf_hist.drop('d_civil_war_zeros', axis=1)

df_conf_hist['d_civil_conflict_zeros'] = consec_zeros_grouped(df,'country','d_civil_conflict')
df_conf_hist['d_civil_conflict_zeros'] = lag_groupped(df_conf_hist,'country','d_civil_conflict_zeros',1)
df_conf_hist['d_civil_conflict_zeros_growth'] = exponential_growth(df_conf_hist['d_civil_conflict_zeros'])
df_conf_hist = df_conf_hist.drop('d_civil_conflict_zeros', axis=1)

### Neighbor history protest ###

# Add neighbors to df
neighbors=pd.read_csv("data/data_out/cy_neighbors.csv",index_col=0)
gw_codes=pd.read_csv("data/df_ccodes_gw.csv")
# Country names in neighbors file and df_codes need to be the same
gw_codes_s=gw_codes.loc[gw_codes["end"]>=1989]
df_neighbors=pd.merge(left=df_out[["year","country","gw_codes","d_protest"]],right=neighbors[["gw_codes","year","neighbors"]],on=["year","gw_codes"],how="left")

df_neighbors["neighbors_protest"]=0
# Loop through every observation
for i in range(len(df_neighbors)):
    
    # If no neighbors pass on
    if pd.isna(df_neighbors["neighbors"].iloc[i]): 
        pass
    else:
        
        # Get list of neighbors and set events to zero
        lst=df_neighbors["neighbors"].iloc[i].split(';')
        counts=0
        
        # For each neighbor 
        for x in lst:
            c=int(gw_codes_s["gw_codes"].loc[gw_codes_s["country"]==x].iloc[0])
            # (if available) sum event counts
            if df_neighbors["d_protest"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].empty==False:
                counts+=int(df_neighbors["d_protest"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].iloc[0])
        
        # If larger than zero, add to df
        if counts>0:
            df_neighbors.iloc[i, df_neighbors.columns.get_loc('neighbors_protest')] = counts

# Dichotomize and lag --> at least one neighbor had more than 25 protest events in previous year
dichotomize(df_neighbors,"neighbors_protest","d_neighbors_proteset",0)
df_neighbors['d_neighbors_proteset_lag1'] = lag_groupped(df_neighbors,'country','d_neighbors_proteset',1)

# Merge
df_conf_hist=pd.merge(left=df_conf_hist,right=df_neighbors[["year","gw_codes","d_neighbors_proteset_lag1"]],on=["year","gw_codes"],how="left")

### Neighbor history riot ###

# Add neighbors to df
neighbors=pd.read_csv("data/data_out/cy_neighbors.csv",index_col=0)
gw_codes=pd.read_csv("data/df_ccodes_gw.csv")
# Country names in neighbors file and df_codes need to be the same
gw_codes_s=gw_codes.loc[gw_codes["end"]>=1989]
df_neighbors=pd.merge(left=df_out[["year","country","gw_codes","d_riot"]],right=neighbors[["gw_codes","year","neighbors"]],on=["year","gw_codes"],how="left")

df_neighbors["neighbors_riot"]=0
# Loop through every observation
for i in range(len(df_neighbors)):
    
    # If no neighbors pass on
    if pd.isna(df_neighbors["neighbors"].iloc[i]): 
        pass
    else:   
        
        # Get list of neighbors and set events to zero
        lst=df_neighbors["neighbors"].iloc[i].split(';')
        counts=0
        
        # For each neighbor 
        for x in lst:
            c=int(gw_codes_s["gw_codes"].loc[gw_codes_s["country"]==x].iloc[0])
            # (if available) sum event counts
            if df_neighbors["d_riot"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].empty==False:
                counts+=int(df_neighbors["d_riot"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].iloc[0])
        
        # If larger than zero, add to df
        if counts>0:
            df_neighbors.iloc[i, df_neighbors.columns.get_loc('neighbors_riot')] = counts

# Dichotomize and lag ---> at least one neighbor had more than 25 riot events in previous year
dichotomize(df_neighbors,"neighbors_riot","d_neighbors_riot",0)
df_neighbors['d_neighbors_riot_lag1'] = lag_groupped(df_neighbors,'country','d_neighbors_riot',1)

# Merge
df_conf_hist=pd.merge(left=df_conf_hist,right=df_neighbors[["year","gw_codes","d_neighbors_riot_lag1"]],on=["year","gw_codes"],how="left")

### Neighbor conflict history terrorism ###

# Add neighbors to df
neighbors=pd.read_csv("data/data_out/cy_neighbors.csv",index_col=0)
gw_codes=pd.read_csv("data/df_ccodes_gw.csv")
# Country names in neighbors file and df_codes need to be the same
gw_codes_s=gw_codes.loc[gw_codes["end"]>=1989]
df_neighbors=pd.merge(left=df_out[["year","country","gw_codes","d_terror"]],right=neighbors[["gw_codes","year","neighbors"]],on=["year","gw_codes"],how="left")

df_neighbors["neighbors_terror"]=0
# Loop through every observation
for i in range(len(df_neighbors)):
    
    # If no neighbors pass on
    if pd.isna(df_neighbors["neighbors"].iloc[i]): 
        pass
    else:
        
        # Get list of neighbors and set events to zero
        lst=df_neighbors["neighbors"].iloc[i].split(';')
        counts=0
        
        # For each neighbor
        for x in lst:
            c=int(gw_codes_s["gw_codes"].loc[gw_codes_s["country"]==x].iloc[0])
            # (if available) sum event counts
            if df_neighbors["d_terror"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].empty==False:
                counts+=int(df_neighbors["d_terror"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].iloc[0])
        
        # If larger than zero, add to df
        if counts>0:
            df_neighbors.iloc[i, df_neighbors.columns.get_loc('neighbors_terror')] = counts

# Dichotomize and lag ---> At least one neighbor had at least on fatality from terrorism in previous year
dichotomize(df_neighbors,"neighbors_terror","d_neighbors_terror",0)
df_neighbors['d_neighbors_terror_lag1'] = lag_groupped(df_neighbors,'country','d_neighbors_terror',1)

# Merge
df_conf_hist=pd.merge(left=df_conf_hist,right=df_neighbors[["year","gw_codes","d_neighbors_terror_lag1"]],on=["year","gw_codes"],how="left")

### Neighbor conflict history sb ###

# Add neighbors to df
neighbors=pd.read_csv("data/data_out/cy_neighbors.csv",index_col=0)
gw_codes=pd.read_csv("data/df_ccodes_gw.csv")
# Country names in neighbors file and df_codes need to be the same
gw_codes_s=gw_codes.loc[gw_codes["end"]>=1989]
df_neighbors=pd.merge(left=df_out[["year","country","gw_codes","d_sb"]],right=neighbors[["gw_codes","year","neighbors"]],on=["year","gw_codes"],how="left")

df_neighbors["neighbors_fat"]=0
# Loop through every observation
for i in range(len(df_neighbors)):
    
    # If no neighbors pass on
    if pd.isna(df_neighbors["neighbors"].iloc[i]): 
        pass
    else: 
        
        # Get list of neighbors and set events to zero
        lst=df_neighbors["neighbors"].iloc[i].split(';')
        counts=0
        
        # For each neighbor 
        for x in lst:
            c=int(gw_codes_s["gw_codes"].loc[gw_codes_s["country"]==x].iloc[0])
            # (if available) sum event counts
            if df_neighbors["d_sb"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].empty==False:
                counts+=int(df_neighbors["d_sb"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].iloc[0])
        
        # If larger than zero, add to df
        if counts>0:
            df_neighbors.iloc[i, df_neighbors.columns.get_loc('neighbors_fat')] = counts

# Dichotomize and lag ---> at least one nieghbor had at least one fatality from civil conflict in previous year
dichotomize(df_neighbors,"neighbors_fat","d_neighbors_sb",0)
df_neighbors['d_neighbors_sb_lag1'] = lag_groupped(df_neighbors,'country','d_neighbors_sb',1)

# Merge
df_conf_hist=pd.merge(left=df_conf_hist,right=df_neighbors[["year","gw_codes","d_neighbors_sb_lag1"]],on=["year","gw_codes"],how="left")

### Neighbor conflict history ns ###

# Add neighbors to df
neighbors=pd.read_csv("data/data_out/cy_neighbors.csv",index_col=0)
gw_codes=pd.read_csv("data/df_ccodes_gw.csv")
# Country names in neighbors file and df_codes need to be the same
gw_codes_s=gw_codes.loc[gw_codes["end"]>=1989]
df_neighbors=pd.merge(left=df_out[["year","country","gw_codes","d_ns"]],right=neighbors[["gw_codes","year","neighbors"]],on=["year","gw_codes"],how="left")

df_neighbors["neighbors_ns"]=0
# Loop through every observation
for i in range(len(df_neighbors)):
    
    # If no neighbors pass on
    if pd.isna(df_neighbors["neighbors"].iloc[i]): 
        pass
    else:   
        
        # Get list of neighbors and set events to zero
        lst=df_neighbors["neighbors"].iloc[i].split(';')
        counts=0
        
        # For each neighbor 
        for x in lst:
            c=int(gw_codes_s["gw_codes"].loc[gw_codes_s["country"]==x].iloc[0])
            # (if available) sum event counts
            if df_neighbors["d_ns"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].empty==False:
                counts+=int(df_neighbors["d_ns"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].iloc[0])
        
        # If larger than zero, add to df
        if counts>0:
            df_neighbors.iloc[i, df_neighbors.columns.get_loc('neighbors_ns')] = counts

# Dichotomize and lag ---> at least one neighbor had at least one fatality from non-state conflict in previous year
dichotomize(df_neighbors,"neighbors_ns","d_neighbors_ns",0)
df_neighbors['d_neighbors_ns_lag1'] = lag_groupped(df_neighbors,'country','d_neighbors_ns',1)

# Merge
df_conf_hist=pd.merge(left=df_conf_hist,right=df_neighbors[["year","gw_codes","d_neighbors_ns_lag1"]],on=["year","gw_codes"],how="left")

### Neighbor conflict history osv ###

# Add neighbors to df
neighbors=pd.read_csv("data/data_out/cy_neighbors.csv",index_col=0)
gw_codes=pd.read_csv("data/df_ccodes_gw.csv")
# Country names in neighbors file and df_codes need to be the same
gw_codes_s=gw_codes.loc[gw_codes["end"]>=1989]
df_neighbors=pd.merge(left=df_out[["year","country","gw_codes","d_osv"]],right=neighbors[["gw_codes","year","neighbors"]],on=["year","gw_codes"],how="left")

df_neighbors["neighbors_osv"]=0
# Loop through every observation
for i in range(len(df_neighbors)):
    
    # If no neighbors pass on
    if pd.isna(df_neighbors["neighbors"].iloc[i]): 
        pass
    else:
        
        # Get list of neighbors and set events to zero
        lst=df_neighbors["neighbors"].iloc[i].split(';')
        counts=0
        
        # For each neighbor 
        for x in lst:
            c=int(gw_codes_s["gw_codes"].loc[gw_codes_s["country"]==x].iloc[0])
            #(if available) sum event counts
            if df_neighbors["d_osv"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].empty==False:
                counts+=int(df_neighbors["d_osv"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].iloc[0])
        
        # If larger than zero, add to df
        if counts>0:
            df_neighbors.iloc[i, df_neighbors.columns.get_loc('neighbors_osv')] = counts

# Dichotomize and lag ---> at least one neighbor had at least one fatality from one-sided violence in previous year
dichotomize(df_neighbors,"neighbors_osv","d_neighbors_osv",0)
df_neighbors['d_neighbors_osv_lag1'] = lag_groupped(df_neighbors,'country','d_neighbors_osv',1)

# Merge
df_conf_hist=pd.merge(left=df_conf_hist,right=df_neighbors[["year","gw_codes","d_neighbors_osv_lag1"]],on=["year","gw_codes"],how="left")

### Neighbor history civil war ###

# Add neighbors to df
neighbors=pd.read_csv("data/data_out/cy_neighbors.csv",index_col=0)
gw_codes=pd.read_csv("data/df_ccodes_gw.csv")
# Country names in neighbors file and df_codes need to be the same
gw_codes_s=gw_codes.loc[gw_codes["end"]>=1989]
df_neighbors=pd.merge(left=df_out[["year","country","gw_codes","d_civil_war"]],right=neighbors[["gw_codes","year","neighbors"]],on=["year","gw_codes"],how="left")

df_neighbors["neighbors_civil_war"]=0
# Loop through every observation
for i in range(len(df_neighbors)):
    
    # If no neighbors pass on
    if pd.isna(df_neighbors["neighbors"].iloc[i]): 
        pass
    else:
        
        # Get list of neighbors and set events to zero
        lst=df_neighbors["neighbors"].iloc[i].split(';')
        counts=0
        
        # For each neighbor 
        for x in lst:
            c=int(gw_codes_s["gw_codes"].loc[gw_codes_s["country"]==x].iloc[0])
            # (if available) sum event counts
            if df_neighbors["d_civil_war"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].empty==False:
                counts+=int(df_neighbors["d_civil_war"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].iloc[0])
        
        # If larger than zero, add to df
        if counts>0:
            df_neighbors.iloc[i, df_neighbors.columns.get_loc('neighbors_civil_war')] = counts

# Dichotomize and lag --> at least one nieghbor had more than 1000 fatalities from civil conflict in previous year
dichotomize(df_neighbors,"neighbors_civil_war","d_neighbors_civil_war",0)
df_neighbors['d_neighbors_civil_war_lag1'] = lag_groupped(df_neighbors,'country','d_neighbors_civil_war',1)

# Merge
df_conf_hist=pd.merge(left=df_conf_hist,right=df_neighbors[["year","gw_codes","d_neighbors_civil_war_lag1"]],on=["year","gw_codes"],how="left")

### Neighbor history civil conflict ###

# Add neighbors to df
neighbors=pd.read_csv("data/data_out/cy_neighbors.csv",index_col=0)
gw_codes=pd.read_csv("data/df_ccodes_gw.csv")
# Country names in neighbors file and df_codes need to be the same
gw_codes_s=gw_codes.loc[gw_codes["end"]>=1989]
df_neighbors=pd.merge(left=df_out[["year","country","gw_codes","d_civil_conflict"]],right=neighbors[["gw_codes","year","neighbors"]],on=["year","gw_codes"],how="left")

df_neighbors["neighbors_civil_conflict"]=0
# Loop through every observation
for i in range(len(df_neighbors)):
    
    # If no neighbors pass on
    if pd.isna(df_neighbors["neighbors"].iloc[i]): 
        pass
    else:
        
        # Get list of neighbors and set events to zero
        lst=df_neighbors["neighbors"].iloc[i].split(';')
        counts=0
        
        # For each neighbor 
        for x in lst:
            c=int(gw_codes_s["gw_codes"].loc[gw_codes_s["country"]==x].iloc[0])
            # (if available) sum event counts
            if df_neighbors["d_civil_conflict"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].empty==False:
                counts+=int(df_neighbors["d_civil_conflict"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].iloc[0])
        
        # If larger than zero, add to df
        if counts>0:
            df_neighbors.iloc[i, df_neighbors.columns.get_loc('neighbors_civil_conflict')] = counts

# Dichotomize and lag --> at least one nieghbor had more than 25 fatalities from civil conflict in previous year
dichotomize(df_neighbors,"neighbors_civil_conflict","d_neighbors_civil_conflict",0)
df_neighbors['d_neighbors_civil_conflict_lag1'] = lag_groupped(df_neighbors,'country','d_neighbors_civil_conflict',1)

# Merge
df_conf_hist=pd.merge(left=df_conf_hist,right=df_neighbors[["year","gw_codes","d_neighbors_civil_conflict_lag1"]],on=["year","gw_codes"],how="left")

### New state ###

base=df_out[["year","gw_codes","country"]].copy()
base['regime_duration']=0

# Count years from start date as defined in GW set up
# http://ksgleditsch.com/data/iisystem.dat
# http://ksgleditsch.com/data/microstatessystem.dat
all_countries=pd.read_csv("data/df_ccodes_gw.csv")
all_countries_s=all_countries.loc[all_countries["end"]>=1989]

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
df_conf_hist=df_conf_hist[['year','gw_codes','country','d_protest_lag1','d_riot_lag1','d_terror_lag1','d_sb_lag1','d_osv_lag1','d_ns_lag1','d_civil_war_lag1','d_civil_conflict_lag1','d_protest_zeros_growth','d_riot_zeros_growth','d_terror_zeros_growth','d_sb_zeros_growth','d_osv_zeros_growth','d_ns_zeros_growth','d_civil_war_zeros_growth','d_civil_conflict_zeros_growth','d_neighbors_proteset_lag1','d_neighbors_riot_lag1','d_neighbors_terror_lag1','d_neighbors_sb_lag1','d_neighbors_ns_lag1','d_neighbors_osv_lag1','d_neighbors_civil_war_lag1','d_neighbors_civil_conflict_lag1','regime_duration']]
print(df_conf_hist.isna().any())
print(df_conf_hist.min())
print(df_conf_hist.duplicated(subset=['year',"country","gw_codes"]).any())
print(df_conf_hist.duplicated(subset=['year',"country"]).any())
print(df_conf_hist.duplicated(subset=['year',"gw_codes"]).any())

# Check datatypes and convert floats to integer if needed
print(df_conf_hist.dtypes)
df_conf_hist['d_protest_lag1']=df_conf_hist['d_protest_lag1'].astype('int64')
df_conf_hist['d_riot_lag1']=df_conf_hist['d_riot_lag1'].astype('int64')
df_conf_hist['d_terror_lag1']=df_conf_hist['d_terror_lag1'].astype('int64')
df_conf_hist['d_sb_lag1']=df_conf_hist['d_sb_lag1'].astype('int64')
df_conf_hist['d_osv_lag1']=df_conf_hist['d_osv_lag1'].astype('int64')
df_conf_hist['d_ns_lag1']=df_conf_hist['d_ns_lag1'].astype('int64')
df_conf_hist['d_civil_war_lag1']=df_conf_hist['d_civil_war_lag1'].astype('int64')
df_conf_hist['d_civil_conflict_lag1']=df_conf_hist['d_civil_conflict_lag1'].astype('int64')
df_conf_hist['d_neighbors_proteset_lag1']=df_conf_hist['d_neighbors_proteset_lag1'].astype('int64')
df_conf_hist['d_neighbors_riot_lag1']=df_conf_hist['d_neighbors_riot_lag1'].astype('int64')
df_conf_hist['d_neighbors_terror_lag1']=df_conf_hist['d_neighbors_terror_lag1'].astype('int64')
df_conf_hist['d_neighbors_sb_lag1']=df_conf_hist['d_neighbors_sb_lag1'].astype('int64')
df_conf_hist['d_neighbors_ns_lag1']=df_conf_hist['d_neighbors_ns_lag1'].astype('int64')
df_conf_hist['d_neighbors_osv_lag1']=df_conf_hist['d_neighbors_osv_lag1'].astype('int64')
df_conf_hist['d_neighbors_civil_war_lag1']=df_conf_hist['d_neighbors_civil_war_lag1'].astype('int64')
df_conf_hist['d_neighbors_civil_conflict_lag1']=df_conf_hist['d_neighbors_civil_conflict_lag1'].astype('int64')

# Save 
df_conf_hist.to_csv("out/df_conf_hist_full.csv")  

########################
### Demography theme ###
########################

# Initiate
df_demog=df_out[["year","gw_codes","country"]].copy()

# Load wb data, previously retrived with the WB api
demog=pd.read_csv("data/demog_wb.csv",index_col=0)
print(demog.min())

### Population size  ###

demog = demog.rename(columns={'SP.POP.TOTL': 'pop'})
df_demog=pd.merge(left=df_demog,right=demog[["year","gw_codes","pop"]],on=["year","gw_codes"],how="left")

### Population density ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
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

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["EN.POP.DNST"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["pop_density"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","pop_density",'pop_density_id']],on=["year","gw_codes"],how="left")

### Urbanization  ###

demog = demog.rename(columns={"SP.URB.TOTL.IN.ZS": 'urb_share'})
df_demog=pd.merge(left=df_demog,right=demog[["year","gw_codes","urb_share"]],on=["year","gw_codes"],how="left")

### Rural ###

demog = demog.rename(columns={"SP.RUR.TOTL.ZS": 'rural_share'})
df_demog=pd.merge(left=df_demog,right=demog[["year","gw_codes","rural_share"]],on=["year","gw_codes"],how="left")

# Load wb data, previously retrived with the WB api
demog=pd.read_csv("data/demog_wb2.csv",index_col=0)
print(demog.min())

### Male total population ###

demog = demog.rename(columns={"SP.POP.TOTL.MA.ZS": 'pop_male_share'})
df_demog=pd.merge(left=df_demog,right=demog[["year","gw_codes","pop_male_share"]],on=["year","gw_codes"],how="left")

### Male total population 0-14 ### 

demog = demog.rename(columns={"SP.POP.0014.MA.ZS": 'pop_male_share_0_14'})
df_demog=pd.merge(left=df_demog,right=demog[["year","gw_codes","pop_male_share_0_14"]],on=["year","gw_codes"],how="left")

### Male total population 15-19  ###

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
erp=pd.read_csv("data/data_out/epr_cy.csv",index_col=0)
base=pd.merge(left=base,right=erp[["year","gw_codes","group_counts","monopoly_share","discriminated_share","powerless_share","dominant_share","ethnic_frac","rel_frac","lang_frac","race_frac"]],on=["year","gw_codes"],how="left")

### Group counts ###

# Simple imputation (linear and if it fails mean)
base_imp_final=linear_imp_grouped(base,"country",["group_counts"])
base_imp_mean=simple_imp_grouped(base,"country",["group_counts"])
base_imp_final['group_counts'] = base_imp_final['group_counts'].fillna(base_imp_mean['group_counts'])
base_imp_final['group_counts_id'] = base["group_counts"].isnull().astype(int)

# Validate
for c in base.country.unique():
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(base["year"].loc[base["country"]==c], base["group_counts"].loc[base["country"]==c],c="black")
    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["group_counts"].loc[base_imp_final["country"]==c],c="black")
    axs[0].set_title(c,size=20)
    if c=="Iraq":
        plt.savefig("out/struc_missin1.eps",dpi=300,bbox_inches='tight')    
    plt.show()

# Merge
df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","group_counts",'group_counts_id']],on=["year","gw_codes"],how="left")

### Monopoly share ###

# Simple imputation (linear and if it fails mean)
base_imp_final=linear_imp_grouped(base,"country",["monopoly_share"])
base_imp_mean=simple_imp_grouped(base,"country",["monopoly_share"])
base_imp_final['monopoly_share'] = base_imp_final['monopoly_share'].fillna(base_imp_mean['monopoly_share'])
base_imp_final['monopoly_share_id'] = base["monopoly_share"].isnull().astype(int)

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["monopoly_share"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["monopoly_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","monopoly_share",'monopoly_share_id']],on=["year","gw_codes"],how="left")

#### Discriminated share ###

# Simple imputation (linear and if it fails mean) 
base_imp_final=linear_imp_grouped(base,"country",["discriminated_share"])
base_imp_mean=simple_imp_grouped(base,"country",["discriminated_share"])
base_imp_final['discriminated_share'] = base_imp_final['discriminated_share'].fillna(base_imp_mean['discriminated_share'])
base_imp_final['discriminated_share_id'] = base["discriminated_share"].isnull().astype(int)

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["discriminated_share"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["discriminated_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","discriminated_share",'discriminated_share_id']],on=["year","gw_codes"],how="left")

### Powerless share ###

# Simple imputation (linear and if it fails mean) 
base_imp_final=linear_imp_grouped(base,"country",["powerless_share"])
base_imp_mean=simple_imp_grouped(base,"country",["powerless_share"])
base_imp_final['powerless_share'] = base_imp_final['powerless_share'].fillna(base_imp_mean['powerless_share'])
base_imp_final['powerless_share_id'] = base["powerless_share"].isnull().astype(int)

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["powerless_share"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["powerless_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","powerless_share",'powerless_share_id']],on=["year","gw_codes"],how="left")

### Dominant share ###

# Simple imputation (linear and if it fails mean)
base_imp_final=linear_imp_grouped(base,"country",["dominant_share"])
base_imp_mean=simple_imp_grouped(base,"country",["dominant_share"])
base_imp_final['dominant_share'] = base_imp_final['dominant_share'].fillna(base_imp_mean['dominant_share'])
base_imp_final['dominant_share_id'] = base["dominant_share"].isnull().astype(int)

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["dominant_share"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["dominant_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","dominant_share",'dominant_share_id']],on=["year","gw_codes"],how="left")

### Ethnic fractionalization ###

# Simple imputation (linear and if it fails mean) 
base_imp_final=linear_imp_grouped(base,"country",["ethnic_frac"])
base_imp_mean=simple_imp_grouped(base,"country",["ethnic_frac"])
base_imp_final['ethnic_frac'] = base_imp_final['ethnic_frac'].fillna(base_imp_mean['ethnic_frac'])
base_imp_final['ethnic_frac_id'] = base["ethnic_frac"].isnull().astype(int)

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["ethnic_frac"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["ethnic_frac"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","ethnic_frac",'ethnic_frac_id']],on=["year","gw_codes"],how="left")

### Religious fractionalization ###

# Simple imputation (linear and if it fails mean) 
base_imp_final=linear_imp_grouped(base,"country",["rel_frac"])
base_imp_mean=simple_imp_grouped(base,"country",["rel_frac"])
base_imp_final['rel_frac'] = base_imp_final['rel_frac'].fillna(base_imp_mean['rel_frac'])
base_imp_final['rel_frac_id'] = base["rel_frac"].isnull().astype(int)

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["rel_frac"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["rel_frac"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","rel_frac",'rel_frac_id']],on=["year","gw_codes"],how="left")

### Language fractionlization ###

# Simple imputation (linear and if it fails mean) 
base_imp_final=linear_imp_grouped(base,"country",["lang_frac"])
base_imp_mean=simple_imp_grouped(base,"country",["lang_frac"])
base_imp_final['lang_frac'] = base_imp_final['lang_frac'].fillna(base_imp_mean['lang_frac'])
base_imp_final['lang_frac_id'] = base["lang_frac"].isnull().astype(int)

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["lang_frac"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["lang_frac"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","lang_frac",'lang_frac_id']],on=["year","gw_codes"],how="left")

### Race fractionlization ###

# Simple imputation (linear and if it fails mean) 
base_imp_final=linear_imp_grouped(base,"country",["race_frac"])
base_imp_mean=simple_imp_grouped(base,"country",["race_frac"])
base_imp_final['race_frac'] = base_imp_final['race_frac'].fillna(base_imp_mean['race_frac'])
base_imp_final['race_frac_id'] = base["race_frac"].isnull().astype(int)

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["race_frac"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["race_frac"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","race_frac",'race_frac_id']],on=["year","gw_codes"],how="left")

# Final df
print(df_demog.isna().any())
print(df_demog.min())
print(df_demog.duplicated(subset=['year',"country","gw_codes"]).any())
print(df_demog.duplicated(subset=['year',"country"]).any())
print(df_demog.duplicated(subset=['year',"gw_codes"]).any())

# Check datatypes and convert floats to integer if needed
print(df_demog.dtypes)
df_demog['group_counts'].unique() # Check unique levels
df_demog['group_counts']=df_demog['group_counts'].astype('int64')

# Save
df_demog.to_csv("out/df_demog_full.csv") 

###################################
### Economy & development theme ###
###################################

# Initiate
df_econ=df_out[["year","gw_codes","country"]].copy()

# Load wb data, previously retrived with the WB api
economy=pd.read_csv("data/economy_wb.csv",index_col=0)
print(economy.min())

### Total natual resource rents % of GDP ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS","NY.GDP.TOTL.RT.ZS"]],on=["year","gw_codes"],how="left")
# Use other rents as additional variables
base_imp=multivariate_imp_bayes(base,"country",["NY.GDP.TOTL.RT.ZS"],vars_add=["NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["NY.GDP.TOTL.RT.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["NY.GDP.TOTL.RT.ZS"])
base_imp_final['NY.GDP.TOTL.RT.ZS'] = base_imp_final['NY.GDP.TOTL.RT.ZS'].fillna(base_imp_mean['NY.GDP.TOTL.RT.ZS'])
base_imp_final["NY.GDP.TOTL.RT.ZS"] = base_imp_final["NY.GDP.TOTL.RT.ZS"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['natres_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NY.GDP.TOTL.RT.ZS": 'natres_share'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GDP.TOTL.RT.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["natres_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","natres_share",'natres_share_id']],on=["year","gw_codes"],how="left")

### Oil rents (% of GDP) ##

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS","NY.GDP.TOTL.RT.ZS"]],on=["year","gw_codes"],how="left")
# Use other rents as additional variables
base_imp=multivariate_imp_bayes(base,"country",["NY.GDP.PETR.RT.ZS"],vars_add=["NY.GDP.TOTL.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["NY.GDP.PETR.RT.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["NY.GDP.PETR.RT.ZS"])
base_imp_final['NY.GDP.PETR.RT.ZS'] = base_imp_final['NY.GDP.PETR.RT.ZS'].fillna(base_imp_mean['NY.GDP.PETR.RT.ZS'])
base_imp_final["NY.GDP.PETR.RT.ZS"] = base_imp_final["NY.GDP.PETR.RT.ZS"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['oil_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NY.GDP.PETR.RT.ZS": 'oil_share'})

# Validate
for c in base.country.unique():
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GDP.PETR.RT.ZS"].loc[base["country"]==c],c="black")
    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["oil_share"].loc[base_imp_final["country"]==c],c="black")
    axs[0].set_title(c,size=20)
    if c=="Equatorial Guinea":
        plt.savefig("out/struc_missin2.eps",dpi=300,bbox_inches='tight')
    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","oil_share",'oil_share_id']],on=["year","gw_codes"],how="left")

### Natural gas rents (% of GDP) ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS","NY.GDP.TOTL.RT.ZS"]],on=["year","gw_codes"],how="left")
# Use other rents as additional variables
base_imp=multivariate_imp_bayes(base,"country",["NY.GDP.NGAS.RT.ZS"],vars_add=["NY.GDP.TOTL.RT.ZS","NY.GDP.PETR.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["NY.GDP.NGAS.RT.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["NY.GDP.NGAS.RT.ZS"])
base_imp_final['NY.GDP.NGAS.RT.ZS'] = base_imp_final['NY.GDP.NGAS.RT.ZS'].fillna(base_imp_mean['NY.GDP.NGAS.RT.ZS'])
base_imp_final["NY.GDP.NGAS.RT.ZS"] = base_imp_final["NY.GDP.NGAS.RT.ZS"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['gas_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NY.GDP.NGAS.RT.ZS": 'gas_share'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GDP.NGAS.RT.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["gas_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","gas_share",'gas_share_id']],on=["year","gw_codes"],how="left")

### Coal rents (% of GDP) ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS","NY.GDP.TOTL.RT.ZS"]],on=["year","gw_codes"],how="left")
# Use other rents as additional variables
base_imp=multivariate_imp_bayes(base,"country",["NY.GDP.COAL.RT.ZS"],vars_add=["NY.GDP.TOTL.RT.ZS","NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["NY.GDP.COAL.RT.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["NY.GDP.COAL.RT.ZS"])
base_imp_final['NY.GDP.COAL.RT.ZS'] = base_imp_final['NY.GDP.COAL.RT.ZS'].fillna(base_imp_mean['NY.GDP.COAL.RT.ZS'])
base_imp_final["NY.GDP.COAL.RT.ZS"] = base_imp_final["NY.GDP.COAL.RT.ZS"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['coal_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NY.GDP.COAL.RT.ZS": 'coal_share'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GDP.COAL.RT.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["coal_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","coal_share",'coal_share_id']],on=["year","gw_codes"],how="left")

### Forest rents (% of GDP) ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS","NY.GDP.TOTL.RT.ZS"]],on=["year","gw_codes"],how="left")
# Use other rents as additional variables
base_imp=multivariate_imp_bayes(base,"country",["NY.GDP.FRST.RT.ZS"],vars_add=["NY.GDP.TOTL.RT.ZS","NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.MINR.RT.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["NY.GDP.FRST.RT.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["NY.GDP.FRST.RT.ZS"])
base_imp_final['NY.GDP.FRST.RT.ZS'] = base_imp_final['NY.GDP.FRST.RT.ZS'].fillna(base_imp_mean['NY.GDP.FRST.RT.ZS'])
base_imp_final["NY.GDP.FRST.RT.ZS"] = base_imp_final["NY.GDP.FRST.RT.ZS"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['forest_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NY.GDP.FRST.RT.ZS": 'forest_share'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GDP.FRST.RT.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["forest_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","forest_share",'forest_share_id']],on=["year","gw_codes"],how="left")

### Minerals rents (% of GDP) ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS","NY.GDP.TOTL.RT.ZS"]],on=["year","gw_codes"],how="left")
# Use other rents as additional variables
base_imp=multivariate_imp_bayes(base,"country",["NY.GDP.MINR.RT.ZS"],vars_add=["NY.GDP.TOTL.RT.ZS","NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["NY.GDP.MINR.RT.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["NY.GDP.MINR.RT.ZS"])
base_imp_final['NY.GDP.MINR.RT.ZS'] = base_imp_final['NY.GDP.MINR.RT.ZS'].fillna(base_imp_mean['NY.GDP.MINR.RT.ZS'])
base_imp_final["NY.GDP.MINR.RT.ZS"] = base_imp_final["NY.GDP.MINR.RT.ZS"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['minerals_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NY.GDP.MINR.RT.ZS": 'minerals_share'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GDP.MINR.RT.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["minerals_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","minerals_share",'minerals_share_id']],on=["year","gw_codes"],how="left")

### GDP per capita ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
# Use GDP growth and GNI per capita as additional variables
base_imp=multivariate_imp_bayes(base,"country",["NY.GDP.PCAP.CD"],vars_add=["NY.GDP.MKTP.KD.ZG","NY.GNP.PCAP.CD"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["NY.GDP.PCAP.CD"])
base_imp_mean=simple_imp_grouped(base,"country",["NY.GDP.PCAP.CD"])
base_imp_final['NY.GDP.PCAP.CD'] = base_imp_final['NY.GDP.PCAP.CD'].fillna(base_imp_mean['NY.GDP.PCAP.CD'])
base_imp_final["NY.GDP.PCAP.CD"] = base_imp_final["NY.GDP.PCAP.CD"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['gdp_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NY.GDP.PCAP.CD": 'gdp'})

# Validate
for c in base.country.unique():
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GDP.PCAP.CD"].loc[base["country"]==c],c="black")
    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["gdp"].loc[base_imp_final["country"]==c],c="black")
    axs[0].set_title(c,size=20)
    axs[0].tick_params(axis='y',labelsize=15)
    axs[1].tick_params(axis='y',labelsize=15)
    if c=="Venezuela":
        plt.savefig("out/struc_missin_Venezuela.eps",dpi=300,bbox_inches='tight')
    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","gdp",'gdp_id']],on=["year","gw_codes"],how="left")

### GNI per capita ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
# Use GDP growth and GDP per capita as additional variables
base_imp=multivariate_imp_bayes(base,"country",["NY.GNP.PCAP.CD"],vars_add=["NY.GDP.MKTP.KD.ZG","NY.GDP.PCAP.CD"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["NY.GNP.PCAP.CD"])
base_imp_mean=simple_imp_grouped(base,"country",["NY.GNP.PCAP.CD"])
base_imp_final['NY.GNP.PCAP.CD'] = base_imp_final['NY.GNP.PCAP.CD'].fillna(base_imp_mean['NY.GNP.PCAP.CD'])
base_imp_final["NY.GNP.PCAP.CD"] = base_imp_final["NY.GNP.PCAP.CD"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['gni_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NY.GNP.PCAP.CD": 'gni'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GNP.PCAP.CD"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["gni"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","gni",'gni_id']],on=["year","gw_codes"],how="left")

### GDP growth ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
# Use GDP per capita and GNI per capita as additional variables
base_imp=multivariate_imp_bayes(base,"country",["NY.GDP.MKTP.KD.ZG"],vars_add=["NY.GDP.PCAP.CD","NY.GNP.PCAP.CD"],max_iter=10,min_val=-100)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["NY.GDP.MKTP.KD.ZG"])
base_imp_mean=simple_imp_grouped(base,"country",["NY.GDP.MKTP.KD.ZG"])
base_imp_final['NY.GDP.MKTP.KD.ZG'] = base_imp_final['NY.GDP.MKTP.KD.ZG'].fillna(base_imp_mean['NY.GDP.MKTP.KD.ZG'])
base_imp_final["NY.GDP.MKTP.KD.ZG"] = base_imp_final["NY.GDP.MKTP.KD.ZG"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['gdp_growth_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NY.GDP.MKTP.KD.ZG": 'gdp_growth'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GDP.MKTP.KD.ZG"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["gdp_growth"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","gdp_growth",'gdp_growth_id']],on=["year","gw_codes"],how="left")

### Unemployment, total ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
# Use unemployment male as additional variable
base_imp=multivariate_imp_bayes(base,"country",["SL.UEM.TOTL.NE.ZS"],vars_add=["SL.UEM.TOTL.MA.NE.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["SL.UEM.TOTL.NE.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["SL.UEM.TOTL.NE.ZS"])
base_imp_final['SL.UEM.TOTL.NE.ZS'] = base_imp_final['SL.UEM.TOTL.NE.ZS'].fillna(base_imp_mean['SL.UEM.TOTL.NE.ZS'])
base_imp_final["SL.UEM.TOTL.NE.ZS"] = base_imp_final["SL.UEM.TOTL.NE.ZS"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['unemploy_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SL.UEM.TOTL.NE.ZS": 'unemploy'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SL.UEM.TOTL.NE.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["unemploy"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","unemploy",'unemploy_id']],on=["year","gw_codes"],how="left")

### Unemployment, men ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
# Use unemployment as additional variable
base_imp=multivariate_imp_bayes(base,"country",["SL.UEM.TOTL.MA.NE.ZS"],vars_add=["SL.UEM.TOTL.NE.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["SL.UEM.TOTL.MA.NE.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["SL.UEM.TOTL.MA.NE.ZS"])
base_imp_final['SL.UEM.TOTL.MA.NE.ZS'] = base_imp_final['SL.UEM.TOTL.MA.NE.ZS'].fillna(base_imp_mean['SL.UEM.TOTL.MA.NE.ZS'])
base_imp_final["SL.UEM.TOTL.MA.NE.ZS"] = base_imp_final["SL.UEM.TOTL.MA.NE.ZS"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['unemploy_male_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SL.UEM.TOTL.MA.NE.ZS": 'unemploy_male'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SL.UEM.TOTL.MA.NE.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["unemploy_male"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","unemploy_male",'unemploy_male_id']],on=["year","gw_codes"],how="left")

### Inflation rate ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
# Use consumer price as additional variable
base_imp=multivariate_imp_bayes(base,"country",["FP.CPI.TOTL.ZG"],vars_add=["FP.CPI.TOTL"],max_iter=10,min_val=-100)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["FP.CPI.TOTL.ZG"])
base_imp_mean=simple_imp_grouped(base,"country",["FP.CPI.TOTL.ZG"])
base_imp_final['FP.CPI.TOTL.ZG'] = base_imp_final['FP.CPI.TOTL.ZG'].fillna(base_imp_mean['FP.CPI.TOTL.ZG'])
base_imp_final["FP.CPI.TOTL.ZG"] = base_imp_final["FP.CPI.TOTL.ZG"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['inflat_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"FP.CPI.TOTL.ZG": 'inflat'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["FP.CPI.TOTL.ZG"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["inflat"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","inflat",'inflat_id']],on=["year","gw_codes"],how="left")

### Consumer price index (2010 = 100) ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
# Use inflation as additional variable
base_imp=multivariate_imp_bayes(base,"country",["FP.CPI.TOTL"],vars_add=["FP.CPI.TOTL.ZG"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["FP.CPI.TOTL"])
base_imp_mean=simple_imp_grouped(base,"country",["FP.CPI.TOTL"])
base_imp_final['FP.CPI.TOTL'] = base_imp_final['FP.CPI.TOTL'].fillna(base_imp_mean['FP.CPI.TOTL'])
base_imp_final["FP.CPI.TOTL"] = base_imp_final["FP.CPI.TOTL"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['conprice_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"FP.CPI.TOTL": 'conprice'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["FP.CPI.TOTL"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["conprice"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","conprice",'conprice_id']],on=["year","gw_codes"],how="left")

### Prevalence of undernourishment (% of population) ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
# Use as additional variables food production, rural driking water, and urban drinking water
base_imp=multivariate_imp_bayes(base,"country",["SN.ITK.DEFC.ZS"],vars_add=["AG.PRD.FOOD.XD","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["SN.ITK.DEFC.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["SN.ITK.DEFC.ZS"])
base_imp_final['SN.ITK.DEFC.ZS'] = base_imp_final['SN.ITK.DEFC.ZS'].fillna(base_imp_mean['SN.ITK.DEFC.ZS'])
base_imp_final["SN.ITK.DEFC.ZS"] = base_imp_final["SN.ITK.DEFC.ZS"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['undernour_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SN.ITK.DEFC.ZS": 'undernour'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SN.ITK.DEFC.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["undernour"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","undernour",'undernour_id']],on=["year","gw_codes"],how="left")

### Food production ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
# Use as additional variables, undernourishment, drinking water rural, and drinking water urban
base_imp=multivariate_imp_bayes(base,"country",["AG.PRD.FOOD.XD"],vars_add=["SN.ITK.DEFC.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["AG.PRD.FOOD.XD"])
base_imp_mean=simple_imp_grouped(base,"country",["AG.PRD.FOOD.XD"])
base_imp_final['AG.PRD.FOOD.XD'] = base_imp_final['AG.PRD.FOOD.XD'].fillna(base_imp_mean['AG.PRD.FOOD.XD'])
base_imp_final["AG.PRD.FOOD.XD"] = base_imp_final["AG.PRD.FOOD.XD"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['foodprod_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"AG.PRD.FOOD.XD": 'foodprod'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["AG.PRD.FOOD.XD"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["foodprod"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","foodprod",'foodprod_id']],on=["year","gw_codes"],how="left")

### People using at least basic drinking water services, rural (% of rural population) ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
# Use as additional variables undernourishment, food production, drinking water urban
base_imp=multivariate_imp_bayes(base,"country",["SH.H2O.BASW.RU.ZS"],vars_add=["SN.ITK.DEFC.ZS","AG.PRD.FOOD.XD","SH.H2O.BASW.UR.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["SH.H2O.BASW.RU.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["SH.H2O.BASW.RU.ZS"])
base_imp_final['SH.H2O.BASW.RU.ZS'] = base_imp_final['SH.H2O.BASW.RU.ZS'].fillna(base_imp_mean['SH.H2O.BASW.RU.ZS'])
base_imp_final["SH.H2O.BASW.RU.ZS"] = base_imp_final["SH.H2O.BASW.RU.ZS"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['water_rural_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SH.H2O.BASW.RU.ZS": 'water_rural'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SH.H2O.BASW.RU.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["water_rural"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","water_rural",'water_rural_id']],on=["year","gw_codes"],how="left")

### People using at least basic drinking water services, urban (% of urban population) ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
# Use as additional variables undernourishment, food production, drinking water rural
base_imp=multivariate_imp_bayes(base,"country",["SH.H2O.BASW.UR.ZS"],vars_add=["SN.ITK.DEFC.ZS","AG.PRD.FOOD.XD","SH.H2O.BASW.RU.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["SH.H2O.BASW.UR.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["SH.H2O.BASW.UR.ZS"])
base_imp_final['SH.H2O.BASW.UR.ZS'] = base_imp_final['SH.H2O.BASW.UR.ZS'].fillna(base_imp_mean['SH.H2O.BASW.UR.ZS'])
base_imp_final["SH.H2O.BASW.UR.ZS"] = base_imp_final["SH.H2O.BASW.UR.ZS"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['water_urb_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SH.H2O.BASW.UR.ZS": 'water_urb'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SH.H2O.BASW.UR.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["water_urb"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","water_urb",'water_urb_id']],on=["year","gw_codes"],how="left")

### Agriculture % of GDP ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
# Use trade % of GDP as additional variable
base_imp=multivariate_imp_bayes(base,"country",["NV.AGR.TOTL.ZS"],vars_add=["NE.TRD.GNFS.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["NV.AGR.TOTL.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["NV.AGR.TOTL.ZS"])
base_imp_final['NV.AGR.TOTL.ZS'] = base_imp_final['NV.AGR.TOTL.ZS'].fillna(base_imp_mean['NV.AGR.TOTL.ZS'])
base_imp_final["NV.AGR.TOTL.ZS"] = base_imp_final["NV.AGR.TOTL.ZS"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['agri_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NV.AGR.TOTL.ZS": 'agri_share'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NV.AGR.TOTL.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["agri_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","agri_share",'agri_share_id']],on=["year","gw_codes"],how="left")

### Trade % of GDP ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
# Use agriculture % of GDP as additional variable
base_imp=multivariate_imp_bayes(base,"country",["NE.TRD.GNFS.ZS"],vars_add=["NV.AGR.TOTL.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["NE.TRD.GNFS.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["NE.TRD.GNFS.ZS"])
base_imp_final['NE.TRD.GNFS.ZS'] = base_imp_final['NE.TRD.GNFS.ZS'].fillna(base_imp_mean['NE.TRD.GNFS.ZS'])
base_imp_final["NE.TRD.GNFS.ZS"] = base_imp_final["NE.TRD.GNFS.ZS"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['trade_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NE.TRD.GNFS.ZS": 'trade_share'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NE.TRD.GNFS.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["trade_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","trade_share",'trade_share_id']],on=["year","gw_codes"],how="left")

### Fertility rate, total (births per woman) ###

base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
base = base.rename(columns={"SP.DYN.TFRT.IN": 'fert'})
df_econ=pd.merge(left=df_econ,right=base[["year","gw_codes","fert"]],on=["year","gw_codes"],how="left")

### Life expectancy at birth, female (years) ###

base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
base = base.rename(columns={"SP.DYN.LE00.FE.IN": 'lifeexp_female'})
df_econ=pd.merge(left=df_econ,right=base[["year","gw_codes","lifeexp_female"]],on=["year","gw_codes"],how="left")

### Life expectancy at birth, male (years) ###

base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
base = base.rename(columns={"SP.DYN.LE00.MA.IN": 'lifeexp_male'})
df_econ=pd.merge(left=df_econ,right=base[["year","gw_codes","lifeexp_male"]],on=["year","gw_codes"],how="left")

### Population growth (annual %) ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
# Use as additional variables fertility rate, life expectancy female, life expectancy male, infant mortality
base_imp=multivariate_imp_bayes(base,"country",["SP.POP.GROW"],vars_add=["SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.DYN.IMRT.IN"],max_iter=10,min_val=-100)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["SP.POP.GROW"])
base_imp_mean=simple_imp_grouped(base,"country",["SP.POP.GROW"])
base_imp_final['SP.POP.GROW'] = base_imp_final['SP.POP.GROW'].fillna(base_imp_mean['SP.POP.GROW'])
base_imp_final["SP.POP.GROW"] = base_imp_final["SP.POP.GROW"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['pop_growth_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SP.POP.GROW": 'pop_growth'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SP.POP.GROW"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["pop_growth"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","pop_growth","pop_growth_id"]],on=["year","gw_codes"],how="left")

### Infant mortality ###

base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")
base = base.rename(columns={"SP.DYN.IMRT.IN": 'inf_mort'})
df_econ=pd.merge(left=df_econ,right=base[["year","gw_codes","inf_mort"]],on=["year","gw_codes"],how="left")

### Exports of goods and services (% of GDP) ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NE.EXP.GNFS.ZS","NE.IMP.GNFS.ZS","SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"]],on=["year","gw_codes"],how="left")
# Use imports as additional variable
base_imp=multivariate_imp_bayes(base,"country",["NE.EXP.GNFS.ZS"],vars_add=["NE.IMP.GNFS.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["NE.EXP.GNFS.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["NE.EXP.GNFS.ZS"])
base_imp_final['NE.EXP.GNFS.ZS'] = base_imp_final['NE.EXP.GNFS.ZS'].fillna(base_imp_mean['NE.EXP.GNFS.ZS'])
base_imp_final["NE.EXP.GNFS.ZS"] = base_imp_final["NE.EXP.GNFS.ZS"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['exports_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NE.EXP.GNFS.ZS": 'exports'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NE.EXP.GNFS.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["exports"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","exports",'exports_id']],on=["year","gw_codes"],how="left")

### Imports of goods and services (% of GDP) ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NE.EXP.GNFS.ZS","NE.IMP.GNFS.ZS","SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"]],on=["year","gw_codes"],how="left")
# Use exports as additional variable
base_imp=multivariate_imp_bayes(base,"country",["NE.IMP.GNFS.ZS"],vars_add=["NE.EXP.GNFS.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["NE.IMP.GNFS.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["NE.IMP.GNFS.ZS"])
base_imp_final['NE.IMP.GNFS.ZS'] = base_imp_final['NE.IMP.GNFS.ZS'].fillna(base_imp_mean['NE.IMP.GNFS.ZS'])
base_imp_final["NE.IMP.GNFS.ZS"] = base_imp_final["NE.IMP.GNFS.ZS"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['imports_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NE.IMP.GNFS.ZS": 'imports'})

# Validate
for c in base.country.unique():
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(base["year"].loc[base["country"]==c], base["NE.IMP.GNFS.ZS"].loc[base["country"]==c],c="black")
    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["imports"].loc[base_imp_final["country"]==c],c="black")
    axs[0].set_title(c,size=20)
    if c=="Afghanistan":
        plt.savefig("out/struc_missin_Afghanistan.eps",dpi=300,bbox_inches='tight')
    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","imports",'imports_id']],on=["year","gw_codes"],how="left")

### School enrollment, primary, female (% gross) ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NE.EXP.GNFS.ZS","NE.IMP.GNFS.ZS","SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"]],on=["year","gw_codes"],how="left")
# Use as additional variables male primary school enrollment, secondary school enrollment male and female, tertiary school enrollment male and female
base_imp=multivariate_imp_bayes(base,"country",["SE.PRM.ENRR.FE"],vars_add=["SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["SE.PRM.ENRR.FE"])
base_imp_mean=simple_imp_grouped(base,"country",["SE.PRM.ENRR.FE"])
base_imp_final['SE.PRM.ENRR.FE'] = base_imp_final['SE.PRM.ENRR.FE'].fillna(base_imp_mean['SE.PRM.ENRR.FE'])
base_imp_final["SE.PRM.ENRR.FE"] = base_imp_final["SE.PRM.ENRR.FE"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['primary_female_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SE.PRM.ENRR.FE": 'primary_female'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SE.PRM.ENRR.FE"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["primary_female"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","primary_female",'primary_female_id']],on=["year","gw_codes"],how="left")

### School enrollment, primary, male (% gross) ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NE.EXP.GNFS.ZS","NE.IMP.GNFS.ZS","SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"]],on=["year","gw_codes"],how="left")
# Use as additional variables primary school enrollment female, secondary school enrollment male and female, tertiary school enrollment male and female
base_imp=multivariate_imp_bayes(base,"country",["SE.PRM.ENRR.MA"],vars_add=["SE.PRM.ENRR.FE","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["SE.PRM.ENRR.MA"])
base_imp_mean=simple_imp_grouped(base,"country",["SE.PRM.ENRR.MA"])
base_imp_final["SE.PRM.ENRR.MA"] = base_imp_final["SE.PRM.ENRR.MA"].fillna(base_imp_mean["SE.PRM.ENRR.MA"])
base_imp_final["SE.PRM.ENRR.MA"] = base_imp_final["SE.PRM.ENRR.MA"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['primary_male_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SE.PRM.ENRR.MA": 'primary_male'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SE.PRM.ENRR.MA"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["primary_male"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","primary_male",'primary_male_id']],on=["year","gw_codes"],how="left")

### School enrollment, secondary, female (% gross) ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NE.EXP.GNFS.ZS","NE.IMP.GNFS.ZS","SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"]],on=["year","gw_codes"],how="left")
# Use as additional variables primary school enrollment male and female, secondary school enrollment male,  tertiary school enrollment male and female
base_imp=multivariate_imp_bayes(base,"country",["SE.SEC.ENRR.FE"],vars_add=["SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["SE.SEC.ENRR.FE"])
base_imp_mean=simple_imp_grouped(base,"country",["SE.SEC.ENRR.FE"])
base_imp_final["SE.SEC.ENRR.FE"] = base_imp_final["SE.SEC.ENRR.FE"].fillna(base_imp_mean["SE.SEC.ENRR.FE"])
base_imp_final["SE.SEC.ENRR.FE"] = base_imp_final["SE.SEC.ENRR.FE"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['second_female_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SE.SEC.ENRR.FE": 'second_female'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SE.SEC.ENRR.FE"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["second_female"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","second_female",'second_female_id']],on=["year","gw_codes"],how="left")

### School enrollment, secondary, male (% gross) ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NE.EXP.GNFS.ZS","NE.IMP.GNFS.ZS","SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"]],on=["year","gw_codes"],how="left")
# Use as additional variables primary school enrollment male and female, secondary school enrollment female,  tertiary school enrollment male and female
base_imp=multivariate_imp_bayes(base,"country",["SE.SEC.ENRR.MA"],vars_add=["SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.TER.ENRR.FE","SE.TER.ENRR.MA"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["SE.SEC.ENRR.MA"])
base_imp_mean=simple_imp_grouped(base,"country",["SE.SEC.ENRR.MA"])
base_imp_final["SE.SEC.ENRR.MA"] = base_imp_final["SE.SEC.ENRR.MA"].fillna(base_imp_mean["SE.SEC.ENRR.MA"])
base_imp_final["SE.SEC.ENRR.MA"] = base_imp_final["SE.SEC.ENRR.MA"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['second_male_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SE.SEC.ENRR.MA": 'second_male'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SE.SEC.ENRR.MA"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["second_male"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","second_male",'second_male_id']],on=["year","gw_codes"],how="left")

### School enrollment, tertiary, female (% gross) ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NE.EXP.GNFS.ZS","NE.IMP.GNFS.ZS","SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"]],on=["year","gw_codes"],how="left")
# Use as additional variables primary school enrollment male and female, secondary school enrollment male and female,  tertiary school enrollment male 
base_imp=multivariate_imp_bayes(base,"country",["SE.TER.ENRR.FE"],vars_add=["SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.MA"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["SE.TER.ENRR.FE"])
base_imp_mean=simple_imp_grouped(base,"country",["SE.TER.ENRR.FE"])
base_imp_final["SE.TER.ENRR.FE"] = base_imp_final["SE.TER.ENRR.FE"].fillna(base_imp_mean["SE.TER.ENRR.FE"])
base_imp_final["SE.TER.ENRR.FE"] = base_imp_final["SE.TER.ENRR.FE"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['tert_female_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SE.TER.ENRR.FE": 'tert_female'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SE.TER.ENRR.FE"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["tert_female"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","tert_female",'tert_female_id']],on=["year","gw_codes"],how="left")

### School enrollment, tertiary, male (% gross) ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NE.EXP.GNFS.ZS","NE.IMP.GNFS.ZS","SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"]],on=["year","gw_codes"],how="left")
# Use as additional variables primary school enrollment male and female, secondary school enrollment male and female,  tertiary school enrollment female 
base_imp=multivariate_imp_bayes(base,"country",["SE.TER.ENRR.MA"],vars_add=["SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["SE.TER.ENRR.MA"])
base_imp_mean=simple_imp_grouped(base,"country",["SE.TER.ENRR.MA"])
base_imp_final["SE.TER.ENRR.MA"] = base_imp_final["SE.TER.ENRR.MA"].fillna(base_imp_mean["SE.TER.ENRR.MA"])
base_imp_final["SE.TER.ENRR.MA"] = base_imp_final["SE.TER.ENRR.MA"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['tert_male_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SE.TER.ENRR.MA": 'tert_male'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SE.TER.ENRR.FE"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["tert_male"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","tert_male",'tert_male_id']],on=["year","gw_codes"],how="left")

### Expected years of schooling ###

# Create base df, perform multiple imputation 
hdi=pd.read_csv("data/data_out/hdi_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=hdi[["year","gw_codes",'eys','eys_male','eys_female','mys','mys_male','mys_female']],on=["year","gw_codes"],how="left")
# Use as additional variables expected schooling male and female
base_imp=multivariate_imp_bayes(base,"country",["eys"],vars_add=['eys_male','eys_female'],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["eys"])
base_imp_mean=simple_imp_grouped(base,"country",["eys"])
base_imp_final["eys"] = base_imp_final["eys"].fillna(base_imp_mean["eys"])
base_imp_final["eys"] = base_imp_final["eys"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['eys_id'] = base_imp["missing_id"]

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["eys"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["eys"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","eys",'eys_id']],on=["year","gw_codes"],how="left")

### Expected years of schooling, male ###

# Create base df, perform multiple imputation 
hdi=pd.read_csv("data/data_out/hdi_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=hdi[["year","gw_codes",'eys','eys_male','eys_female','mys','mys_male','mys_female']],on=["year","gw_codes"],how="left")
# Use as additional variables expected schooling total and female
base_imp=multivariate_imp_bayes(base,"country",["eys_male"],vars_add=['eys','eys_female'],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["eys_male"])
base_imp_mean=simple_imp_grouped(base,"country",["eys_male"])
base_imp_final["eys_male"] = base_imp_final["eys_male"].fillna(base_imp_mean["eys_male"])
base_imp_final["eys_male"] = base_imp_final["eys_male"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['eys_male_id'] = base_imp["missing_id"]

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["eys_male"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["eys_male"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","eys_male",'eys_male_id']],on=["year","gw_codes"],how="left")

### Expected years of schooling, female ###

# Create base df, perform multiple imputation 
hdi=pd.read_csv("data/data_out/hdi_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=hdi[["year","gw_codes",'eys','eys_male','eys_female','mys','mys_male','mys_female']],on=["year","gw_codes"],how="left")
# Use as additional variables expected schooling total and male
base_imp=multivariate_imp_bayes(base,"country",["eys_female"],vars_add=['eys','eys_male'],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["eys_female"])
base_imp_mean=simple_imp_grouped(base,"country",["eys_female"])
base_imp_final["eys_female"] = base_imp_final["eys_female"].fillna(base_imp_mean["eys_female"])
base_imp_final["eys_female"] = base_imp_final["eys_female"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['eys_female_id'] = base_imp["missing_id"]

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["eys_female"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["eys_female"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","eys_female",'eys_female_id']],on=["year","gw_codes"],how="left")

### Mean years of schooling ###

# Create base df, perform multiple imputation 
hdi=pd.read_csv("data/data_out/hdi_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=hdi[["year","gw_codes",'eys','eys_male','eys_female','mys','mys_male','mys_female']],on=["year","gw_codes"],how="left")
# Use as additional variables mean schooling male and female
base_imp=multivariate_imp_bayes(base,"country",["mys"],vars_add=['mys_male','mys_female'],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["mys"])
base_imp_mean=simple_imp_grouped(base,"country",["mys"])
base_imp_final["mys"] = base_imp_final["mys"].fillna(base_imp_mean["mys"])
base_imp_final["mys"] = base_imp_final["mys"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['mys_id'] = base_imp["missing_id"]

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["mys"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["mys"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","mys",'mys_id']],on=["year","gw_codes"],how="left")

### Mean years of schooling, male ###

# Create base df, perform multiple imputation 
hdi=pd.read_csv("data/data_out/hdi_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=hdi[["year","gw_codes",'eys','eys_male','eys_female','mys','mys_male','mys_female']],on=["year","gw_codes"],how="left")
# Use as additional variables mean schooling total and female
base_imp=multivariate_imp_bayes(base,"country",["mys_male"],vars_add=['mys','mys_female'],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["mys_male"])
base_imp_mean=simple_imp_grouped(base,"country",["mys_male"])
base_imp_final["mys_male"] = base_imp_final["mys_male"].fillna(base_imp_mean["mys_male"])
base_imp_final["mys_male"] = base_imp_final["mys_male"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['mys_male_id'] = base_imp["missing_id"]

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["mys_male"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["mys_male"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","mys_male",'mys_male_id']],on=["year","gw_codes"],how="left")

### Mean years of schooling, female ###

# Create base df, perform multiple imputation
hdi=pd.read_csv("data/data_out/hdi_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=hdi[["year","gw_codes",'eys','eys_male','eys_female','mys','mys_male','mys_female']],on=["year","gw_codes"],how="left")
# Use as additional variables mean schooling total and male
base_imp=multivariate_imp_bayes(base,"country",["mys_female"],vars_add=['mys','mys_male'],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["mys_female"])
base_imp_mean=simple_imp_grouped(base,"country",["mys_female"])
base_imp_final["mys_female"] = base_imp_final["mys_female"].fillna(base_imp_mean["mys_female"])
base_imp_final["mys_female"] = base_imp_final["mys_female"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['mys_female_id'] = base_imp["missing_id"]

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["mys_female"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["mys_female"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","mys_female",'mys_female_id']],on=["year","gw_codes"],how="left")

# Final df
print(df_econ.isna().any().any())
print(df_econ.min())
print(df_econ.duplicated(subset=['year',"country","gw_codes"]).any())
print(df_econ.duplicated(subset=['year',"country"]).any())
print(df_econ.duplicated(subset=['year',"gw_codes"]).any())

# Check datatypes 
for c in df_econ.columns: 
    print(c,df_econ[c].dtypes)

# Save
df_econ.to_csv("out/df_econ_full.csv")  

###############################
### Regime and policy theme ###
###############################

# Initiate
df_pol=df_out[["year","gw_codes","country"]].copy()

# Load wb data, previously retrived with the WB api
pol=pd.read_csv("data/pol_wb.csv",index_col=0)
print(pol.min())

### Armed forces personnel (% of total labor force) ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")
# Use military expenditure as additional variable
base_imp=multivariate_imp_bayes(base,"country",["MS.MIL.TOTL.TF.ZS"],vars_add=["MS.MIL.XPND.GD.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["MS.MIL.TOTL.TF.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["MS.MIL.TOTL.TF.ZS"])
base_imp_final["MS.MIL.TOTL.TF.ZS"] = base_imp_final["MS.MIL.TOTL.TF.ZS"].fillna(base_imp_mean["MS.MIL.TOTL.TF.ZS"])
base_imp_final["MS.MIL.TOTL.TF.ZS"] = base_imp_final["MS.MIL.TOTL.TF.ZS"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['armedforces_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"MS.MIL.TOTL.TF.ZS": 'armedforces_share'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["MS.MIL.TOTL.TF.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["armedforces_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","armedforces_share",'armedforces_share_id']],on=["year","gw_codes"],how="left")

### Military expenditure (% of GDP) ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")
# Use armed forces personnel as additional variable
base_imp=multivariate_imp_bayes(base,"country",["MS.MIL.XPND.GD.ZS"],vars_add=["MS.MIL.TOTL.TF.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["MS.MIL.XPND.GD.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["MS.MIL.XPND.GD.ZS"])
base_imp_final["MS.MIL.XPND.GD.ZS"] = base_imp_final["MS.MIL.XPND.GD.ZS"].fillna(base_imp_mean["MS.MIL.XPND.GD.ZS"])
base_imp_final["MS.MIL.XPND.GD.ZS"] = base_imp_final["MS.MIL.XPND.GD.ZS"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['milex_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"MS.MIL.XPND.GD.ZS": 'milex_share'})

# Validate
for c in base.country.unique():
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(base["year"].loc[base["country"]==c], base["MS.MIL.XPND.GD.ZS"].loc[base["country"]==c],c="black")
    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["milex_share"].loc[base_imp_final["country"]==c],c="black")
    axs[0].set_title(c)
    axs[0].set_title(c,size=20)
    if c=="Bhutan":
        plt.savefig("out/struc_missin6.eps",dpi=300,bbox_inches='tight')    
    plt.show()
    
# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","milex_share",'milex_share_id']],on=["year","gw_codes"],how="left")

### Control of Corruption: Estimate ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")
# Use other QoG indicators and tax revenue as additional variables
base_imp=multivariate_imp_bayes(base,"country",["CC.EST"],vars_add=["GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS"],max_iter=10,min_val=-10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["CC.EST"])
base_imp_mean=simple_imp_grouped(base,"country",["CC.EST"])
base_imp_final["CC.EST"] = base_imp_final["CC.EST"].fillna(base_imp_mean["CC.EST"])
base_imp_final["CC.EST"] = base_imp_final["CC.EST"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['corruption_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"CC.EST": 'corruption'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["CC.EST"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["corruption"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","corruption",'corruption_id']],on=["year","gw_codes"],how="left")

### Government Effectiveness: Estimate ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")
# Use other QoG indicators and tax revenue as additional variables
base_imp=multivariate_imp_bayes(base,"country",["GE.EST"],vars_add=["CC.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS"],max_iter=10,min_val=-10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["GE.EST"])
base_imp_mean=simple_imp_grouped(base,"country",["GE.EST"])
base_imp_final["GE.EST"] = base_imp_final["GE.EST"].fillna(base_imp_mean["GE.EST"])
base_imp_final["GE.EST"] = base_imp_final["GE.EST"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['effectiveness_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"GE.EST": 'effectiveness'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["GE.EST"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["effectiveness"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","effectiveness",'effectiveness_id']],on=["year","gw_codes"],how="left")

### Political Stability and Absence of Violence/Terrorism: Estimate ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")
# Use other QoG indicators and tax revenue as additional variables
base_imp=multivariate_imp_bayes(base,"country",["PV.EST"],vars_add=["CC.EST","GE.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS"],max_iter=10,min_val=-10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["PV.EST"])
base_imp_mean=simple_imp_grouped(base,"country",["PV.EST"])
base_imp_final["PV.EST"] = base_imp_final["PV.EST"].fillna(base_imp_mean["PV.EST"])
base_imp_final["PV.EST"] = base_imp_final["PV.EST"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['polvio_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"PV.EST": 'polvio'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["PV.EST"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["polvio"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","polvio",'polvio_id']],on=["year","gw_codes"],how="left")

### Regulatory Quality: Estimate ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")
# Use other QoG indicators and tax revenue as additional variables
base_imp=multivariate_imp_bayes(base,"country",["RQ.EST"],vars_add=["CC.EST","GE.EST","PV.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS"],max_iter=10,min_val=-10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["RQ.EST"])
base_imp_mean=simple_imp_grouped(base,"country",["RQ.EST"])
base_imp_final["RQ.EST"] = base_imp_final["RQ.EST"].fillna(base_imp_mean["RQ.EST"])
base_imp_final["RQ.EST"] = base_imp_final["RQ.EST"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['regu_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"RQ.EST": 'regu'})

# Validate
for c in base.country.unique():
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(base["year"].loc[base["country"]==c], base["RQ.EST"].loc[base["country"]==c],c="black")
    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["regu"].loc[base_imp_final["country"]==c],c="black")
    axs[0].set_title(c,size=20)
    if c=="Ivory Coast":
        plt.savefig("out/struc_missin3.eps",dpi=300,bbox_inches='tight')    
    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","regu",'regu_id']],on=["year","gw_codes"],how="left")

### Rule of Law: Estimate ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")
# Use other QoG indicators and tax revenue as additional variables
base_imp=multivariate_imp_bayes(base,"country",["RL.EST"],vars_add=["CC.EST","GE.EST","PV.EST","RQ.EST","VA.EST","GC.TAX.TOTL.GD.ZS"],max_iter=10,min_val=-10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["RL.EST"])
base_imp_mean=simple_imp_grouped(base,"country",["RL.EST"])
base_imp_final["RL.EST"] = base_imp_final["RL.EST"].fillna(base_imp_mean["RL.EST"])
base_imp_final["RL.EST"] = base_imp_final["RL.EST"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['law_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"RL.EST": 'law'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["RL.EST"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["law"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","law",'law_id']],on=["year","gw_codes"],how="left")

### Voice and Accountability: Estimate ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")
# Use other QoG indicators and tax revenue as additional variables
base_imp=multivariate_imp_bayes(base,"country",["VA.EST"],vars_add=["CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","GC.TAX.TOTL.GD.ZS"],max_iter=10,min_val=-10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["VA.EST"])
base_imp_mean=simple_imp_grouped(base,"country",["VA.EST"])
base_imp_final["VA.EST"] = base_imp_final["VA.EST"].fillna(base_imp_mean["VA.EST"])
base_imp_final["VA.EST"] = base_imp_final["VA.EST"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['account_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"VA.EST": 'account'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["VA.EST"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["account"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","account",'account_id']],on=["year","gw_codes"],how="left")

### Tax revenue (% of GDP) ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")
# Use QoG indicators as additional variables
base_imp=multivariate_imp_bayes(base,"country",["GC.TAX.TOTL.GD.ZS"],vars_add=["CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["GC.TAX.TOTL.GD.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["GC.TAX.TOTL.GD.ZS"])
base_imp_final["GC.TAX.TOTL.GD.ZS"] = base_imp_final["GC.TAX.TOTL.GD.ZS"].fillna(base_imp_mean["GC.TAX.TOTL.GD.ZS"])
base_imp_final["GC.TAX.TOTL.GD.ZS"] = base_imp_final["GC.TAX.TOTL.GD.ZS"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['tax_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"GC.TAX.TOTL.GD.ZS": 'tax'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["GC.TAX.TOTL.GD.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["tax"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","tax",'tax_id']],on=["year","gw_codes"],how="left")

### Fixed broadband subscriptions (per 100 people) ###

# Create base df, perform multiple imputation
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")
# Use as additional variables telephone subscriptions, Internet access, and mobile phone subscriptions
base_imp=multivariate_imp_bayes(base,"country",["IT.NET.BBND.P2"],vars_add=["IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["IT.NET.BBND.P2"])
base_imp_mean=simple_imp_grouped(base,"country",["IT.NET.BBND.P2"])
base_imp_final["IT.NET.BBND.P2"] = base_imp_final["IT.NET.BBND.P2"].fillna(base_imp_mean["IT.NET.BBND.P2"])
base_imp_final["IT.NET.BBND.P2"] = base_imp_final["IT.NET.BBND.P2"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['broadband_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"IT.NET.BBND.P2": 'broadband'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["IT.NET.BBND.P2"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["broadband"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","broadband",'broadband_id']],on=["year","gw_codes"],how="left")

### Fixed telephone subscriptions (per 100 people) ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")
# Use as additional variables broadband subscriptions, Internet access, and mobile subsciptions
base_imp=multivariate_imp_bayes(base,"country",["IT.MLT.MAIN.P2"],vars_add=["IT.NET.BBND.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["IT.MLT.MAIN.P2"])
base_imp_mean=simple_imp_grouped(base,"country",["IT.MLT.MAIN.P2"])
base_imp_final["IT.MLT.MAIN.P2"] = base_imp_final["IT.MLT.MAIN.P2"].fillna(base_imp_mean["IT.MLT.MAIN.P2"])
base_imp_final["IT.MLT.MAIN.P2"] = base_imp_final["IT.MLT.MAIN.P2"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['telephone_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"IT.MLT.MAIN.P2": 'telephone'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["IT.MLT.MAIN.P2"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["telephone"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","telephone",'telephone_id']],on=["year","gw_codes"],how="left")

### Individuals using the Internet (% of population) ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")
# Use as additional variables broadband, telephone and mobile subsciptions
base_imp=multivariate_imp_bayes(base,"country",["IT.NET.USER.ZS"],vars_add=["IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.CEL.SETS.P2"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["IT.NET.USER.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["IT.NET.USER.ZS"])
base_imp_final["IT.NET.USER.ZS"] = base_imp_final["IT.NET.USER.ZS"].fillna(base_imp_mean["IT.NET.USER.ZS"])
base_imp_final["IT.NET.USER.ZS"] = base_imp_final["IT.NET.USER.ZS"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['internet_use_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"IT.NET.USER.ZS": 'internet_use'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["IT.NET.USER.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["internet_use"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","internet_use",'internet_use_id']],on=["year","gw_codes"],how="left")

### Mobile cellular subscriptions (per 100 people) ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")
# Use as additional variables broadband, telephone and Internat
base_imp=multivariate_imp_bayes(base,"country",["IT.CEL.SETS.P2"],vars_add=["IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["IT.CEL.SETS.P2"])
base_imp_mean=simple_imp_grouped(base,"country",["IT.CEL.SETS.P2"])
base_imp_final["IT.CEL.SETS.P2"] = base_imp_final["IT.CEL.SETS.P2"].fillna(base_imp_mean["IT.CEL.SETS.P2"])
base_imp_final["IT.CEL.SETS.P2"] = base_imp_final["IT.CEL.SETS.P2"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['mobile_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"IT.CEL.SETS.P2": 'mobile'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["IT.CEL.SETS.P2"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["mobile"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","mobile",'mobile_id']],on=["year","gw_codes"],how="left")

### Electoral democracy index ###

vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
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

# Simple imputation (linear and if it fails mean) 
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
base_imp_final=linear_imp_grouped(base,"country",["v2x_libdem"])
base_imp_mean=simple_imp_grouped(base,"country",["v2x_libdem"])
base_imp_final["v2x_libdem"] = base_imp_final["v2x_libdem"].fillna(base_imp_mean["v2x_libdem"])

# Add missing id and rename variable
base_imp_final = base_imp_final.rename(columns={"v2x_libdem": 'libdem'})
base_imp_final['libdem_id'] = base["v2x_libdem"].isnull().astype(int)

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["v2x_libdem"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["libdem"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","libdem",'libdem_id']],on=["year","gw_codes"],how="left")

### Participatory democracy index ###

vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
base = base.rename(columns={"v2x_partipdem": 'partipdem'})
df_pol=pd.merge(left=df_pol,right=base[["year","gw_codes","partipdem"]],on=["year","gw_codes"],how="left")

### Deliberative democracy index ###

vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
base = base.rename(columns={"v2x_delibdem": 'delibdem'})
df_pol=pd.merge(left=df_pol,right=base[["year","gw_codes","delibdem"]],on=["year","gw_codes"],how="left")

### Egalitarian democracy index ###

vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
base = base.rename(columns={"v2x_egaldem": 'egaldem'})
df_pol=pd.merge(left=df_pol,right=base[["year","gw_codes","egaldem"]],on=["year","gw_codes"],how="left")

### Civil liberties index ###

vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
base = base.rename(columns={"v2x_civlib": 'civlib'})
df_pol=pd.merge(left=df_pol,right=base[["year","gw_codes","civlib"]],on=["year","gw_codes"],how="left")

### Physical violence index ###

vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
base = base.rename(columns={"v2x_clphy": 'phyvio'})
df_pol=pd.merge(left=df_pol,right=base[["year","gw_codes","phyvio"]],on=["year","gw_codes"],how="left")

### Political civil liberties index ###

vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
base = base.rename(columns={"v2x_clpol": 'pollib'})
df_pol=pd.merge(left=df_pol,right=base[["year","gw_codes","pollib"]],on=["year","gw_codes"],how="left")

### Private civil liberties index ###

vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
base = base.rename(columns={"v2x_clpriv": 'privlib'})
df_pol=pd.merge(left=df_pol,right=base[["year","gw_codes","privlib"]],on=["year","gw_codes"],how="left")

### Exclusion by Socio-Economic Group index ###

# Create base df, perform multiple imputation 
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
# Use other exclusion indicators as additional variables
base_imp=multivariate_imp_bayes(base,"country",["v2xpe_exlecon"],vars_add=["v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["v2xpe_exlecon"])
base_imp_mean=simple_imp_grouped(base,"country",["v2xpe_exlecon"])
base_imp_final["v2xpe_exlecon"] = base_imp_final["v2xpe_exlecon"].fillna(base_imp_mean["v2xpe_exlecon"])
base_imp_final["v2xpe_exlecon"] = base_imp_final["v2xpe_exlecon"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['execon_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"v2xpe_exlecon": 'execon'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["v2xpe_exlecon"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["execon"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","execon",'execon_id']],on=["year","gw_codes"],how="left")

### Exclusion by Gender index ###

# Create base df, perform multiple imputation 
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
# Use other exclusion indicators as additional variables
base_imp=multivariate_imp_bayes(base,"country",["v2xpe_exlgender"],vars_add=["v2xpe_exlecon","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["v2xpe_exlgender"])
base_imp_mean=simple_imp_grouped(base,"country",["v2xpe_exlgender"])
base_imp_final["v2xpe_exlgender"] = base_imp_final["v2xpe_exlgender"].fillna(base_imp_mean["v2xpe_exlgender"])
base_imp_final["v2xpe_exlgender"] = base_imp_final["v2xpe_exlgender"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['exgender_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"v2xpe_exlgender": 'exgender'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["v2xpe_exlgender"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["exgender"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","exgender",'exgender_id']],on=["year","gw_codes"],how="left")

### Exclusion by Urban-Rural Location index ###

# Create base df, perform multiple imputation 
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
# Use other exclusion indicators as additional variables
base_imp=multivariate_imp_bayes(base,"country",["v2xpe_exlgeo"],vars_add=["v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlpol","v2xpe_exlsocgr"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["v2xpe_exlgeo"])
base_imp_mean=simple_imp_grouped(base,"country",["v2xpe_exlgeo"])
base_imp_final["v2xpe_exlgeo"] = base_imp_final["v2xpe_exlgeo"].fillna(base_imp_mean["v2xpe_exlgeo"])
base_imp_final["v2xpe_exlgeo"] = base_imp_final["v2xpe_exlgeo"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['exgeo_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"v2xpe_exlgeo": 'exgeo'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["v2xpe_exlgeo"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["exgeo"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","exgeo",'exgeo_id']],on=["year","gw_codes"],how="left")

### Exclusion by Political Group index ###

# Create base df, perform multiple imputation 
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
# Use other exclusion indicators as additional variables
base_imp=multivariate_imp_bayes(base,"country",["v2xpe_exlpol"],vars_add=["v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlsocgr"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["v2xpe_exlpol"])
base_imp_mean=simple_imp_grouped(base,"country",["v2xpe_exlpol"])
base_imp_final["v2xpe_exlpol"] = base_imp_final["v2xpe_exlpol"].fillna(base_imp_mean["v2xpe_exlpol"])
base_imp_final["v2xpe_exlpol"] = base_imp_final["v2xpe_exlpol"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['expol_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"v2xpe_exlpol": 'expol'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["v2xpe_exlpol"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["expol"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","expol",'expol_id']],on=["year","gw_codes"],how="left")

### Exclusion by Social Group index ###

# Create base df, perform multiple imputation 
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
# Use other exclusion indicators as additional variables
base_imp=multivariate_imp_bayes(base,"country",["v2xpe_exlsocgr"],vars_add=["v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["v2xpe_exlsocgr"])
base_imp_mean=simple_imp_grouped(base,"country",["v2xpe_exlsocgr"])
base_imp_final["v2xpe_exlsocgr"] = base_imp_final["v2xpe_exlsocgr"].fillna(base_imp_mean["v2xpe_exlsocgr"])
base_imp_final["v2xpe_exlsocgr"] = base_imp_final["v2xpe_exlsocgr"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['exsoc_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"v2xpe_exlsocgr": 'exsoc'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["v2xpe_exlsocgr"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["exsoc"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","exsoc",'exsoc_id']],on=["year","gw_codes"],how="left")

### Government Internet shut down in practice ###

# Create base df, perform multiple imputation 
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
# Use Internet filter as additional variable
base_imp=multivariate_imp_bayes(base,"country",["v2smgovshut"],vars_add=["v2smgovfilprc"],max_iter=10,min_val=-10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["v2smgovshut"])
base_imp_mean=simple_imp_grouped(base,"country",["v2smgovshut"])
base_imp_final["v2smgovshut"] = base_imp_final["v2smgovshut"].fillna(base_imp_mean["v2smgovshut"])
base_imp_final["v2smgovshut"] = base_imp_final["v2smgovshut"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['shutdown_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"v2smgovshut": 'shutdown'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["v2smgovshut"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["shutdown"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","shutdown",'shutdown_id']],on=["year","gw_codes"],how="left")

### Government Internet filtering in practice ###

# Create base df, perform multiple imputation 
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
# Use Internet shut down as additional variable
base_imp=multivariate_imp_bayes(base,"country",["v2smgovfilprc"],vars_add=["v2smgovshut"],max_iter=10,min_val=-10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["v2smgovfilprc"])
base_imp_mean=simple_imp_grouped(base,"country",["v2smgovfilprc"])
base_imp_final["v2smgovfilprc"] = base_imp_final["v2smgovfilprc"].fillna(base_imp_mean["v2smgovfilprc"])
base_imp_final["v2smgovfilprc"] = base_imp_final["v2smgovfilprc"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['filter_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"v2smgovfilprc": 'filter'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["v2smgovfilprc"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["filter"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","filter",'filter_id']],on=["year","gw_codes"],how="left")

### Number of months that leader has been in power ###

reign=pd.read_csv("data/data_out/reign_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=reign[["year",
                                    "gw_codes",
                                    "tenure_months", # Number of months that leader has been in power 
                                    "dem_duration", # Logged number of months that a country is democratic
                                    "elections", # Election for leadership taking place in that year
                                    "lastelection" # Time since the last election for leadership (decay function)
                                    ]],on=["year","gw_codes"],how="left")

base_imp_final=linear_imp_grouped(base,"country",["tenure_months"])
base_imp_mean=simple_imp_grouped(base,"country",["tenure_months"])
base_imp_final["tenure_months"] = base_imp_final["tenure_months"].fillna(base_imp_mean["tenure_months"])
base_imp_final['tenure_months_id'] = base["tenure_months"].isnull().astype(int)

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["tenure_months"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["tenure_months"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","tenure_months",'tenure_months_id']],on=["year","gw_codes"],how="left")
  
### Logged number of months that a country is democratic ###

reign=pd.read_csv("data/data_out/reign_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=reign[["year","gw_codes","tenure_months","dem_duration","elections","lastelection"]],on=["year","gw_codes"],how="left")
base_imp_final=linear_imp_grouped(base,"country",["dem_duration"])
base_imp_mean=simple_imp_grouped(base,"country",["dem_duration"])
base_imp_final["dem_duration"] = base_imp_final["dem_duration"].fillna(base_imp_mean["dem_duration"])
base_imp_final['dem_duration_id'] = base["dem_duration"].isnull().astype(int)

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["dem_duration"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["dem_duration"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","dem_duration",'dem_duration_id']],on=["year","gw_codes"],how="left")

### Election for leadership taking place in that year ###

reign=pd.read_csv("data/data_out/reign_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=reign[["year","gw_codes","tenure_months","dem_duration","elections","lastelection"]],on=["year","gw_codes"],how="left") 
base_imp_final=linear_imp_grouped(base,"country",["elections"])
base_imp_mean=simple_imp_grouped(base,"country",["elections"])
base_imp_final["elections"] = base_imp_final["elections"].fillna(base_imp_mean["elections"])
base_imp_final['elections_id'] = base["elections"].isnull().astype(int)

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["elections"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["elections"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Dichotomize post hoc
base_imp_final.loc[base_imp_final["elections"]>0.5,"elections"]=1
base_imp_final.loc[base_imp_final["elections"]<=0.5,"elections"]=0
base_imp_final["elections"].unique()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","elections",'elections_id']],on=["year","gw_codes"],how="left")

### Time since the last election for leadership (decay function) ###

reign=pd.read_csv("data/data_out/reign_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=reign[["year","gw_codes","tenure_months","dem_duration","elections","lastelection"]],on=["year","gw_codes"],how="left")
base_imp_final=linear_imp_grouped(base,"country",["lastelection"])
base_imp_mean=simple_imp_grouped(base,"country",["lastelection"])
base_imp_final["lastelection"] = base_imp_final["lastelection"].fillna(base_imp_mean["lastelection"])
base_imp_final['lastelection_id'] = base["lastelection"].isnull().astype(int)

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["lastelection"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["lastelection"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","lastelection",'lastelection_id']],on=["year","gw_codes"],how="left")
  
# Final df
print(df_pol.isna().any().any())
print(df_pol.min())

# Check datatypes and convert floats to integer
for c in df_pol.columns: 
    print(c,df_pol[c].dtypes)
df_pol['elections']=df_pol['elections'].astype('int64')

# Note that tenure_months is theoretically an integer variable, but the imputation
# returns floats when interpolating. Due to the imputation, the variables is kept as float. 
df_pol['tenure_months'].unique() # check levels
  
# Save
df_pol.to_csv("out/df_pol_full.csv") 
print(df_pol.duplicated(subset=['year',"country","gw_codes"]).any())
print(df_pol.duplicated(subset=['year',"country"]).any())
print(df_pol.duplicated(subset=['year',"gw_codes"]).any())

##############################################
### Geography, environment & climate theme ###
##############################################

# Initiate
df_geog=df_out[["year","gw_codes","country"]].copy()

# Load wb data, previously retrived with the WB api
geog=pd.read_csv("data/geog_wb.csv",index_col=0)
print(geog.min())

### Land area (sq. km) ###

base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=geog[["year","gw_codes","AG.LND.TOTL.K2","AG.LND.FRST.ZS","AG.LND.PRCP.MM","ER.H2O.FWST.ZS","EN.GHG.CO2.MT.CE.AR5","AG.LND.AGRI.ZS","AG.LND.ARBL.ZS"]],on=["year","gw_codes"],how="left")
base_imp_final=linear_imp_grouped(base,"country",["AG.LND.TOTL.K2"])
base_imp_mean=simple_imp_grouped(base,"country",["AG.LND.TOTL.K2"])
base_imp_final["AG.LND.TOTL.K2"] = base_imp_final["AG.LND.TOTL.K2"].fillna(base_imp_mean["AG.LND.TOTL.K2"])
base_imp_final = base_imp_final.rename(columns={"AG.LND.TOTL.K2": 'land'})
base_imp_final['land_id'] = base["AG.LND.TOTL.K2"].isnull().astype(int)

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["AG.LND.TOTL.K2"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["land"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_geog=pd.merge(left=df_geog,right=base_imp_final[["year","gw_codes","land",'land_id']],on=["year","gw_codes"],how="left")

### Average Mean Surface Air Temperature ###

temp=pd.read_csv("data/data_out/temp_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=temp[["year","gw_codes","temp"]],on=["year","gw_codes"],how="left")  
base_imp_final=linear_imp_grouped(base,"country",["temp"])
base_imp_mean=simple_imp_grouped(base,"country",["temp"])
base_imp_final["temp"] = base_imp_final["temp"].fillna(base_imp_mean["temp"])
base_imp_final['temp_id'] = base["temp"].isnull().astype(int)

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["temp"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["temp"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

df_geog=pd.merge(left=df_geog,right=base_imp_final[["year","gw_codes","temp",'temp_id']],on=["year","gw_codes"],how="left")

### Forest area (% of land area) ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=geog[["year","gw_codes","AG.LND.TOTL.K2","AG.LND.FRST.ZS","AG.LND.PRCP.MM","ER.H2O.FWST.ZS","EN.GHG.CO2.MT.CE.AR5","AG.LND.AGRI.ZS","AG.LND.ARBL.ZS"]],on=["year","gw_codes"],how="left")
# Use land area and CO2 emissions as additional variables
base_imp=multivariate_imp_bayes(base,"country",["AG.LND.FRST.ZS"],vars_add=["AG.LND.TOTL.K2","EN.GHG.CO2.MT.CE.AR5"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["AG.LND.FRST.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["AG.LND.FRST.ZS"])
base_imp_final["AG.LND.FRST.ZS"] = base_imp_final["AG.LND.FRST.ZS"].fillna(base_imp_mean["AG.LND.FRST.ZS"])
base_imp_final["AG.LND.FRST.ZS"] = base_imp_final["AG.LND.FRST.ZS"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['forest_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"AG.LND.FRST.ZS": 'forest'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["AG.LND.FRST.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["forest"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_geog=pd.merge(left=df_geog,right=base_imp_final[["year","gw_codes","forest",'forest_id']],on=["year","gw_codes"],how="left")

### CO2 emissions (kt) ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=geog[["year","gw_codes","AG.LND.TOTL.K2","AG.LND.FRST.ZS","AG.LND.PRCP.MM","ER.H2O.FWST.ZS","EN.GHG.CO2.MT.CE.AR5","AG.LND.AGRI.ZS","AG.LND.ARBL.ZS"]],on=["year","gw_codes"],how="left")
# Use land area and forest area as additional variables
base_imp=multivariate_imp_bayes(base,"country",["EN.GHG.CO2.MT.CE.AR5"],vars_add=["AG.LND.TOTL.K2","AG.LND.FRST.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["EN.GHG.CO2.MT.CE.AR5"])
base_imp_mean=simple_imp_grouped(base,"country",["EN.GHG.CO2.MT.CE.AR5"])
base_imp_final["EN.GHG.CO2.MT.CE.AR5"] = base_imp_final["EN.GHG.CO2.MT.CE.AR5"].fillna(base_imp_mean["EN.GHG.CO2.MT.CE.AR5"])
base_imp_final["EN.GHG.CO2.MT.CE.AR5"] = base_imp_final["EN.GHG.CO2.MT.CE.AR5"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['co2_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"EN.GHG.CO2.MT.CE.AR5": 'co2'})

# Validate
for c in base.country.unique():
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(base["year"].loc[base["country"]==c], base["EN.GHG.CO2.MT.CE.AR5"].loc[base["country"]==c],c="black")
    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["co2"].loc[base_imp_final["country"]==c],c="black")
    axs[0].set_title(c,size=20)
    if c=="Serbia":
        plt.savefig("out/struc_missin4.eps",dpi=300,bbox_inches='tight')
    plt.show()

# Merge
df_geog=pd.merge(left=df_geog,right=base_imp_final[["year","gw_codes","co2",'co2_id']],on=["year","gw_codes"],how="left")

### Average precipitation in depth (mm per year) ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=geog[["year","gw_codes","AG.LND.TOTL.K2","AG.LND.FRST.ZS","AG.LND.PRCP.MM","ER.H2O.FWST.ZS","EN.GHG.CO2.MT.CE.AR5","AG.LND.AGRI.ZS","AG.LND.ARBL.ZS"]],on=["year","gw_codes"],how="left")
# Use water stress as additional variable
base_imp=multivariate_imp_bayes(base,"country",["AG.LND.PRCP.MM"],vars_add=["ER.H2O.FWST.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["AG.LND.PRCP.MM"])
base_imp_mean=simple_imp_grouped(base,"country",["AG.LND.PRCP.MM"])
base_imp_final["AG.LND.PRCP.MM"] = base_imp_final["AG.LND.PRCP.MM"].fillna(base_imp_mean["AG.LND.PRCP.MM"])
base_imp_final["AG.LND.PRCP.MM"] = base_imp_final["AG.LND.PRCP.MM"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['percip_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"AG.LND.PRCP.MM": 'percip'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["AG.LND.PRCP.MM"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["percip"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_geog=pd.merge(left=df_geog,right=base_imp_final[["year","gw_codes","percip",'percip_id']],on=["year","gw_codes"],how="left")

### Level of water stress: freshwater withdrawal as a proportion of available freshwater resources ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=geog[["year","gw_codes","AG.LND.TOTL.K2","AG.LND.FRST.ZS","AG.LND.PRCP.MM","ER.H2O.FWST.ZS","EN.GHG.CO2.MT.CE.AR5","AG.LND.AGRI.ZS","AG.LND.ARBL.ZS"]],on=["year","gw_codes"],how="left")
# Use percipitation as additional variable
base_imp=multivariate_imp_bayes(base,"country",["ER.H2O.FWST.ZS"],vars_add=["AG.LND.PRCP.MM"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["ER.H2O.FWST.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["ER.H2O.FWST.ZS"])
base_imp_final["ER.H2O.FWST.ZS"] = base_imp_final["ER.H2O.FWST.ZS"].fillna(base_imp_mean["ER.H2O.FWST.ZS"])
base_imp_final["ER.H2O.FWST.ZS"] = base_imp_final["ER.H2O.FWST.ZS"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['waterstress_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"ER.H2O.FWST.ZS": 'waterstress'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["ER.H2O.FWST.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["waterstress"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_geog=pd.merge(left=df_geog,right=base_imp_final[["year","gw_codes","waterstress",'waterstress_id']],on=["year","gw_codes"],how="left")

### Agricultural land (% of land area) ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=geog[["year","gw_codes","AG.LND.TOTL.K2","AG.LND.FRST.ZS","AG.LND.PRCP.MM","ER.H2O.FWST.ZS","EN.GHG.CO2.MT.CE.AR5","AG.LND.AGRI.ZS","AG.LND.ARBL.ZS"]],on=["year","gw_codes"],how="left")
# Use arable land as additional variable
base_imp=multivariate_imp_bayes(base,"country",["AG.LND.AGRI.ZS"],vars_add=["AG.LND.ARBL.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["AG.LND.AGRI.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["AG.LND.AGRI.ZS"])
base_imp_final["AG.LND.AGRI.ZS"] = base_imp_final["AG.LND.AGRI.ZS"].fillna(base_imp_mean["AG.LND.AGRI.ZS"])
base_imp_final["AG.LND.AGRI.ZS"] = base_imp_final["AG.LND.AGRI.ZS"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['agri_land_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"AG.LND.AGRI.ZS": 'agri_land'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["AG.LND.AGRI.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["agri_land"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_geog=pd.merge(left=df_geog,right=base_imp_final[["year","gw_codes","agri_land",'agri_land_id']],on=["year","gw_codes"],how="left")

### Arable land (% of land area) ###

# Create base df, perform multiple imputation 
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=geog[["year","gw_codes","AG.LND.TOTL.K2","AG.LND.FRST.ZS","AG.LND.PRCP.MM","ER.H2O.FWST.ZS","EN.GHG.CO2.MT.CE.AR5","AG.LND.AGRI.ZS","AG.LND.ARBL.ZS"]],on=["year","gw_codes"],how="left")
# Use agricultural land as additional variable
base_imp=multivariate_imp_bayes(base,"country",["AG.LND.ARBL.ZS"],vars_add=["AG.LND.AGRI.ZS"],max_iter=10)

# Simple imputation (linear and if it fails mean), and if both fail
# use imputation from multiple 
base_imp_final=linear_imp_grouped(base,"country",["AG.LND.ARBL.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["AG.LND.ARBL.ZS"])
base_imp_final["AG.LND.ARBL.ZS"] = base_imp_final["AG.LND.ARBL.ZS"].fillna(base_imp_mean["AG.LND.ARBL.ZS"])
base_imp_final["AG.LND.ARBL.ZS"] = base_imp_final["AG.LND.ARBL.ZS"].fillna(base_imp["imp"])

# Add missing id and rename variable
base_imp_final['arable_land_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"AG.LND.ARBL.ZS": 'arable_land'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["AG.LND.ARBL.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["arable_land"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_geog=pd.merge(left=df_geog,right=base_imp_final[["year","gw_codes","arable_land",'arable_land_id']],on=["year","gw_codes"],how="left")

### Terrain Ruggedness Index ###

base=df_out[["year","gw_codes","country"]].copy()
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
            
### No neighbors ###

# Load neighbors, and replace missing with 0, missing means that country has no neighbors
neighbors=pd.read_csv("data/data_out/cy_neighbors.csv",index_col=0)
neighbors['neighbors'].fillna(0, inplace=True)

# If no neighbors, code as 1, zero otherwise
neighbors["no_neigh"]=0
neighbors.loc[neighbors["neighbors"]==0,"no_neigh"]=1

# Merge
df_geog=pd.merge(left=df_geog,right=neighbors[["year","gw_codes","no_neigh"]],on=["year","gw_codes"],how="left")

### Neighbor not democratic  ###

# First obtain value for regime type, libdem from vdem

# Simple imputation (linear and if it fails mean) 
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")
base_imp_final=linear_imp_grouped(base,"country",["v2x_libdem"])
base_imp_mean=simple_imp_grouped(base,"country",["v2x_libdem"])
base_imp_final["v2x_libdem"] = base_imp_final["v2x_libdem"].fillna(base_imp_mean["v2x_libdem"])
base_imp_final = base_imp_final.rename(columns={"v2x_libdem": 'libdem'})
base_imp_final['libdem_id_neigh'] = base["v2x_libdem"].isnull().astype(int)

# Dichotomize, takes value 1 if libdem  > 0.5
dichotomize(base_imp_final,"libdem","d_libdem",0.5)

# Second obtain value for neighbors

# Merge neighbors to df
neighbors=pd.read_csv("data/data_out/cy_neighbors.csv",index_col=0)
gw_codes=pd.read_csv("data/df_ccodes_gw.csv")
# Country names in neighbors file and df_codes need to be the same
gw_codes_s=gw_codes.loc[gw_codes["end"]>=1989]
base_imp_final=pd.merge(left=base_imp_final,right=neighbors[["gw_codes","year","neighbors"]],on=["year","gw_codes"],how="left")

base_imp_final["neighbors_non_dem"]=0
# Loop through every observation
for i in range(len(base_imp_final)):
    # If no neighbors pass on
    if pd.isna(base_imp_final["neighbors"].iloc[i]): 
        pass
    
    # Get list of neighbors and set counter to zero
    else:   
        lst=base_imp_final["neighbors"].iloc[i].split(';')
        counts=0
        
        # For each neighbor       
        for x in lst:
            c=int(gw_codes_s["gw_codes"].loc[gw_codes_s["country"]==x].iloc[0])
            #  (if available) 
            if base_imp_final["d_libdem"].loc[(base_imp_final["year"]==base_imp_final["year"].iloc[i])&(base_imp_final["gw_codes"]==c)].empty==False:
                # Count the ones which are not democratic, if d_libdem==0
                if base_imp_final["d_libdem"].loc[(base_imp_final["year"]==base_imp_final["year"].iloc[i])&(base_imp_final["gw_codes"]==c)].iloc[0]==0:
                    counts=+1

        # If larger than zero, add to df        
        if counts>0:
            base_imp_final.iloc[i, base_imp_final.columns.get_loc('neighbors_non_dem')] = counts
            
# Dichotomize and merge ---> Country has at least one non-democratic neighbor
dichotomize(base_imp_final,"neighbors_non_dem","d_neighbors_non_dem",0)
df_geog=pd.merge(left=df_geog,right=base_imp_final[["year","gw_codes","d_neighbors_non_dem",'libdem_id_neigh']],on=["year","gw_codes"],how="left")

# Final df
print(df_geog.isna().any())
print(df_geog.min())

# Check datatypes and convert floats to integer
print(df_geog.dtypes)
df_geog['cont_africa']=df_geog['cont_africa'].astype('int64')
df_geog['cont_asia']=df_geog['cont_asia'].astype('int64')

# Save
df_geog.to_csv("out/df_geog_full.csv")
print(df_geog.duplicated(subset=['year',"country","gw_codes"]).any())
print(df_geog.duplicated(subset=['year',"country"]).any())
print(df_geog.duplicated(subset=['year',"gw_codes"]).any())




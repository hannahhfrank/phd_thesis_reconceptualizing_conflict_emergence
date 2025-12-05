import pandas as pd
import numpy as np
     
# Ethnic Power Relations (EPR) Dataset Family 2021 
# Available here: https://icr.ethz.ch/data/epr/
# Codebook: https://icr.ethz.ch/data/epr/core/EPR_2021_Codebook_EPR.pdf
erp = pd.read_csv("EPR-2021.csv")

# STEP 1: Convert data to ethnic group-year level

# Define out df
df=pd.DataFrame()

# Loop through every entry in data
for i in range(len(erp)):
    # Get the range 
    date = list(range(erp['from'].iloc[i],erp['to'].iloc[i]+1))
    # and obtain one onservation per year
    for x in range(0, len(date)):
        s = {'year':date[x],'gw_codes':erp['gwid'].iloc[i],'country':erp['statename'].iloc[i],'group':erp['group'].iloc[i],'group_id':erp['groupid'].iloc[i],'gwgroupid':erp['gwgroupid'].iloc[i],'group_size':erp['size'].iloc[i],'group_status':erp['status'].iloc[i]}
        s = pd.DataFrame(data=s,index=[i])
        df = pd.concat([df,s])  
        
# Add religion, language, race data (within ethnic group)        

# Ethnic Power Relations - Ethnic Dimensions (EPR-ED)
# Available here: https://icr.ethz.ch/data/epr/
# Codebook: https://icr.ethz.ch/data/epr/ed/EPR_2021_Codebook_ED.pdf
erp2 = pd.read_csv("ED-2021.csv")

# Merge on gwgroupid
df=pd.merge(df,erp2[["gwgroupid","rel1_size","rel2_size","rel3_size","lang1_size","lang2_size","lang3_size","pheno1_size","pheno2_size","pheno3_size"]],on=["gwgroupid"],how="left")

# Some ethnic groups have missings, because they are not included in erp2
# Countries with missing ethnic groups are Haiti, Dominican Republic, 
# Jamaica, Barbados, Chile, Ireland, Netherlands, Portugal, Germany,
# German Democratic Republic, Hungary, Czech Republic, Sweden, 
# Norway, Denmark, Burkina Faso, Central African Republic, Chad, Somalia, Lesotho, 
# Swaziland, Tunisia, Iraq, Yemen, People's Republic of, United Arab Emirates, Oman, 
# Korea, People's Republic of., Korea, Republic of, East Timor

# Missing values are dropped when calculating the fractionalization below. 
# For cases with only missing values (e.g., missing values in 'rel1_size','rel2_size','rel3_size'), 
# fractionalization takes a value of zero.

# STEP 2: Calculate within group fractionalization

# Add empty columns
df['rel_frac'] = np.nan
df['lang_frac'] = np.nan
df['race_frac'] = np.nan

# Loop through every observation
for i in range(len(df)):
    
    # (1) Religious fractionalization
    # If all ethnic groups have missing values, fractionlization takes value of 0
    if len(df[['rel1_size','rel2_size','rel3_size']].iloc[i].dropna())==0: 
        df['rel_frac'].iloc[i]=0
    # else drop missing values and calculate fractionalization
    else: 
        df['rel_frac'].iloc[i]=1-(np.sum(np.square(df[['rel1_size','rel2_size','rel3_size']].iloc[i].dropna().values)))
   
    # (2) Linguistic fractionalization
    # If all ethnic groups have missing values, fractionlization takes value of 0   
    if len(df[['lang1_size','lang2_size','lang3_size']].iloc[i].dropna())==0: 
        df['lang_frac'].iloc[i]=0
    # else drop missing values and calculate fractionalization
    else:
        df['lang_frac'].iloc[i]=1-(np.sum(np.square(df[['lang1_size','lang2_size','lang3_size']].iloc[i].dropna().values)))
   
    # (3) Race fractionalization
    # If all ethnic groups have missing values, fractionlization takes value of 0
    if len(df[['pheno1_size','pheno2_size','pheno3_size']].iloc[i].dropna())==0: 
        df['race_frac'].iloc[i]=0
    # else drop missing values and calculate fractionalization
    else:
        df['race_frac'].iloc[i]=1-(np.sum(np.square(df[['pheno1_size','pheno2_size','pheno3_size']].iloc[i].dropna().values)))

# Check some nonsensical values 
# Some values are negative because the proportions of relative sizes do not sum
# up to one --> cap at zero
df['rel_frac'] = df['rel_frac'].apply(lambda x: max(0, x))
df['race_frac'] = df['race_frac'].apply(lambda x: max(0, x))
# The affected ethnic groups are 71005100 and 66303000 (could this be a coding error?)

# Note: Another odd value occurs for race fractionalization, when 
# 'pheno1_size','pheno2_size','pheno3_size' all have a value of 0, the fractionalization
# becomes 1. The value is kept like this (although zero might mean na?) 

# STEP 3: Aggregate data from group to country-year level

# For status, the share of each ethnic group falling in one status is summed
# The between group ethnic fractionalization is calculated
# And the within group mean fractionalization is calculated

# Warning: This approach does not take into account the start and end dates for countries
# and therefore should always be merged with the appropriate sample.

# Define out df
df_agg=pd.DataFrame()

# Get years
date = list(range(1989, 2022, 1))

# Loop through every country
for c in df.gw_codes.unique():
    # For each year
    for d in date:
        # Subset data to add
        s = {'year':d,
             'gw_codes':c,
             'country':df["country"].loc[(df["gw_codes"]==c)].iloc[0],
             'group_counts':len(df.loc[(df["gw_codes"]==c)&(df["year"]==d)]),
             'group_names':"-".join(df["group"].loc[(df["gw_codes"]==c)&(df["year"]==d)]),
             'monopoly_share':df["group_size"].loc[(df["gw_codes"]==c)&(df["year"]==d)&(df["group_status"]=="MONOPOLY")].sum(),
             'discriminated_share':df["group_size"].loc[(df["gw_codes"]==c)&(df["year"]==d)&(df["group_status"]=="DISCRIMINATED")].sum(),
             'powerless_share':df["group_size"].loc[(df["gw_codes"]==c)&(df["year"]==d)&(df["group_status"]=="POWERLESS")].sum(),
             'dominant_share':df["group_size"].loc[(df["gw_codes"]==c)&(df["year"]==d)&(df["group_status"]=="DOMINANT")].sum(),             
             'ethnic_frac':1-(np.sum(np.square(df["group_size"].loc[(df["gw_codes"]==c)&(df["year"]==d)].values))),
             'rel_frac':df["rel_frac"].loc[(df["gw_codes"]==c)&(df["year"]==d)].mean(),
             'lang_frac':df["lang_frac"].loc[(df["gw_codes"]==c)&(df["year"]==d)].mean(),
             'race_frac':df["race_frac"].loc[(df["gw_codes"]==c)&(df["year"]==d)].mean(),
             }
        s = pd.DataFrame(data=s,index=[0])
        df_agg = pd.concat([df_agg,s])  

# Manual fix: Replace ethnic fractionalization with na if group count is 0.
# Based on the fractionalization formula, a value of 1 is returned if group sizes take only na, 
# which is a nonsensical value. 
df_agg.loc[df_agg["group_counts"]==0,"ethnic_frac"]=np.nan

# Check if df is complete
counts = df_agg.groupby('country').size()

# Save data 
df_agg = df_agg.sort_values(by=["gw_codes","year"])
df_agg.reset_index(drop=True,inplace=True)
df_agg.to_csv("data_out/epr_cy.csv",sep=',')
print(df_agg.duplicated(subset=['year',"country","gw_codes"]).any())
print(df_agg.duplicated(subset=['year',"country"]).any())
print(df_agg.duplicated(subset=['year',"gw_codes"]).any())
print(df_agg.isnull().any())
df_agg.dtypes







import pandas as pd
import wbgapi as wb

# The World Bank data is loaded via the API, using the wbgapi library: https://pypi.org/project/wbgapi/
# But can be manually downloaded here: https://databank.worldbank.org/source/world-development-indicators

########################
### (1) Demography I ###
########################

# Specify variables for call
feat_dev = ["SP.POP.TOTL", # Population, total
            "EN.POP.DNST", # Population density (people per sq. km of land area)'
            "SP.URB.TOTL.IN.ZS", # Urban population (% of total population)
            "SP.RUR.TOTL.ZS", # Rural population (% of total population)
            ] 

# Import country codes
df_ccodes = pd.read_csv("df_ccodes.csv")

# Specify countries for call  
c_list=list(df_ccodes.iso_alpha3)
c_list=[char for char in c_list if char != "XYZ"] # Exclude Taiwan

# Specify years for call
years=list(range(1989, 2024, 1))

# Define out df
wdi = pd.DataFrame()

# Get data for each year and merge
for i in years:
    print(i)
    wdi_s = wb.data.DataFrame(feat_dev, c_list, [i])
    wdi_s.reset_index(inplace=True)
    wdi_s["year"] = i
    wdi = pd.concat([wdi, wdi_s], ignore_index=True)  
    
# Add country and country codes: Merge GW and ACLED country codes over WB country codes 
df_ccodes_s = df_ccodes[["country",'gw_codes', "acled_codes", "iso_alpha3"]]
wdi_final = pd.merge(wdi, df_ccodes_s, how='left', left_on=['economy'], right_on=['iso_alpha3'])

# Drop duplicates WB country codes
wdi_final = wdi_final.drop(columns=['economy'])
        
# Sort columns, so that year, country and country codes appear at beginning
wdi_final = wdi_final[['country','year','acled_codes','iso_alpha3','gw_codes'] + [c for c in wdi_final.columns if c not in ['country','year','acled_codes','iso_alpha3','gw_codes']]]
 
# Print head of df to confirm load   
print("Obtained data")
print(wdi_final.head())
    
# Sort and reset index
wdi_final = wdi_final.sort_values(by=["iso_alpha3", 'year'])
wdi_final = wdi_final.reset_index(drop=True)

# Save data  
wdi_final.to_csv("demog_wb.csv") 
print(wdi_final.duplicated(subset=["year","gw_codes","country"]).any())
print(wdi_final.duplicated(subset=["year","country"]).any())
print(wdi_final.duplicated(subset=["year","gw_codes"]).any())
print(wdi_final.dtypes)

#########################
### (2) Demography II ###
#########################

# Specify variables for call
feat_dev = ["SP.POP.TOTL.MA.ZS", # Population, male (% of total population)
            "SP.POP.0014.MA.ZS", # 	Population ages 0-14, male (% of male population)
            "SP.POP.1519.MA.5Y", # Population ages 15-19, male (% of male population)
            "SP.POP.2024.MA.5Y", # 	Population ages 20-24, male (% of male population)
            "SP.POP.2529.MA.5Y", # 	Population ages 25-29, male (% of male population)
            "SP.POP.3034.MA.5Y"] # 	Population ages 30-34, male (% of male population)

# Import country codes
df_ccodes = pd.read_csv("df_ccodes.csv")

# Specify countries for call  
c_list=list(df_ccodes.iso_alpha3)
c_list=[char for char in c_list if char != "XYZ"] # Exclude Taiwan

# Specify years for call
years=list(range(1989, 2024, 1))

# Define out df
wdi = pd.DataFrame()

# Get data for each year and merge
for i in years:
    print(i)
    wdi_s = wb.data.DataFrame(feat_dev, c_list, [i])
    wdi_s.reset_index(inplace=True)
    wdi_s["year"] = i
    wdi = pd.concat([wdi, wdi_s], ignore_index=True)  
    
# Add country and country codes: Merge GW and ACLED country codes over WB country codes 
df_ccodes_s = df_ccodes[["country",'gw_codes', "acled_codes", "iso_alpha3"]]
wdi_final = pd.merge(wdi, df_ccodes_s, how='left', left_on=['economy'], right_on=['iso_alpha3'])

# Drop duplicates WB country codes
wdi_final = wdi_final.drop(columns=['economy'])
        
# Sort columns, so that year, country and country codes appear at beginning
wdi_final = wdi_final[['country','year','acled_codes','iso_alpha3','gw_codes'] + [c for c in wdi_final.columns if c not in ['country','year','acled_codes','iso_alpha3','gw_codes']]]
 
# Print head of df to confirm load   
print("Obtained data")
print(wdi_final.head())
    
# Sort and reset index
wdi_final = wdi_final.sort_values(by=["iso_alpha3", 'year'])
wdi_final = wdi_final.reset_index(drop=True)

# Save data  
wdi_final.to_csv("demog_wb2.csv") 
print(wdi_final.duplicated(subset=["year","gw_codes","country"]).any())
print(wdi_final.duplicated(subset=["year","country"]).any())
print(wdi_final.duplicated(subset=["year","gw_codes"]).any())
print(wdi_final.dtypes)

###################
### (3) Economy ###
###################

# Specify variables for call
feat_dev = ["NY.GDP.TOTL.RT.ZS", # Total natual resource rents % of GDP
            "NY.GDP.PETR.RT.ZS", # 	Oil rents (% of GDP)
            "NY.GDP.NGAS.RT.ZS", # Natural gas rents (% of GDP)
            "NY.GDP.COAL.RT.ZS", # 	Coal rents (% of GDP)
            "NY.GDP.FRST.RT.ZS", # Forest rents (% of GDP) 
            "NY.GDP.MINR.RT.ZS", # Mineral rents (% of GDP)
            "NY.GDP.PCAP.CD", # GDP per capita (current US$)
            "NY.GNP.PCAP.CD", # GNI per capita, Atlas method (current US$)
            "NY.GDP.MKTP.KD.ZG", # GDP growth (annual %) 
            "SL.UEM.TOTL.NE.ZS", # Unemployment, total (% of total labor force)
            "SL.UEM.TOTL.MA.NE.ZS", # Unemployment, male (% of male labor force)
            "FP.CPI.TOTL.ZG", # Inflation, consumer prices (annual %)
            "SN.ITK.DEFC.ZS", # Prevalence of undernourishment (% of population)
            "SP.DYN.IMRT.IN", # Mortality rate, infant (per 1,000 live births)
            "AG.PRD.FOOD.XD", # Food production index (2014-2016 = 100)
            "NV.AGR.TOTL.ZS", # Agriculture % of GDP
            "NE.TRD.GNFS.ZS", # Trade % of GDP
            "SH.H2O.BASW.RU.ZS", # People using at least basic drinking water services, rural (% of rural population)
            "SH.H2O.BASW.UR.ZS", # People using at least basic drinking water services, urban (% of urban population)
            "FP.CPI.TOTL", # Consumer price index (2010 = 100)
            "SP.DYN.TFRT.IN", # Fertility rate, total (births per woman)
            "SP.DYN.LE00.FE.IN", # Life expectancy at birth, female (years) 
            "SP.DYN.LE00.MA.IN", # Life expectancy at birth, male (years)
            "SP.POP.GROW", # Population growth (annual %)
            "NE.EXP.GNFS.ZS", # Exports of goods and services (% of GDP)
            "NE.IMP.GNFS.ZS", # Imports of goods and services (% of GDP)
            "SE.PRM.ENRR.FE", # School enrollment, primary, female (% gross)
            "SE.PRM.ENRR.MA", # School enrollment, primary, male (% gross)
            "SE.SEC.ENRR.FE", # School enrollment, secondary, female (% gross)
            "SE.SEC.ENRR.MA", # School enrollment, secondary, male (% gross)
            "SE.TER.ENRR.FE", # School enrollment, tertiary, female (% gross)
            "SE.TER.ENRR.MA", # School enrollment, tertiary, male (% gross)
            ]

# Import country codes 
df_ccodes = pd.read_csv("df_ccodes.csv")

# Specify countries for call  
c_list=list(df_ccodes.iso_alpha3)
c_list=[char for char in c_list if char != "XYZ"] # Exclude Taiwan

# Specify years for call
years=list(range(1989, 2024, 1))

# Define out df
wdi = pd.DataFrame()

# Get data for each year and merge
for i in years:
    print(i)
    wdi_s = wb.data.DataFrame(feat_dev, c_list, [i])
    wdi_s.reset_index(inplace=True)
    wdi_s["year"] = i
    wdi = pd.concat([wdi, wdi_s], ignore_index=True)  
    
# Add country and country codes: Merge GW and ACLED country codes over WB country codes 
df_ccodes_s = df_ccodes[["country",'gw_codes', "acled_codes", "iso_alpha3"]]
wdi_final = pd.merge(wdi, df_ccodes_s, how='left', left_on=['economy'], right_on=['iso_alpha3'])

# Drop duplicates WB country codes
wdi_final = wdi_final.drop(columns=['economy'])
        
# Sort columns, so that year, country and country codes appear at beginning
wdi_final = wdi_final[['country','year','acled_codes','iso_alpha3','gw_codes'] + [c for c in wdi_final.columns if c not in ['country','year','acled_codes','iso_alpha3','gw_codes']]]
 
# Print head of df to confirm load   
print("Obtained data")
print(wdi_final.head())
    
# Sort and reset index
wdi_final = wdi_final.sort_values(by=["iso_alpha3", 'year'])
wdi_final = wdi_final.reset_index(drop=True)

# Save data  
wdi_final.to_csv("economy_wb.csv") 
print(wdi_final.duplicated(subset=["year","gw_codes","country"]).any())
print(wdi_final.duplicated(subset=["year","country"]).any())
print(wdi_final.duplicated(subset=["year","gw_codes"]).any())
print(wdi_final.dtypes)

####################
### (4) Politics ###
####################

# Specify variables for call
feat_dev = ["MS.MIL.TOTL.TF.ZS", # Armed forces personnel (% of total labor force)
            "MS.MIL.XPND.GD.ZS", # Military expenditure (% of GDP)
            "CC.EST", # Control of Corruption: Estimate
            "GE.EST", # Government Effectiveness: Estimate
            "PV.EST", # Political Stability and Absence of Violence/Terrorism: Estimate
            "RQ.EST", # Regulatory Quality: Estimate
            "RL.EST", # Rule of Law: Estimate
            "VA.EST", # Voice and Accountability: Estimate
            "GC.TAX.TOTL.GD.ZS", # Tax revenue (% of GDP)
            "IT.NET.BBND.P2", # Fixed broadband subscriptions (per 100 people)
            "IT.MLT.MAIN.P2", # Fixed telephone subscriptions (per 100 people)
            "IT.NET.USER.ZS", # Individuals using the Internet (% of population)
            "IT.CEL.SETS.P2" # Mobile cellular subscriptions (per 100 people)
            ]

# Import country codes 
df_ccodes = pd.read_csv("df_ccodes.csv")

# Specify countries for call  
c_list=list(df_ccodes.iso_alpha3)
c_list=[char for char in c_list if char != "XYZ"] # Exclude Taiwan

# Specify years for call
years=list(range(1989, 2024, 1))

# Define out df
wdi = pd.DataFrame()

# Get data for each year and merge
for i in years:
    print(i)
    wdi_s = wb.data.DataFrame(feat_dev, c_list, [i])
    wdi_s.reset_index(inplace=True)
    wdi_s["year"] = i
    wdi = pd.concat([wdi, wdi_s], ignore_index=True)  
    
# Add country and country codes: Merge GW and ACLED country codes over WB country codes 
df_ccodes_s = df_ccodes[["country",'gw_codes', "acled_codes", "iso_alpha3"]]
wdi_final = pd.merge(wdi, df_ccodes_s, how='left', left_on=['economy'], right_on=['iso_alpha3'])

# Drop duplicates WB country codes
wdi_final = wdi_final.drop(columns=['economy'])
        
# Sort columns, so that year, country and country codes appear at beginning
wdi_final = wdi_final[['country','year','acled_codes','iso_alpha3','gw_codes'] + [c for c in wdi_final.columns if c not in ['country','year','acled_codes','iso_alpha3','gw_codes']]]
 
# Print head of df to confirm load   
print("Obtained data")
print(wdi_final.head())
    
# Sort and reset index
wdi_final = wdi_final.sort_values(by=["iso_alpha3", 'year'])
wdi_final = wdi_final.reset_index(drop=True)

# Save data  
wdi_final.to_csv("pol_wb.csv") 
print(wdi_final.duplicated(subset=["year","gw_codes","country"]).any())
print(wdi_final.duplicated(subset=["year","country"]).any())
print(wdi_final.duplicated(subset=["year","gw_codes"]).any())
print(wdi_final.dtypes)

#####################
### (5) Geography ###
#####################

# Specify variables for call
feat_dev = ["AG.LND.TOTL.K2", # Land area (sq. km)
            "AG.LND.FRST.ZS", # Forest area (% of land area)
            "AG.LND.PRCP.MM", # Average precipitation in depth (mm per year)
            "ER.H2O.FWST.ZS", # Level of water stress: freshwater withdrawal as a proportion of available freshwater resources
            "EN.GHG.CO2.MT.CE.AR5", # Carbon dioxide (CO2) emissions (total) excluding LULUCF (Mt CO2e)
            "AG.LND.AGRI.ZS", # Agricultural land (% of land area)
            "AG.LND.ARBL.ZS", # Arable land (% of land area)
            ]

# Import country codes
df_ccodes = pd.read_csv("df_ccodes.csv")

# Specify countries for call  
c_list=list(df_ccodes.iso_alpha3)
c_list=[char for char in c_list if char != "XYZ"] # Exclude Taiwan

# Specify years for call
years=list(range(1989, 2024, 1))

# Define out df
wdi = pd.DataFrame()

# Get data for each year and merge
for i in years:
    print(i)
    wdi_s = wb.data.DataFrame(feat_dev, c_list, [i])
    wdi_s.reset_index(inplace=True)
    wdi_s["year"] = i
    wdi = pd.concat([wdi, wdi_s], ignore_index=True)  
    
# Add country and country codes: Merge GW and ACLED country codes over WB country codes 
df_ccodes_s = df_ccodes[["country",'gw_codes', "acled_codes", "iso_alpha3"]]
wdi_final = pd.merge(wdi, df_ccodes_s, how='left', left_on=['economy'], right_on=['iso_alpha3'])

# Drop duplicates WB country codes
wdi_final = wdi_final.drop(columns=['economy'])
        
# Sort columns, so that year, country and country codes appear at beginning
wdi_final = wdi_final[['country','year','acled_codes','iso_alpha3','gw_codes'] + [c for c in wdi_final.columns if c not in ['country','year','acled_codes','iso_alpha3','gw_codes']]]
 
# Print head of df to confirm load   
print("Obtained data")
print(wdi_final.head())
    
# Sort and reset index
wdi_final = wdi_final.sort_values(by=["iso_alpha3", 'year'])
wdi_final = wdi_final.reset_index(drop=True)

# Save data  
wdi_final.to_csv("geog_wb.csv") 
print(wdi_final.duplicated(subset=["year","gw_codes","country"]).any())
print(wdi_final.duplicated(subset=["year","country"]).any())
print(wdi_final.duplicated(subset=["year","gw_codes"]).any())
print(wdi_final.dtypes)





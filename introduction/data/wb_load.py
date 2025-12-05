import pandas as pd
import wbgapi as wb

# The World Bank data is loaded via the API, using the wbgapi library: https://pypi.org/project/wbgapi/
# But can be manually downloaded here: https://databank.worldbank.org/source/world-development-indicators

# Specify variables for call
feat_dev = ["NY.GDP.PCAP.CD", # GDP per capita (current US$)
            "NY.GDP.MKTP.KD.ZG", # GDP growth (annual %) 
            "NY.GDP.PETR.RT.ZS", # Oil rents (% of GDP)
            "SP.POP.TOTL", # Population size
            "SP.DYN.IMRT.IN", # Mortality rate, infant (per 1,000 live births)
            'SP.POP.2024.MA.5Y', # Population ages 20-24, male (% of male population)
            "ER.H2O.FWTL.ZS", # Annual freshwater withdrawals, total (% of internal resources)          
            ]

# Import country codes
df_ccodes = pd.read_csv("df_ccodes.csv")

# Specify countries for call  
c_list = list(df_ccodes.iso_alpha3)
c_list = [char for char in c_list if char != "XYZ"] # Exclude Taiwan

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
df_ccodes_s = df_ccodes[["country",'gw_codes',"acled_codes","iso_alpha3"]]
wdi_final = pd.merge(wdi, df_ccodes_s, how='left', left_on=['economy'], right_on=['iso_alpha3'])

# Drop duplicates WB country codes
wdi_final = wdi_final.drop(columns=['economy'])

# Sort columns, so that year, country and country codes appear at beginning
wdi_final = wdi_final[['country','year','acled_codes','iso_alpha3','gw_codes'] + [c for c in wdi_final.columns if c not in ['country','year','acled_codes','iso_alpha3','gw_codes']]]

# Print head of df to confirm load
print("Obtained data")
print(wdi_final.head())

# Sort and reset index
wdi_final=wdi_final.sort_values(by=["iso_alpha3", 'year'])
wdi_final=wdi_final.reset_index(drop=True)

# Save data  
wdi_final.to_csv("economy_wb.csv") 
print(wdi_final.duplicated(subset=["year","gw_codes","country"]).any())
print(wdi_final.duplicated(subset=["year","country"]).any())
print(wdi_final.duplicated(subset=["year","gw_codes"]).any())
wdi_final.dtypes








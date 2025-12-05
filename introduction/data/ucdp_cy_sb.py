import pandas as pd

# Load data
# UCDP Georeferenced Event Dataset 
# Available here: https://ucdp.uu.se/downloads/
# Version 24: https://ucdp.uu.se/downloads/ged/ged241-csv.zip
# Codebook: https://ucdp.uu.se/downloads/ged/ged241.pdf
ucdp = pd.read_csv("GEDEvent_v24_1 3.csv",low_memory=False)

# Only keep civil conflict
ucdp_s = ucdp[(ucdp["type_of_violence"]==1)].copy()

# Remove events between states
u = ucdp_s[['dyad_name']].drop_duplicates().reset_index(drop=True)
ucdp_ss = ucdp_s.loc[(ucdp_s["dyad_name"] != "Government of Afghanistan - Government of United Kingdom, Government of United States of America") &
                    (ucdp_s["dyad_name"] != "Government of Cambodia (Kampuchea) - Government of Thailand") &
                    (ucdp_s["dyad_name"] != "Government of Cameroon - Government of Nigeria") &
                    (ucdp_s["dyad_name"] != "Government of Djibouti - Government of Eritrea") &
                    (ucdp_s["dyad_name"] != "Government of Ecuador - Government of Peru") &
                    (ucdp_s["dyad_name"] != "Government of Eritrea - Government of Ethiopia") &
                    (ucdp_s["dyad_name"] != "Government of India - Government of Pakistan") &
                    (ucdp_s["dyad_name"] != "Government of China - Government of India") &  
                    (ucdp_s["dyad_name"] != "Government of Iran - Government of Israel") &                    
                    (ucdp_s["dyad_name"] != "Government of Iraq - Government of Kuwait") &
                    (ucdp_s["dyad_name"] != "Government of Australia, Government of United Kingdom, Government of United States of America - Government of Iraq") &
                    (ucdp_s["dyad_name"] != "Government of Kyrgyzstan - Government of Tajikistan") &                    
                    (ucdp_s["dyad_name"] != "Government of Panama - Government of United States of America") &
                    (ucdp_s["dyad_name"] != "Government of Russia (Soviet Union) - Government of Ukraine") &                   
                    (ucdp_s["dyad_name"] != "Government of South Sudan - Government of Sudan") ].copy(deep=True)

# Aggregate event counts to country year
agg_year = pd.DataFrame(ucdp_ss.groupby(["year","country_id"]).size())
agg_year = agg_year.reset_index()
agg_year.rename(columns={0:"count"},inplace=True)

# Aggregate fatalities to country year
best_est = pd.DataFrame(ucdp_ss.groupby(["year","country_id","country"])['best'].sum())
high_est = pd.DataFrame(ucdp_ss.groupby(["year","country_id","country"])['high'].sum())
low_est = pd.DataFrame(ucdp_ss.groupby(["year","country_id","country"])['low'].sum())
fatalities1 = pd.concat([best_est, high_est], axis=1)
fatalities =  pd.concat([fatalities1, low_est], axis=1) 
fatalities = fatalities.reset_index()

# Merge event counts and fatality counts
fatalities = pd.merge(left=fatalities,right=agg_year[["year","country_id","count"]],left_on=["year","country_id"],right_on=["year","country_id"])
print(fatalities.head())

# Add missing observations (those with zero events) 
countries = ucdp_ss.country_id.unique()
years = ucdp_ss.year.unique()

# Loop through every country and every year available in sample
for i in range(0, len(countries)):
    for x in range(0, len(years)):
        # Check if country-year in data, if False add
        if ((fatalities['year']==years[x])&(fatalities['country_id']==countries[i])).any()==False:
            s = {'year':fatalities['year'].loc[(fatalities["year"]==years[x])].iloc[0],'country':fatalities['country'].loc[(fatalities["country_id"]==countries[i])].iloc[0],'country_id':countries[i],'best':0,'high':0,'low':0,"count":0}
            s = pd.DataFrame(data=s,index=[0])
            fatalities = pd.concat([fatalities,s]) 

# Rename country id
fatalities.rename(columns = {'country_id':'gw_codes'},inplace = True)

# Add missing observations for countries completely missing 
# Take Gleditsch and Ward as point of departure
# http://ksgleditsch.com/data/iisystem.dat
# http://ksgleditsch.com/data/microstatessystem.dat

# Load GW country definitions
all_countries=pd.read_csv("df_ccodes_gw.csv")
all_countries_s=all_countries.loc[all_countries["end"]>=1989]

# Check which countries are in GW but not in my data
countries_ucdp=fatalities["gw_codes"].unique()
countries=all_countries_s["gw_codes"].unique()
add = list(filter(lambda x: x not in countries_ucdp,countries))
             
# Add missing observations for countries completely missing 

# Load country codes to add
df_ccodes = pd.read_csv("df_ccodes.csv")

# Loop through every country in add
for i in range(0, len(add)):
    # Check if country in data (for safety), if False add
    if (fatalities['gw_codes']==add[i]).any()==False:
        for x in years:
            s = {'year':x,'country': df_ccodes[df_ccodes["gw_codes"]==add[i]]["country"].values[0],'gw_codes':add[i],'best':0,'high':0,'low':0,"count":0}
            s = pd.DataFrame(data=s,index=[0])
            fatalities = pd.concat([fatalities,s])  

# Sort and reset index
ucdp_final = fatalities.sort_values(by=["gw_codes","year"])
ucdp_final.reset_index(drop=True,inplace=True)

# Check that df is complete
print(ucdp_final.groupby(['country']).size().unique())

# Check independence and remove observations
# Check Gleditsch and Ward: http://ksgleditsch.com/data-4.html

### Dissolution ###

# Czechoslovakia, 31:12:1992
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==315)&(ucdp_final["year"]>1992))]

# Yemen, People's Republic of, 21:05:1990
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==680)&(ucdp_final["year"]>1989))]

# German Democratic Republic, 02:10:1990
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==265)&(ucdp_final["year"]>1989))]

# Yugoslavia, 04:06:2006
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==345)&(ucdp_final["year"]>2005))]

### New states ###

# Namibia, 21:03:1990	
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==565)&(ucdp_final["year"]<1990))]

# For dissolution of Soviet Union also check Wikipedia, when succession was recognized.
# https://en.wikipedia.org/wiki/Dissolution_of_the_Soviet_Union#Chronology_of_declarations

# Estonia, Latvia, and Lithuania start in 1991.
# The remaining post-Soviet states start in 1992.

# Turkmenistan, 27:10:1991, recognized 26 December 1991 ---> remove complete 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==701)&(ucdp_final["year"]<=1991))]

# Tajikistan, 09:09:1991, recognized 26 December 1991 ---> remove complete 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==702)&(ucdp_final["year"]<=1991))]

# Kyrgyztan, 31:08:1991, recognized 26 December 1991  ---> remove complete 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==703)&(ucdp_final["year"]<=1991))]

# Uzbekistan, 31:08:1991, recognized 26 December 1991 ---> remove complete 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==704)&(ucdp_final["year"]<=1991))]

# Kazakhstan, 16:12:1991, recognized 26 December 1991 ---> remove complete 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==705)&(ucdp_final["year"]<=1991))]

# Ukraine, 01:12:1991, recognized 26 December 1991 ---> remove complete 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==369)&(ucdp_final["year"]<=1991))]

# Armenia, 21:12:1991, recognized 26 December 1991 ---> remove complete 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==371)&(ucdp_final["year"]<=1991))]

# Georgia, 21:12:1991, recognized 26 December 1991 ---> remove complete 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==372)&(ucdp_final["year"]<=1991))]

# Azerbeijan, 21:12:1991, recognized 26 December 1991 ---> remove complete 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==373)&(ucdp_final["year"]<=1991))]

# Belarus, 25:08:1991, recognized 26 December 1991 ---> remove complete 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==370)&(ucdp_final["year"]<=1991))]

# Moldova, 27:08:1991, recognized 26 December 1991 ---> remove complete 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==359)&(ucdp_final["year"]<=1991))]

# Latvia, 06:09:1991, recognized 6 September 1991 ---> keep 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==367)&(ucdp_final["year"]<1991))]

# Estonia, 06:09:1991, recognized 6 September 1991 ---> keep 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==366)&(ucdp_final["year"]<1991))]

# Lithuanua, 06:09:1991, recognized 6 September 1991 ---> keep 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==368)&(ucdp_final["year"]<1991))]

# Macedonia, 20:11:1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==343)&(ucdp_final["year"]<1991))]

# Bosnia-Herzegovina, 27:04:1992
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==346)&(ucdp_final["year"]<1992))]

# Montenegro, 03:06:2006
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==341)&(ucdp_final["year"]<2006))]

# Kosovo, 17:02:2008	
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==347)&(ucdp_final["year"]<2008))]

# Czech Republic, 01:01:1993	
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==316)&(ucdp_final["year"]<1993))]

# Slovakia, 01:01:1993
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==317)&(ucdp_final["year"]<1993))]

# Slovenia, 27:04:1992
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==349)&(ucdp_final["year"]<1992))]

# Croatia, 27:04:1992	
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==344)&(ucdp_final["year"]<1992))]

# Eritrea, 24:05:1993
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==531)&(ucdp_final["year"]<1993))]

# Palau, 01:10:1994	
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==986)&(ucdp_final["year"]<1994))]

# East Timor, 20:05:2002	
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==860)&(ucdp_final["year"]<2002))]

# Serbia, 05:06:2006
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==340)&(ucdp_final["year"]<2006))]

# South Ossetia, 26:08:2008
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==397)&(ucdp_final["year"]<2008))]

# Abkhazia, 26:08:2008	
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==396)&(ucdp_final["year"]<2008))]

# South Sudan, 09:07:2011	
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==626)&(ucdp_final["year"]<2011))]

# Save data 
ucdp_final = ucdp_final.sort_values(by=["gw_codes","year"])
ucdp_final.reset_index(drop=True,inplace=True)
ucdp_final.to_csv("data_out/ucdp_cy_sb.csv",sep=',')
print(ucdp_final.duplicated(subset=['year',"country","gw_codes"]).any())
print(ucdp_final.duplicated(subset=['year',"country"]).any())
print(ucdp_final.duplicated(subset=['year',"gw_codes"]).any())
ucdp_final.dtypes








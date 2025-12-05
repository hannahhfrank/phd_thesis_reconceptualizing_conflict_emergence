import pandas as pd
        
# Load data
# UCDP Georeferenced Event Dataset 
# Available here: https://ucdp.uu.se/downloads/
# Version 24: https://ucdp.uu.se/downloads/ged/ged241-csv.zip
# Codebook: https://ucdp.uu.se/downloads/ged/ged241.pdf
ucdp = pd.read_csv("GEDEvent_v24_1 3.csv",low_memory=False)

# Only use non-state conflict
ucdp_s = ucdp[(ucdp["type_of_violence"]==2)].copy(deep=True)

# Add dates 
ucdp_s["dd_date_start"] = pd.to_datetime(ucdp_s['date_start'],format='%Y-%m-%d %H:%M:%S.000')
ucdp_s["dd_date_end"] = pd.to_datetime(ucdp_s['date_end'],format='%Y-%m-%d %H:%M:%S.000')

# Only store month
ucdp_s["month_date_start"] = ucdp_s["dd_date_start"].dt.strftime('%m')
ucdp_s["month_date_end"] = ucdp_s["dd_date_end"].dt.strftime('%m')
ucdp_date = ucdp_s[["year","dd_date_start","dd_date_end","active_year","country","country_id","date_prec","best","high","low","month_date_start","month_date_end"]].copy(deep=True)

# Sort and reset index 
ucdp_date = ucdp_date.sort_values(by=["country", "year"],ascending=True)
ucdp_date.reset_index(drop=True, inplace=True)

# Loop through data and delete observations which comprise more than one month 
ucdp_final = ucdp_date.copy()
for i in range(0,len(ucdp_date)):
    if ucdp_date["month_date_start"].loc[i]!=ucdp_date["month_date_end"].loc[i]:
        ucdp_final = ucdp_final.drop(index=i, axis=0)       

# Generate year_month variable 
ucdp_final['dd'] = pd.to_datetime(ucdp_final['dd_date_start'],format='%Y-%m').dt.to_period('M')

# Aggregate event counts to country-month
agg_month = pd.DataFrame(ucdp_final.groupby(["dd","year","country_id"]).size())
agg_month = agg_month.reset_index()
agg_month.rename(columns={0:"count"},inplace=True)

# Aggregate fatalities to country-month 
best_est = pd.DataFrame(ucdp_final.groupby(["dd","country_id"])['best'].sum())
high_est = pd.DataFrame(ucdp_final.groupby(["dd","country_id"])['high'].sum())
low_est = pd.DataFrame(ucdp_final.groupby(["dd","country_id"])['low'].sum())
fatalities1 = pd.concat([best_est, high_est], axis=1)
fat = pd.concat([fatalities1, low_est], axis=1)
ucdp_fat = fat.reset_index()

# Re-obtain year  
ucdp_fat["year"] = None
for i in ucdp_fat.index:
    ucdp_fat.loc[i, "year"] = ucdp_fat.loc[i, 'dd'].year
    
# Re-obtain country        
ucdp_cc = ucdp[["country_id", "country"]].drop_duplicates().reset_index(drop=True)
ucdp_final = pd.merge(ucdp_fat,ucdp_cc,how='left',on='country_id')

# Merge fatalities and event counts
ucdp_final = pd.merge(left=ucdp_final,right=agg_month[["dd","country_id","count"]],left_on=["dd","country_id"],right_on=["dd","country_id"])

# Add missing observations to time series, those have zero fatalities 

# Get countries
countries = list(ucdp_final.country_id.unique())

# Specify range
date = list(pd.date_range(start="1989-01",end="2023-12",freq="MS"))
date = pd.to_datetime(date, format='%Y-%m').to_period('M')

# Loop through every country
for i in range(0, len(countries)):
    # and every month
    for x in range(0, len(date)):
        # Check if country-month in data, if False add
        if ((ucdp_final['dd']==date[x])&(ucdp_final['country_id']==countries[i])).any()==False:
            # Subset data to add
            s = {'dd':date[x],'year':date[x].year,'country':ucdp_final['country'].loc[(ucdp_final["country_id"]==countries[i])].iloc[0],'country_id':countries[i],'best':0,'high':0,'low':0,'count':0}
            s = pd.DataFrame(data=s,index=[0])
            ucdp_final = pd.concat([ucdp_final,s])  

# Rename country id
ucdp_final.rename(columns = {'country_id':'gw_codes'},inplace = True)

# Add missing observations for countries completely missing 
# Take Gleditsch and Ward as point of departure
# http://ksgleditsch.com/data/iisystem.dat
# http://ksgleditsch.com/data/microstatessystem.dat

# Load GW country definitions
all_countries=pd.read_csv("df_ccodes_gw.csv")
all_countries_s=all_countries.loc[all_countries["end"]>=1989]

# Check which countries are in GW but not in my data
countries_ucdp=ucdp_final["gw_codes"].unique()
countries=all_countries_s["gw_codes"].unique()
add = list(filter(lambda x: x not in countries_ucdp,countries))

# Add missing observations for countries completely missing 

# Load country codes to add
df_ccodes = pd.read_csv("df_ccodes.csv")

# Loop through every country in add
for i in range(0, len(add)):
    print(df_ccodes[df_ccodes["gw_codes"]==add[i]]["country"].values[0])
    # Check if country in data (for safety), if False add
    if (ucdp_final['gw_codes'] == add[i]).any() == False:
        # For each month        
        for d in range(0, len(date)):
            # Subset data to add            
            s = {'dd':date[d],'year':date[d].year,'country':df_ccodes[df_ccodes["gw_codes"]==add[i]]["country"].values[0],'gw_codes':add[i],'best':0,'high':0,'low':0,'count':0}
            s = pd.DataFrame(data=s,index=[0])
            ucdp_final = pd.concat([ucdp_final, s])  

# Subset needed columns, sort and reset index
ucdp_final = ucdp_final[["dd","year","country","gw_codes","best","high","low","count"]]
ucdp_final = ucdp_final.sort_values(by=["country","year","dd"])
ucdp_final.reset_index(drop=True,inplace=True)

# Check that df is complete
print(ucdp_final.groupby(['country', 'year']).size().unique())

# Check independence and remove observations
# Only the full year is considered. 
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
ucdp_final = ucdp_final.sort_values(by=["gw_codes","year","dd"])
ucdp_final.reset_index(drop=True,inplace=True)
ucdp_final.to_csv("data_out/ucdp_cm_ns.csv",sep=',')
print(ucdp_final.duplicated(subset=['dd',"country","gw_codes"]).any())
print(ucdp_final.duplicated(subset=['dd',"country"]).any())
print(ucdp_final.duplicated(subset=['dd',"gw_codes"]).any())
ucdp_final.dtypes # dd --> period[M] but will become object outside of python




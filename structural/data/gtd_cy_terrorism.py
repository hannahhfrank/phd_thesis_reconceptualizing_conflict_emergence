import pandas as pd
        
# GTD data 
# Codebook: https://www.start.umd.edu/sites/default/files/2024-10/Codebook.pdf 
# Manually downloaded from: https://www.start.umd.edu/data-tools/GTD
#df = pd.read_excel("globalterrorismdb_0522dist.xlsx") # Manually converted excel into csv
df = pd.read_csv("globalterrorismdb_0522dist.csv") 

# Remove events with high uncertainty
gtd_suicide = df[(df["imonth"]!=0)] # Month is unknown
gtd_suicide = gtd_suicide[(gtd_suicide["doubtterr"]!=1)] # There is doubt that the event was terrorism

# Subset columns 
gtd_s = gtd_suicide[[
            #"eventid"
            "iyear",
            #"imonth",
            #"iday",
            #"approxdate",
            #"extended",
            #"resolution",
            "country",
            "country_txt",
            #"region",
            #"region_txt",
            #"provstate",
            #"city",
            #"latitude", # 
            #"longitude", # 
            #"specificity",
            #"vicinity",
            #"location",
            #"summary",
            #"crit1",
            #"crit2",
            #"crit3",
            #"doubtterr",
            #"alternative",
            #"alternative_txt",
            #"multiple",
            #"success",
            #"suicide",
            #"attacktype1",
            #"attacktype1_txt",
            #"attacktype2",
            #"attacktype2_txt",
            #"attacktype3",
            #"attacktype3_txt",
            #"targtype1",
            #"targtype1_txt",
            #"targsubtype1"
            #"targsubtype1_txt",
            #"corp1",
            #"target1",
            #"natlty1",
            #"natlty1_txt",
            #"targtype2",
            #"targtype2_txt",
            #"targsubtype2",
            #"targsubtype2_txt",
            #"corp2",
            #"target2",
            #"natlty2",
            #"natlty2_txt",
            #"targtype3",
            #"targtype3_txt",
            #"targsubtype3",
            #"targsubtype3_txt",
            #"corp3",
            #"target3",
            #"natlty3",
            #"natlty3_txt",
            #"gname",
            #"gsubname",
            #"gname2",
            #"gsubname2",
            #"gname3",
            #"gsubname3",
            #"motive",
            #"guncertain1",
            #"guncertain2",
            #"guncertain3",
            #"individual",
            #"nperps",
            #"nperpcap",
            #"claimed",
            #"claimmode",
            #"claimmode_txt",
            #"claim2",
            #"claimmode2",
            #"claimmode2_txt",
            #"claim3",
            #"claimmode3",
            #"claimmode3_txt",
            #"compclaim",
            #"weaptype1",
            #"weaptype1_txt",
            #"weapsubtype1",
            #"weapsubtype1_txt",
            #"weaptype2",
            #"weaptype2_txt",
            #"weapsubtype2",
            #"weapsubtype2_txt",
            #"weaptype3",
            #"weaptype3_txt",
            #"weapsubtype3",
            #"weapsubtype3_txt",
            #"weaptype4",
            #"weaptype4_txt",
            #"weapsubtype4",
            #"weapsubtype4_txt",
            #"weapdetail",
            "nkill",
            #"nkillus",
            #"nkillter",
            #"nwound",
            #"nwoundus",
            #"nwoundte",
            #"property",
            #"propextent",
            #"propextent_txt",
            #"propvalue",
            #"propcomment",
            #"ishostkid",
            #"nhostkid",
            #"nhostkidus",
            #"nhours",
            #"ndays",
            #"divert",
            #"kidhijcountry",
            #"ransom",
            #"ransomamt",
            #"ransomamtus",
            #"ransompaid",
            #"ransompaidus",
            #"ransomnote",
            #"hostkidoutcome",
            #"hostkidoutcome_txt",
            #"nreleased",
            #"addnotes",
            #"scite1",
            #"scite2",
            #"scite3",
            #"dbsource",
            #"INT_LOG",
            #"INT_IDEO",
            #"INT_MISC",
            #"INT_ANY",
            #"related",
            ]].copy()

# GTD documents events on a number of additional countries, which can be merged
# with the GW set up: 
# West Bank and Gaza Strip --> Israel
# North Yemen --> Yemen
# People's Republic of the Congo --> Congo
# Rhodesia --> Zimbabwe
# Serbia-Montenegro --> Yugoslavia
# Soviet Union --> Russia
# West Germany (FRG) --> Germany
# Zaire --> DRC

gtd_s.loc[gtd_s["country_txt"]=="West Bank and Gaza Strip","country_txt"]="Israel"
gtd_s.loc[gtd_s["country_txt"]=="Israel","country"]=97

gtd_s.loc[gtd_s["country_txt"]=="North Yemen","country_txt"]="Yemen"
gtd_s.loc[gtd_s["country_txt"]=="Yemen","country"]=228

gtd_s.loc[gtd_s["country_txt"]=="People's Republic of the Congo","country_txt"]="Republic of the Congo"
gtd_s.loc[gtd_s["country_txt"]=="Republic of the Congo","country"]=47

gtd_s.loc[gtd_s["country_txt"]=="Rhodesia","country_txt"]="Zimbabwe"
gtd_s.loc[gtd_s["country_txt"]=="Zimbabwe","country"]=231

gtd_s.loc[gtd_s["country_txt"]=="Serbia-Montenegro","country_txt"]="Yugoslavia"
gtd_s.loc[gtd_s["country_txt"]=="Yugoslavia","country"]=235

gtd_s.loc[gtd_s["country_txt"]=="Soviet Union","country_txt"]="Russia"
gtd_s.loc[gtd_s["country_txt"]=="Russia","country"]=167

gtd_s.loc[gtd_s["country_txt"]=="West Germany (FRG)","country_txt"]="Germany"
gtd_s.loc[gtd_s["country_txt"]=="Germany","country"]=75

gtd_s.loc[gtd_s["country_txt"]=="Zaire","country_txt"]="Democratic Republic of the Congo"
gtd_s.loc[gtd_s["country_txt"]=="Democratic Republic of the Congo","country"]=229

# Aggregate event counts to year level 
agg_year = pd.DataFrame(gtd_s.groupby(["iyear","country","country_txt"]).size())
agg_year = agg_year.reset_index()
agg_year.rename(columns={0:"count"},inplace=True)

# Aggregate fatalities to year level, and rename columns
fat = pd.DataFrame(gtd_s.groupby(["iyear","country"])['nkill'].sum())
agg_year = pd.merge(left=agg_year,right=fat,left_on=["iyear","country"],right_on=["iyear","country"])
agg_year = agg_year.reset_index(drop=True)
agg_year.columns = ["year","gtd_codes","country","n_attack","fatalities"]

# Add missing observation to time series, those with zero events 

# Get years and countries in data
years = list(agg_year.year.unique()) # 1993 is missing
countries = list(agg_year.gtd_codes.unique())

# Loop through every country and add missing observations with zeros

# For every country
for i in range(0, len(countries)):
    # and year
    for x in years:
        # Check if observation in data, if False add
        if ((agg_year['year']==x)&(agg_year['gtd_codes']==countries[i])).any()==False:               
            # Subset data to add
            s = {'year':x,'country':agg_year['country'].loc[(agg_year["gtd_codes"]==countries[i])].iloc[0],'gtd_codes':countries[i],'n_attack':0,'fatalities':0}
            s = pd.DataFrame(data=s,index=[0])
            agg_year = pd.concat([agg_year,s])  
 
# Sort and reset index            
agg_year = agg_year.sort_values(by=["country","year"])
agg_year.reset_index(drop=True,inplace=True)

# Merge country codes and remove countries not in GW

# Load GW codes and merge with gtd data
df_ccodes = pd.read_csv("df_ccodes.csv")
agg_year = pd.merge(left=agg_year,right=df_ccodes[["gw_codes","gtd_codes"]],left_on=["gtd_codes"],right_on=["gtd_codes"],how="left")

# Remove territories included in gtd but not available in GW definitions
agg_year = agg_year[~agg_year['country'].isin(["Falkland Islands",
                                               "French Guiana",
                                               "French Polynesia",
                                               "Guadeloupe",
                                               "Hong Kong",
                                               "International",
                                               "Macau",
                                               "Martinique",
                                               "New Caledonia",
                                               "New Hebrides",
                                               "Vatican City",
                                               "Wallis and Futuna", 
                                               "Western Sahara"])]

# Add missing observations for countries completely missing 
# Take countries in Gleditsch and Ward as point of departure
# http://ksgleditsch.com/data/iisystem.dat
# http://ksgleditsch.com/data/microstatessystem.dat

# Load GW country definitions
all_countries=pd.read_csv("df_ccodes_gw.csv")
all_countries_s=all_countries.loc[all_countries["end"]>=1970]

# Add Vietnam, Republic of manually because not included in sample
obs={"country":"Vietnam, Republic of", 
     "gw_codes":817,
     "iso_alpha3":99999999,
     "M49":"XYZ", 
     "StateAbb":99999999,
     "acled_codes":99999999,
     "gtd_codes":428,
     "vdem_codes":99999999}
df_ccodes = pd.concat([df_ccodes, pd.DataFrame(obs,index=[0])])

# Check which countries are in GW but not in my data
countries_gtd=agg_year["gw_codes"].unique()
countries=all_countries_s["gw_codes"].unique()
add = list(filter(lambda x: x not in countries_gtd,countries))

# Countries not included in GTD: 
# 402: Cape Verde	
# 698: Oman
# 712: Mongolia
# 817: Vietnam, Republic of
# 57: Saint Vincent and the Grenadines
# 221: Monaco
# 223: Liechtenstein
# 331: San Marino	
# 396: Abkhazia
# 397: South Ossetia	
# 403: São Tomé and Principe	
# 970: Kiribati
# 971: Nauru
# 972: Tonga
# 973: Tuvalu
# 983: Marshall Islands	
# 986: Palau
# 987: Federated States of Micronesia	
# 990: Samoa/Western Samoa

# Add countries completely missing. 

# The GTD does not collect data on these countries (?), at least not for "target" of terrorism.
# However, to allow for a fair comparison, the sample should have at least
# the same spatial coverage --> only affects three countries: Cape Verde, Oman, Mongolia
# because the others are not included in sample anyways.

# Loop through every country in add
for i in range(0, len(add)):
    # Check if country in data (for safety), if False add
    if (agg_year['gw_codes']==add[i]).any()==False:
        for d in years:
            s = {'year':d,'country':df_ccodes[df_ccodes["gw_codes"]==add[i]]["country"].values[0],'gw_codes':add[i],'gtd_codes':df_ccodes[df_ccodes["gw_codes"]==add[i]]["gtd_codes"].values[0],'n_attack':0,'fatalities':0}
            s = pd.DataFrame(data=s,index=[0])
            agg_year = pd.concat([agg_year, s])  

# Subset needed columns, sort and reset index
gtd_final = agg_year[["year","country","gw_codes","n_attack","fatalities"]]
gtd_final = gtd_final.sort_values(by=["country","year"])
gtd_final.reset_index(drop=True,inplace=True)

# Check that df is complete
print(gtd_final.groupby(['country']).size().unique())

# Check independence and remove observations
# Check Gleditsch and Ward: http://ksgleditsch.com/data-4.html

### Dissolution ###

# Czechoslovakia, 31:12:1992
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==315)&(gtd_final["year"]>1992))]

# Yemen, People's Republic of, 21:05:1990
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==680)&(gtd_final["year"]>1989))]

# German Democratic Republic, 02:10:1990
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==265)&(gtd_final["year"]>1989))]

# Yugoslavia, 04:06:2006
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==345)&(gtd_final["year"]>2005))]

# Vietnam, Republic of, 30:04:1975
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==817)&(gtd_final["year"]>1974))]

### New states ###

# Namibia, 21:03:1990	
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==565)&(gtd_final["year"]<1990))]

# For dissolution of Soviet Union also check Wikipedia, when succession was recognized.
# https://en.wikipedia.org/wiki/Dissolution_of_the_Soviet_Union#Chronology_of_declarations

# Estonia, Latvia, and Lithuania start in 1991.
# The remaining post-Soviet states start in 1992.

# Turkmenistan, 27:10:1991, recognized 26 December 1991 ---> remove complete 1991
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==701)&(gtd_final["year"]<=1991))]

# Tajikistan, 09:09:1991, recognized 26 December 1991 ---> remove complete 1991
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==702)&(gtd_final["year"]<=1991))]

# Kyrgyztan, 31:08:1991, recognized 26 December 1991  ---> remove complete 1991
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==703)&(gtd_final["year"]<=1991))]

# Uzbekistan, 31:08:1991, recognized 26 December 1991 ---> remove complete 1991
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==704)&(gtd_final["year"]<=1991))]

# Kazakhstan, 16:12:1991, recognized 26 December 1991 ---> remove complete 1991
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==705)&(gtd_final["year"]<=1991))]

# Ukraine, 01:12:1991, recognized 26 December 1991 ---> remove complete 1991
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==369)&(gtd_final["year"]<=1991))]

# Armenia, 21:12:1991, recognized 26 December 1991 ---> remove complete 1991
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==371)&(gtd_final["year"]<=1991))]

# Georgia, 21:12:1991, recognized 26 December 1991 ---> remove complete 1991
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==372)&(gtd_final["year"]<=1991))]

# Azerbeijan, 21:12:1991, recognized 26 December 1991 ---> remove complete 1991
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==373)&(gtd_final["year"]<=1991))]

# Belarus, 25:08:1991, recognized 26 December 1991 ---> remove complete 1991
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==370)&(gtd_final["year"]<=1991))]

# Moldova, 27:08:1991, recognized 26 December 1991 ---> remove complete 1991
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==359)&(gtd_final["year"]<=1991))]

# Latvia, 06:09:1991, recognized 6 September 1991 ---> keep 1991
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==367)&(gtd_final["year"]<1991))]

# Estonia, 06:09:1991, recognized 6 September 1991 ---> keep 1991
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==366)&(gtd_final["year"]<1991))]

# Lithuanua, 06:09:1991, recognized 6 September 1991 ---> keep 1991
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==368)&(gtd_final["year"]<1991))]

# Macedonia, 20:11:1991
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==343)&(gtd_final["year"]<1991))]

# Bosnia-Herzegovina, 27:04:1992
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==346)&(gtd_final["year"]<1992))]

# Montenegro, 03:06:2006
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==341)&(gtd_final["year"]<2006))]

# Kosovo, 17:02:2008	
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==347)&(gtd_final["year"]<2008))]

# Czech Republic, 01:01:1993	
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==316)&(gtd_final["year"]<1993))]

# Slovakia, 01:01:1993
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==317)&(gtd_final["year"]<1993))]

# Slovenia, 27:04:1992
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==349)&(gtd_final["year"]<1992))]

# Croatia, 27:04:1992	
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==344)&(gtd_final["year"]<1992))]

# Eritrea, 24:05:1993
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==531)&(gtd_final["year"]<1993))]

# Palau, 01:10:1994	
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==986)&(gtd_final["year"]<1994))]

# East Timor, 20:05:2002	
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==860)&(gtd_final["year"]<2002))]

# Serbia, 05:06:2006
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==340)&(gtd_final["year"]<2006))]

# South Ossetia, 26:08:2008
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==397)&(gtd_final["year"]<2008))]

# Abkhazia, 26:08:2008	
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==396)&(gtd_final["year"]<2008))]

# South Sudan, 09:07:2011	
gtd_final=gtd_final.loc[~((gtd_final["gw_codes"]==626)&(gtd_final["year"]<2011))]

# Save  
gtd_final = gtd_final.sort_values(by=["gw_codes","year"])
gtd_final.reset_index(drop=True,inplace=True)
gtd_final["gw_codes"] = gtd_final["gw_codes"].astype(int)
gtd_final["n_attack"] = gtd_final["n_attack"].astype(int)
gtd_final["fatalities"] = gtd_final["fatalities"].astype(int)
gtd_final.to_csv("data_out/gtd_cy_attacks.csv",sep=',')
print(gtd_final.duplicated(subset=['year',"country","gw_codes"]).any())
print(gtd_final.duplicated(subset=['year',"country"]).any())
print(gtd_final.duplicated(subset=['year',"gw_codes"]).any())
gtd_final.dtypes










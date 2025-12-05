import pandas as pd

# Load Rulers, Elections and Irregular Governance Dataset (REIGN) 
# Available at: https://oefdatascience.github.io/REIGN.github.io/menu/reign_current.html
# Codebook: https://raw.githubusercontent.com/OEFDataScience/REIGN.github.io/gh-pages/documents/REIGN_CODEBOOK.pdf
reign = pd.read_csv("reign_8_21.csv", encoding='latin1')

# Fix countries, REIGN uses COW country codes, which are almost identical with
# GW country codes. The manual fixes are performed below
# http://ksgleditsch.com/data-4.html
com =  reign[['ccode', 'country']].drop_duplicates().reset_index(drop=True)

# (1) Check Germany, coded as 255 (Germany (Prussia)), 260 (Germany), 265 (East Germany)
# --> replace Prussia with Germany to match GW definition
reign.loc[reign["ccode"]==255, "ccode"] = 260 

# (2) Check Serbia and Yugoslavia, both coded as 345
# --> name changes from Yugoslavia to Serbia in 1992
# Yugoslavia and Serbia are separate countries in GW --> split
reign.loc[(reign["ccode"]==345)&(reign["year"]<=2005), "ccode"] = 345 
reign.loc[(reign["ccode"]==345)&(reign["year"]<=2005), "country"] = "Yugoslavia" 
reign.loc[(reign["ccode"]==345)&(reign["year"]>=2006), "ccode"] = 340 
reign.loc[(reign["ccode"]==345)&(reign["year"]>=2006), "country"] = "Serbia" 

# (3) Check Russia and Soviet Union both coded as 365
# --> only name changes from Soviet Union to Russia

# (4) Check Yemen coded as 678 and 679 and Yemen South as 680
#  --> 679 does not exist in GW codes, replace with 678
reign.loc[reign["ccode"]==679, "ccode"] = 678

# (5) Check Vietnam coded as 816 and 817 --> check 817 is not included in sample

# (6) Check Kiribati coded as 946
# replace with GW code
reign.loc[reign["country"]=="Kiribati", "ccode"] = 970

# (7) Check Tuvalu coded as 947
# replace with GW code
reign.loc[reign["country"]=="Tuvalu", "ccode"] = 973

# (8) Check Tonga coded as 955
# replace with GW code
reign.loc[reign["country"]=="Tonga", "ccode"] = 972

# (9) Check Nauru coded as 970
# replace with GW code
reign.loc[reign["country"]=="Nauru", "ccode"] = 971

# Import country codes and add 
df_ccodes = pd.read_csv("df_ccodes.csv")
df_ccodes_s = df_ccodes[["gw_codes","iso_alpha3","acled_codes"]]
reign = pd.merge(reign,df_ccodes_s,how='left',left_on=['ccode'],right_on=['gw_codes'])

# Add dates 
reign['dd'] = pd.to_datetime(reign['month'].astype(str)+reign['year'].astype(str),format='%m%Y').dt.to_period('M')

# Only keep needed columns
reign = reign[["country",
               #"ccode",
               "gw_codes",
               "iso_alpha3",
               "acled_codes",
               "dd",
               "year",
               "month",
               #"leader", # Provides the de-facto leader’s name.
               #"elected", # whether the de facto leader had previously been elected
               #"age", # leader’s age
               #"male", # ex of the de facto leader
               #"militarycareer", # career in the military, police force or defense ministry.
               "tenure_months", # months that a leader has been in power
               #"government", # regime type
               #"gov_democracy", # either a parliamentary democracy or presidential democrac
               "dem_duration", # logged number of months that a country has had a democratic government
               #"anticipation", # here is an election for the de facto leadership position coming within the next six-months.
               #"ref_ant", # here is a constitutional referendum coming within the next six-months
               #"leg_ant", # here is a legislative election to determine the de facto leader coming within the next six-months
               #"exec_ant", # here is an executive election to determine the de facto leader coming within the next six-months
               #"irreg_lead_ant", # an irregular election to determine the de facto leader is expected within the next six months
               "election_now", # there is an election for the de facto leadership position taking place in that country-month
               #"election_recent", # there is an election for the de facto leadership position that took place in the previous six months.
               #"leg_recent", # here is a legislative election took place in the previous six months
               #"exec_recent", # here is an executive election took place in the previous six months
               #"lead_recent", # if any electoral opportunity (non-referendum) to change leadership took place in the previous six months
               #"ref_recent", # there is a constitutional referendum took place in the previous six months
               #"direct_recent", # a direct (popular) election took place in the previous six months.
               #"indirect_recent", # an indirect (elite) election took place in the previous six months
               #"victory_recent", # an incumbent political party/leader won an election in the previous six months
               #"defeat_recent", #  an incumbent political party/leader won an election in the previous six months
               #"change_recent", # the de facto leader changed due to an election in the previous six months
               #"nochange_recent", # the de facto leader did not change following an election in the previous six months.
               #"delayed", # previously scheduled/expected election is cancelled by choice or through exogenous factors 
               "lastelection", # time since the last election
               #"loss", # 
               #"irregular", # 
               #"political_violence", # elative level (z-score) of political violence 
               #"prev_conflict", # umber of on-going violent civil and inter-state conflicts that the country was involved in during the previous month. 
               #"pt_suc", # a successful coup event took place in that month
               #"pt_attempt", #  coup attempt, regardless of success, took place in that month
               #"precip", #  measures the Standardized Precipitation Index (SPI) for each country month
               #"couprisk", # estimated probability of the risk of a military coup attempt taking place in the country-month.
               #"pctile_risk" # the percentile risk for each country’s estimated risk of a military coup attempt that month.
                  ]]

# Fix Vietnam South manually, not included in sample
reign.loc[reign["country"]=="Vietnam South", "gw_codes"] = 817
reign.loc[reign["country"]=="Vietnam South", "acled_codes"] = 99999999

# Aggregate from month to year level

# (1) Election: Elections takes value one if there was an election in that year.

# Get the maximum of election_now over year --> if value is one, there was an election in that year
group=pd.DataFrame(reign.groupby(["year","country"])['election_now'].max())
group=group.reset_index()

# Merge with reign (on country-month)
group.columns=["year","country","elections"]
reign=pd.merge(reign,group,on=["year","country"],how="left")

# (2) Tenure_months, dem_duration and lastelection.

# For the other variables the value at the end of each year is used. 

# Subset reign data to only include the last month
reign=reign.loc[reign["month"]==12]
# Note that this drops the last year (2021) from the data because it does not
# have all 12 months available.

# Only include needed variables 
reign=reign[['country','gw_codes','year','tenure_months','dem_duration','elections','lastelection']]

# The data has duplicates
duplicates = reign.duplicated(subset=['country', 'year'],keep=False)
duplicate_rows = reign[duplicates]
# when there is a change in leadership, the country-month is coded twice (sometimes?)

# Remove duplicates --> keep the one with the old leader, or rather remove the new
reign=reign[~reign.index.isin(duplicate_rows.loc[duplicate_rows["tenure_months"]==1].index)]

# Check --> no duplicates
duplicates = reign.duplicated(subset=['country', 'year'],keep=False)
duplicate_rows = reign[duplicates]

# Save
reign["gw_codes"]=reign["gw_codes"].astype(int)
reign=reign.sort_values(by=['gw_codes', 'year'])
reign=reign.reset_index(drop=True)
reign.to_csv("data_out/reign_cy.csv",sep=',')
print(reign.duplicated(subset=['year',"country","gw_codes"]).any())
print(reign.duplicated(subset=['year',"country"]).any())
print(reign.duplicated(subset=['year',"gw_codes"]).any())
reign.dtypes


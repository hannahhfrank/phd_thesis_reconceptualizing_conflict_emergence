import pandas as pd
from functions import dichotomize,lag_groupped,consec_zeros_grouped,exponential_growth,simple_imp_grouped,linear_imp_grouped
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

# List of microstates: 
# http://ksgleditsch.com/data-4.html
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

# Load
ucdp_sb = pd.read_csv("data/data_out/ucdp_cy_sb.csv",index_col=0)
df = ucdp_sb[["year","gw_codes","country","best"]][~ucdp_sb['gw_codes'].isin(list(micro_states.values())+list(exclude.values())+list(exclude2.values()))]
df.columns=["year","gw_codes","country","sb_fatalities"]

# (1) t-1  
df["sb_fatalities_lag1"]=lag_groupped(df,"country","sb_fatalities",1)

# (2) Time since civil conflict 
dichotomize(df,"sb_fatalities","d_civil_conflict",25)
df['d_civil_conflict_zeros'] = consec_zeros_grouped(df,'country','d_civil_conflict')
df["d_civil_conflict_zeros"]=lag_groupped(df,"country","d_civil_conflict_zeros",1)
df['d_civil_conflict_zeros_growth'] = exponential_growth(df['d_civil_conflict_zeros'])
df = df.drop('d_civil_conflict', axis=1)
df = df.drop('d_civil_conflict_zeros', axis=1)

# (3) Time since civil war
dichotomize(df,"sb_fatalities","d_civil_war",1000)
df['d_civil_war_zeros'] = consec_zeros_grouped(df,'country','d_civil_war')
df["d_civil_war_zeros"]=lag_groupped(df,"country","d_civil_war_zeros",1)
df['d_civil_war_zeros_growth'] = exponential_growth(df['d_civil_war_zeros'])
df = df.drop('d_civil_war', axis=1)
df = df.drop('d_civil_war_zeros', axis=1)

# Neighbor conflict history sb fatalities 
neighbors=pd.read_csv("data/data_out/cy_neighbors.csv",index_col=0)
gw_codes=pd.read_csv("data/df_ccodes_gw.csv")
# Country names in neighbors file and df_codes need to be the same
gw_codes_s=gw_codes.loc[gw_codes["end"]>=1989]
df_neighbors=pd.merge(left=df[["year","country","gw_codes","sb_fatalities"]],right=neighbors[["gw_codes","year","neighbors"]],on=["year","gw_codes"],how="left")

df_neighbors["neighbors_fat"]=0
# Loop through every observation
for i in range(len(df_neighbors)):
    
    # If no neighbors pass on
    if pd.isna(df_neighbors["neighbors"].iloc[i]): 
        pass
    else:
        
        # Get list of neighbors and set fatalities to zero
        lst=df_neighbors["neighbors"].iloc[i].split(';')
        counts=0
        
        # For each neighbor
        for x in lst:
            # get gw code
            c=int(gw_codes_s["gw_codes"].loc[gw_codes_s["country"]==x].iloc[0])
            
            # If neighbor exists in data (e.g., microstates are dropped), obtain fatalities and add to count
            if df_neighbors["sb_fatalities"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].empty==False:
                counts+=int(df_neighbors["sb_fatalities"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].iloc[0])
       
        # If count is larger than zero, add to df
        if counts>0:
            df_neighbors.iloc[i, df_neighbors.columns.get_loc('neighbors_fat')] = counts

# Dichotomize and lag --> At least one neighbor had at least on fatality in previous year
dichotomize(df_neighbors,"neighbors_fat","d_neighbors_sb_fatalities",0)
df_neighbors['d_neighbors_sb_fatalities_lag1'] = lag_groupped(df_neighbors,'country','d_neighbors_sb_fatalities',1)
df=pd.merge(left=df,right=df_neighbors[["year","gw_codes","d_neighbors_sb_fatalities_lag1"]],on=["year","gw_codes"],how="left")

#######################
### World Bank data ###
#######################

# Load wb data, previously retrived with the WB api
economy=pd.read_csv("data/economy_wb.csv",index_col=0) 

# (4) GDP per capita

# Imputation
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD"]],on=["year","gw_codes"],how="left")
base_imp=linear_imp_grouped(base,"country",["NY.GDP.PCAP.CD"])
base_imp_mean=simple_imp_grouped(base,"country",["NY.GDP.PCAP.CD"])
base_imp["NY.GDP.PCAP.CD"] = base_imp["NY.GDP.PCAP.CD"].fillna(base_imp_mean["NY.GDP.PCAP.CD"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GDP.PCAP.CD"].loc[base["country"]==c])
#    axs[1].plot(base_imp["year"].loc[base_imp["country"]==c], base_imp["NY.GDP.PCAP.CD"].loc[base_imp["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df=pd.merge(left=df,right=base_imp[["year","gw_codes","NY.GDP.PCAP.CD"]],on=["year","gw_codes"],how="left")
df.rename(columns={"NY.GDP.PCAP.CD": 'gdp'}, inplace=True)

# (5) GDP growth (annual %)

# Imputation
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.MKTP.KD.ZG"]],on=["year","gw_codes"],how="left")
base_imp=linear_imp_grouped(base,"country",["NY.GDP.MKTP.KD.ZG"])
base_imp_mean=simple_imp_grouped(base,"country",["NY.GDP.MKTP.KD.ZG"])
base_imp["NY.GDP.MKTP.KD.ZG"] = base_imp["NY.GDP.MKTP.KD.ZG"].fillna(base_imp_mean["NY.GDP.MKTP.KD.ZG"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GDP.MKTP.KD.ZG"].loc[base["country"]==c])
#    axs[1].plot(base_imp["year"].loc[base_imp["country"]==c], base_imp["NY.GDP.MKTP.KD.ZG"].loc[base_imp["country"]==c])
#    axs[0].set_title(c)
#    plt.show()
    
# Merge
df=pd.merge(left=df,right=base_imp[["year","gw_codes","NY.GDP.MKTP.KD.ZG"]],on=["year","gw_codes"],how="left")
df.rename(columns={"NY.GDP.MKTP.KD.ZG": 'growth'}, inplace=True)

# (6) Oil rents (% of GDP)

# Imputation
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PETR.RT.ZS"]],on=["year","gw_codes"],how="left")
base_imp=linear_imp_grouped(base,"country",["NY.GDP.PETR.RT.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["NY.GDP.PETR.RT.ZS"])
base_imp["NY.GDP.PETR.RT.ZS"] = base_imp["NY.GDP.PETR.RT.ZS"].fillna(base_imp_mean["NY.GDP.PETR.RT.ZS"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GDP.PETR.RT.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["oil_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df=pd.merge(left=df,right=base_imp[["year","gw_codes","NY.GDP.PETR.RT.ZS"]],on=["year","gw_codes"],how="left")
df.rename(columns={"NY.GDP.PETR.RT.ZS": 'oil_share'}, inplace=True)

# (7) Population size

# Imputation
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","SP.POP.TOTL"]],on=["year","gw_codes"],how="left")
base_imp=linear_imp_grouped(base,"country",["SP.POP.TOTL"])
base_imp_mean=simple_imp_grouped(base,"country",["SP.POP.TOTL"])
base_imp["SP.POP.TOTL"] = base_imp["SP.POP.TOTL"].fillna(base_imp_mean["SP.POP.TOTL"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SP.POP.TOTL"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["pop"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df=pd.merge(left=df,right=base_imp[["year","gw_codes","SP.POP.TOTL"]],on=["year","gw_codes"],how="left")
df.rename(columns={"SP.POP.TOTL": 'pop'}, inplace=True)

# (8) Mortality rate, infant (per 1,000 live births)

# Imputation
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SP.POP.TOTL","NY.GDP.PETR.RT.ZS","SP.DYN.IMRT.IN",'SP.POP.2024.MA.5Y',"ER.H2O.FWTL.ZS"]],on=["year","gw_codes"],how="left")
base_imp=linear_imp_grouped(base,"country",["SP.DYN.IMRT.IN"])
base_imp_mean=simple_imp_grouped(base,"country",["SP.DYN.IMRT.IN"])
base_imp["SP.DYN.IMRT.IN"] = base_imp["SP.DYN.IMRT.IN"].fillna(base_imp_mean["SP.DYN.IMRT.IN"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SP.DYN.IMRT.IN"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["inf_mort"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df=pd.merge(left=df,right=base_imp[["year","gw_codes","SP.DYN.IMRT.IN"]],on=["year","gw_codes"],how="left")
df.rename(columns={"SP.DYN.IMRT.IN": 'inf_mort'}, inplace=True)

# (9) Male total population 15-19

# Imputation
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SP.POP.TOTL","NY.GDP.PETR.RT.ZS","SP.DYN.IMRT.IN",'SP.POP.2024.MA.5Y',"ER.H2O.FWTL.ZS"]],on=["year","gw_codes"],how="left")
base_imp=linear_imp_grouped(base,"country",["SP.POP.2024.MA.5Y"])
base_imp_mean=simple_imp_grouped(base,"country",["SP.POP.2024.MA.5Y"])
base_imp["SP.POP.2024.MA.5Y"] = base_imp["SP.POP.2024.MA.5Y"].fillna(base_imp_mean["SP.POP.2024.MA.5Y"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SP.POP.2024.MA.5Y"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["male_youth_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df=pd.merge(left=df,right=base_imp[["year","gw_codes","SP.POP.2024.MA.5Y"]],on=["year","gw_codes"],how="left")
df.rename(columns={"SP.POP.2024.MA.5Y": 'male_youth_share'}, inplace=True)

# (10) Average Mean Surface Air Temperature 

# Imputation
temp=pd.read_csv("data/data_out/temp_cy.csv",index_col=0)
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=temp[["gw_codes","year","temp"]],on=["year","gw_codes"],how="left")
base_imp=linear_imp_grouped(base,"country",["temp"])
base_imp_mean=simple_imp_grouped(base,"country",["temp"])
base_imp["temp"] = base_imp["temp"].fillna(base_imp_mean["temp"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["temp"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["temp"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df=pd.merge(left=df,right=base_imp[["year","gw_codes","temp"]],on=["year","gw_codes"],how="left")

# (11) Annual freshwater withdrawals, total (% of internal resources)  

# Imputation
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SP.POP.TOTL","NY.GDP.PETR.RT.ZS","SP.DYN.IMRT.IN",'SP.POP.2024.MA.5Y',"ER.H2O.FWTL.ZS"]],on=["year","gw_codes"],how="left")
base_imp=linear_imp_grouped(base,"country",["ER.H2O.FWTL.ZS"])
base_imp_mean=simple_imp_grouped(base,"country",["ER.H2O.FWTL.ZS"])
base_imp["ER.H2O.FWTL.ZS"] = base_imp["ER.H2O.FWTL.ZS"].fillna(base_imp_mean["ER.H2O.FWTL.ZS"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["ER.H2O.FWTL.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["ER.H2O.FWTL.ZS"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df=pd.merge(left=df,right=base_imp[["year","gw_codes","ER.H2O.FWTL.ZS"]],on=["year","gw_codes"],how="left")
df.rename(columns={"ER.H2O.FWTL.ZS": 'withdrawl'}, inplace=True)

###########
### EPR ###
###########

# (12) Ethnic fractionalization

# Imputation
base=df[["year","gw_codes","country"]].copy()
erp=pd.read_csv("data/data_out/epr_cy.csv",index_col=0)
base=pd.merge(left=base,right=erp[["year","gw_codes","ethnic_frac"]],on=["year","gw_codes"],how="left")
base_imp=linear_imp_grouped(base,"country",["ethnic_frac"])
base_imp_mean=simple_imp_grouped(base,"country",["ethnic_frac"])
base_imp["ethnic_frac"] = base_imp["ethnic_frac"].fillna(base_imp_mean["ethnic_frac"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["ethnic_frac"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["ethnic_frac"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df=pd.merge(left=df,right=base_imp[["year","gw_codes","ethnic_frac"]],on=["year","gw_codes"],how="left")

############
### UNDP ###
############

# (13) Expected years of schooling, male

# Imputation
hdi=pd.read_csv("data/data_out/hdi_cy.csv",index_col=0)
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=hdi[["year","gw_codes",'eys','eys_male','eys_female','mys','mys_male','mys_female']],on=["year","gw_codes"],how="left")
base_imp=linear_imp_grouped(base,"country",["eys_male"])
base_imp_mean=simple_imp_grouped(base,"country",["eys_male"])
base_imp["eys_male"] = base_imp["eys_male"].fillna(base_imp_mean["eys_male"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["eys_male"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["eys_male"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df=pd.merge(left=df,right=base_imp[["year","gw_codes","eys_male"]],on=["year","gw_codes"],how="left")

#############
### V-Dem ###
#############

# (14) Electoral democracy index 
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)

# Merge
df=pd.merge(left=df,right=vdem[["year","gw_codes","v2x_polyarchy"]],on=["year","gw_codes"],how="left")
df = df.rename(columns={"v2x_polyarchy": 'polyarchy'})

# (15) Liberal democracy index

# Imputation
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_egaldem","v2x_civlib","v2xpe_exlsocgr"]],on=["year","gw_codes"],how="left")
base_imp=linear_imp_grouped(base,"country",["v2x_libdem"])
base_imp_mean=simple_imp_grouped(base,"country",["v2x_libdem"])
base_imp["v2x_libdem"] = base_imp["v2x_libdem"].fillna(base_imp_mean["v2x_libdem"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["v2x_libdem"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["libdem"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df=pd.merge(left=df,right=base_imp[["year","gw_codes","v2x_libdem"]],on=["year","gw_codes"],how="left")
df.rename(columns={"v2x_libdem": 'libdem'}, inplace=True)

# (16) Egalitarian democracy index  
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)

# Merge
df=pd.merge(left=df,right=vdem[["year","gw_codes","v2x_egaldem"]],on=["year","gw_codes"],how="left")
df = df.rename(columns={"v2x_egaldem": 'egaldem'})

# (17) Civil liberties index  
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)

# Merge
df=pd.merge(left=df,right=vdem[["year","gw_codes","v2x_civlib"]],on=["year","gw_codes"],how="left")
df = df.rename(columns={"v2x_civlib": 'civlib'})

# (18 )Exclusion by Social Group index

# Imputation
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_egaldem","v2x_civlib","v2xpe_exlsocgr"]],on=["year","gw_codes"],how="left")
base_imp=linear_imp_grouped(base,"country",["v2xpe_exlsocgr"])
base_imp_mean=simple_imp_grouped(base,"country",["v2xpe_exlsocgr"])
base_imp["v2xpe_exlsocgr"] = base_imp["v2xpe_exlsocgr"].fillna(base_imp_mean["v2xpe_exlsocgr"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["v2xpe_exlsocgr"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["exlsocgr"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df=pd.merge(left=df,right=base_imp[["year","gw_codes","v2xpe_exlsocgr"]],on=["year","gw_codes"],how="left")
df.rename(columns={"v2xpe_exlsocgr": 'exlsocgr'}, inplace=True)

# Check datatypes and convert floats to integer if needed
df.dtypes
df['sb_fatalities_lag1']=df['sb_fatalities_lag1'].astype('int64')
df['d_neighbors_sb_fatalities_lag1']=df['d_neighbors_sb_fatalities_lag1'].astype('int64')

# Save 
df = df[~df['country'].isin(['Montenegro', 'Somalia', 'Kuwait', 'Solomon Islands'])] 
print(df.isnull().any())
df=df.reset_index(drop=True)
df.to_csv("out/data_examples.csv")
print(df.duplicated(subset=["year","gw_codes","country"]).any())
print(df.duplicated(subset=["year","country"]).any())
print(df.duplicated(subset=["year","gw_codes"]).any())




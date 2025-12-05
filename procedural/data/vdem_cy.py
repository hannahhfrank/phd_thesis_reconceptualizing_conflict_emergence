import pandas as pd

# Load V-dem data Version 14
# Manually downloaded from: https://v-dem.net/data/the-v-dem-dataset/
# Codebook: https://v-dem.net/documents/38/V-Dem_Codebook_v14.pdf
vdem = pd.read_csv("V-Dem-CY-Full+Others-v14.csv")

# Add country codes from df_ccodes file
df_ccodes = pd.read_csv("df_ccodes.csv")
df_ccodes_s = df_ccodes[["gw_codes","iso_alpha3","acled_codes","vdem_codes"]]
vdem = pd.merge(vdem,df_ccodes_s,how='left',left_on=['country_id'],right_on=['vdem_codes'])

# Keep needed columns and rename
vdem = vdem[["country_name","year","country_id","gw_codes","iso_alpha3","acled_codes",
             "v2x_polyarchy",
             "v2x_libdem",
             "v2x_partipdem",
             "v2x_delibdem",
             "v2x_egaldem",
             "v2x_civlib",
             "v2x_clphy",
             "v2x_clpol",
             "v2x_clpriv",
             "v2xpe_exlecon",
             "v2xpe_exlgender",
             "v2xpe_exlgeo",
             "v2xpe_exlpol",
             "v2xpe_exlsocgr",
             "v2smgovshut",
             "v2smgovfilprc"]]

vdem.columns=["country","year","vdem_ccode","gw_codes","iso_alpha3","acled_codes",
             "v2x_polyarchy",
             "v2x_libdem",
             "v2x_partipdem",
             "v2x_delibdem",
             "v2x_egaldem",
             "v2x_civlib",
             "v2x_clphy",
             "v2x_clpol",
             "v2x_clpriv",
             "v2xpe_exlecon",
             "v2xpe_exlgender",
             "v2xpe_exlgeo",
             "v2xpe_exlpol",
             "v2xpe_exlsocgr",
             "v2smgovshut",
             "v2smgovfilprc"]

# Check which countries are not matched and add codes manually
# Make sure no countries get missing that should be in sample
# http://ksgleditsch.com/data-4.html
vdem.loc[vdem["country"]=="Republic of Vietnam", "gw_codes"] = 817
vdem.loc[vdem["country"]=="Palestine/West Bank", "gw_codes"] = 99999999
vdem.loc[vdem["country"]=="Palestine/Gaza", "gw_codes"] = 99999999
vdem.loc[vdem["country"]=="Somaliland", "gw_codes"] = 99999999
vdem.loc[vdem["country"]=="Hong Kong", "gw_codes"] = 99999999
vdem.loc[vdem["country"]=="Palestine/British Mandate", "gw_codes"] = 99999999
vdem.loc[vdem["country"]=="Zanzibar", "gw_codes"] = 99999999
vdem.loc[vdem["country"]=="Baden", "gw_codes"] = 267
vdem.loc[vdem["country"]=="Bavaria", "gw_codes"] = 245
vdem.loc[vdem["country"]=="Modena", "gw_codes"] = 332
vdem.loc[vdem["country"]=="Parma", "gw_codes"] = 335
vdem.loc[vdem["country"]=="Saxony", "gw_codes"] = 269
vdem.loc[vdem["country"]=="Tuscany", "gw_codes"] = 337
vdem.loc[vdem["country"]=="Würtemberg", "gw_codes"] = 271
vdem.loc[vdem["country"]=="Two Sicilies", "gw_codes"] = 329
vdem.loc[vdem["country"]=="Hanover", "gw_codes"] = 240
vdem.loc[vdem["country"]=="Hesse-Kassel", "gw_codes"] = 273
vdem.loc[vdem["country"]=="Hesse-Darmstadt", "gw_codes"] = 275
vdem.loc[vdem["country"]=="Mecklenburg Schwerin", "gw_codes"] = 280
vdem.loc[vdem["country"]=="Papal States", "gw_codes"] = 327
vdem.loc[vdem["country"]=="Hamburg", "gw_codes"] = 99999999
vdem.loc[vdem["country"]=="Brunswick", "gw_codes"] = 99999999
vdem.loc[vdem["country"]=="Oldenburg", "gw_codes"] = 99999999
vdem.loc[vdem["country"]=="Saxe-Weimar-Eisenach", "gw_codes"] = 99999999
vdem.loc[vdem["country"]=="Nassau", "gw_codes"] = 99999999
vdem.loc[vdem["country"]=="Piedmont-Sardinia", "gw_codes"] = 325

# Convert datatype
vdem['gw_codes']=vdem['gw_codes'].astype('int64')

# Save data 
vdem = vdem.sort_values(by=["country", "year"])
vdem=vdem.reset_index(drop=True)
vdem.to_csv("data_out/vdem_cy.csv",sep=',')
print(vdem.duplicated(subset=['year',"country","gw_codes"]).any())
print(vdem.duplicated(subset=['year',"country"]).any())
vdem.dtypes

# gw_codes has duplicates because missing countries are coded as 99999999






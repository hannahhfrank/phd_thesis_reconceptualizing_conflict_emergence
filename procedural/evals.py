import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import gridspec
from functions import evals
import matplotlib as mpl
import os
from matplotlib.ticker import MaxNLocator
os.environ['PATH'] = "/Library/TeX/texbin:" + os.environ.get('PATH', '')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}\usepackage[T1]{fontenc}'
plt.rcParams['xtick.labelsize'] = 20  
plt.rcParams['ytick.labelsize'] = 20  
plt.rcParams['axes.labelsize'] = 20 

# Load predictions and get evaluations

# Shape finder
shape_finder=pd.read_csv("out/sf.csv",index_col=0)
evals_shape_finder = evals(shape_finder,"actuals","preds","country")
print(f"Onset Score {evals_shape_finder['Onset Score']}")
print(f"Mean MSE {evals_shape_finder['Mean MSE']}")
print(f"Normalized MSE {evals_shape_finder['Normalized MSE']}")
        
# VIEWS
views=pd.read_csv("out/views.csv",index_col=0)
evals_views = evals(views, "actuals", "preds", "country")
print(f"Onset Score {evals_views['Onset Score']}")
print(f"Mean MSE {evals_views['Mean MSE']}")
print(f"Normalized MSE {evals_views['Normalized MSE']}")

# Negative-binomial 
zinb=pd.read_csv("out/zinb.csv",index_col=0)
evals_zinb = evals(zinb, "sb_fatalities", "preds", "country")
print(f"Onset Score {evals_zinb['Onset Score']}")
print(f"Mean MSE {evals_zinb['Mean MSE']}")
print(f"Normalized MSE {evals_zinb['Normalized MSE']}")

# Onset catcher
catcher=pd.read_csv("out/catcher.csv",index_col=0)
evals_catcher = evals(catcher, "sb_fatalities", "preds", "country")      
print(f"Onset Score {evals_catcher['Onset Score']}")
print(f"Mean MSE {evals_catcher['Mean MSE']}")
print(f"Normalized MSE {evals_catcher['Normalized MSE']}")

### Ensemble ###

# Create an ensemble between Onset catcher and VIEWS. A simple selection rule
# is used. For countries where the onset catcher predicts more than 1 onset
# regardless of whether the onset is correct, the Onset catcher returns the 
# final predictions. For all other countries, VIEWS is used. The VIEWS
# predictions are evaluated against state-based fatalities, and the Onset catcher
# against civil conflict fatalities.

# Count onsets per country
catcher_countries=pd.DataFrame()
# Loop through every country 
for c in catcher["country"].unique():
    df_s = catcher[catcher["country"] == c]
    # and count number of predicted onsets
    counts={"country":c,"onsets":len(df_s["preds"][(df_s["preds"].shift(1)==0)&(df_s["preds"]>0)].index)}
    counts=pd.DataFrame(counts,index=[0])
    catcher_countries=pd.concat([catcher_countries,counts],ignore_index=True)
        
# Get countries where count is higher than 1
select_catcher=catcher_countries["country"].loc[catcher_countries["onsets"]>1]
print(select_catcher)

# Subset onset catcher predictions to only include those countries
catcher_s = catcher[catcher['country'].isin(select_catcher)]

# Make base df and merge selected onset catcher predictions
base = catcher[["country","dd","sb_fatalities"]]
ensemble=pd.merge(base,catcher_s[["country","dd","preds"]],on=["country","dd"],how="left")

# Careful: VIEWS predicts state-based conflict and not civil conflict. 
# If VIEWS is selected in the ensemble, the value on the outcome needs to be replaced. 

# Step 1: replace sb_fatalities with nan if preds is nan, which means that onset catcher was not selected
ensemble.loc[ensemble['preds'].isna(),'sb_fatalities'] = np.nan

# Step 2: Replace missings in preds and sb_fatalities with VIEWS values
ensemble['preds'] = ensemble['preds'].fillna(views['preds'])
ensemble['sb_fatalities'] = ensemble['sb_fatalities'].fillna(views['actuals'])

# Get evaluations for ensemble
evals_ensemble = evals(ensemble, "sb_fatalities", "preds", "country")
print(f"Onset Score {evals_ensemble['Onset Score']}")
print(f"Mean MSE {evals_ensemble['Mean MSE']}")
print(f"Normalized MSE {evals_ensemble['Normalized MSE']}")

# Structural model
structural=pd.read_csv("out/structural.csv",index_col=0)
evals_structural = evals(structural, "sb_fatalities", "preds", "country")
print(f"Onset Score {evals_structural['Onset Score']}")
print(f"Mean MSE {evals_structural['Mean MSE']}")
print(f"Normalized MSE {evals_structural['Normalized MSE']}")

#################
### Main plot ###
#################

fig,ax = plt.subplots(figsize=(12,8))

# (1) Views
mse_views = np.mean(evals_views['Normalized MSE by Country'])
onset_views = np.mean(evals_views["Onset Scores by Country"])
mse_views_std = np.std(evals_views['Normalized MSE by Country'])
onset_views_std = np.std(evals_views["Onset Scores by Country"])
ax.scatter(mse_views, onset_views, color="black", s=50)
# Add confidence intervals
ax.errorbar(mse_views,onset_views,xerr=1.65*mse_views_std/np.sqrt(len(evals_views['Normalized MSE by Country'])),yerr=1.65*onset_views_std/np.sqrt(len(evals_views['Onset Scores by Country'])),linewidth=1,color="gray")
plt.text(mse_views-0.02, onset_views+0.001, "VIEWS", size=20, color='gray')

# (2) Negative-binomial (trim means and sd)
mse_zinb = np.mean(evals_zinb['Normalized MSE by Country'])
onset_zinb = np.mean(evals_zinb["Onset Scores by Country"])
mse_zinb_std = np.std(evals_zinb['Normalized MSE by Country'])
onset_zinb_std = np.std(evals_zinb["Onset Scores by Country"])
ax.scatter(mse_zinb, onset_zinb, color="black", s=50)
# Add confidence intervals
ax.errorbar(mse_zinb, onset_zinb, xerr=1.65*mse_zinb_std/np.sqrt(len(evals_zinb['Normalized MSE by Country'])), yerr=1.65*onset_zinb_std/np.sqrt(len(evals_zinb['Onset Scores by Country'])),linewidth=1,color="gray")
plt.text(mse_zinb+0.21, onset_zinb+0.001, "ZINB", size=20, color='gray')

# (3) Onset catcher
mse_catcher = np.mean(evals_catcher['Normalized MSE by Country'])
onset_catcher = np.mean(evals_catcher["Onset Scores by Country"])
mse_catcher_std = np.std(evals_catcher['Normalized MSE by Country'])
onset_catcher_std = np.std(evals_catcher["Onset Scores by Country"])
ax.scatter(mse_catcher, onset_catcher, color="black", s=50)
# Add confidence intervals
ax.errorbar(mse_catcher, onset_catcher, xerr=1.65*mse_catcher_std/np.sqrt(len(evals_catcher['Normalized MSE by Country'])), yerr=1.65*onset_catcher_std/np.sqrt(len(evals_catcher['Onset Scores by Country'])),linewidth=1,color="gray")
plt.text(mse_catcher-0.02, onset_catcher+0.001, "Onset catcher", size=20, color='gray')

# (4) Ensemble
mse_ens = np.mean(evals_ensemble['Normalized MSE by Country'])
onset_ens = np.mean(evals_ensemble["Onset Scores by Country"])
mse_ens_std = np.std(evals_ensemble['Normalized MSE by Country'])
onset_ens_std = np.std(evals_ensemble["Onset Scores by Country"])
ax.scatter(mse_ens, onset_ens, color="black", s=70, marker="x")
plt.text(mse_ens-0.02, onset_ens+0.001, "Ensemble", size=20, color='black')

# (5) Structural
mse_struc = np.mean(evals_structural['Normalized MSE by Country'])
onset_struc = np.mean(evals_structural["Onset Scores by Country"])
mse_struc_std = np.std(evals_structural['Normalized MSE by Country'])
onset_struc_std = np.std(evals_structural["Onset Scores by Country"])
ax.scatter(mse_struc, onset_struc, color="black", s=50)
# Add confidence intervals
ax.errorbar(mse_struc, onset_struc, xerr=1.65*mse_struc_std/np.sqrt(len(evals_structural['Normalized MSE by Country'])), yerr=1.65*onset_struc_std/np.sqrt(len(evals_structural['Onset Scores by Country'])),linewidth=1,color="gray")
plt.text(mse_struc-0.02, onset_struc+0.001, "Structural", size=20, color='gray')

# Ticks and labels
ax.invert_xaxis() # revert x axis
plt.xlabel("Mean squared logarithmic error (reverted)")
plt.ylabel("Onset score")
ax.set_xticks([0,0.5,1,1.5,2,2.5])
ax.set_xlim([2.5,0])
ax.set_ylim([-0.01,0.087])
ax.set_yticks([0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08])

# Save
plt.savefig("out/proc_main.eps",dpi=300,bbox_inches='tight')        
  
##############
### Robust ###
##############

# Add Shape finder

fig,ax = plt.subplots(figsize=(12,8))

# (1) Shape finder
mse_sf = np.mean(evals_shape_finder['Normalized MSE by Country'])
onset_sf = np.mean(evals_shape_finder["Onset Scores by Country"])
mse_sf_std = np.std(evals_shape_finder['Normalized MSE by Country'])
onset_sf_std = np.std(evals_shape_finder["Onset Scores by Country"])
ax.scatter(mse_sf, onset_sf, color="steelblue", s=50)
# Add confidence intervals
ax.errorbar(mse_sf, onset_sf, xerr=1.65*mse_sf_std/np.sqrt(len(evals_shape_finder['Normalized MSE by Country'])),yerr=1.65*onset_sf_std/np.sqrt(len(evals_shape_finder['Onset Scores by Country'])),linewidth=1,color="steelblue")
plt.text(mse_sf-0.02, onset_sf+0.001, "Shape finder", size=20, color='steelblue')

# (2) Views
mse_views = np.mean(evals_views['Normalized MSE by Country'])
onset_views = np.mean(evals_views["Onset Scores by Country"])
mse_views_std = np.std(evals_views['Normalized MSE by Country'])
onset_views_std = np.std(evals_views["Onset Scores by Country"])
ax.scatter(mse_views, onset_views, color="black", s=50)
# Add confidence intervals
ax.errorbar(mse_views, onset_views, xerr=1.65*mse_views_std/np.sqrt(len(evals_views['Normalized MSE by Country'])),yerr=1.65*onset_views_std/np.sqrt(len(evals_views['Onset Scores by Country'])),linewidth=1,color="gray")
plt.text(mse_views-0.02, onset_views+0.001, "VIEWS", size=20, color='gray')

# (2) Negative-binomial (trim means and sd)
mse_zinb = np.mean(evals_zinb['Normalized MSE by Country'])
onset_zinb = np.mean(evals_zinb["Onset Scores by Country"])
mse_zinb_std = np.std(evals_zinb['Normalized MSE by Country'])
onset_zinb_std = np.std(evals_zinb["Onset Scores by Country"])
ax.scatter(mse_zinb, onset_zinb, color="black", s=50)
# Add confidence intervals
ax.errorbar(mse_zinb, onset_zinb, xerr=1.65*mse_zinb_std/np.sqrt(len(evals_zinb['Normalized MSE by Country'])), yerr=1.65*onset_zinb_std/np.sqrt(len(evals_zinb['Onset Scores by Country'])),linewidth=1,color="gray")
plt.text(mse_zinb+0.21, onset_zinb+0.001, "ZINB", size=20, color='gray')

# (4) Onset catcher
mse_catcher = np.mean(evals_catcher['Normalized MSE by Country'])
onset_catcher = np.mean(evals_catcher["Onset Scores by Country"])
mse_catcher_std = np.std(evals_catcher['Normalized MSE by Country'])
onset_catcher_std = np.std(evals_catcher["Onset Scores by Country"])
ax.scatter(mse_catcher, onset_catcher, color="black", s=50)
# Add confidence intervals
ax.errorbar(mse_catcher, onset_catcher, xerr=1.65*mse_catcher_std/np.sqrt(len(evals_catcher['Normalized MSE by Country'])), yerr=1.65*onset_catcher_std/np.sqrt(len(evals_catcher['Onset Scores by Country'])),linewidth=1,color="gray")
plt.text(mse_catcher-0.02, onset_catcher+0.001, "Onset catcher", size=20, color='gray')

# (6) Structural
mse_struc = np.mean(evals_structural['Normalized MSE by Country'])
onset_struc = np.mean(evals_structural["Onset Scores by Country"])
mse_struc_std = np.std(evals_structural['Normalized MSE by Country'])
onset_struc_std = np.std(evals_structural["Onset Scores by Country"])
ax.scatter(mse_struc, onset_struc, color="black", s=50)
# Add confidence intervals
ax.errorbar(mse_struc, onset_struc, xerr=1.65*mse_struc_std/np.sqrt(len(evals_structural['Normalized MSE by Country'])), yerr=1.65*onset_struc_std/np.sqrt(len(evals_structural['Onset Scores by Country'])),linewidth=1,color="gray")
plt.text(mse_struc-0.02, onset_struc+0.001, "Structural", size=20, color='gray')

# Ticks and labels
ax.invert_xaxis() # revert x axis
plt.xlabel("Mean squared logarithmic error (reverted)")
plt.ylabel("Onset score")
ax.set_xticks([0,0.5,1,1.5,2,2.5])
ax.set_xlim([2.5,0])
ax.set_ylim([-0.01,0.087])
ax.set_yticks([0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08])

# Save
plt.savefig("out/proc_main_robust.eps",dpi=300,bbox_inches='tight')        
          
##################  
### By country ###
##################  

# For each model get scores for each country and save as df
catcher_countries = pd.DataFrame({'country': catcher.country.unique(),'mse_catcher': evals_catcher["Normalized MSE by Country"],'onset_catcher': evals_catcher["Onset Scores by Country"]})
views_countries = pd.DataFrame({'country': views.country.unique(),'mse_views': evals_views["Normalized MSE by Country"],'onset_views': evals_views["Onset Scores by Country"]})
struc_countries = pd.DataFrame({'country': structural.country.unique(),'mse_struc': evals_structural["Normalized MSE by Country"],'onset_struc': evals_structural["Onset Scores by Country"]})

# Merge the evaluations across countries
evals_country=pd.merge(catcher_countries,views_countries,on=["country"])
evals_country=pd.merge(evals_country,struc_countries,on=["country"])

# Get total sum (log + 1) of fatalities for country --> use for size of marker

# (1) Onset catcher, civil conflict
catcher["sb_fatalities_log"]=np.log(catcher["sb_fatalities"]+1)
summary=catcher.groupby('country').agg({'sb_fatalities_log':'mean'})

# (2) VIEWS, state-based conflict
views["actuals_log"]=np.log(views["actuals"]+1)
summary2=views.groupby('country').agg({'actuals_log':'mean'})

# Plot evaluations for each country
fig,ax = plt.subplots(figsize=(12,8))
plt.scatter(evals_country["mse_catcher"],evals_country["onset_catcher"],c="black",s=(summary+0.1)*100,alpha=0.5)
plt.scatter(evals_country["mse_views"],evals_country["onset_views"],c="forestgreen",s=(summary2+0.1)*100,alpha=0.5)
plt.scatter(evals_country["mse_struc"],evals_country["onset_struc"],c="steelblue",s=(summary+0.1)*100,alpha=0.5)

# Ticks and labels
ax.invert_xaxis() # revert x axis
plt.xlabel("Mean squared logarithmic error (reverted)")
plt.ylabel("Onset score")
ax.set_xticks([0,2,4,6,8,10,12,14,16,18,20,22,24,26])
ax.set_xlim([26,-0.5])
ax.set_yticks([0,0.25,0.5,0.75,1,1.25,1.5,1.75,2])
ax.set_ylim([-0.2,2.1])

# Rename some countries to avoid cluttering
evals_country.loc[evals_country["country"]=="Russia (Soviet Union)","country"]="Russia"
evals_country.loc[evals_country["country"]=="Yemen (North Yemen)","country"]="Yemen"
evals_country.loc[evals_country["country"]=="Central African Republic","country"]="CAR"
evals_country.loc[evals_country["country"]=="DR Congo (Zaire)","country"]="DRC"

# Add lables for each country which has a non zero onset score for the Onset catcher
add=evals_country.loc[evals_country["onset_catcher"]!=0]
for i in range(len(add)):
    plt.annotate(f'{add.iloc[i]["country"]}',(add.iloc[i]["mse_catcher"],add.iloc[i]["onset_catcher"]),ha='center',size=12,textcoords="offset points",xytext=(0,4))

# Add lables for each country which has a non zero onset score for the structural model
add=evals_country.loc[evals_country["onset_struc"]!=0]
for i in range(len(add)):
    plt.annotate(f'{add.iloc[i]["country"]}',(add.iloc[i]["mse_struc"],add.iloc[i]["onset_struc"]),ha='center',size=12,textcoords="offset points",xytext=(0,4),color="steelblue")

# Save
plt.savefig("out/proc_scatter.png",dpi=300,bbox_inches='tight')    

#########################
### By country robust ###
#########################

# Compare with Shape finder

# For each model get scores for each country and save as df
catcher_countries = pd.DataFrame({'country': catcher.country.unique(),'mse_catcher': evals_catcher["Normalized MSE by Country"],'onset_catcher': evals_catcher["Onset Scores by Country"]})
sf_countries = pd.DataFrame({'country': shape_finder.country.unique(),'mse_sf': evals_shape_finder["Normalized MSE by Country"],'onset_sf': evals_shape_finder["Onset Scores by Country"]})

# Merge the evaluations across countries
evals_country=pd.merge(catcher_countries,sf_countries,on=["country"])

# Get total sum (log + 1) of fatalities for country --> use for size of marker

# (1) Onset catcher, civil conflict
catcher["sb_fatalities_log"]=np.log(catcher["sb_fatalities"]+1)
summary=catcher.groupby('country').agg({'sb_fatalities_log': 'mean'})

# (2) Shape finder, state-based conflict
shape_finder["actuals_log"]=np.log(shape_finder["actuals"]+1)
summary2=shape_finder.groupby('country').agg({'actuals_log': 'mean'})

# Plot evaluations for each country
fig,ax = plt.subplots(figsize=(12,8))
plt.scatter(evals_country["mse_catcher"],evals_country["onset_catcher"],c="black",s=(summary+0.1)*100,alpha=0.5)
plt.scatter(evals_country["mse_sf"],evals_country["onset_sf"],c="steelblue",s=(summary2+0.1)*100,alpha=0.5)

# Ticks and labels
ax.invert_xaxis() # revert x axis
plt.xlabel("Mean squared logarithmic error (reverted)")
plt.ylabel("Onset score")
ax.set_xticks([0,2,4,6,8,10,12,14,16,18])
ax.set_xlim([19,-0.7])
ax.set_yticks([0,0.25,0.5,0.75,1,1.25,1.5,1.75,2])
ax.set_ylim([-0.2,2.1])

# Rename some countries to avoid cluttering
evals_country.loc[evals_country["country"]=="Russia (Soviet Union)","country"]="Russia"
evals_country.loc[evals_country["country"]=="Yemen (North Yemen)","country"]="Yemen"
evals_country.loc[evals_country["country"]=="Central African Republic","country"]="CAR"
evals_country.loc[evals_country["country"]=="DR Congo (Zaire)","country"]="DRC"

# Add lables for each country which has a non zero onset score for the Onset catcher
add=evals_country.loc[evals_country["onset_catcher"]!=0]
for i in range(len(add)):
    plt.annotate(f'{add.iloc[i]["country"]}',(add.iloc[i]["mse_catcher"], add.iloc[i]["onset_catcher"]), ha='center', size=12, textcoords="offset points", xytext=(0,4))

# Add lables for each country which has a non zero onset score for the Shape finder
add=evals_country.loc[evals_country["onset_sf"]!=0]
for i in range(len(add)):
    plt.annotate(f'{add.iloc[i]["country"]}',(add.iloc[i]["mse_sf"], add.iloc[i]["onset_sf"]), ha='center', size=12, textcoords="offset points", xytext=(0,4), color="steelblue")

# Save
plt.savefig("out/proc_scatter_robsut.png",dpi=300,bbox_inches='tight')    

########################
### Prediction plots ###
########################

# Load onset catcher predictions, and add year variable
catcher = pd.read_csv("out/catcher.csv",index_col=0)
catcher['year'] = catcher['dd'].str[:4]
catcher['year'] = catcher['year'].astype(int)

# Get prediction plot for each country
for c in catcher.country.unique():
    
    # Specify figure grid, two columns, no space between the two plots
    fig = plt.figure(figsize=(12, 5))
    grid = gridspec.GridSpec(1, 2, figure=fig)
    grid.update(wspace=0)    
    
    # Subset country
    df_s=catcher.loc[catcher["country"]==c]
    
    # Fix axis for Lebanon manually, which has only zeros in 2022
    if c=="Lebanon":
        
        # Plot 2022 # 
        
        # Access grid
        ax1 = fig.add_subplot(grid[0])
        # Plot actuals and predictions
        ax1.plot(df_s["dd"].loc[df_s["year"]==2022],df_s["sb_fatalities"].loc[df_s["year"]==2022],color="black")
        ax1.plot(df_s["dd"].loc[df_s["year"]==2022],df_s["preds"].loc[df_s["year"]==2022],color="black",marker="x",markersize=9,linestyle="dashed")   
        # Set ticks and range
        ax1.set_xticks([0,2,4,6,8,10],["01-22","03-22","05-22","07-22","09-22","11-22"])
        ax1.set_yticks([0,1,2,3,4,5],[0,1,2,3,4,5])
        ax1.set_ylim([-0.5,5])
    
        # Plot 2023 # 
        
        # Access grid
        ax2 = fig.add_subplot(grid[1])  
        # Plot actuals and predictions
        ax2.plot(df_s["dd"].loc[df_s["year"]==2023],df_s["sb_fatalities"].loc[df_s["year"]==2023],color="black")
        ax2.plot(df_s["dd"].loc[df_s["year"]==2023],df_s["preds"].loc[df_s["year"]==2023],color="black",marker="x",markersize=9,linestyle="dashed") 
        # Set labels 
        ax2.set_xticks([0,2,4,6,8,10],["01-23","03-23","05-23","07-23","09-23","11-23"])
        # Move y axis to right and make sure that y axis only contain integers (no floats)      
        ax2.yaxis.set_ticks_position('right')
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    else: 
    
        # Plot 2022 # 
        
        # Access grid
        ax1 = fig.add_subplot(grid[0])
        # Plot actuals and predictions
        ax1.plot(df_s["dd"].loc[df_s["year"]==2022],df_s["sb_fatalities"].loc[df_s["year"]==2022],color="black")
        ax1.plot(df_s["dd"].loc[df_s["year"]==2022],df_s["preds"].loc[df_s["year"]==2022],color="black",marker="x",markersize=9,linestyle="dashed")   
        # Set ticks and make sure that y axis only contains integers (no floats)
        ax1.set_xticks([0,2,4,6,8,10],["01-22","03-22","05-22","07-22","09-22","11-22"])
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    
        # Plot 2023 # 
        
        # Access grid
        ax2 = fig.add_subplot(grid[1])  
        # Plot actuals and predictions
        ax2.plot(df_s["dd"].loc[df_s["year"]==2023],df_s["sb_fatalities"].loc[df_s["year"]==2023],color="black")
        ax2.plot(df_s["dd"].loc[df_s["year"]==2023],df_s["preds"].loc[df_s["year"]==2023],color="black",marker="x",markersize=9,linestyle="dashed") 
        ax2.set_xticks([0,2,4,6,8,10],["01-23","03-23","05-23","07-23","09-23","11-23"])
        # Move y axis to right and make sure that y axis only contain integers (no flots) 
        ax2.yaxis.set_ticks_position('right')
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    #fig.suptitle(catcher["country"].loc[catcher["country"]==c].iloc[0],size=30)
    
    # Save well working examples
    if c=="Lebanon" or c=="Indonesia" or c=="Armenia" or c=="Azerbaijan" or c=="Uganda" or c=="Chad" or c=="Burundi"  or c=="Philippines" or c=="Israel" or c=="Sudan":
        plt.savefig(f"out/proc_preds_{c}.eps",dpi=300,bbox_inches='tight')  
        
    if c=="Russia (Soviet Union)":
        c="Russia"
        plt.savefig(f"out/proc_preds_{c}.eps",dpi=300,bbox_inches='tight')  

    # Save poorly working examples
    if c=="Togo" or c=="Mexico" or c=="Pakistan" or c=="Mozambique":
        plt.savefig(f"out/proc_preds_{c}.eps",dpi=300,bbox_inches='tight') 
    plt.show()   

########################
### Balancing factor ###
########################

# Validate the effect of the balancing factor in the onset score. 

print(1*np.exp(-0.001*0))
print(1*round(np.exp(-0.001*1),4))
print(1*round(np.exp(-0.001*2),4))
print(1*round(np.exp(-0.001*3),4))

print(1*np.exp(-0.005*0))
print(1*round(np.exp(-0.005*1),4))
print(1*round(np.exp(-0.005*2),4))
print(1*round(np.exp(-0.005*3),4))

print(1*np.exp(-0.01*0))
print(1*round(np.exp(-0.01*1),4))
print(1*round(np.exp(-0.01*2),4))
print(1*round(np.exp(-0.01*3),4))

print(1*np.exp(-0.05*0))
print(1*round(np.exp(-0.05*1),4))
print(1*round(np.exp(-0.05*2),4))
print(1*round(np.exp(-0.05*3),4))

print(1*np.exp(-0.1*0))
print(1*round(np.exp(-0.1*1),4))
print(1*round(np.exp(-0.1*2),4))
print(1*round(np.exp(-0.1*3),4))




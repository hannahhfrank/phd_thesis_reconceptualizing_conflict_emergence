import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.inspection import partial_dependence
import shap
from PyALE import ale
import matplotlib as mpl
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import random
random.seed(42)
import seaborn as sns
import os 
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from matplotlib.ticker import FormatStrFormatter
os.environ['PATH'] = "/Library/TeX/texbin:" + os.environ.get('PATH', '')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}\usepackage[T1]{fontenc}'

grid = {"n_estimators": [10, 231, 452, 673, 894, 1115, 1336, 1557, 1778, 2000],              
        "max_features": ["sqrt", "log2", None],
        "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]}

# Load data
df=pd.read_csv("out/data_examples.csv",index_col=0)

# Check distributions
for var in ['sb_fatalities_lag1','d_civil_conflict_zeros_growth','d_civil_war_zeros_growth','d_neighbors_sb_fatalities_lag1','gdp','growth','oil_share','pop','inf_mort','male_youth_share','temp','withdrawl','ethnic_frac','eys_male','polyarchy','libdem','egaldem','civlib','exlsocgr']:
    fig,ax = plt.subplots()
    df[var].hist()
    
# Transforms
df["sb_fatalities_log"]=np.log(df["sb_fatalities"]+1)
df["sb_fatalities_lag1_log"]=np.log(df["sb_fatalities_lag1"]+1)
df["gdp"]=np.log(df["gdp"]+1)
df["oil_share"]=np.log(df["oil_share"]+1)
df["pop"]=np.log(df["pop"]+1)
df["withdrawl"]=np.log(df["withdrawl"]+1)
   
# Specify model    
target='sb_fatalities_log'
inputs=['sb_fatalities_lag1_log','d_civil_conflict_zeros_growth','d_civil_war_zeros_growth','d_neighbors_sb_fatalities_lag1','gdp','growth','oil_share','pop','inf_mort','male_youth_share','temp','withdrawl','ethnic_frac','eys_male','polyarchy','libdem','egaldem','civlib','exlsocgr']
y=df[["year",'country','sb_fatalities_log']]
x=df[["year",'country','sb_fatalities_lag1_log','d_civil_conflict_zeros_growth','d_civil_war_zeros_growth','d_neighbors_sb_fatalities_lag1','gdp','growth','oil_share','pop','inf_mort','male_youth_share','temp','withdrawl','ethnic_frac','eys_male','polyarchy','libdem','egaldem','civlib','exlsocgr']]

# Split data
training_y = pd.DataFrame()
testing_y = pd.DataFrame()
training_x = pd.DataFrame()
testing_x = pd.DataFrame()
splits=[] 

for c in y.country.unique():
    y_s = y.loc[y["country"] == c]
    x_s = x.loc[x["country"] == c]
    y_training = y_s[["country","year"]+[target]][:int(0.8*len(y_s))]
    x_training = x_s[["country","year"]+inputs][:int(0.8*len(x_s))]
    y_testing = y_s[["country","year"]+[target]][int(0.8*len(y_s)):]
    x_testing = x_s[["country","year"]+inputs][int(0.8*len(x_s)):]
    training_y = pd.concat([training_y, y_training])
    testing_y = pd.concat([testing_y, y_testing])
    training_x = pd.concat([training_x, x_training])
    testing_x = pd.concat([testing_x, x_testing])  
    val_training_ids = list(y_training[:int(0.8*len(y_training))].index)
    val_testing_ids = list(y_training[int(0.8*len(y_training)):].index)   
    splits+=[-1] * len(val_training_ids) + [0] * len(val_testing_ids)
    
training_y_d=training_y.drop(columns=["country","year"])
training_x_d=training_x.drop(columns=["country","year"])
testing_y_d=testing_y.drop(columns=["country","year"])
testing_x_d=testing_x.drop(columns=["country","year"])

# Optimize model
splits = PredefinedSplit(test_fold=splits)
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=1), param_grid=grid, cv=splits, verbose=0, n_jobs=-1)
grid_search.fit(training_x_d, training_y_d.values.ravel())
best_params = grid_search.best_params_
model=RandomForestRegressor(random_state=1,**best_params)
model.fit(training_x_d, training_y_d.values.ravel())
pred = pd.DataFrame(model.predict(testing_x_d))
mse = mean_squared_error(testing_y_d.values, pred)
print(mse)

################################
### Interpretability methods ###
################################

names={"sb_fatalities_lag1_log":"t-1 lag of the number of fatalities (log + 1)",
       "d_civil_conflict_zeros_growth":"Exponential time since last civil conflict",
       "d_civil_war_zeros_growth":"Exponential time since last civil war",
       "d_neighbors_sb_fatalities_lag1":"At least one fatality in neighborhood (t - 1)",
       "inf_mort":"Mortality rate, infant",       
       "gdp":"GDP per capita (log + 1)",
       "growth":"GDP growth",
       "oil_share":"Oil rents (log + 1)",
       "pop":"Population size (log + 1)",
       "male_youth_share":"Share of male population 15-19 years",
       "ethnic_frac":"Ethnolinguistic fractionalization (ELF) index",
       "eys_male":"Expected years of schooling, male",
       "temp":"Average mean surface air temperature",
       "withdrawl":"Annual freshwater withdrawals",
       "polyarchy":"Electoral democracy index",
       "libdem":"Liberal democracy index",
       "egaldem":"Egalitarian democracy index",
       "civlib":"Civil liberties index",
       "exlsocgr":"Exclusion by Social Group index"}

x_d=x.drop(columns=["country","year"])
y_d=y.drop(columns=["country","year"])

# (1) Shap
explainer = shap.Explainer(model)
shap_values = explainer.shap_values(x_d)

# Get Shap plot for each variable
for count in range(0,len(inputs)):
    
    # If variable is binary, return violine plot
    if len(x_d[inputs[count]].unique())==2:
        
        # Plot
        fig,ax = plt.subplots(figsize=(12, 8))
        
        # Get x values (column count)
        x_vals=x_d.iloc[:, [count]]
        
        # Get shap values (column count)
        y_vals=pd.DataFrame(shap_values)[count]
        
        # Get shap values for x=0, plot and change color
        shap0=y_vals.loc[x_vals[inputs[count]]==0]
        vio0=ax.violinplot(shap0,positions=[0],widths=0.5,showmeans=False,showextrema=False,showmedians=False)
        for pc in vio0['bodies']:
            pc.set_facecolor('gray')
            
        # Get shap values for x=1, plot and change color
        shap1=y_vals.loc[x_vals[inputs[count]]==1]
        vio1=ax.violinplot(shap1,positions=[1],widths=0.5,showmeans=False,showextrema=False,showmedians=False)
        for pc in vio1['bodies']:
            pc.set_facecolor('gray')
        
        # Add labels
        plt.xlabel(names[inputs[count]],size=20)
        plt.ylabel("SHAP value",size=20)
        plt.yticks(size=20)
        plt.xticks([0,1],["0","1"],size=20)
        plt.xlim(-0.5, 1.5)  
        
        # And ruggplot
        sns.rugplot(x_d[inputs[count]],height=0.05,color='black')
        plt.show()
        
    # If variable is continous, return scatter plot
    else:
        
        # Plot
        fig,ax = plt.subplots(figsize=(12, 8))
        
        # Get x values (column count)
        x_vals=x_d.iloc[:, [count]]
        
        # Get shap values (column count)
        y_vals=pd.DataFrame(shap_values)[count]
        
        # Make scatter plot
        plt.scatter(x_vals,y_vals,color="black",s=60)
        
        # Add labels
        plt.xlabel(names[inputs[count]],size=20)
        plt.ylabel("SHAP value",size=20)
        plt.yticks(size=20)
        plt.xticks(size=20)
        
        # And ruggplot
        sns.rugplot(x_d[inputs[count]],height=0.05,color='black')

        # Save plot for oil
        if count==6: 
            plt.savefig(f"out/intro_shap_depend_{inputs[count]}.png",dpi=300,bbox_inches='tight')
        plt.show()
        
# (2) Partial dependency

# Get pd for each variable 
for count in range(0,len(inputs)):
    
    # If variable is dichotomous
    if len(x_d[inputs[count]].unique())==2:
        
        # Plot
        fig,ax = plt.subplots(figsize=(12, 8))
        
        # Get pd values
        pd_vals=partial_dependence(model,x_d,features=count,kind="both",grid_resolution=10,percentiles=(0,1))
        
        # Get pd values for x=0, plot and change color
        pd0=pd_vals["individual"][0][:, 0]
        vio0=ax.violinplot(pd0,positions=[0],widths=0.5,showmeans=False,showextrema=False,showmedians=False)
        for pc in vio0['bodies']:
            pc.set_facecolor('gray')
        
        # Get pd values for x=1, plot and change color
        pd1=pd_vals["individual"][0][:, 1]
        vio1=ax.violinplot(pd1,positions=[1],widths=0.5,showmeans=False,showextrema=False,showmedians=False)
        for pc in vio1['bodies']:
            pc.set_facecolor('gray')
            
        # Add mean on second axis    
        ax2 = ax.twinx()
        ax2.plot(pd_vals["grid_values"][0],pd_vals["average"][0],marker='o',linestyle='None',color="black",markersize=10)
        
        # Add labels
        ax2.set_xticks([0,1],["0","1"],size=20)
        ax2.set_xlim(-0.5, 1.5)        
        ax2.tick_params(axis='y', which='major', labelsize=20)
        ax.set_xlabel(names[inputs[count]],size=20)
        ax.set_ylabel("Partial dependence",size=20)
        ax2.set_ylabel("Partial dependence (average)",size=20)
    
        # Add ruggplot
        sns.rugplot(x_d[inputs[count]], height=0.05, color='black')
        plt.show()
        
    # If variable is continuous
    else: 
        
        # Plot
        fig,ax = plt.subplots(figsize=(12, 8))
        
        # Get pd values
        pd_vals=partial_dependence(model,x_d,features=count,kind="both",grid_resolution=10,percentiles=(0,1))
        
        # Filter out ice curves which have the highest range in pd values to avoid cluttering
        
        # For each case save the index and the range
        no_flat={"id":[],"range":[]} 
        for i in range(len(pd_vals["individual"][0])): 
            no_flat["id"].append(i)
            no_flat["range"].append(np.ptp(pd_vals["individual"][0][i]))
            
        # Convert dictionary to df and save top 500 cases with highest range
        no_flat=pd.DataFrame(no_flat)
        cases_plt=no_flat['range'].nlargest(500).index
        
        # Plot non flat curves
        for i in cases_plt:
            ax.plot(pd_vals["grid_values"][0],pd_vals["individual"][0][i],color="gray",linewidth=1,alpha=0.2) 
        
        # Add second axis with mean
        ax2 = ax.twinx()
        ax2.plot(pd_vals["grid_values"][0],pd_vals["average"][0],color="black",linewidth=3)
    
        # Add labels
        ax.set_xlabel(names[inputs[count]],size=20)
        ax.set_ylabel("Partial dependence (ICE)",size=20)
        ax2.set_ylabel("Partial dependence (average)",size=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax2.tick_params(axis='y', which='major', labelsize=20)
        
        # Add ruggplot
        sns.rugplot(x_d[inputs[count]],height=0.05,color='black')
        
        # Plot for time since civil war
        if count==2:
            
            # Adjust the limits for the y axis
            ax2.set_ylim(1.11,1.25)
            
            # Save
            plt.savefig(f"out/intro_partial_depend_{inputs[count]}.png",dpi=300,bbox_inches='tight')
        plt.show()
      
# (3) ALE

# Get ale for each variable 
for count in range(0,len(inputs)):
    
    # If variable dichotomous
    if len(x_d[inputs[count]].unique())==2:
        
        # Plot
        fig,ax = plt.subplots(figsize=(12, 8))        

        # Get ale values, grid size 1 for binary predictors
        ale_out=ale(X=x_d,model=model,feature=[inputs[count]],grid_size=1,include_CI=True,plot=False)
        ale_out=ale_out.fillna(0) # First ci is always na --> fill with zero
        
        # Get x values
        x_vals=ale_out.index
        
        # Get ale values
        ale_vals=ale_out["eff"].values
        
        # Plot values
        plt.plot(x_vals,ale_vals,marker='o',linestyle='None',color="black",markersize=10)
        
        # Add confidence intervals
        for i in range(len(ale_out["eff"])):
            plt.plot([ale_out["eff"].index[i],ale_out["eff"].index[i]],[ale_out["lowerCI_95%"].values[i],ale_out["upperCI_95%"].values[i]],color='gray',linewidth=2)
        
        # Add labels
        plt.xlabel(names[inputs[count]],size=20)
        plt.ylabel("Accumulated local effect",size=20)
        plt.yticks(size=20)
        plt.xticks([0,1],size=20)
        plt.xlim(-0.5, 1.5)
        
        # Add ruggplot
        sns.rugplot(x_d[inputs[count]], height=0.05, color='black')
        
        # Save plot for neighborhood
        if count==3: 
            plt.savefig(f"out/intro_ale_{inputs[count]}.png",dpi=300,bbox_inches='tight')
        plt.show()
       
    # If variable is continuous      
    else: 
        
        # Plot
        fig,ax = plt.subplots(figsize=(12, 8))
        
        # Get ale values
        ale_out=ale(X=x_d,model=model,feature=[inputs[count]],grid_size=40,include_CI=True,plot=False)
        ale_out=ale_out.fillna(0) # First ci is always na --> fill with zero
        
        # Get x values
        x_vals=ale_out.index
        
        # Get ale values
        ale_vals=ale_out["eff"].values
        
        # Plot values
        plt.plot(x_vals,ale_vals,marker='o',linestyle='None',color="black",markersize=10)
        
        # Add confidence intervals
        for i in range(len(ale_out["eff"])):
            plt.plot([ale_out["eff"].index[i],ale_out["eff"].index[i]],[ale_out["lowerCI_95%"].values[i],ale_out["upperCI_95%"].values[i]],color='gray',linewidth=2)
        
        # Add labels
        plt.xlabel(names[inputs[count]],size=20)
        plt.ylabel("Accumulated local effect",size=20)
        plt.yticks(size=20)
        plt.xticks(size=20)
        
        # Add rugg plot
        sns.rugplot(x_d[inputs[count]], height=0.05, color='black')
        plt.show()

# Make custom colormap, which has blue for negative values and green for positive 
colors=np.vstack((plt.cm.Greens(np.linspace(1,0,100)),plt.cm.bone(np.linspace(1,0,100))))
cmap=ListedColormap(colors)
cmap

# (4) Interactive ALE

# For each variable, get an interactive ALE plot
for i in inputs:
    
    # Drop var from inputs list
    vars_list=[s for s in inputs if s != i]
    for x in vars_list:
        
        # Plot
        fig,ax = plt.subplots(figsize=(12, 8))     
        
        # Get ale value for a grid
        ale_out=ale(X=x_d,model=model,feature=[i,x],grid_size=40,include_CI=True,plot=False)
        
        # Make colormap center around zero with different ranges in negative and positive space
        map_n=TwoSlopeNorm(vmin=ale_out.min().min(),vcenter=0,vmax=ale_out.max().max())
        
        # Plot heatmap
        h_map = ax.imshow(ale_out,cmap=cmap,norm=map_n,aspect='auto') 
        
        # Thin out ticks, only includ every third tick
        xt = np.arange(ale_out.shape[1])[::3]
        yt = np.arange(ale_out.shape[0])[::3]
        
        # Set ticks
        ax.set_xticks(xt)
        ax.set_yticks(yt)
        
        # And round corresponding tick labels and set
        ax.set_xticklabels([f"{l:.2f}" for l in ale_out.columns[::3]],rotation=0,ha='center',size=15)
        ax.set_yticklabels([f"{l:.2f}" for l in ale_out.index[::3]],rotation=0,ha='right',size=15)

        # Add axis labels
        plt.xlabel(names[x],size=20)
        plt.ylabel(names[i],size=20)   
        
        # Add colorbar
        color_bar = fig.colorbar(h_map,ax=ax,pad=0.01,shrink=0.6,format=FormatStrFormatter('%.3f'))
        color_bar.ax.tick_params(labelsize=15)
        color_bar.set_label('Accumulated local effect', size=20)

        # Save plot for time since civil war
        if i=="d_civil_war_zeros_growth" and x=="male_youth_share":            
            plt.savefig(f"out/intro_ale_{i}_{x}.png",dpi=300,bbox_inches='tight')
            
        # Save plot for oil rents
        if i=="oil_share" and x=="gdp":            
            plt.savefig(f"out/intro_ale_{i}_{x}.png",dpi=300,bbox_inches='tight')
        plt.show()


#################
### OLS model ###
#################

# Get country dummies, drop_first drops first dummy
dummies_c = pd.get_dummies(df['country'],prefix='c',drop_first=True).astype(int)
df = pd.concat([df, dummies_c], axis=1)
country_dummies = list(dummies_c.columns)

# Model I
X=df[['sb_fatalities_lag1_log','d_civil_conflict_zeros_growth','d_civil_war_zeros_growth','d_neighbors_sb_fatalities_lag1']+country_dummies]
X = sm.add_constant(X)
y=df[['sb_fatalities_log']]
est1 = sm.OLS(y, X)
est1 = est1.fit(cov_type='cluster', cov_kwds={'groups': df['country']})
est1.summary()

# Check if full rank because of warning
print(np.linalg.matrix_rank(X)==X.shape[1])

# Model II
X=df[['sb_fatalities_lag1_log','d_civil_conflict_zeros_growth','d_civil_war_zeros_growth','d_neighbors_sb_fatalities_lag1','inf_mort','gdp','growth','oil_share']+country_dummies]
X = sm.add_constant(X)
y=df[['sb_fatalities_log']]
est2 = sm.OLS(y, X)
est2 = est2.fit(cov_type='cluster', cov_kwds={'groups': df['country']})
est2.summary()

# Check if full rank because of warning
print(np.linalg.matrix_rank(X)==X.shape[1])

# Model III
X=df[['sb_fatalities_lag1_log','d_civil_conflict_zeros_growth','d_civil_war_zeros_growth','d_neighbors_sb_fatalities_lag1','inf_mort','gdp','growth','oil_share','pop','male_youth_share','ethnic_frac','eys_male']+country_dummies]
X = sm.add_constant(X)
y=df[['sb_fatalities_log']]
est3 = sm.OLS(y, X)
est3 = est3.fit(cov_type='cluster', cov_kwds={'groups': df['country']})
est3.summary()

# Check if full rank because of warning
print(np.linalg.matrix_rank(X)==X.shape[1])

# Model IV
X=df[['sb_fatalities_lag1_log','d_civil_conflict_zeros_growth','d_civil_war_zeros_growth','d_neighbors_sb_fatalities_lag1','inf_mort','gdp','growth','oil_share','pop','male_youth_share','ethnic_frac','eys_male','temp','withdrawl']+country_dummies]
X = sm.add_constant(X)
y=df[['sb_fatalities_log']]
est4 = sm.OLS(y, X)
est4 = est4.fit(cov_type='cluster', cov_kwds={'groups': df['country']})
est4.summary()

# Check if full rank because of warning
print(np.linalg.matrix_rank(X)==X.shape[1])

# Model V
X=df[['sb_fatalities_lag1_log','d_civil_conflict_zeros_growth','d_civil_war_zeros_growth','d_neighbors_sb_fatalities_lag1','inf_mort','gdp','growth','oil_share','pop','male_youth_share','ethnic_frac','eys_male','temp','withdrawl','polyarchy','libdem','egaldem','civlib','exlsocgr']+country_dummies]
X = sm.add_constant(X)
y=df[['sb_fatalities_log']]
est5 = sm.OLS(y, X)
est5 = est5.fit(cov_type='cluster', cov_kwds={'groups': df['country']})
est5.summary()

# Check if full rank because of warning
print(np.linalg.matrix_rank(X)==X.shape[1])
 
# Output table
summary = summary_col([est1,est2,est3,est4,est5],float_format='%0.3f',stars=[0.1, 0.05, 0.01],regressor_order=['sb_fatalities_lag1_log','d_civil_conflict_zeros_growth','d_civil_war_zeros_growth','d_neighbors_sb_fatalities_lag1','inf_mort','gdp','growth','oil_share','pop','male_youth_share','ethnic_frac','eys_male','temp','withdrawl','polyarchy','libdem','egaldem','civlib','exlsocgr']+country_dummies)
print(summary.as_latex())

# Save table
out = summary.as_latex()
with open("out/reg_table.tex", "w") as f:
    f.write(out)




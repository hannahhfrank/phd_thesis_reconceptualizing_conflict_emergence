import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import os 
os.environ['PATH'] = "/Library/TeX/texbin:" + os.environ.get('PATH', '')
import random
random.seed(42)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}\usepackage[T1]{fontenc}'
from alibi.explainers import KernelShap
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, ListedColormap

grid = {"n_estimators": [10, 231, 452, 673, 894, 1115, 1336, 1557, 1778, 2000],              
        "max_features": ["sqrt", "log2", None],
        "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]}

# Load data
df=pd.read_csv("data/df_interpret.csv",index_col=0)

# Transforms
df["d_neighbors_sb_fatalities_lag1_log"]=np.log(df["d_neighbors_sb_fatalities_lag1"]+1)
df["gdp"]=np.log(df["gdp"]+1)
df["pop"]=np.log(df["pop"]+1)

# Rename columns
df.columns=['year', 
            'gw_codes', 
            'country', 
            'sb_fatalities',
            'onset',
            'onset2', 
            'd_neighbors_sb_fatalities_lag1', 
            'GDP per capita', 
            'GDP growth', 
            'Population size', 
            'Oil rents', 
            'Male education',
            'Regulatory quality', 
            'Temperature', 
            'Ethnic exclusion', 
            'Liberal democracy', 
            'Fatalities in neighborhood']

# Specify model
target='onset2'
inputs=['Fatalities in neighborhood','GDP per capita','GDP growth','Population size','Oil rents','Temperature','Ethnic exclusion','Male education','Liberal democracy',"Regulatory quality",]
y=df[["year",'country','onset2']]
x=df[["year",'country','Fatalities in neighborhood','GDP per capita','GDP growth','Population size','Oil rents','Temperature','Ethnic exclusion','Male education','Liberal democracy',"Regulatory quality",]]

# Split data into training and validation data. No test data is needed for this application.
splits=[]
  
for c in y.country.unique():
    y_s = y.loc[y["country"] == c]
    x_s = x.loc[x["country"] == c]
    y_training = y_s[["country","year"]+[target]][:int(0.7*len(y_s))]
    y_testing = y_s[["country","year"]+[target]][int(0.7*len(y_s)):]
    val_training_ids = list(y_training.index)
    val_testing_ids = list(y_testing.index)    
    splits += [-1] * len(val_training_ids) + [0] * len(val_testing_ids)
    
# Decision tree example
model = DecisionTreeClassifier(random_state=42,max_depth=3)
model.fit(df[['GDP per capita','Fatalities in neighborhood','GDP growth','Population size','Oil rents','Ethnic exclusion',"Regulatory quality",]],df[[target]].values.ravel())    

# Plot tree
fig, ax = plt.subplots(figsize=(23, 10))
plot_tree(model,feature_names=df[['GDP per capita','Fatalities in neighborhood','GDP growth','Population size','Oil rents','Ethnic exclusion',"Regulatory quality",]].columns,filled=False,fontsize=25)

# In the second node, the values are shown as 5488.0 and 55.0 --> Remove .0 manually

# Obtain components of plot
sections=ax.get_children()

# Loop through components and replace text manually
for sec in sections[:13]: 
    sec.set_text(sec.get_text().replace('5488.0','5488'))
    sec.set_text(sec.get_text().replace('55.0','55'))
        
# Save        
plt.savefig("out/decion_tree.eps",dpi=300,bbox_inches='tight')
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/interpretability/out/decion_tree.eps",dpi=300,bbox_inches='tight')        
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/decion_tree.eps",dpi=300,bbox_inches='tight')
plt.show()

# Train model  
splits = PredefinedSplit(test_fold=splits)
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=1),param_grid=grid,cv=splits,verbose=0,n_jobs=-1)
grid_search.fit(df[inputs], df[[target]].values.ravel())
best_params = grid_search.best_params_
model = RandomForestClassifier(random_state=1,**best_params)
model.fit(df[inputs], df[[target]].values.ravel())

###################
### SHAP values ###
###################

# Remove country, year variables
x_d=df[inputs]
y_d=df[[target]]

# Subset onset cases
onsets=df.loc[list(y_d.loc[y_d["onset2"]==1].index)]
onsets=onsets.reset_index(drop=True)

# Add index, starting at one and save as table for Appendix
onsets.index = range(1, len(onsets) + 1)
onsets[["country","year"]].to_latex("out/onsets.tex")
 
# Fit Shap value explainer on data
# https://docs.seldon.io/projects/alibi/en/latest/methods/KernelSHAP.html                           
preds_f = lambda x_d: model.predict_proba(x_d)[:, 1]
ex_shap = KernelShap(preds_f,feature_names=list(x_d.columns),link='logit')
ex_shap.fit(x_d.values)

# Subset onset cases
onset_cases = x_d.loc[y_d.loc[y_d["onset2"]==1].index].values

# Get Shap values for onset cases
exp_values = ex_shap.explain(onset_cases)  
shap_vals = exp_values.shap_values

# Convert shap values to df (observations in columns ---> transpose) and save 
shaps = pd.DataFrame(shap_vals[0].T,index=x_d.columns,columns=onsets.index)
shaps.to_csv("out/df_shaps.csv")

##################
### Clustering ###
##################

# Clustering fit using kMeans, select k=8
inertia=[]
for k in [3,4,5,6,7,8,9,10]:
    kmeans = KMeans(n_clusters=k,random_state=1).fit(shaps.T) # Transpose to have cases in rows
    inertia.append(kmeans.inertia_)

# Plot inertia for each k
plt.figure(figsize=(12,8))
plt.plot([3,4,5,6,7,8,9,10],inertia,c="black")
plt.ylabel("Inertia", size=25)
plt.xlabel("Number of clusters $k$", size=25)
plt.yticks(size=25)
plt.xticks(size=25)

# Save 
plt.savefig("out/k_select.eps",dpi=300,bbox_inches='tight') 
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/interpretability/out/k_select.eps",dpi=300,bbox_inches='tight')        
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/k_select.eps",dpi=300,bbox_inches='tight')
plt.show()

# Make custom colormap, which has blue for negative values and green for positive 
colors=np.vstack((plt.cm.Greens(np.linspace(1,0,50)),plt.cm.bone(np.linspace(1,0,50))))
cmap=ListedColormap(colors)
cmap

# Cluster shap values, hierachical clustering using ward's method and euclidean distance over columns (observations in columns)
# (only to get clustering solution, pretty version of plot is produced below)
cluster_map=sns.clustermap(shaps,method="ward",metric="euclidean",col_cluster=True,row_cluster=False)

# Get the order of onset cases based on the clustering
cols_order=cluster_map.dendrogram_col.reordered_ind

# Add one to each case, because my onsets start from 1 and not 0.
for i in range(len(cols_order)):
    cols_order[i] += 1

# Apply hierachical clustering using ward's method and euclidean distance 
mat=linkage(shaps.T,method="ward",metric="euclidean") # Transpose to have cases in rows
# Get cluster labels for k=8
cluster_labs=fcluster(mat,8,criterion="maxclust")

# Plot clustermap in pretty
maps=sns.clustermap(shaps,cmap=cmap,linewidths=0,center=0,method="ward",metric="euclidean",col_cluster=True,row_cluster=False,yticklabels=12,figsize=(20,5),cbar_pos=(0.99,0.1,0.013,0.65))

# Move variable labels to left side
maps.ax_heatmap.yaxis.set_ticks_position('left')
maps.ax_heatmap.yaxis.set_label_position('left')
maps.ax_heatmap.tick_params(axis='y',labelleft=True)

# Add variable names
maps.ax_heatmap.set_yticks(ticks=np.arange(len(shaps))+0.5,labels=inputs,size=25,rotation=0,ha="right")

# Add Onset numbers
maps.ax_heatmap.set_xticks(ticks=np.arange(len(cols_order))+0.5,labels=cols_order,size=15)

# Set ticks on colorbar
col_bar = maps.ax_heatmap.collections[0].colorbar
col_bar.set_ticks([0,0.5,1,1.5,2,2.5,3,3.5])
col_bar.set_ticklabels(["0","0.5","1","1.5","2","2.5","3","3.5"])
col_bar.ax.tick_params(labelsize=15)   

# Add horizontal lines to denote clusters: (1) First get position for breaks

# Sort cluster labels
cluster_labs_sorted=np.sort(cluster_labs)

# And obtain differenced series to denote position where cluster increases by one
cluster_breaks=np.diff(cluster_labs_sorted)

# Get positions and add one, so that breaks occur after last case in cluster
breaks=np.where(cluster_breaks!=0)[0]+1

# (2) Add lines to plot by looping over breaks, and add a horizontal line
for i in breaks:
    maps.ax_heatmap.vlines(x=i,ymin=0,ymax=10,colors="black",linewidth=1)
    
# Save
plt.savefig("out/inter_shaps.eps",dpi=300,bbox_inches='tight') 
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/interpretability/out/inter_shaps.eps",dpi=300,bbox_inches='tight')        
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/inter_shaps.eps",dpi=300,bbox_inches='tight')
plt.show()

#############################
### Independent variables ###
#############################

# Get matrix for independent variables (transpose to have cases in columns)
matrix=x_d.loc[list(y_d.loc[y_d["onset2"]==1].index)].T
matrix.columns=onsets.index

# Order cases according to clustering
matrix=matrix[cols_order]

# Save table for each cluster for Appendix
print(breaks)
# Define ranges of clusters
for a, b in zip([0,3,16,27,31,39,42,46],[3,16,27,31,39,42,46,58]):    
    # Get observations for each cluster
    sub_df=matrix.iloc[:, a:b]
    # Add sample means for reference
    sub_df["Means"]=x_d.mean()
    # Save as tex file
    sub_df.to_latex(f"out/vars_{a}.tex",float_format="%.2f")

# To plot the independent variables a figure grid is produced, 
# which is then filled in a loop, one row with each variable. 
# This allows to customize the appearance, for example, the color
# map and the min and max values. 

# Make custom colormap, which has blue for negative values and green for positive 
colors=np.vstack((plt.cm.Greens(np.linspace(1,0,100)),plt.cm.bone(np.linspace(1,0,100))))
cmap=ListedColormap(colors)
cmap

# Initiate plot 
fig = plt.figure(figsize=(20,5))

# Create plot grid with two columns, one for variables and one for colorbar
grid = gridspec.GridSpec(10,2,width_ratios=[23,0.3],height_ratios=[10,10,10,10,10,10,10,10,10,10],hspace=0.1,wspace=0.01)

# Matrix for independent variables 
matrix_vals=matrix.values

# Prepare values for each variable (colorbar, min and max values, and ticks for colorbar)
colormaps = ['bone_r','bone_r',"seismic",'bone_r','bone_r',"seismic",'bone_r','bone_r', 'bone_r',"seismic"]
vars_min = matrix_vals.min(axis=1)
vars_max = matrix_vals.max(axis=1)
lst_ticks = [[0,5,10],[4,6.5,9],[-50,0,45],[15,17.5,20],[0,25,50],[-3,0,20],[0,0.4,0.8],[10,50,90],[0.1,0.3,0.5],[-2,0,1]]

# For each variable
for i in range(10):
    
    # Access plot grid for variable
    ax_vars = fig.add_subplot(grid[i, 0])
    
    # Access plot grid for colorbar   
    ax_col_bar = fig.add_subplot(grid[i, 1]) 
    
    # Get variable data 
    df_var = matrix_vals[i, :].reshape(1, -1)
    
    # Plot with normal colormap for variables that only take positive values
    if colormaps[i]=="bone_r":
        plot = ax_vars.imshow(df_var,aspect='auto',cmap=colormaps[i],vmin=vars_min[i],vmax=vars_max[i])
   
    # Plot with centered colormap for variables which take both positive and negative values
    elif colormaps[i]=="seismic":
        # Center colormap around zero using the costume colormap produced above
        cmap_norm = TwoSlopeNorm(vmin=vars_min[i],vcenter=0,vmax=vars_max[i])
        plot = ax_vars.imshow(df_var,aspect='auto',cmap=cmap,norm=cmap_norm)
        
    # Add lines between clusters, adjust with 0.5 to position correctly  
    for x in breaks:
        ax_vars.vlines(x=x-0.5,ymin=-0.5,ymax=0.5,color='black',linewidth=2)
    
    # Remove ticks 
    ax_vars.set_yticks([])
    ax_vars.set_xticks([])
    
    # Set variable name
    ax_vars.set_ylabel(inputs[i],rotation=0,ha='right',fontsize=25,va="center")
   
    # Add colorbar
    color_bar = plt.colorbar(plot,cax=ax_col_bar,ticks=lst_ticks[i])
    color_bar.ax.tick_params(labelsize=9)
    
    # Set case numbers after last row
    if i==9:
        ax_vars.set_xticks(np.arange(0,58,1),matrix.columns,size=15)
        
# Save
plt.tight_layout()
plt.savefig("out/inter_vars.eps",dpi=300,bbox_inches='tight')
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/interpretability/out/inter_vars.eps",dpi=300,bbox_inches='tight')        
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/inter_vars.eps",dpi=300,bbox_inches='tight')
plt.show()






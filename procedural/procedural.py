import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import PredefinedSplit,GridSearchCV
from functions import evals
import matplotlib.pyplot as plt
from functions import lag_groupped
import random
from joblib import parallel_backend
random.seed(42)
import matplotlib as mpl
import os
os.environ['PATH'] = "/Library/TeX/texbin:" + os.environ.get('PATH', '')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}\usepackage[T1]{fontenc}'

# Load data
df=pd.read_csv("data/data_out/ucdp_cm_sb.csv",index_col=0)

# Merge ensemble predictions with df
ensemble_ens=pd.read_csv("out/ensemble_ens_df_cm_live.csv",index_col=0)
df=pd.merge(ensemble_ens, df[["country","dd","best"]],on=["country","dd"],how="left")

# Reobtain year
df['year'] = df['dd'].str[:4]

# Rename columns and sort columns
df.columns=["country","dd","preds_proba","sb_fatalities","year"]
df=df[["country","year","dd","preds_proba","sb_fatalities"]]

#######################################################
### Plot predicted probability of collective action ###
#######################################################

# Make a random selection of 10 countries to save for paper
selects=random.choices(df.country.unique(), k=10)

# Make plot for every country
for c in df.country.unique():
        
    # Get unique years
    years=df["year"].loc[df["country"]==c].unique()
    
    # Specify plot grid to loop over
    fig, axes = plt.subplots(7, 5, figsize=(8, 6))
    plt.subplots_adjust(wspace=0, hspace=0.5)
    axes = axes.flatten()
    
    # Plot each year in one subplot
    for i,y in zip(range(35),years):
        # Subset data and plot
        df_s=df.loc[(df["country"]==c)&(df["year"]==y)]
        
        # Plot, use no clip to avoid that lines are cut off
        axes[i].plot(df_s.dd,df_s.preds_proba,color="black",linewidth=2,clip_on=False)
        
        # Remove ticks and frames
        axes[i].set_xticks([],[])
        axes[i].set_yticks([])
        axes[i].set_yticks([])
        axes[i].spines['bottom'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['left'].set_visible(False)
        axes[i].spines['top'].set_visible(False)

        # If preds proba is zero, set ylim to global max
        if df_s.preds_proba.max()==0:
            axes[i].set_ylim(0,df.loc[(df["country"]==c)].preds_proba.max())
        # If preds proba is not zero, set ylim to within year max to ensure better visibility    
        else: 
            axes[i].set_ylim(0,df_s.preds_proba.max())
            
    # Remove subplots which are not filled in the last row
    for i in range(len(years), len(axes)):
        fig.delaxes(axes[i])
    
    # If country in selects or if Germany, save
    if c=="Armenia" or c=="Germany" or c=="Benin" or c=="Iran":
        plt.savefig(f"out/proc_latent_{c}.eps",dpi=300,bbox_inches='tight')    
    plt.show()   

#####################################    
### Predictions for Onset Catcher ### 
#####################################    

# Optimization 
#grid = {'n_estimators': [10, 231, 452, 673, 894, 1115, 1336, 1557, 1778, 2000]}

# No optimization
grid=None

# Define out df to get scores for different hyperparameters
vals=pd.DataFrame()

# Check distribution of preds_proba
df["preds_proba"].hist()

# For different number of temporal lags
test_score=0
for ar in [2,3,4,5,6]:
    
    # Specify list for inputs
    lags=[]

    # Create temporal lags
    for i in range(1,ar):
        df[f"preds_proba_lag{i}"]=lag_groupped(df,"country","preds_proba",i)
        # And append to lags list used to specify model below
        lags.append(f"preds_proba_lag{i}")
       
    # For different cut values
    for cut in [0.15,0.2,0.25,0.3,0.35,0.4]:
        
        # Split observations based on preceding risk of collective action (t-1)
        # Only df_nonzero are considered in the next step.
        df_zero=df.loc[df["preds_proba_lag1"]<=cut]
        df_nonzero=df.loc[df["preds_proba_lag1"]>cut]
        
        print(f"{cut}, with {len(df_nonzero)/len(df_zero)*100} observations passed on")
        
        ##################
        ### Validation ###
        ##################
        
        # Hyperparameters ar and cut are optimized in the validation data. 
        # The model is trained, using all data until 2019, and out-of-sample
        # predictions are obtained for the validation data (2020-2021).

        # Data split
        training_y = pd.DataFrame()
        testing_y = pd.DataFrame()
        training_x = pd.DataFrame()
        testing_x = pd.DataFrame()  
        splits=[]
        
        for c in df.country.unique():
            df_nonzero_s = df_nonzero.loc[df_nonzero["country"] == c]
            y_training = df_nonzero_s[["country","dd","sb_fatalities"]].loc[df_nonzero_s["dd"]<="2019-12"]
            x_training = df_nonzero_s[["country","dd"]+lags].loc[df_nonzero_s["dd"]<="2019-12"]
            y_testing = df_nonzero_s[["country","dd","sb_fatalities"]].loc[(df_nonzero_s["dd"]>="2020-01")&(df_nonzero_s["dd"]<="2021-12")]
            x_testing = df_nonzero_s[["country","dd"]+lags].loc[(df_nonzero_s["dd"]>="2020-01")&(df_nonzero_s["dd"]<="2021-12")]
            training_y = pd.concat([training_y, y_training])
            testing_y = pd.concat([testing_y, y_testing])
            training_x = pd.concat([training_x, x_training])
            testing_x = pd.concat([testing_x, x_testing])        
            val_training_ids = list(y_training[:int(0.8*len(y_training))].index)
            val_testing_ids = list(y_training[int(0.8*len(y_training)):].index)   
            splits += [-1] * len(val_training_ids) + [0] * len(val_testing_ids)
                   
        training_y_d=training_y.drop(columns=["country","dd"])
        training_x_d=training_x.drop(columns=["country","dd"])
        testing_y_d=testing_y.drop(columns=["country","dd"])
        testing_x_d=testing_x.drop(columns=["country","dd"])
       
        # If optimization
        if grid is not None:
            splits = PredefinedSplit(test_fold=splits)
            with parallel_backend('threading'):
                grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=1),param_grid=grid,cv=splits,verbose=0,n_jobs=-1)                    
                grid_search.fit(training_x_d, training_y_d.values.ravel())
            best_params = grid_search.best_params_
            model=RandomForestRegressor(random_state=1,**best_params)
            model.fit(training_x_d, training_y_d.values.ravel())
                       
            # Predictions (out-of-sample)
            pred = pd.DataFrame(model.predict(testing_x_d))
            pred["country"]=testing_x.country.values
            pred["dd"]=testing_x.dd.values
            
            # Get base df and merge predictions
            base=df[["country","dd","sb_fatalities"]].loc[(df["dd"]>="2020-01")&(df["dd"]<="2021-12")]
            val=pd.merge(base,pred,on=["country","dd"],how="left")
            val.columns=["country","dd","sb_fatalities","preds"]
            
            # Missing values are filled with zero, those are cases which do
            # not pass to the second stage
            val=val.fillna(0)
            
            # Sort df and reset index, which is needed for evaluation
            val=val.sort_values(by=["country","dd"])
            val=val.reset_index(drop=True)
            
            # Get evaluations
            evals_out = evals(val,"sb_fatalities","preds","country")
            print(evals_out['Onset Score'],evals_out['Normalized MSE'])
            
        # If no optimization
        else:
            model=RandomForestRegressor(random_state=1)
            model.fit(training_x_d, training_y_d.values.ravel())
            
            # Predictions (out-of-sample)
            pred = pd.DataFrame(model.predict(testing_x_d))
            pred["country"]=testing_x.country.values
            pred["dd"]=testing_x.dd.values
            
            # Get base df and merge predictions
            base=df[["country","dd","sb_fatalities"]].loc[(df["dd"]>="2020-01")&(df["dd"]<="2021-12")]
            val=pd.merge(base,pred,on=["country","dd"],how="left")
            val.columns=["country","dd","sb_fatalities","preds"]
            
            # Missing values are filled with zero, those are cases which do
            # not pass to the second stage
            val=val.fillna(0)
            
            # Sort df and reset index, which is needed for evaluation
            val=val.sort_values(by=["country","dd"])
            val=val.reset_index(drop=True)
            
            # Get evaluations
            evals_out = evals(val,"sb_fatalities","preds","country")
            print(evals_out['Onset Score'],evals_out['Normalized MSE'])
            
        # Save onset score for different hyperparameter combinations for Appendix
        vals_s={"ar":ar,"thres":cut,"onset":evals_out['Onset Score'],"mse":evals_out['Normalized MSE']}
        vals_d=pd.DataFrame(vals_s,index=[0])
        vals=pd.concat([vals,vals_d],ignore_index=True)
        vals=vals.reset_index(drop=True)

        # Live predictions are only returned if onset score is higher than test
        if evals_out['Onset Score']>test_score:
            test_score=evals_out['Onset Score']
            print(f"Best: ar={ar}, cut={cut}")
            
            #####################################
            ### Get live predictions for 2022 ###
            #####################################
            
            # Train model until 2020, use 2021 as input to make predictions for 2022. 
            
            training_y = pd.DataFrame()
            training_x = pd.DataFrame()
            splits=[]
                
            for c in df.country.unique():
                df_nonzero_s = df_nonzero.loc[df_nonzero["country"] == c]
                y_training = df_nonzero_s[["country","dd","sb_fatalities"]].loc[df_nonzero_s["dd"]<="2020-12"]
                x_training = df_nonzero_s[["country","dd"]+lags].loc[df_nonzero_s["dd"]<="2020-12"]
                training_y = pd.concat([training_y, y_training])
                training_x = pd.concat([training_x, x_training])                
                val_training_ids = list(y_training[:int(0.8*len(y_training))].index)
                val_testing_ids = list(y_training[int(0.8*len(y_training)):].index)           
                splits += [-1] * len(val_training_ids) + [0] * len(val_testing_ids)
                
            training_y_d=training_y.drop(columns=["country","dd"])
            training_x_d=training_x.drop(columns=["country","dd"])
            
            # If optimization
            if grid is not None:
                splits = PredefinedSplit(test_fold=splits)
                with parallel_backend('threading'):
                    grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=1),param_grid=grid,cv=splits,verbose=0,n_jobs=-1)                    
                    grid_search.fit(training_x_d, training_y_d.values.ravel())
                best_params = grid_search.best_params_
                model=RandomForestRegressor(random_state=1,**best_params)
                model.fit(training_x_d, training_y_d.values.ravel())
            
                # Use last 12 months as input when making predictions
                testing_x_s=df_nonzero[["country","dd"]+lags].loc[(df_nonzero["dd"]>="2021-01")&(df_nonzero["dd"]<="2021-12")]
                testing_x_d=testing_x_s.drop(columns=["country","dd"])
            
                # Predictions
                pred2022 = pd.DataFrame(model.predict(testing_x_d))
                
                # Create df, replace year 2021 with year 2022
                pred2022["country"]=testing_x_s.country.values
                pred2022["dd"]=testing_x_s.dd.values
                pred2022['dd'] = pred2022['dd'].str.replace('2021', '2022')
                
                # Get base df and merge predictions
                base=df[["country","dd","sb_fatalities"]].loc[(df["dd"]>="2022-01")&(df["dd"]<="2022-12")]
                catcher2022=pd.merge(base,pred2022,on=["country","dd"],how="left")
                catcher2022.columns=["country","dd","sb_fatalities","preds"]
 
                # Missing values are filled with zero, those are cases which do
                # not pass to the second stage
                catcher2022=catcher2022.fillna(0)
                catcher2022=catcher2022.sort_values(by=["country","dd"])
                
            # If no optimization
            else:
                model=RandomForestRegressor(random_state=1)
                model.fit(training_x_d, training_y_d.values.ravel())
            
                # Use last 12 months as input when making predictions
                testing_x_s=df_nonzero[["country","dd"]+lags].loc[(df_nonzero["dd"]>="2021-01")&(df_nonzero["dd"]<="2021-12")]
                testing_x_d=testing_x_s.drop(columns=["country","dd"])
            
                # Predictions
                pred2022 = pd.DataFrame(model.predict(testing_x_d))
                
                # Create df, replace year 2021 with year 2022
                pred2022["country"]=testing_x_s.country.values
                pred2022["dd"]=testing_x_s.dd.values
                pred2022['dd'] = pred2022['dd'].str.replace('2021', '2022')
                
                # Get base df and merge predictions
                base=df[["country","dd","sb_fatalities"]].loc[(df["dd"]>="2022-01")&(df["dd"]<="2022-12")]
                catcher2022=pd.merge(base,pred2022,on=["country","dd"],how="left")
                catcher2022.columns=["country","dd","sb_fatalities","preds"]
 
                # Missing values are filled with zero, those are cases which do
                # not pass to the second stage
                catcher2022=catcher2022.fillna(0)
                catcher2022=catcher2022.sort_values(by=["country","dd"])
            
            #####################################
            ### Get live predictions for 2023 ###
            #####################################
            
            # Train model until 2021, use 2022 as input to make predictions for 2023. 
             
            training_y = pd.DataFrame()
            training_x = pd.DataFrame()
            splits=[]
             
            for c in df.country.unique():
                df_nonzero_s = df_nonzero.loc[df_nonzero["country"] == c]
                y_training = df_nonzero_s[["country","dd","sb_fatalities"]].loc[df_nonzero_s["dd"]<="2021-12"]
                x_training = df_nonzero_s[["country","dd"]+lags].loc[df_nonzero_s["dd"]<="2021-12"]
                training_y = pd.concat([training_y, y_training])
                training_x = pd.concat([training_x, x_training])
                val_training_ids = list(y_training[:int(0.8*len(y_training))].index)
                val_testing_ids = list(y_training[int(0.8*len(y_training)):].index)       
                splits += [-1] * len(val_training_ids) + [0] * len(val_testing_ids)
                
            training_y_d=training_y.drop(columns=["country","dd"])
            training_x_d=training_x.drop(columns=["country","dd"])
                       
            # If optimization
            if grid is not None:
                splits = PredefinedSplit(test_fold=splits)
                with parallel_backend('threading'):
                    grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=1), param_grid=grid, cv=splits, verbose=0, n_jobs=-1)
                    
                    grid_search.fit(training_x_d, training_y_d.values.ravel())
                best_params = grid_search.best_params_
                model=RandomForestRegressor(random_state=1,**best_params)
                model.fit(training_x_d, training_y_d.values.ravel())
            
                # Use last 12 months as input when making predictions
                testing_x_s=df_nonzero[["country","dd"]+lags].loc[(df_nonzero["dd"]>="2022-01")&(df_nonzero["dd"]<="2022-12")]
                testing_x_d=testing_x_s.drop(columns=["country","dd"])
            
                # Predictions
                pred2023 = pd.DataFrame(model.predict(testing_x_d))
                
                # Create df, replace year 2022 with year 2023
                pred2023["country"]=testing_x_s.country.values
                pred2023["dd"]=testing_x_s.dd.values
                pred2023['dd'] = pred2023['dd'].str.replace('2022', '2023')
                
                # Get base df and merge predictions
                base=df[["country","dd","sb_fatalities"]].loc[(df["dd"]>="2023-01")&(df["dd"]<="2023-12")]
                catcher2023=pd.merge(base,pred2023,on=["country","dd"],how="left")
                catcher2023.columns=["country","dd","sb_fatalities","preds"]
 
                # Missing values are filled with zero, those are cases which do
                # not pass to the second stage
                catcher2023=catcher2023.fillna(0)
                catcher2023=catcher2023.sort_values(by=["country","dd"])
                
            # If no optimization
            else:
                model=RandomForestRegressor(random_state=1)
                model.fit(training_x_d, training_y_d.values.ravel())
            
                # Use last 12 months as input when making predictions
                testing_x_s=df_nonzero[["country","dd"]+lags].loc[(df_nonzero["dd"]>="2022-01")&(df_nonzero["dd"]<="2022-12")]
                testing_x_d=testing_x_s.drop(columns=["country","dd"])
            
                # Predictions
                pred2023 = pd.DataFrame(model.predict(testing_x_d))
                
                # Create df, replace year 2022 with year 2023
                pred2023["country"]=testing_x_s.country.values
                pred2023["dd"]=testing_x_s.dd.values
                pred2023['dd'] = pred2023['dd'].str.replace('2022', '2023')
                
                # Get base df and merge predictions
                base=df[["country","dd","sb_fatalities"]].loc[(df["dd"]>="2023-01")&(df["dd"]<="2023-12")]
                catcher2023=pd.merge(base,pred2023,on=["country","dd"],how="left")
                catcher2023.columns=["country","dd","sb_fatalities","preds"]
 
                # Missing values are filled with zero, those are cases which do
                # not pass to the second stage
                catcher2023=catcher2023.fillna(0)
                catcher2023=catcher2023.sort_values(by=["country","dd"])
               
            # Merge years 2022 and 2023
            catcher = pd.concat([catcher2022, catcher2023], axis=0, ignore_index=True)
            
            # Sort and reset (needed for evaluation)
            catcher=catcher.sort_values(by=["country","dd"])
            catcher=catcher.reset_index(drop=True)
            
            # Save
            catcher.to_csv("out/catcher.csv") 

    
# Save onset score for different hyperparameter combinations for Appendix
vals=vals.sort_values(by=["onset"],ascending=False)    
with open('out/eval_socres.tex', 'w') as tab:
     tab.write(vals.to_latex(index=False))   
    
    

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
from sklearn.impute import SimpleImputer
from joblib import parallel_backend
from sklearn.metrics import mean_squared_log_error,mean_squared_error
from scipy.stats import trim_mean

def lag_groupped(df, group_var, var, lag):
    return df.groupby(group_var)[var].shift(lag).fillna(0)

def multivariate_imp_bayes(df, country, vars_input, vars_add=None,max_iter=10,min_val=0,last_train=2019):
    
    # Split data 
    train = pd.DataFrame()
    test = pd.DataFrame()
    for c in df.country.unique():
        df_s = df.loc[df["country"] == c]
        train_s = df_s.loc[df_s["year"]<=last_train]
        test_s = df_s.loc[df_s["year"]>last_train]
        train = pd.concat([train, train_s])
        test = pd.concat([test, test_s])
    
    # (1) Training
    
    # Create data matrix
    df_imp = train[vars_input]
    df_imp = df_imp.copy(deep=True)
    
    # Get temporal lags
    df_imp['lag1'] = lag_groupped(train, country, vars_input, 1)
    df_imp['lag2'] = lag_groupped(train, country, vars_input, 2)
    df_imp['lag3'] = lag_groupped(train, country, vars_input, 3)
    df_imp['lag4'] = lag_groupped(train, country, vars_input, 4)
    df_imp['lag5'] = lag_groupped(train, country, vars_input, 5)

    # Add vars_add and fill missing values in vars_add with zero
    df_filled = train.fillna(0)
    df_imp = pd.concat([df_imp, df_filled[vars_add]], axis=1)

    # Make a base df 
    feat_complete = train.drop(columns=vars_add)
    feat_complete=feat_complete.reset_index(drop=True)
    
    # Fit imputer in training data
    imputer = IterativeImputer(estimator=linear_model.Ridge(), random_state=1, max_iter=max_iter, min_value=min_val)
    imputer.fit(df_imp)
    
    # Impute missing values and merge imputed column with base df
    df_imp_trans = imputer.transform(df_imp)
    df_imp_trans_df = pd.DataFrame(df_imp_trans)
    # Obtain first row in df, which is the imputed variable        
    df_imp_trans_df = df_imp_trans_df.iloc[:, :len(vars_input)]
    df_imp_trans_df = df_imp_trans_df.rename(columns={0: 'imp'})
    train_imp = pd.concat([feat_complete, df_imp_trans_df],axis=1)
    
    # Add variable, indicating if observation was imputed
    train_imp['missing_id'] = train_imp[vars_input].isnull().astype(int)
    
    # (2) Test data

    # Create data matrix
    df_imp = test[vars_input]
    df_imp = df_imp.copy(deep=True)
    
    # Get temporal lags
    df_imp['lag1'] = lag_groupped(test, country, vars_input, 1)
    df_imp['lag2'] = lag_groupped(test, country, vars_input, 2)
    df_imp['lag3'] = lag_groupped(test, country, vars_input, 3)
    df_imp['lag4'] = lag_groupped(test, country, vars_input, 4)
    df_imp['lag5'] = lag_groupped(test, country, vars_input, 5)

    # Add vars_add and fill missing values in vars_add with zero   
    df_filled = test.fillna(0)
    df_imp = pd.concat([df_imp, df_filled[vars_add]], axis=1)
    
    # Make a base df 
    feat_complete = test.drop(columns=vars_add)
    feat_complete=feat_complete.reset_index(drop=True)
 
    # Use trained imputer model to fill in missings in the test data
    # merge imputed column with base df
    df_imp_trans = imputer.transform(df_imp)
    df_imp_trans_df = pd.DataFrame(df_imp_trans)
    # Obtain first row in df, which is the imputed variable        
    df_imp_trans_df = df_imp_trans_df.iloc[:, :len(vars_input)]
    df_imp_trans_df = df_imp_trans_df.rename(columns={0: 'imp'})
    test_imp = pd.concat([feat_complete, df_imp_trans_df], axis=1)
    
    # Add variable, indicating if observation was imputed   
    test_imp['missing_id'] = test_imp[vars_input].isnull().astype(int)
    
    # Merge train and test
    out = pd.concat([train_imp, test_imp])
    out=out.sort_values(by=["gw_codes","year"])
    out=out.reset_index(drop=True)

    return out

def simple_imp_grouped(df, group, vars_input,last_train=2019):
    
    # Split data 
    train = pd.DataFrame()
    test = pd.DataFrame()
    for c in df.country.unique():
        df_s = df.loc[df["country"] == c]
        train_s = df_s.loc[df_s["year"]<=last_train]
        test_s = df_s.loc[df_s["year"]>last_train]
        train = pd.concat([train, train_s])
        test = pd.concat([test, test_s])
        
    # Fill
    df_filled = pd.DataFrame()
    for c in df[group].unique():
        
        # Train
        df_s = train.loc[train[group] == c]
        df_imp = df_s[vars_input]
        df_imp = df_imp.copy(deep=True)
        imputer = SimpleImputer(strategy='mean')
        imputer.fit(df_imp)
        df_imp_train = imputer.transform(df_imp)
        df_imp_train_df = pd.DataFrame(df_imp_train)
        
        # Test
        df_s = test.loc[test[group] == c]
        df_imp = df_s[vars_input]
        df_imp = df_imp.copy(deep=True)        
        df_imp_test = imputer.transform(df_imp)
        df_imp_test_df = pd.DataFrame(df_imp_test)   
        
        # Merge
        df_imp_trans_df = pd.concat([df_imp_train_df, df_imp_test_df])
        df_filled = pd.concat([df_filled, df_imp_trans_df])

    # Merge
    df_filled.columns = vars_input
    df_filled = df_filled.reset_index(drop=True)
    feat_complete = df.drop(columns=vars_input)
    out = pd.concat([feat_complete, df_filled], axis=1)
    out=out.reset_index(drop=True)
    
    return out

def linear_imp_grouped(df, group, vars_input,last_train=2019):
    
    # Split data 
    train = pd.DataFrame()
    test = pd.DataFrame()
    for c in df.country.unique():
        df_s = df.loc[df["country"] == c]
        train_s = df_s.loc[df_s["year"]<=last_train]
        test_s = df_s.loc[df_s["year"]>last_train]
        train = pd.concat([train, train_s])
        test = pd.concat([test, test_s])
    
    # Fill
    df_filled = pd.DataFrame()
    for c in df[group].unique():
        
        # Train
        df_s = train.loc[train[group] == c]
        df_imp = df_s[vars_input]
        df_imp = df_imp.copy(deep=True)
        df_imp_train_df = df_imp.interpolate(limit_direction="both")
        
        # Test
        df_s = test.loc[test[group] == c]
        df_imp = df_s[vars_input]
        df_imp = df_imp.copy(deep=True)        
        df_imp_test_df = df_imp.interpolate(limit_direction="forward")
        
        # Merge
        df_imp_trans_df = pd.concat([df_imp_train_df, df_imp_test_df])
        df_filled = pd.concat([df_filled, df_imp_trans_df])
        
    # Merge
    df_filled.columns = vars_input
    df_filled = df_filled.reset_index(drop=True)
    feat_complete = df.drop(columns=vars_input)
    out = pd.concat([feat_complete, df_filled], axis=1)
    out=out.reset_index(drop=True)
    
    return out

def dichotomize(df,var,var_out,thresh):
    df[var_out]=0
    df.loc[df[var]>thresh,var_out]=1
    
def consec_zeros_grouped(df,group,var):
    zeros=[]
    for c in df.country.unique():
        # Start counting with zero                
        counts=0
        df_s=df[var].loc[df[group]==c]
        for i in range(len(df_s)): 
            if df_s.iloc[i] == 0:
                zeros.append(counts)
                counts+=1
            elif df_s.iloc[i]==1:
                # If there is an event, reset counter to zero
                counts=0
                zeros.append(0)
    return zeros

def exponential_decay(time):
    return 2**(-time/12)

def gen_model_live(y,x,target,inputs,grid=None,model_fit=RandomForestClassifier(random_state=0)):
    
    ### STEP 1 Get predictions for validation data ###
    # Train data until 2019 and get predictions for 2020 and 2021. 
    
    ##################
    ### Data split ###
    ##################
    
    training_y = pd.DataFrame()
    testing_y = pd.DataFrame()
    training_x = pd.DataFrame()
    testing_x = pd.DataFrame()  
    splits=[]
    
    for c in y.country.unique():
        y_s = y.loc[y["country"] == c]
        x_s = x.loc[x["country"] == c]    
        y_training = y_s[["country","dd"]+[target]].loc[y_s["dd"]<="2019-12"]
        x_training = x_s[["country","dd"]+inputs].loc[x_s["dd"]<="2019-12"]
        y_testing = y_s[["country","dd"]+[target]].loc[(y_s["dd"]>="2020-01")&(y_s["dd"]<="2021-12")]
        x_testing = x_s[["country","dd"]+inputs].loc[(x_s["dd"]>="2020-01")&(x_s["dd"]<="2021-12")]
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
    
    ###########
    ### Fit ###    
    ###########
    
    # If optimization
    if grid is not None:
        splits = PredefinedSplit(test_fold=splits)
        with parallel_backend('threading'):
            grid_search = RandomizedSearchCV(estimator=model_fit,param_distributions=grid,cv=splits,verbose=0,n_jobs=-1,n_iter=50,random_state=1)
            grid_search.fit(training_x_d, training_y_d.values.ravel())
        
        best_params = grid_search.best_params_
        model_fit.set_params(**best_params)
        model = model_fit
        model.fit(training_x_d, training_y_d.values.ravel())
        pred_proba=pd.DataFrame(model.predict_proba(testing_x_d)[:, 1])
    
        # Evaluations
        brier = brier_score_loss(testing_y_d[target], pred_proba)
        precision, recall, thres = precision_recall_curve(testing_y_d[target], pred_proba)
        aupr = auc(recall, precision)
        auroc = roc_auc_score(testing_y_d[target], pred_proba)
        evals = {"brier": brier, "aupr": aupr, "auroc": auroc}
        
    # If no optimization
    else:
        model = model_fit
        model.fit(training_x_d, training_y_d.values.ravel())
        pred_proba=pd.DataFrame(model.predict_proba(testing_x_d)[:, 1])
        
        # Evaluations
        brier = brier_score_loss(testing_y_d[target], pred_proba)
        precision, recall, thres = precision_recall_curve(testing_y_d[target], pred_proba)
        aupr = auc(recall, precision)
        auroc = roc_auc_score(testing_y_d[target], pred_proba)
        evals = {"brier": brier, "aupr": aupr, "auroc": auroc}
        
        
    ### Step 2 Get live prediction for 2022 ###
    # Train model until 2020 and use 2021 as inputs for 2022. 
    
    ##################
    ### Data split ###
    ##################
    
    training_y = pd.DataFrame()
    training_x = pd.DataFrame()
    splits=[]
        
    for c in y.country.unique():
        y_s = y.loc[y["country"] == c]
        x_s = x.loc[x["country"] == c]    
        y_training = y_s[["country","dd"]+[target]].loc[y_s["dd"]<="2020-12"]
        x_training = x_s[["country","dd"]+inputs].loc[x_s["dd"]<="2020-12"]
        training_y = pd.concat([training_y, y_training])
        training_x = pd.concat([training_x, x_training])
        val_training_ids = list(y_training[:int(0.8*len(y_training))].index)
        val_testing_ids = list(y_training[int(0.8*len(y_training)):].index)        
        splits += [-1] * len(val_training_ids) + [0] * len(val_testing_ids)
        
    training_y_d=training_y.drop(columns=["country","dd"])
    training_x_d=training_x.drop(columns=["country","dd"])
       
    ###########
    ### Fit ###    
    ###########
    
    # If optimization
    if grid is not None:
        splits = PredefinedSplit(test_fold=splits)
        with parallel_backend('threading'):
            grid_search = RandomizedSearchCV(estimator=model_fit,param_distributions=grid,cv=splits,verbose=0,n_jobs=-1,n_iter=50,random_state=1)
            grid_search.fit(training_x_d, training_y_d.values.ravel())
        
        best_params = grid_search.best_params_
        model_fit.set_params(**best_params)
        model = model_fit
        model.fit(training_x_d, training_y_d.values.ravel())
    
        # Use last 12 months as input when making predictions
        testing_x_s=x[["country","dd"]+inputs].loc[(x["dd"]>="2021-01")&(x["dd"]<="2021-12")]
        testing_x_d=testing_x_s.drop(columns=["country","dd"])
    
        # Predictions
        pred_proba2022=pd.DataFrame(model.predict_proba(testing_x_d)[:, 1])
        
        # Create df, replace year 2021 with year 2022
        pred_proba2022["country"]=testing_x_s.country.values
        pred_proba2022["dd"]=testing_x_s.dd.values
        pred_proba2022['dd'] = pred_proba2022['dd'].str.replace('2021', '2022')
        
        # Merge with outcome
        base=y[["country","dd",target]].loc[(y["dd"]>="2022-01")&(y["dd"]<="2022-12")]
        out2022=pd.merge(base,pred_proba2022,on=["country","dd"],how="left")   
        out2022.columns=["country","dd",target,"preds"]
        
    # If no optimization
    else:
        model = model_fit
        model.fit(training_x_d, training_y_d.values.ravel())
    
        # Use last 12 months as input when making predictions
        testing_x_s=x[["country","dd"]+inputs].loc[(x["dd"]>="2021-01")&(x["dd"]<="2021-12")]
        testing_x_d=testing_x_s.drop(columns=["country","dd"])
    
        # Predictions
        pred_proba2022=pd.DataFrame(model.predict_proba(testing_x_d)[:, 1])
        
        # Create df, replace year 2021 with year 2022
        pred_proba2022["country"]=testing_x_s.country.values
        pred_proba2022["dd"]=testing_x_s.dd.values
        pred_proba2022['dd'] = pred_proba2022['dd'].str.replace('2021', '2022')
        
        # Merge with outcome
        base=y[["country","dd",target]].loc[(y["dd"]>="2022-01")&(y["dd"]<="2022-12")]
        out2022=pd.merge(base,pred_proba2022,on=["country","dd"],how="left")   
        out2022.columns=["country","dd",target,"preds"]
       
    ### Step 3 Get live predictions for 2023 and in sample predictions for 1989-2021 ###
    # Train model until 2021 and use 2022 as inputs for 2023. 
    # In sample predictions are returned <=2021.
    
    ##################
    ### Data split ###
    ##################
    
    training_y = pd.DataFrame()
    training_x = pd.DataFrame()
    splits=[]
    
    for c in y.country.unique():
        y_s = y.loc[y["country"] == c]
        x_s = x.loc[x["country"] == c]    
        y_training = y_s[["country","dd"]+[target]].loc[y_s["dd"]<="2021-12"]
        x_training = x_s[["country","dd"]+inputs].loc[x_s["dd"]<="2021-12"]
        training_y = pd.concat([training_y, y_training])
        training_x = pd.concat([training_x, x_training])
        val_training_ids = list(y_training[:int(0.8*len(y_training))].index)
        val_testing_ids = list(y_training[int(0.8*len(y_training)):].index)       
        splits += [-1] * len(val_training_ids) + [0] * len(val_testing_ids)
        
    training_y_d=training_y.drop(columns=["country","dd"])
    training_x_d=training_x.drop(columns=["country","dd"])
    
    ###########
    ### Fit ###    
    ###########
    
    # If optimization
    if grid is not None:
        splits = PredefinedSplit(test_fold=splits)
        with parallel_backend('threading'):
            grid_search = RandomizedSearchCV(estimator=model_fit,param_distributions=grid,cv=splits,verbose=0,n_jobs=-1,n_iter=50,random_state=1)
            grid_search.fit(training_x_d, training_y_d.values.ravel())
        best_params = grid_search.best_params_
        model_fit.set_params(**best_params)
        model = model_fit
        model.fit(training_x_d, training_y_d.values.ravel())
        
        # In sample predictions
        pred_proba_in_sample=pd.DataFrame(model.predict_proba(training_x_d)[:, 1])
         
        # Create df
        in_df = pd.concat([training_y.reset_index(drop=True), pred_proba_in_sample.reset_index(drop=True)], axis=1)
        in_df.columns = ["country","dd",target,"preds"]  
        in_df=in_df.sort_values(by=["country","dd"])
            
        # Use last 12 months as input when making predictions
        testing_x_s=x[["country","dd"]+inputs].loc[(x["dd"]>="2022-01")&(x["dd"]<="2022-12")]
        testing_x_d=testing_x_s.drop(columns=["country","dd"])
    
        # Predictions       
        pred_proba2023=pd.DataFrame(model.predict_proba(testing_x_d)[:, 1])
        
        # Create df, replace year 2022 with year 2023
        pred_proba2023["country"]=testing_x_s.country.values
        pred_proba2023["dd"]=testing_x_s.dd.values
        pred_proba2023['dd'] = pred_proba2022['dd'].str.replace('2022', '2023')
        
        # Merge with outcome
        base=y[["country","dd",target]].loc[(y["dd"]>="2023-01")&(y["dd"]<="2023-12")]
        out2023=pd.merge(base,pred_proba2023,on=["country","dd"],how="left")   
        out2023.columns=["country","dd",target,"preds"]
        
    # If no optimization
    else:
        model = model_fit
        model.fit(training_x_d, training_y_d.values.ravel())
        
        # In sample predictions
        pred_proba_in_sample=pd.DataFrame(model.predict_proba(training_x_d)[:, 1])
         
        # Create df
        in_df = pd.concat([training_y.reset_index(drop=True), pred_proba_in_sample.reset_index(drop=True)], axis=1)
        in_df.columns = ["country","dd",target,"preds"]  
        in_df=in_df.sort_values(by=["country","dd"])
            
        # Use last 12 months as input when making predictions
        testing_x_s=x[["country","dd"]+inputs].loc[(x["dd"]>="2022-01")&(x["dd"]<="2022-12")]
        testing_x_d=testing_x_s.drop(columns=["country","dd"])
    
        # Predictions       
        pred_proba2023=pd.DataFrame(model.predict_proba(testing_x_d)[:, 1])
        
        # Create df, replace year 2022 with year 2023
        pred_proba2023["country"]=testing_x_s.country.values
        pred_proba2023["dd"]=testing_x_s.dd.values
        pred_proba2023['dd'] = pred_proba2022['dd'].str.replace('2022', '2023')
        
        # Merge with outcome
        base=y[["country","dd",target]].loc[(y["dd"]>="2023-01")&(y["dd"]<="2023-12")]
        out2023=pd.merge(base,pred_proba2023,on=["country","dd"],how="left")   
        out2023.columns=["country","dd",target,"preds"]
        
    # Merge in sample, live 2022 and live 2023
    df_out=pd.concat([in_df.reset_index(drop=True),out2022.reset_index(drop=True),out2023.reset_index(drop=True)], axis=0)
    df_out=df_out.sort_values(by=["country","dd"])        
   
    
    return df_out,evals


def evals(df, actuals, preds, country, tol=3, trim=False):

    onset_scores_counties = []
    mse_all = []
    mse_norm_countries = []
    
    # Get evaluations for each country
    for c in df[country].unique():
        df_s = df[df[country] == c]
        
        # Step 1: Onset score 
        
        # Get index of true onsets, country-months with non-zero fatalities
        # preceded by a value of zero
        true_onsets = df_s[actuals][(df_s[actuals].shift(1) == 0) & (df_s[actuals] > 0)].index
        
        # Get index of predicted onsets
        pred_onsets = df_s[preds][(df_s[preds].shift(1) == 0) & (df_s[preds] > 0)].index
        
        # For each true onset, check if there was a predicted one within an error
        # tolerance of "tol" 
        count = []
        # Compare each true onsets
        for i in list(true_onsets):
            # With every predicted one
            for x in list(pred_onsets): 
                # And if the temporal error is below the error tolerance
                if np.abs(x-i) <=tol:
                    # Count the onset after applying an exp penalty
                    error=np.abs(x-i)
                    count.append(1*np.exp(-0.01*error))  
                    
        # If there are true onsets, divide the number of detected onsets by the number
        # of true onsets
        if len(true_onsets)!=0:
            onset_score = np.sum(count)/len(true_onsets) 
        # If there are no onsets, onset score is zero (avoid division by zero)
        else:
            onset_score=0
        
        # Append onset score by country
        onset_scores_counties.append(onset_score)    
    
        # Step 2: Mean squared error 
        
        # Normal MSE
        mse=mean_squared_error(df_s[actuals].values, df_s[preds].values)
        mse_all.append(mse)
            
        # Normalized MSE (MSLE)
        
        # If predictions are negative, calculation fails --> clip at 0
        preds_clip = np.clip(df_s[preds].values, a_min=0, a_max=None)
        mse_norm = mean_squared_log_error(df_s[actuals].values,preds_clip)
        mse_norm_countries.append(mse_norm)
        
        # Step 3: Calculate the means 
                
        if trim==True: 
            # Calculate the mean across countries, after trimming
            mean_mse = np.mean(trim_mean(mse_all, proportiontocut=0.01))
            normalized_mse = np.mean(trim_mean(mse_norm_countries, proportiontocut=0.01))
            onset_score = np.mean(onset_scores_counties)
        else: 
            # Calculate the mean across countries
            mean_mse = np.mean(mse_all)
            normalized_mse = np.mean(mse_norm_countries)
            onset_score = np.mean(onset_scores_counties)            

    return {"Onset Score": onset_score,"Mean MSE": mean_mse,"Normalized MSE": normalized_mse,"Onset Scores by Country": onset_scores_counties,"MSE by Country": mse_all,"Normalized MSE by Country": mse_norm_countries}





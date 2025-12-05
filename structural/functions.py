import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import PredefinedSplit,RandomizedSearchCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
from sklearn.impute import SimpleImputer
import shap
from joblib import parallel_backend

def lag_groupped(df, group_var, var, lag):
    return df.groupby(group_var)[var].shift(lag).fillna(0)

def multivariate_imp_bayes(df, country, vars_input, vars_add=None,max_iter=10,min_val=0,last_train=2016):
    
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

def simple_imp_grouped(df, group, vars_input,last_train=2016):
    
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
        feat_imp = df_s[vars_input]
        df_imp = feat_imp.copy(deep=True)
        imputer = SimpleImputer(strategy='mean')
        imputer.fit(df_imp)
        df_imp_train = imputer.transform(df_imp)
        df_imp_train_df = pd.DataFrame(df_imp_train)
        
        # Test
        df_s = test.loc[test[group] == c]
        feat_imp = df_s[vars_input]
        df_imp = feat_imp.copy(deep=True)        
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

def linear_imp_grouped(df, group, vars_input,last_train=2016):
    
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
        imp = df_s[vars_input]
        df_imp = imp.copy(deep=True)
        df_imp_train_df = df_imp.interpolate(limit_direction="both")
        
        # Test
        df_s = test.loc[test[group] == c]
        imp = df_s[vars_input]
        df_imp = imp.copy(deep=True)        
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

def exponential_growth(time):
    return 2**(time/12)

def gen_model(y, x, target, inputs, model_fit=RandomForestClassifier(random_state=0), grid=None, int_methods=False, last_train=2016):

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
        y_training = y_s[["country","year"]+[target]].loc[y_s["year"]<=last_train]
        x_training = x_s[["country","year"]+inputs].loc[x_s["year"]<=last_train]
        y_testing = y_s[["country","year"]+[target]].loc[(y_s["year"]>last_train)]
        x_testing = x_s[["country","year"]+inputs].loc[(x_s["year"]>last_train)]
        training_y = pd.concat([training_y, y_training])
        testing_y = pd.concat([testing_y, y_testing])
        training_x = pd.concat([training_x, x_training])
        testing_x = pd.concat([testing_x, x_testing])
        val_training_ids = list(y_training[:int(0.8*len(y_training))].index)
        val_testing_ids = list(y_training[int(0.8*len(y_training)):].index)       
        splits += [-1] * len(val_training_ids) + [0] * len(val_testing_ids)
                 
    training_y_d=training_y.drop(columns=["country","year"])
    training_x_d=training_x.drop(columns=["country","year"])
    testing_x_d=testing_x.drop(columns=["country","year"])
        
    ###########
    ### Fit ###    
    ###########
     
    if grid is not None:
        splits = PredefinedSplit(test_fold=splits)
        with parallel_backend('threading'):
            grid_search = RandomizedSearchCV(estimator=model_fit,param_distributions=grid,cv=splits,verbose=0,n_jobs=-1,n_iter=100,random_state=1)  
            grid_search.fit(training_x_d, training_y_d.values.ravel())
        best_params = grid_search.best_params_
        model_fit.set_params(**best_params)
        model = model_fit
        model.fit(training_x_d, training_y_d.values.ravel())
        pred = pd.DataFrame(model.predict(testing_x_d))
       
        # Get out df
        out_df = pd.concat([testing_y.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
        out_df["preds_proba"]=pd.DataFrame(model.predict_proba(testing_x_d)[:, 1]) 
        out_df.columns = ["country","year",target,"preds","preds_proba"]

  
    # If no optimization
    else:
        model = model_fit
        model.fit(training_x_d, training_y_d.values.ravel())
        pred = pd.DataFrame(model.predict(testing_x_d))
        
        # Get out df
        out_df = pd.concat([testing_y.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
        out_df["preds_proba"]=pd.DataFrame(model.predict_proba(testing_x_d)[:, 1]) 
        out_df.columns = ["country","year",target,"preds","preds_proba"]
            
    ##################
    ### Evaluation ###
    ##################

    t=out_df.loc[(out_df["year"]>=2019)&(out_df["year"]<=2023)]
    brier = brier_score_loss(t[target], t["preds_proba"])
    precision, recall, thres = precision_recall_curve(t[target], t["preds_proba"])
    aupr = auc(recall, precision)
    auroc = roc_auc_score(t[target], t["preds_proba"])
    evals = {"brier": brier, "aupr": aupr, "auroc": auroc}
        
    val=out_df.loc[(out_df["year"]>=2017)&(out_df["year"]<=2018)]
    brier = brier_score_loss(val[target], val["preds_proba"])
    precision, recall, thres = precision_recall_curve(val[target], val["preds_proba"])
    aupr = auc(recall, precision)
    auroc = roc_auc_score(val[target], val["preds_proba"])
    evals_val = {"brier": brier, "aupr": aupr, "auroc": auroc}

    if int_methods == False:
            
        return out_df, evals, evals_val

    else:
              
        # SHAP values 
        x_d=x[inputs]
        
        # Auto
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(x_d)
        
        return out_df, evals, evals_val, shap_values



















import pandas as pd
from sklearn.impute import SimpleImputer

def lag_groupped(df, group_var, var, lag):
    return df.groupby(group_var)[var].shift(lag).fillna(0)

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

def simple_imp_grouped(df, group, vars_input):
    
    # Split data 
    train = pd.DataFrame()
    test = pd.DataFrame()
    for c in df.country.unique():
        df_s = df.loc[df["country"] == c]
        train_s = df_s[:int(0.8*len(df_s))]
        test_s = df_s[int(0.8*len(df_s)):]
        train = pd.concat([train, train_s])
        test = pd.concat([test, test_s])
        
    # Fill
    df_filled = pd.DataFrame()
    for c in df[group].unique():
        
        # Training
        df_s = train.loc[train[group] == c]
        imp = df_s[vars_input]
        df_imp = imp.copy(deep=True)
        imputer = SimpleImputer(strategy='mean')
        imputer.fit(df_imp)
        df_imp_train = imputer.transform(df_imp)
        df_imp_train_df = pd.DataFrame(df_imp_train)
        
        # Test
        df_s = test.loc[test[group] == c]
        imp = df_s[vars_input]
        df_imp = imp.copy(deep=True)        
        df_imp_test = imputer.transform(df_imp)
        df_imp_test_df = pd.DataFrame(df_imp_test)        

        # Merge
        df_imp_final = pd.concat([df_imp_train_df, df_imp_test_df])
        df_filled = pd.concat([df_filled, df_imp_final])

    # Merge
    df_filled.columns = vars_input
    df_filled = df_filled.reset_index(drop=True)
    feat_complete = df.drop(columns=vars_input)
    out = pd.concat([feat_complete, df_filled], axis=1)
    out=out.reset_index(drop=True)
    
    return out


def linear_imp_grouped(df, group, vars_input):
    
    # Split data 
    train = pd.DataFrame()
    test = pd.DataFrame()
    for c in df.country.unique():
        df_s = df.loc[df["country"] == c]
        train_s = df_s[:int(0.8*len(df_s))]
        test_s = df_s[int(0.8*len(df_s)):]
        train = pd.concat([train, train_s])
        test = pd.concat([test, test_s])
    
    # Fill
    df_filled = pd.DataFrame()
    for c in df[group].unique():
        
        # Training
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
        df_imp_final = pd.concat([df_imp_train_df, df_imp_test_df])
        df_filled = pd.concat([df_filled, df_imp_final])
        
    # Merge
    df_filled.columns = vars_input
    df_filled = df_filled.reset_index(drop=True)
    feat_complete = df.drop(columns=vars_input)
    out = pd.concat([feat_complete, df_filled], axis=1)
    out=out.reset_index(drop=True)
    
    return out




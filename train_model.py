#Rob Hodde 
#Fall 2018 IS622 Homework 2
#use RandomForestClassifier to train a model to predict survival in Titanic dataset
#adapted from: https://github.com/cuny-sps-msda-data622-2017fall/homework-2-jelikish/blob/master/train_model.py

import pandas as pd
import numpy as np
import os
import statsmodels.imputation.mice as mice
from sklearn.preprocessing import Imputer  
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import pickle

model_file = 'model.pkl' #trained model 


#retrieve training and test files, cleanse, create 1-hot predictors, write to disk
def eda():

    files = ['train.csv','test.csv']  
    for file in files:
    
        #read csv file
        try:
            df = pd.read_csv(file)
        except:
            print(file + " not found")
        
        #remove columns unlikely to contain inferential value
        df = df.drop(['Cabin','PassengerId','Name','Ticket'], axis=1)
        
        #convert gender and port into numerics, 
        #so we can do imputation on missing Age values 
        df1 = df.copy(deep=True)
        df1['Sex'] = pd.factorize(df['Sex'])[0]
        df1['Embarked'] = pd.factorize(df['Embarked'])[0]
        
        # use MICE imputation to fix ages
        imp = mice.MICEData(df1)
        imp.update_all(100)  #creates new data frame with imputed values
        df = df.drop(['Age'], axis=1) #drop the original age column
        df = pd.concat([df, imp.data['Age']], axis=1) #add the imputed column back in
        
        #tried binning but could not factorize these in ascending order 
        #and did not like bin labels.
        #bin Age into Toddler, Child, Adolescent, Adult, Elderly
        #df = df.filter(['Age'], axis=1)
        #print(df)
        #age_bins = [0, 2, 7, 21, 60, 100]   
        #out = pd.cut(df['Age'], bins=age_bins)
        #df = pd.concat((df, out), axis=1)
        #df.columns.values[1] = "Age_Bin"
        #df = df.drop(['Age'], axis=1)
        #df = pd.concat([df, df], axis=1)
        
        #create five categories for age, representing boundaries between social mores
        df.loc[df['Age'] < 3, 'Age_Bin'] = '1-Toddler'
        df.loc[(df['Age'] >= 3)  & (df['Age'] < 13), 'Age_Bin'] = '2-Child'
        df.loc[(df['Age'] >= 13) & (df['Age'] < 20), 'Age_Bin'] = '3-Teen'
        df.loc[(df['Age'] >= 20) & (df['Age'] < 60), 'Age_Bin'] = '4-Adult'
        df.loc[df['Age']  >= 60, 'Age_Bin'] = '5-Senior'
        
        #distinguish between traveling alone, small families, and large families
        df['family_size'] = df['SibSp'] + df['Parch']
        df.loc[df['family_size'] == 0, 'Family'] = '2-None'
        df.loc[(df['family_size'] > 0) & (df['family_size'] < 4), 'Family'] = '1-Small'
        df.loc[df['family_size'] >= 4, 'Family'] = '3-Large'
        
        #create 1-hot variables for each category value (level), then drop the original columns
        df = pd.concat([df,pd.get_dummies(df['Age_Bin'], prefix='Age_Bin')],axis=1)
        df = pd.concat([df,pd.get_dummies(df['Sex'], prefix='Gender')],axis=1)
        df = pd.concat([df,pd.get_dummies(df['Embarked'], prefix='Embarked')],axis=1)
        df = pd.concat([df,pd.get_dummies(df['Family'], prefix='Family')],axis=1)
        df.drop(['Age_Bin','Sex','Embarked','Family'],axis=1, inplace=True)
        
        try:
            df.to_csv('mod_'+ file, encoding='utf-8')  #save results to disk
            print('Successful writing file mod_' + file)
        except:
            print("Could not write file mod_" + file)
            
    return()



#Function takes in X = training dataset and y=target and returns pipeline object including a trained model using RandomForestClassifier
def ml(file):
   
#    #to tune the Random Forest learner parameters, use Random Hyperparameter Grid
#    takes several hours to run - don't repeat for production. just use the optimal parameter settings
#    #adapted from https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
#    n_estimators = [int(x) for x in np.linspace(start = 50, stop = 2000, num = 10)] # Number of trees in random forest
#    max_features = ['auto', 'sqrt']  # Number of features to consider at every split
#    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)] # Maximum number of levels in tree
#    max_depth.append(None)
#    min_samples_split = [2, 4, 6] # Minimum number of samples required to split a node
#    min_samples_leaf  = [2, 4, 6] # Minimum number of samples required at each leaf node
#    bootstrap = [True, False] # Method of selecting samples for training each tree
#    # Create the random grid
#    random_grid = {'n_estimators': n_estimators,
#                   'max_features': max_features,
#                   'max_depth': max_depth,
#                   'min_samples_split': min_samples_split,
#                   'min_samples_leaf': min_samples_leaf,
#                   'bootstrap': bootstrap}
#    # Use the random grid to search for best hyperparameters
#    # First create the base model to tune
#    rf = RandomForestRegressor()
#    # Random search of parameters, using 3 fold cross validation, 
#    # search across 100 different combinations, and use all available cores
#    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
#    # Fit the random search model
#    rf_random.fit(X, y)
#    rf_random.best_params_

    #read csv file
    try:
        df = pd.read_csv(file)
    except:
        print(file + " not found")
        
    y = df['Survived'] # predictors
    X = df.drop('Survived', axis=1)  # outcome

    # instantiate an imputer object and randomforestclassifier
    #min_samples_split = min number of data points placed in a node before the node is split
    #min_samples_leaf = min number of data points allowed in a leaf node
    imp1 = Imputer(missing_values='NaN', strategy='mean', axis=0)  #here I am not imputing Age (already did above), because this method uses Mean.
    f1 = RandomForestClassifier(max_depth=10, min_samples_split=3, min_samples_leaf=2, n_estimators=100, random_state=1)

    # list steps for pipline
    steps = [('imputation', imp1), ('random_forest', f1)]

    # instatiate pipeline
    pipeline = Pipeline(steps)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #save predictors and outcome to text files
    try:
        X_test.to_csv('x_test.csv')
        y_test.to_csv('y_test.csv')
        print('Sucessful saving test set to disk.')
    except:
        print("Could not save test set.")

    # fit the model
    try:
        model = pipeline.fit(X_train, y_train)
        print('Successful fitting model.')
    except:
        print("Could not fit model.")

    return(model)


#Function takes trained model and file name as arguments and saves the trained model to the specified file.
def model_write(model, model_file):
    try:
        p = open(model_file, 'wb')
        pickle.dump(model, p)
        print('Successful writing model to disk.')
    except:
        print("Could not save model to a file.")
    p.close()


def main():
    eda()
    model = ml('mod_train.csv')
    model_write(model, model_file)


if  __name__ =='__main__':
    main()
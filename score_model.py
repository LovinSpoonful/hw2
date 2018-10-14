#Rob Hodde
#IS622 Hw2

#This script scores the model that was created in 'train_model'
#by retrieving the model and enhanced test data
#then using sci-kit learn's prediction capability to test the accuracy of the model
# adapted from https://github.com/cuny-sps-msda-data622-2017fall/homework-2-jelikish/blob/master/score_model.py

import pickle
import pandas as pd
from sklearn.metrics import classification_report

#file with trained model
model_file = 'model.pkl'

#Read the trained model and the enhanced test set
def read_files():
    try:
        test_df = pd.read_csv("mod_test.csv")
        X_test = pd.read_csv("x_test.csv")
        y_test = pd.read_csv("y_test.csv")
    except:
        print("Could not read files")

    X_test.drop(X_test.columns[[0]], axis=1, inplace=True)
    y_test.drop(y_test.columns[[0]], axis=1, inplace=True)
    X_test.drop(X_test.index[0], inplace=True)
    return([X_test, y_test, test_df])


def model_read(file):
    try:
        p = open(file, 'rb')
        openmodel = pickle.load(p)
    except:
        print("Could not open", file)
    return(openmodel)


def main():
    X_test, y_test, test_df = read_files()
    model = model_read(model_file)
    score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    test_pred = model.predict(test_df)
    class_report = classification_report(y_test, y_pred, )

    print("-------------------------------------------------------")
    print("Classification Report on Test Set from train.csv ")
    print(class_report)
    print("-------------------------------------------------------")
    print("Model score on Test set from train.csv:", score)
    print("-------------------------------------------------------")


    input("Press Enter to continue...")
    print("Predictions on test.csv")
    print(format(test_pred))

    print("-------------------------------------------------------")

if  __name__ =='__main__':
    main()
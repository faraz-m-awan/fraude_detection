import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from datetime import datetime

def trainModel(data_dir:str, class_label:str)-> RandomForestClassifier:
    
    # Loading training data
    train_df=pd.read_csv(f'{data_dir}/training_data.csv')
    x_train=train_df.drop(columns=[class_label])
    y_train=train_df[class_label]
    
    # Training Random Forest Classifier
    model= RandomForestClassifier(n_estimators=500, random_state=42)
    model.fit(x_train, y_train)

    return model

def testModel(model: RandomForestClassifier, data_dir:str, class_label:str,rev_cap:int,test_data_perc:float)-> None:
    # Loading test data
    
    test_df=pd.read_csv(f'{data_dir}/test_data.csv')
    x_test=test_df.drop(columns=[class_label])
    y_test=test_df[class_label]

    

    # Predicting the class label
    y_pred=model.predict(x_test)
    print(classification_report(y_test, y_pred))

    predict_prob=model.predict_proba(x_test) 
    test_df['fraudulent_prob']=predict_prob[:,1] # Assignining fraudulent probability to transactions
    test_df.sort_values(by=['fraudulent_prob'],ascending=False,inplace=True) # Sorting data frame in descending order with respect to fraudulent probability

    test_review_capacity= int((rev_cap*12) * test_data_perc) # Calculating the review capacity for the test data
    
    high_risk_df= test_df.head(test_review_capacity) # Getting top n=review_capacity transactions
    
    detected_fraud=high_risk_df[high_risk_df[class_label]==1].shape[0] # Calculating the total number of detected fraudulent transactions in top n transactions
    total_fraud=y_test[y_test==1].shape[0] # Getting total number of fraudulent transactions in the test data
    perc_fraud_prevented=(detected_fraud/total_fraud)*100 # Calculating the percentage of the fraud prevented
    print(f'Percentage of fraud prevented: {perc_fraud_prevented}')

    return



if __name__=='__main__':

    data_dir= 'fraud_detection/data'
    class_label='fraudulent'
    review_capacity=400
    test_data_perc=0.3

    print(f'{datetime.now()} Model training started...')
    model=trainModel(data_dir,class_label)
    print(f'{datetime.now()} Model training completed!')

    #model=joblib.load('fraud_detection/model/random_forest_trained.joblib')
    print(f'{datetime.now()} Testing model...')

    testModel(model,data_dir,class_label,review_capacity,test_data_perc)
    print(f'{datetime.now()} Model testing completed!')

    save_model=input('Do you want to save model? (Y/N):')

    if save_model.lower()=='y':
        joblib.dump(model,'fraud_detection/model/random_forest_trained.joblib')
        print(f'{datetime.now()}: Model Saved!')


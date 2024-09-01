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

def testModel(model: RandomForestClassifier, data_dir:str, class_label:str)-> None:
    # Loading test data
    test_df=pd.read_csv(f'{data_dir}/test_data.csv')
    x_test=test_df.drop(columns=[class_label])
    y_test=test_df[class_label]

    # Predicting the class label
    y_pred=model.predict(x_test)
    print(classification_report(y_test, y_pred))
    return



if __name__=='__main__':

    data_dir= 'fraud_detection/data'
    class_label='fraudulent'

    print(f'{datetime.now()} Model training started...')
    model=trainModel(data_dir,class_label)
    print(f'{datetime.now()} Model training completed!')
    print(f'{datetime.now()} Testing model...')
    testModel(model,data_dir,class_label)
    print(f'{datetime.now()} Model testing completed!')

    save_model=input('Do you want to save model? (Y/N):')

    if save_model.lower()=='y':
        joblib.dump(model,'fraud_detection/model/random_forest_trained.joblib')
        print(f'{datetime.now()}: Model Saved!')


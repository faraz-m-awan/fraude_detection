
# Imporing necessary libraries
import pandas as pd
from glob import glob
import os
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from datetime import datetime


def processData(feature_file:str, label_file:str,output_dir:str)->None:
    
    

    obs_df=pd.read_csv(feature_file,parse_dates=['transactionTime'])
    label_df=pd.read_csv(label_file,parse_dates=['reportedTime'])

    # Feature engineering

    
    obs_df['fraudulent']=obs_df['eventId'].isin(label_df['eventId']).astype(int) # Joining transactions and labels dataframse

    # Getting time related features
    obs_df['hour']=obs_df['transactionTime'].dt.hour
    obs_df['isWeekend']= obs_df['transactionTime'].apply(lambda x: 1 if x.day_of_week>=5 else 0)

    # Getting amount related features
    obs_df['avgTransactionAmount']=obs_df.groupby(['accountNumber'])['transactionAmount'].transform(lambda x: x.mean()) # Average transaction amount /Account Number
    obs_df['avgFraudulentTransaction']=obs_df.groupby(['accountNumber'])['fraudulent'].transform(lambda x: x.mean()) # Fraudulent transaction average/Account Number

    # Getting dummy variable of categorical variables
    obs_df=pd.get_dummies(obs_df, columns=['mcc', 'posEntryMode'], drop_first=True)

    # The created dummy variables are in boolean format, coverting them into numeric format (0 or 1)
    bool_cols=obs_df.select_dtypes(include='bool').columns
    obs_df[bool_cols]=obs_df[bool_cols].astype(int)
    obs_df.drop(columns=['merchantId','transactionTime','eventId','accountNumber','merchantCountry','merchantZip','availableCash'],inplace=True)

    # Performing normalization on the numeric columns
    label_df=obs_df['fraudulent']
    obs_df.drop(columns='fraudulent', inplace=True)


    # Data Balancing
    x_train, x_test, y_train, y_test=train_test_split(obs_df,label_df,test_size=0.3,random_state=42, stratify=label_df)

    smote=SMOTE(random_state=42)
    x_resampled,y_resampled=smote.fit_resample(x_train,y_train)

    x_resampled=pd.concat([x_resampled,y_resampled],axis=1)
    x_test=pd.concat([x_test,y_test],axis=1)

    # Saving processed data
    x_resampled.to_csv(f'{output_dir}/training_data.csv',index=False)
    x_test.to_csv(f'{output_dir}/test_data.csv',index=False)

    return


if __name__=='__main__':
    # Loading data
    data_dir=f'fraud_detection/data'
    feature_file=f'{data_dir}/transactions_obf.csv'
    label_file=f'{data_dir}/labels_obf.csv'

    processData(feature_file, label_file, data_dir)
    print('Preprocessing Completed...')



import pandas as pd
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics as sklm
from math import sqrt

# Load data
cust_df = pd.read_csv('../AdvWorksCusts.csv')
spend_df = pd.read_csv('../AW_AveMonthSpend.csv')
bike_df = pd.read_csv('../AW_BikeBuyer.csv')
test_df = pd.read_csv('../AW_test.csv')

# Clean and merge data (decision to keep first occurrence is given by the question)
cust_df.drop_duplicates(subset='CustomerID', keep='first', inplace=True)
spend_df.drop_duplicates(subset='CustomerID', keep='first', inplace=True)
bike_df.drop_duplicates(subset='CustomerID', keep='first', inplace=True)

# Check df shapes
cust_df.shape, bike_df.shape
cust_df.head(), bike_df.head()

# Visualize label
spend_df['AveMonthSpend'].plot.hist()

# Encode categorical features
def encode_cat(cat_features):
    ## If category is an object, encode to numeric first, 
    ## else onehotencode directly
    if cat_features.dtype == 'object':
        ## Encode strings to numeric categories
        enc = preprocessing.LabelEncoder()
        enc_cat_features = enc.fit_transform(cat_features)
    else:
        enc_cat_features = np.array(cat_features)
    ## One hot encoding
    ohe = preprocessing.OneHotEncoder(categories='auto')
    encoded = ohe.fit_transform(enc_cat_features.reshape(-1,1)).toarray()
    return encoded

def process_data(cust_df):
    # Compute Age on 1998-01-01 (Scenario given by question)
    start_date = pd.to_datetime('1998-01-01')
    cust_df['Age'] = cust_df['BirthDate'].apply(lambda bd : relativedelta(start_date, pd.to_datetime(bd)).years)
    
    # Remove irrelevant columns
    columns_drop = ['Title', 'FirstName', 'MiddleName', 'LastName', 'Suffix', 
                    'AddressLine1', 'AddressLine2', 'City', 'StateProvinceName',
                    'PostalCode', 'PhoneNumber', 'BirthDate']
    cust_df.drop(columns=columns_drop, inplace=True)

    categorical_columns = ['CountryRegionName', 'Education', 'Occupation', 'Gender',
                           'MaritalStatus']
    
    # Init an array onehot encoded with the column of data that's already numeric 'HomeOwnerFlag'
    features = encode_cat(cust_df['HomeOwnerFlag'])
    
    # Onehot all other categoricals and concat to features array
    for col in categorical_columns:
        temp = encode_cat(cust_df[col])
        features = np.concatenate((features, temp), axis=1)
    
    numeric_columns = ['NumberCarsOwned', 'NumberChildrenAtHome', 
                       'TotalChildren', 'YearlyIncome', 'Age']
        
    # Concactenate encoded features with other features
    features = np.concatenate((features, np.array(cust_df[numeric_columns])), axis=1)
    return features

# =============================================================================
# # Preprocessing
# =============================================================================
labels = np.array(spend_df['AveMonthSpend'])
features = process_data(cust_df)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

# Data scaling
sc = preprocessing.StandardScaler()

X_train[:, -5:] = sc.fit_transform(X_train[:, -5:])
X_test[:, -5:] = sc.transform(X_test[:, -5:])

# =============================================================================
# # Modelling
# =============================================================================
# Weighted model due to class imbalance of approx 1:2, Buyers to Non-Buyers
linear_mod = LinearRegression()
linear_mod.fit(X_train, y_train)
pred = linear_mod.predict(X_test)

# =============================================================================
# # Evaluation
# =============================================================================
rmse = sqrt(sklm.mean_squared_error(y_test, pred))
print('RMSE is %0.4f' % rmse)

# =============================================================================
# # Predictions
# =============================================================================
features_real = process_data(test_df)
features_real[:, -5:] = sc.transform(features_real[:, -5:])
pred_real = linear_mod.predict(features_real)

# =============================================================================
# # Submission
# =============================================================================
submission_df = pd.DataFrame(pred_real)
submission_df.to_csv('./submission.csv')


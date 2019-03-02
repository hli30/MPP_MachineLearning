import pandas as pd
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

# Check for class imbalance
bike_df.groupby('BikeBuyer').count()


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

def score_model(probs, threshold):
    return np.array([1 if x > threshold else 0 for x in probs[:,1]])

def print_metrics(labels, scores):
    metrics = sklm.precision_recall_fscore_support(labels, scores)
    conf = sklm.confusion_matrix(labels, scores)
    print('                 Confusion matrix')
    print('                 Score positive    Score negative')
    print('Actual positive    %6d' % conf[0,0] + '             %5d' % conf[0,1])
    print('Actual negative    %6d' % conf[1,0] + '             %5d' % conf[1,1])
    print('')
    print('Accuracy  %0.2f' % sklm.accuracy_score(labels, scores))
    print('RMSE  %0.4f' % sqrt(sklm.mean_squared_error(labels, scores)))
    print(' ')
    print('           Positive      Negative')
    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])

def plot_auc(labels, probs):
    ## Compute the false positive rate, true positive rate
    ## and threshold along with the AUC
    fpr, tpr, threshold = sklm.roc_curve(labels, probs[:,1])
    auc = sklm.auc(fpr, tpr)
    
    ## Plot the result
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color = 'orange', label = 'AUC = %0.3f' % auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# =============================================================================
# # Preprocessing
# =============================================================================
labels = np.array(bike_df['BikeBuyer'])
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
logistic_mod = LogisticRegression(class_weight = {0:0.33, 1:0.67})
logistic_mod.fit(X_train, y_train)
probabilities = logistic_mod.predict_proba(X_test)

# =============================================================================
# # Evaluation
# =============================================================================
scores = score_model(probabilities, 0.65)
print_metrics(y_test, scores)
plot_auc(y_test, probabilities)  

# =============================================================================
# # Optimization
# =============================================================================
# Optimize decision threshold
# def test_threshold(probs, labels, threshold):
#     scores = score_model(probs, threshold)
#     print('')
#     print('For threshold = ' + str(threshold))
#     print_metrics(labels, scores)
#     
# thresholds = [0.65, 0.6, 0.55, 0.5, 0.45, 0.40, 0.35]
# for t in thresholds:
#     test_threshold(probabilities, y_test, t)
# =============================================================================

# =============================================================================
# # Predictions
# =============================================================================
features_real = process_data(test_df)
features_real[:, -5:] = sc.transform(features_real[:, -5:])
prob_real = logistic_mod.predict_proba(features_real)
scores_real = score_model(prob_real, 0.65)

# =============================================================================
# # Submission
# =============================================================================
submission_df = pd.DataFrame(scores_real)
submission_df.to_csv('./submission.csv')


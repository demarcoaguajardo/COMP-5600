# Importing Libraries

import pandas as panda # Data loading, manipulation, cleaning, and visualization

from sklearn.model_selection import train_test_split # Split data into training and testing sets
from sklearn.model_selection import cross_validate # Uses multiple metrics and returns dictionary of scores

from sklearn.linear_model import LogisticRegression # Logistic Regression Model
from sklearn.naive_bayes import GaussianNB # Naive Bayes Model 

from sklearn.metrics import accuracy_score # Measures accuracy
from sklearn.metrics import precision_score # Measures precision
from sklearn.metrics import recall_score # Measures recall
from sklearn.metrics import f1_score # Measures F1 Score
from sklearn.metrics import roc_auc_score # Measures ROC-AUC Score

from sklearn.preprocessing import StandardScaler # Standard Scaler
from sklearn.preprocessing import LabelEncoder # Encodes labels as ints

from sklearn.impute import SimpleImputer # Handles missing data

# ---------- Load data from CSV ----------

try:
    data = panda.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
except FileNotFoundError:
    print("File not found. Please check the file path.")
    exit()

# ---------- Preprocess the data ----------

# Handle missing values using most common value for LR
imputerMostFrequent = SimpleImputer(strategy='most_frequent')
dataImputedMostFrequent = panda.DataFrame(imputerMostFrequent.fit_transform(data),
                                          columns=data.columns)

# Handle missing values using default value (0) for NB
imputerDefault = SimpleImputer(strategy='constant', fill_value=0)
dataImputedDefault = panda.DataFrame(imputerDefault.fit_transform(data), 
                                     columns=data.columns)

# Encode categorical data
labelEncoders = {}
for column in dataImputedMostFrequent.select_dtypes(include=['object']).columns:
    if column != 'customerID':
        labEnc = LabelEncoder()
        dataImputedMostFrequent[column] = labEnc.fit_transform(dataImputedMostFrequent[column])
        dataImputedDefault[column] = labEnc.fit_transform(dataImputedDefault[column])
        labelEncoders[column] = labEnc

# Scale Numerical Features
scaler = StandardScaler()
numericalFeatures = ['tenure', 'MonthlyCharges', 'TotalCharges'] # Numerical Features of data
   # Fit and transform most frequent dataset
dataImputedMostFrequent[numericalFeatures] = scaler.fit_transform(
    dataImputedMostFrequent[numericalFeatures])
   # Fit and transform default dataset   
dataImputedDefault[numericalFeatures] = scaler.fit_transform(
    dataImputedDefault[numericalFeatures])

# ---------- Split Data Into Features and Target ----------

# (X = features, y = target)
# Exclude customer ID and target variable from features
# Include only target variable in target
X_MostFrequent = dataImputedMostFrequent.drop(columns=['customerID', 'Churn'])
y_MostFrequent = dataImputedMostFrequent['Churn']

X_Default = dataImputedDefault.drop(columns=['customerID', 'Churn'])
y_Default = dataImputedDefault['Churn']

# ---------- Split Data Into Training and Testing Sets ----------

# X_TrainMF = Training Features for Most Frequent Dataset
# X_TestMF = Testing Features for Most Frequent Dataset
# y_TrainMF = Training Target for Most Frequent Dataset
# y_TestMF = Testing Target for Most Frequent Dataset
# test_size = 20% for testing, 80% for training
# random_state = 88 for reproducibility
X_TrainMF, X_TestMF, y_TrainMF, y_TestMF = train_test_split(
    X_MostFrequent, y_MostFrequent, test_size=0.2, random_state=88) 

# X_TrainDF = Training Features for Default Dataset
# X_TestDF = Testing Features for Default Dataset
# y_TrainDF = Training Target for Default Dataset
# y_TestDF = Testing Target for Default Dataset
# test_size = 20% for testing, 80% for training
# random_state = 88 for reproducibility
X_TrainDF, X_TestDF, y_TrainDF, y_TestDF = train_test_split(
    X_Default, y_Default, test_size=0.2, random_state=88)

# ---------- Initialize Models ----------

# Logistic Regression Model
logReg = LogisticRegression(max_iter=1000)

# Naive Bayes Model
naiveBayes = GaussianNB()

# ---------- Perform 5-Fold Cross-Validation and Evaluate Models ----------

def evaluateModel(model, X_Train, y_Train):
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    scores = cross_validate(model, X_Train, y_Train, cv=5, scoring=scoring)

    # Returns dictionary of average scores for each metric
    return {metric: scores[f'test_{metric}'].mean() for metric in scoring}

# Evaluate Logistic Regression Model
logRegResults = evaluateModel(logReg, X_TrainMF, y_TrainMF)

# Evaluate Naive Bayes Model
naiveBayesResults = evaluateModel(naiveBayes, X_TrainDF, y_TrainDF)

# ---------- Print Results ----------
print("\nLogistic Regression Results (Most Frequent):")
for metric, score in logRegResults.items():
    print(f"{metric}: {score:.4f}")

print("\nNaive Bayes Results (Default):")
for metric, score in naiveBayesResults.items():
    print(f"{metric}: {score:.4f}")


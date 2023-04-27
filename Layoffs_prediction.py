import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('Layoff.csv', encoding='utf-8')
df = df.drop(['Company', 'City', 'Effective Date', 'Closure/Layoff','Temporary/Permanent','Union','Region','County'], axis=1)
df['WARN Received Date'] = pd.to_datetime(df['WARN Received Date'])
df['by_month'] = df['WARN Received Date'].dt.strftime("%Y-%m")
df = df[(df.by_month<'2020-03')|(df.by_month>'2021-09')].copy()
# Check for missing values
df.isna().sum()

df.hist(figsize=(20,20))

fig, ax = plt.subplots(figsize=(15,15))    
#sns.heatmap(df.corr(), annot=True, ax=ax)

# Visualize the relationship between two numerical columns using a scatter plot
# plt.scatter(df['GDP'], df['unemployment_rate'])
# plt.xlabel('GDP')
# plt.ylabel('Unemployment Rate')
# plt.show()

sortedDF = df.sort_values(by=['Number of Workers'], ascending=False)
fig, ax = plt.subplots(figsize=(10, 10))
 
# plt.barh(y='State', width='Number of Workers', data=sortedDF)
# plt.xlabel('# of employees affected')
# plt.title('Employees laid off by state since 2000 - Per WARN')
# plt.show()

sortedDF = df.sort_values(by=['WARN Received Date'], ascending=True)
fig, ax = plt.subplots(figsize=(10, 10))
 
# plt.plot(sortedDF['WARN Received Date'], sortedDF['Number of Workers'])
# plt.xlabel('Layoff Date')
# plt.ylabel('# of Employees affected')
# plt.show()

# Day month year of layoffs
df['Day'] = df['WARN Received Date'].dt.day
df['Month'] = df['WARN Received Date'].dt.month
df['Year'] = df['WARN Received Date'].dt.year

df['Price Volatility'] = df['High'] - df['Low']

# One hot encode
df = pd.get_dummies(df, columns=['Industry'])

quantiles = df['Number of Workers'].quantile([0.25, 0.5, 0.75])
df['layoff_category'] = pd.cut(df['Number of Workers'], 
                               bins=[0, quantiles.iloc[0], 
                                     quantiles.iloc[1], quantiles.iloc[2],
                                     df['Number of Workers'].max()], 
                                     labels=[1,2,3,4])

clf = MLPClassifier(solver='lbfgs', 
                    alpha=1e-5, 
                    hidden_layer_sizes=(5, 2), 
                    random_state=1)

df = df.dropna()
df = df.drop(['by_month'], axis=1)

df = df.iloc[:,3:33]

X = df.iloc[:,0:28]
y = df.iloc[:,[29]]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# clf.fit(X, y)

# train = clf.predict(X)

# accuracy = metrics.accuracy_score(y, train)

# print(accuracy)

# #Initiate SVM model
# svm_model = SVC(kernel="linear")

# #Fit the SVM model on the training data
# svm_model.fit(X_train, y_train)

# #Display summary of the SVM model
# print(svm_model)

# #Predict on the training data using the SVM model
# svm_model_predicted_training = svm_model.predict(X_train)

# #Predict on the test data using the SVM model
# svm_model_predicted_test = svm_model.predict(X_test)

# # Accuracy 
# svm_model_training_accuracy = metrics.accuracy_score(y_train, svm_model_predicted_training)
# svm_model_test_accuracy = metrics.accuracy_score(y_test, svm_model_predicted_test)

# print('Training Accuracy: ' + str(svm_model_training_accuracy))
# print('Test Accuracy: ' + str(svm_model_test_accuracy))

# #Summarize the fit of the SVM model on the training data
# print(metrics.classification_report(y_train, svm_model_predicted_training))
# print(metrics.confusion_matrix(y_train, svm_model_predicted_training))

# #Summarize the fit of the SVM model on the test data
# print(metrics.classification_report(y_test, svm_model_predicted_test))
# print(metrics.confusion_matrix(y_test, svm_model_predicted_test))

#Initiate Kernel SVM model
kernel_svm_model = SVC(kernel="rbf")

#Fit the Kernel SVM model on the training data
kernel_svm_model.fit(X_train, y_train)

#Display summary of the Kernel SVM model
print(kernel_svm_model)

#Predict on the training data using the Kernel SVM model
kernel_svm_model_predicted_training = kernel_svm_model.predict(X_train)

#Predict on the test data using the Kernel SVM model
kernel_svm_model_predicted_test = kernel_svm_model.predict(X_test)

# Accuracy 
kernel_svm_model_training_accuracy = metrics.accuracy_score(y_train, kernel_svm_model_predicted_training)
kernel_svm_model_test_accuracy = metrics.accuracy_score(y_test, kernel_svm_model_predicted_test)

print('Training Accuracy: ' + str(kernel_svm_model_training_accuracy))
print('Test Accuracy: ' + str(kernel_svm_model_test_accuracy))

#Summarize the fit of the Kernel SVM model on the training data
print(metrics.classification_report(y_train, kernel_svm_model_predicted_training))
print(metrics.confusion_matrix(y_train, kernel_svm_model_predicted_training))

#Summarize the fit of the Kernel SVM model on the test data
print(metrics.classification_report(y_test, kernel_svm_model_predicted_test))
print(metrics.confusion_matrix(y_test, kernel_svm_model_predicted_test))

#Initiate simple neural network model
neural_network_model = MLPClassifier(alpha=1, hidden_layer_sizes = (20, 10))

#Fit the simple neural network model on the training data
neural_network_model.fit(X_train, y_train)

#Display summary of the simple neural network model
print(neural_network_model)

#Predict on the training data using the simple neural network model
neural_network_model_predicted_training  = neural_network_model.predict(X_train)

#Predict on the test data using the simple neural network model
neural_network_model_predicted_test = neural_network_model.predict(X_test)

#Accuracy 
neural_network_model_training_accuracy = metrics.accuracy_score(y_train, neural_network_model_predicted_training)
neural_network_model_test_accuracy = metrics.accuracy_score(y_test, neural_network_model_predicted_test)

print('Training Accuracy: ' + str(neural_network_model_training_accuracy))
print('Test Accuracy: ' + str(neural_network_model_test_accuracy))

#Summarize the fit of the simple neural network model on the training data
print(metrics.classification_report(y_train, neural_network_model_predicted_training))
print(metrics.confusion_matrix(y_train, neural_network_model_predicted_training))

#Summarize the fit of the simple neural network model on the test data
print(metrics.classification_report(y_test, neural_network_model_predicted_test))
print(metrics.confusion_matrix(y_test, neural_network_model_predicted_test))

# #Define the K-Fold evaluation method 
# kfold_cv = KFold(n_splits = 10, 
#                  random_state = 5, 
#                  shuffle = True)

# #Initialize the grid
# grid = dict()
# grid['alpha'] = np.arange(0.04, 0.71, 0.01)

# #Define the grid search
# grid_search = GridSearchCV(neural_network_model, 
#                            grid, 
#                            scoring = 'neg_mean_squared_error', 
#                            cv = kfold_cv, 
#                            n_jobs = -1)

# #Performing the grid search
# grid_search_results = grid_search.fit(X_train, y_train)

# print(grid_search_results.best_params_)
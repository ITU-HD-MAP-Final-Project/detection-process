import time
import joblib
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import normalize
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# params
read_model = False
model_name = ''

df_train = pd.read_csv('../point_clouds/all_-1_1_train_ds02_optimalneighborhood_cleaned_featured.txt')
df_test = pd.read_csv('../point_clouds/all_-1_1_test_ds02_optimalneighborhood_cleaned_featured.txt')

# "X","Y","Z","Classification","Intensity","OptimalKNN","OptimalRadius","Linearity","Planarity","EigenvalueSum","Verticality","SurfaceVariation"
train_mean_radius = df_train.loc[:, 'OptimalRadius'].mean()
train_mean_knn = df_train.loc[:, 'OptimalKNN'].mean()
train_mean_linearity = df_train.loc[:, 'Linearity'].mean()
train_mean_planarity = df_train.loc[:, 'Planarity'].mean()
train_mean_eigenvaluesum = df_train.loc[:, 'EigenvalueSum'].mean()
train_mean_verticality = df_train.loc[:, 'Verticality'].mean()
train_mean_surfacevariation = df_train.loc[:, 'SurfaceVariation'].mean()

print("OptimalRadius Mean: " + str(train_mean_radius))
print("KNN Mean: " + str(train_mean_knn))
print("Linearity Mean: " + str(train_mean_linearity))
print("Planarity Mean: " + str(train_mean_planarity))
print("EigenvalueSum Mean: " + str(train_mean_eigenvaluesum))
print("Verticality Mean: " + str(train_mean_verticality))
print("SurfaceVariation Mean: " + str(train_mean_surfacevariation))

train_radius_counter = 0
train_knn_counter = 0
train_linearity_counter = 0
train_planarity_counter = 0
train_eigenvaluesum_counter = 0
train_verticality_counter = 0
train_surfacevariation_counter = 0

for index, row in df_train.iterrows():
    if row['OptimalRadius'] == 0.0:
        row['OptimalRadius'] = train_mean_radius
        train_radius_counter += 1
    if row['OptimalKNN'] == 0.0:
        row['OptimalKNN'] = train_mean_knn
        train_knn_counter += 1
    if row['Linearity'] == 0.0:
        row['Linearity'] = train_mean_linearity
        train_linearity_counter += 1
    if row['Planarity'] == 0.0:
        row['Planarity'] = train_mean_planarity
        train_planarity_counter += 1
    if row['EigenvalueSum'] == 0.0:
        row['EigenvalueSum'] = train_mean_eigenvaluesum
        train_eigenvaluesum_counter += 1
    if row['Verticality'] == 0.0:
        row['Verticality'] = train_mean_verticality
        train_verticality_counter += 1
    if row['SurfaceVariation'] == 0.0:
        row['SurfaceVariation'] = train_mean_surfacevariation
        train_surfacevariation_counter += 1
print("------------------- \n0.0 values are replaced with mean values")

# normalize the columns
df_train['OptimalRadius'] = normalize(df_train[['OptimalRadius']], axis=0)
df_train['OptimalKNN'] = normalize(df_train[['OptimalKNN']], axis=0)
df_train['Linearity'] = normalize(df_train[['Linearity']], axis=0)
df_train['Planarity'] = normalize(df_train[['Planarity']], axis=0)
df_train['EigenvalueSum'] = normalize(df_train[['EigenvalueSum']], axis=0)
df_train['Verticality'] = normalize(df_train[['Verticality']], axis=0)
df_train['SurfaceVariation'] = normalize(df_train[['SurfaceVariation']], axis=0)
print("------------------- \nnormalization is applied")


# SEPERATE THE CLASSES AND BALANCE THE DATASET
df_train_0 = df_train[df_train.Classification == 0]
df_train_1 = df_train[df_train.Classification == 1]
df_train_2 = df_train[df_train.Classification == 2]
df_train_3 = df_train[df_train.Classification == 3]
df_train_4 = df_train[df_train.Classification == 4]
df_train_5 = df_train[df_train.Classification == 5]
df_train_6 = df_train[df_train.Classification == 6]
df_train_7 = df_train[df_train.Classification == 7]
df_train_8 = df_train[df_train.Classification == 8]
df_train_9 = df_train[df_train.Classification == 9]
df_train_10 = df_train[df_train.Classification == 10]
df_train_11 = df_train[df_train.Classification == 11]
df_train_12 = df_train[df_train.Classification == 12]
df_train_13 = df_train[df_train.Classification == 13]
df_train_14 = df_train[df_train.Classification == 14]

df_train_1 = resample(df_train_1,
             replace=True,
             n_samples=len(df_train_0),
             random_state=42)
df_train_2 = resample(df_train_2,
             replace=True,
             n_samples=len(df_train_0),
             random_state=42)
df_train_3 = resample(df_train_3,
             replace=True,
             n_samples=len(df_train_0),
             random_state=42)
df_train_4 = resample(df_train_4,
             replace=True,
             n_samples=len(df_train_0),
             random_state=42)
df_train_5 = resample(df_train_5,
             replace=True,
             n_samples=len(df_train_0),
             random_state=42)
df_train_6 = resample(df_train_6,
             replace=True,
             n_samples=len(df_train_0),
             random_state=42)
df_train_7 = resample(df_train_7,
             replace=True,
             n_samples=len(df_train_0),
             random_state=42)
df_train_8 = resample(df_train_8,
             replace=True,
             n_samples=len(df_train_0),
             random_state=42)
df_train_9 = resample(df_train_9,
             replace=True,
             n_samples=len(df_train_0),
             random_state=42)
df_train_10 = resample(df_train_10,
             replace=True,
             n_samples=len(df_train_0),
             random_state=42)
df_train_11 = resample(df_train_11,
             replace=True,
             n_samples=len(df_train_0),
             random_state=42)
df_train_12 = resample(df_train_12,
             replace=True,
             n_samples=len(df_train_0),
             random_state=42)
df_train_13 = resample(df_train_13,
             replace=True,
             n_samples=len(df_train_0),
             random_state=42)
df_train_14 = resample(df_train_14,
             replace=True,
             n_samples=len(df_train_0),
             random_state=42)

df_train = pd.concat([df_train_0, df_train_1, df_train_2, df_train_3,
                      df_train_4, df_train_5, df_train_6, df_train_7,
                      df_train_8, df_train_9, df_train_10, df_train_11,
                      df_train_12, df_train_13, df_train_14])


# CREATE THE TRAINING SET
labels = df_train.columns[4:]
X = df_train[labels]
y = df_train["Classification"]
X_train, X_val, y_train, y_val= train_test_split(X, y, test_size=0.2, random_state=1)


# XGBOOST
eval_set = [(X_val, y_val)]
xg = XGBClassifier()
if read_model:
    xg.load_model('../models/' + model_name)
else:
    start_time = time.time()
    xg.fit(X_train, y_train, early_stopping_rounds=10, eval_set=eval_set, verbose=True)
    print("--- %s seconds ---" % (time.time() - start_time))

y_pred = xg.predict(X_val)

print("Accuracy - Dependent Test XG: " + str(sk.metrics.accuracy_score(y_val, y_pred)))
# print("Precision: " + str(sk.metrics.precision_score(y_val, y_pred, average='macro')))
# print("Recall: " + str(sk.metrics.recall_score(y_val, y_pred, average='macro')))
# print("F1: " + str(sk.metrics.f1_score(y_val, y_pred, average='macro')))
# print("Confusion Matrix: \n" + str(sk.metrics.confusion_matrix(y_val, y_pred)))

xg.save_model('../models/xgboost_model.txt')


# RANDOM FOREST
rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_val)

print("Accuracy: - Dependent Test RF" + str(sk.metrics.accuracy_score(y_val, y_pred)))
# print("Precision: " + str(sk.metrics.precision_score(y_val, y_pred, average='macro')))
# print("Recall: " + str(sk.metrics.recall_score(y_val, y_pred, average='macro')))
# print("F1: " + str(sk.metrics.f1_score(y_val, y_pred, average='macro')))
# print("Confusion Matrix: \n" + str(sk.metrics.confusion_matrix(y_val, y_pred)))

joblib.dump(rf, '../models/random_forest_model.joblib')




# "X","Y","Z","Classification","Intensity","OptimalKNN","OptimalRadius","Linearity","Planarity","EigenvalueSum","Verticality","SurfaceVariation"
test_mean_radius = df_test.loc[:, 'OptimalRadius'].mean()
test_mean_knn = df_test.loc[:, 'OptimalKNN'].mean()
test_mean_linearity = df_test.loc[:, 'Linearity'].mean()
test_mean_planarity = df_test.loc[:, 'Planarity'].mean()
test_mean_eigenvaluesum = df_test.loc[:, 'EigenvalueSum'].mean()
test_mean_verticality = df_test.loc[:, 'Verticality'].mean()
test_mean_surfacevariation = df_test.loc[:, 'SurfaceVariation'].mean()

print("OptimalRadius Mean: " + str(test_mean_radius))
print("KNN Mean: " + str(test_mean_knn))
print("Linearity Mean: " + str(test_mean_linearity))
print("Planarity Mean: " + str(test_mean_planarity))
print("EigenvalueSum Mean: " + str(test_mean_eigenvaluesum))
print("Verticality Mean: " + str(test_mean_verticality))
print("SurfaceVariation Mean: " + str(test_mean_surfacevariation))

test_radius_counter = 0
test_knn_counter = 0
test_linearity_counter = 0
test_planarity_counter = 0
test_eigenvaluesum_counter = 0
test_verticality_counter = 0
test_surfacevariation_counter = 0

for index, row in df_test.iterrows():
    if row['OptimalRadius'] == 0.0:
        row['OptimalRadius'] = test_mean_radius
        test_radius_counter += 1
    if row['OptimalKNN'] == 0.0:
        row['OptimalKNN'] = test_mean_knn
        test_knn_counter += 1
    if row['Linearity'] == 0.0:
        row['Linearity'] = test_mean_linearity
        test_linearity_counter += 1
    if row['Planarity'] == 0.0:
        row['Planarity'] = test_mean_planarity
        test_planarity_counter += 1
    if row['EigenvalueSum'] == 0.0:
        row['EigenvalueSum'] = test_mean_eigenvaluesum
        test_eigenvaluesum_counter += 1
    if row['Verticality'] == 0.0:
        row['Verticality'] = test_mean_verticality
        test_verticality_counter += 1
    if row['SurfaceVariation'] == 0.0:
        row['SurfaceVariation'] = test_mean_surfacevariation
        test_surfacevariation_counter += 1
print("------------------- \n0.0 values are replaced with mean values")

# normalize the columns
df_test['OptimalRadius'] = normalize(df_test[['OptimalRadius']], axis=0)
df_test['OptimalKNN'] = normalize(df_test[['OptimalKNN']], axis=0)
df_test['Linearity'] = normalize(df_test[['Linearity']], axis=0)
df_test['Planarity'] = normalize(df_test[['Planarity']], axis=0)
df_test['EigenvalueSum'] = normalize(df_test[['EigenvalueSum']], axis=0)
df_test['Verticality'] = normalize(df_test[['Verticality']], axis=0)
df_test['SurfaceVariation'] = normalize(df_test[['SurfaceVariation']], axis=0)
print("------------------- \nnormalization is applied")

# create the test and validation data of the TEST AREA
labels_TESTAREA = df_test.columns[4:]
TESTAREA_test = df_test[labels_TESTAREA]
print(TESTAREA_test)
TESTAREA_val = df_test["Classification"]
print(TESTAREA_val)

# XGBOOST
y_pred_TESTAREA_xg = xg.predict(TESTAREA_test)

print("Accuracy - Independent Test XG: " + str(sk.metrics.accuracy_score(TESTAREA_val, y_pred_TESTAREA_xg)))
# print("Precision: " + str(sk.metrics.precision_score(TESTAREA_val, y_pred_TESTAREA_xg, average='macro')))
# print("Recall: " + str(sk.metrics.recall_score(TESTAREA_val, y_pred_TESTAREA_xg, average='macro')))
# print("F1: " + str(sk.metrics.f1_score(TESTAREA_val, y_pred_TESTAREA_xg, average='macro')))
# print("Confusion Matrix: \n" + str(sk.metrics.confusion_matrix(TESTAREA_val, y_pred_TESTAREA_xg)))

df_predicted_class_xg = pd.DataFrame({'Classification': y_pred_TESTAREA_xg})
df_test_result_xg = pd.concat([df_test, df_predicted_class_xg], axis=1)
df_test_result_xg.to_csv('../results/xg_predicted.csv', index=False)


# RANDOM FOREST
y_pred_TESTAREA_rf = rf.predict(TESTAREA_test)

print("Accuracy:  - Independent Test RF" + str(sk.metrics.accuracy_score(TESTAREA_val, y_pred_TESTAREA_rf)))
# print("Precision: " + str(sk.metrics.precision_score(TESTAREA_val, y_pred_TESTAREA_rf, average='macro')))
# print("Recall: " + str(sk.metrics.recall_score(TESTAREA_val, y_pred_TESTAREA_rf, average='macro')))
# print("F1: " + str(sk.metrics.f1_score(TESTAREA_val, y_pred_TESTAREA_rf, average='macro')))
# print("Confusion Matrix: \n" + str(sk.metrics.confusion_matrix(TESTAREA_val, y_pred_TESTAREA_rf)))

df_predicted_class_rf = pd.DataFrame({'Classification': y_pred_TESTAREA_rf})
df_test_result_rf = pd.concat([df_test, df_predicted_class_rf], axis=1)
df_test_result_rf.to_csv('../results/rf_predicted.csv', index=False)

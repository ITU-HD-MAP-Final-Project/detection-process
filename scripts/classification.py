import time
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# params
read_model = False
model_name = ''

df = pd.read_csv('../point_clouds/all_-1_1_generalized_train_ds02_optimalneighborhood_cleaned_featured.txt')
print("all_-1_1_generalized_train_ds02_optimalneighborhood_cleaned.txt")
print(df.head())

# "X","Y","Z","Classification","Intensity","OptimalKNN","OptimalRadius","Linearity","Planarity","EigenvalueSum","Verticality","SurfaceVariation"

mean_radius = df.loc[:, 'OptimalRadius'].mean()
mean_knn = df.loc[:, 'OptimalKNN'].mean()
mean_linearity = df.loc[:, 'Linearity'].mean()
mean_planarity = df.loc[:, 'Planarity'].mean()
mean_eigenvaluesum = df.loc[:, 'EigenvalueSum'].mean()
mean_verticality = df.loc[:, 'Verticality'].mean()
mean_surfacevariation = df.loc[:, 'SurfaceVariation'].mean()

print("OptimalRadius Mean: " + str(mean_radius))
print("KNN Mean: " + str(mean_knn))
print("Linearity Mean: " + str(mean_linearity))
print("Planarity Mean: " + str(mean_planarity))
print("EigenvalueSum Mean: " + str(mean_eigenvaluesum))
print("Verticality Mean: " + str(mean_verticality))
print("SurfaceVariation Mean: " + str(mean_surfacevariation))

radius_counter = 0
knn_counter = 0
linearity_counter = 0
planarity_counter = 0
eigenvaluesum_counter = 0
verticality_counter = 0
surfacevariation_counter = 0

for index, row in df.iterrows():
    if row['OptimalRadius'] == 0.0:
        row['OptimalRadius'] = mean_radius
        radius_counter += 1
    if row['OptimalKNN'] == 0.0:
        row['OptimalKNN'] = mean_knn
        knn_counter += 1
    if row['Linearity'] == 0.0:
        row['Linearity'] = mean_linearity
        linearity_counter += 1
    if row['Planarity'] == 0.0:
        row['Planarity'] = mean_planarity
        planarity_counter += 1
    if row['EigenvalueSum'] == 0.0:
        row['EigenvalueSum'] = mean_eigenvaluesum
        eigenvaluesum_counter += 1
    if row['Verticality'] == 0.0:
        row['Verticality'] = mean_verticality
        verticality_counter += 1
    if row['SurfaceVariation'] == 0.0:
        row['SurfaceVariation'] = mean_surfacevariation
        surfacevariation_counter += 1
print("------------------- \n0.0 values are replaced with mean values")

# normalize the columns
df['OptimalRadius'] = normalize(df[['OptimalRadius']], axis=0)
df['OptimalKNN'] = normalize(df[['OptimalKNN']], axis=0)
df['Linearity'] = normalize(df[['Linearity']], axis=0)
df['Planarity'] = normalize(df[['Planarity']], axis=0)
df['EigenvalueSum'] = normalize(df[['EigenvalueSum']], axis=0)
df['Verticality'] = normalize(df[['Verticality']], axis=0)
df['SurfaceVariation'] = normalize(df[['SurfaceVariation']], axis=0)
print("------------------- \nnormalization is applied")

labels = df.columns[4:]
X = df[labels]
print(X)
y = df["Classification"]
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

start_time = time.time()

model = XGBClassifier()
eval_set = [(X_test, y_test)]
if read_model:
    model.load_model('../models/' + model_name)
else:
    model.fit(X_train, y_train, early_stopping_rounds=10, verbose=True)

y_pred = model.predict(X_test)

print("Accuracy: " + str(sk.metrics.accuracy_score(y_test, y_pred)))
print("Precision: " + str(sk.metrics.precision_score(y_test, y_pred, average='macro')))
print("Recall: " + str(sk.metrics.recall_score(y_test, y_pred, average='macro')))
print("F1: " + str(sk.metrics.f1_score(y_test, y_pred, average='macro')))
print("Confusion Matrix: \n" + str(sk.metrics.confusion_matrix(y_test, y_pred)))

# print out the time it took to run the script
print("--- %s seconds ---" % (time.time() - start_time))

# save the model
model.save_model('../models/xgboost_model.txt')


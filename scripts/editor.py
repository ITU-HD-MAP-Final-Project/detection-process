import pandas as pd

df = pd.read_csv('../point_clouds/all_-1_1_train_ds02_optimalneighborhood.txt')

print(df.head())
print("-------------------")
print("OptimalRadius Mean: " + str(df.loc[:, 'OptimalRadius'].mean()))
print("OptimalRadius KNN: " + str(df.loc[:, 'OptimalKNN'].mean()))

mean_radius = df.loc[:, 'OptimalRadius'].mean()
mean_knn = df.loc[:, 'OptimalKNN'].mean()

radius_counter = 0
knn_counter = 0
for index, row in df.iterrows():
    if row['OptimalRadius'] == 0.0:
        row['OptimalRadius'] = mean_radius
        radius_counter += 1
    if row['OptimalKNN'] == 0.0:
        row['OptimalKNN'] = mean_knn
        knn_counter += 1
print("-------------------")
# print out the counters
print("OptimalRadius Counter: " + str(radius_counter))
print("OptimalKNN Counter: " + str(knn_counter))

df.to_csv("../point_clouds/all_-1_1_train_ds02_optimalneighborhood_cleaned.txt", index=False)



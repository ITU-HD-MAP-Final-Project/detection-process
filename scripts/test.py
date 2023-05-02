import pdal

# json = '''[
#     "../point_clouds/all_-1_1_train_ds02.txt",
#     {
#         "type":"filters.optimalneighborhood"
#     },
#     {
#         "type":"filters.covariancefeatures",
#         "knn": 8,
#         "threads": 10,
#         "optimized":true,
#         "feature_set": "Dimensionality"
#     },
#     {
#         "type":"writers.text",
#         "filename":"../point_clouds/all_-1_1_train_ds02_f.txt"
#     }
# ]'''

data = "/home/ataparlar/projects/fp/point_clouds/all_-1_1_train_ds03.txt"

pipeline = pdal.Reader(data).pipeline()
print(pipeline.execute())
print(pipeline.arrays)
# print(pipeline.log)

# pipeline_filter = pdal.Filter(type="uint64",
#                               filter="filters.optimalneighborhood",
#                               knn=8,
#                               threads=20,
#                               optimized=True,
#                               feature_set="Linearity").pipeline()

pipeline_filter = pdal.Filter(type="filters.optimalneighborhood").pipeline()
pipeline_filter.inputs = [pipeline.pipeline(), pipeline.arrays]
print(pipeline_filter.execute())
print(pipeline_filter.arrays)


# pipeline2 = pdal.Filter("filters.optimalneighborhood", pipeline=pipeline)


# pipeline = pdal.Pipeline(json)
# count = pipeline.execute()
# arrays = pipeline.arrays
# metadata = pipeline.metadata
# log = pipeline.log
# filter = pdal.Filter(json, verbose=True)
# filter.execute(
#     verbose=True
# )
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Python script to run benchmark on a query with a file path.
Usage:
 spark-submit als_model.py <train data> <val data>
'''


# Import command line arguments and helper functions
import sys
import pyspark
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
# Import the requisite packages
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator


def main(spark, file_path1, file_path2):
    
    parquetFile = spark.read.parquet(file_path1)
    parquetFile.createOrReplaceTempView("parquetFile")
    df1 = spark.sql('SELECT user_id, count(*) FROM parquetFile GROUP BY user_id HAVING count(*) >= 10')
    df1.createOrReplaceTempView("df1")
    train = spark.sql('SELECT * from parquetFile where user_id \
                          in (SELECT Distinct(user_id) from df1)')
    val = spark.read.parquet(file_path2)
    val.createOrReplaceTempView('val')
    
    parquetFile.createOrReplaceTempView("train")
    #ALS model only allows int values for userCol and ItemCol so creating int ids for each user_id and track_id
    user_stringIdx = StringIndexer(inputCol ="user_id", outputCol = "user_id_index")
    t_user_stringIdx_model = user_stringIdx.fit(train)
    item_stringIdx = StringIndexer(inputCol ="track_id", outputCol = "track_id_index")
    t_item_stringIdx_model = item_stringIdx.fit(train)
    train = t_user_stringIdx_model.transform(train)
    train = t_item_stringIdx_model.transform(train)
    train_ids = train.drop('user_id','track_id')
    v_user_stringIdx_model = user_stringIdx.fit(val)
    v_item_stringIdx_model = item_stringIdx.fit(val)
    val = v_user_stringIdx_model.transform(val)
    train = v_item_stringIdx_model.transform(val)
    train_ids = train.drop('user_id','track_id')
    val_ids = train.drop('user_id','track_id')
    #return train.limit(10).show()

    # Create ALS model
    als = ALS(
             userCol="user_id_index", 
             itemCol="track_id_index",
             ratingCol="count", 
             nonnegative = True, 
             implicitPrefs = True,
             coldStartStrategy="drop")

    # Add hyperparameters and their respective values to param_grid
    param_grid = ParamGridBuilder() \
                .addGrid(als.rank, [10, 50, 100, 150]) \
                .addGrid(als.maxIter, [5,10,20]) \
                .addGrid(als.regParam, [,.001, .01, .05, .1]) \
                .addGrid(als.alpha, [.001,.01,1]) \
                .build()
    # Define evaluator as RMSE and print length of evaluator
    evaluator = RegressionEvaluator(
               metricName="rmse", 
               labelCol="count", 
               predictionCol="prediction") 
    print ("Num models to be tested: ", len(param_grid))

    # Build cross validation using CrossValidator
    cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)


    #Fit cross validator to the 'train' dataset
    model = cv.fit(train_ids)
    #Extract best model from the cv model above
    best_model = model.bestModel


    print("**Best Model**")
    # Print "Rank"
    print("  Rank:", best_model._java_obj.parent().getRank())
    # Print "MaxIter"
    print("  MaxIter:", best_model._java_obj.parent().getMaxIter())
    # Print "RegParam"
    print("  RegParam:", best_model._java_obj.parent().getRegParam())

    # View the predictions
    test_predictions = best_model.transform(val_ids)
    RMSE = evaluator.evaluate(test_predictions)
    print(RMSE)

    # Generate n Recommendations for all users
    recommendations = best_model.recommendForAllUsers(5)
    # Converting int ids back to original character form
    user_labels = v_user_stringIdx_model.labels
    item_labels = v_item_stringIdx_model.labels
    converter = IndexToString(inputCol="user_id_index", outputCol="user_id",labels = user_labels)
    converted = converter.transform(recommendations)
    n = converted.count()
    item_labels_ = array(*[lit(x) for x in item_labels])

    recs= array(*[struct(
        item_labels_[col("recommendations")[i]["track_id_index"]].alias("trackId"),
        col("recommendations")[i]["rating"].alias("rating")
            ) for i in range(n)])

    converted = converted.withColumn("recommendations", recs)
    converted = converted.select('user_id','recommendations.trackId')


    return converted.limit(10).show()

def get_mat_sparsity(ratings):
    # Count the total number of ratings in the dataset
    count_nonzero = ratings.select("count").count()

    # Count the number of distinct userIds and distinct trackIds
    total_elements = ratings.select("user_id").distinct().count() * ratings.select("track_id").distinct().count()

    # Divide the numerator by the denominator
    sparsity = (1.0 - (count_nonzero *1.0)/total_elements)*100
    print("The ratings dataframe is ", "%.2f" % sparsity + "% sparse.")
    
# get_mat_sparsity(ratings)

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.config("spark.executor.memory", "4g").config("spark.driver.memory", "15g").appName('part2').getOrCreate()

    # Get file_path for dataset to analyze
    file_path1 = sys.argv[1]
    file_path2 = sys.argv[2]

    main(spark, file_path1, file_path2)

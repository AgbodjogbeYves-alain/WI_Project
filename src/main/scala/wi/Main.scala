package wi

import breeze.linalg.max
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

object Main extends App {
  val file = "data-students.json" // Should be some file on your system
  val spark = SparkSession.builder.appName("Wi_App").config("spark.master", "local").getOrCreate()
  val dfJson = spark.read.json(file)
  dfJson.printSchema()
  val nDataFrame = Cleaner.timestamp(dfJson)
  val nnDataFrame = Cleaner.impid(nDataFrame)

  val assembler = new VectorAssembler().setInputCols(Array("os")).setOutputCol("features")
  //val training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

  val lr = new LogisticRegression()
    .setMaxIter(10)
    .setRegParam(0.3)
    .setElasticNetParam(0.8)
    .setFeaturesCol("features")


  val output = assembler.transform(nnDataFrame)

  //Fit the model
  //val lrModel = lr.fit(output)
  val lrModel = lr.fit(nnDataFrame)


  // Print the coefficients and intercept for logistic regression
  println(s"Coefficients: ${lrModel.coefficients} intercept: ${lrModel.intercept}")

  //We can also use the multinomial family for binary classification
 /* val mlr = new LogisticRegression()
    .setMaxIter(10)
    .setRegParam(0.3)
    .setElasticNetParam(0.8)
    .setFamily("multinomial")

  val mlrModel = mlr.fit(dfJson)*/

  // Print the coefficients and intercepts for logistic regression with multinomial family
 /* println(s"Multinomial coefficients: ${mlrModel.coefficientMatrix}")
  println(s"Multinomial intercepts: ${mlrModel.interceptVector}")*/

  // Extract the summary from the returned LogisticRegressionModel instance trained in the earlier
  val trainingSummary = lrModel.binarySummary

  // Obtain the objective per iteration.
  val objectiveHistory = trainingSummary.objectiveHistory
  println("objectiveHistory:")
  objectiveHistory.foreach(loss => println(loss))

  // Obtain the metrics useful to judge performance on test data.
  // We cast the summary to a BinaryLogisticRegressionSummary since the problem is a
  // binary classification problem.
  //val binarySummary = trainingSummary.asInstanceOf[BinaryLogisticRegressionSummary]
  // Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
  /*val roc = trainingSummary.roc
  roc.show()
  println(s"areaUnderROC: ${trainingSummary.areaUnderROC}")

  // Set the model threshold to maximize F-Measure
  val fMeasure = trainingSummary.fMeasureByThreshold
  val maxFMeasure = fMeasure.select(max("F-Measure")).head().getDouble(0)

  val bestThreshold = fMeasure.where($"F-Measure" === maxFMeasure)
    .select("threshold").head().getDouble(0)
  lrModel.setThreshold(bestThreshold)*/

  spark.stop()
}
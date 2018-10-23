package wi

import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.classification.LogisticRegression

object Main extends App {
  val file = "data-students.json" // Should be some file on your system
  

  val spark = SparkSession.builder.appName("Wi_App").config("spark.master", "local").getOrCreate()
  val dfJson = spark.read.json(file)
  val cleanOs = Cleaner.os(dfJson)
  val splits = Cleaner.label(cleanOs).randomSplit(Array(0.6, 0.4),11L)
  val training = splits(0).cache()
  val test = splits(1)
  val selectTraining = training.select("label","os")

  val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3)//.setElasticNetParam(0.8)
  val assembler = new VectorAssembler().setInputCols(Array("os")).setOutputCol("features")
  val output = assembler.transform(selectTraining)
  // Fit the model
  val lrModel = lr.fit(output)

  println(lrModel)
  // Print the coefficients and intercept for logistic regression
  println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
  //training.show()
  spark.stop()

  
}
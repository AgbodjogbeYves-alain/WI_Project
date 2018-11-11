package wi

import breeze.linalg.max
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.GeneralizedLinearRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

import scala.io.StdIn

object Main extends App {
  val file = "data-students.json" // Should be some file on your system
  val spark = SparkSession.builder.appName("Wi_App").config("spark.master", "local").getOrCreate()
  val dfJson = spark.read.json(file)
  val dfCleaned = Cleaner.prepareDF(dfJson)
  dfCleaned.show()
  val splits = dfCleaned.randomSplit(Array(0.7, 0.3))
  var (trainingData, testData) = (splits(0), splits(1))
  trainingData = trainingData.select("features", "label")
  testData = testData.select("features", "label")

  val lr = new LogisticRegression()
    .setFeaturesCol("features")
    .setLabelCol("label")
    .setRegParam(0.0)
    .setElasticNetParam(0.0)
    .setMaxIter(10)
    .setTol(1E-6)
    .setFitIntercept(true)

  val model = lr.fit(trainingData)
  println(s"-------------------------Coefficients: ${model.coefficients}")
  println(s"-------------------------Intercept: ${model.interceptVector}")

  //model.write.save("target/tmp/WILogisticRegression")
  
  val prediction = model.transform(testData)
  prediction.printSchema()
  prediction.select ("label", "prediction","rawPrediction").show()

  val evaluator = new BinaryClassificationEvaluator()
    .setMetricName("areaUnderROC")
    .setRawPredictionCol("rawPrediction")
    .setLabelCol("label")

  val eval = evaluator.evaluate(prediction)
  println("Test set areaunderROC/accuracy = " + eval)

  prediction.select("label", "prediction").write.format("csv").option("header","true").save("target/tmp/Resulst")
  spark.stop()
}
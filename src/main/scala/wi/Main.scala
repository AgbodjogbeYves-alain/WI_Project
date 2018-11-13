package wi

import breeze.linalg.max
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.log4j.Logger

import scala.io.StdIn

object Main extends App {
  val file = "data-students.json" // File data-student to train
  //Logger.getLogger("org").setLevel(Level.ERROR)
  val spark = SparkSession.builder.appName("Wi_App").config("spark.master", "local").getOrCreate() //Create spark session
  val dfJson = spark.read.json(file) //get dataFrame from the file
  val dfCleaned = Cleaner.prepareDF(dfJson) //Clean dataFrame
  dfCleaned.show()
  val splits = dfCleaned.randomSplit(Array(0.7, 0.3)) //split the dataFrame into training and test part
  var (trainingData, testData) = (splits(0), splits(1))
  trainingData = trainingData.select("features", "label") //DataFrame to train with only label and features columnn
  testData = testData.select("features", "label") //DataFrame to test with only label and features columnn

  //Create LogisticRegression 
  val lr = new LogisticRegression()
    .setFeaturesCol("features")
    .setLabelCol("label")
    .setRegParam(0.0)
    .setElasticNetParam(0.0)
    .setMaxIter(10)
    .setTol(1E-6)
    .setFitIntercept(true)

  //Create RandomForrest
  val rf = new RandomForestClassifier()
    .setFeaturesCol("features")
    .setLabelCol("label")

  //Create both model with the trianing dataFrame
  val model = lr.fit(trainingData)
  val rfModel = rf.fit(trainingData)

  println(s"-------------------------Coefficients: ${model.coefficients}")
  println(s"-------------------------Intercept: ${model.interceptVector}")

  //Save the models 
  //model.write.save("target/tmp/WILogisticRegression")
  //rfModel.write.save("target/tmp/WIRandomForrest")

  //Use the model to predict the label column
  val prediction = model.transform(testData)
  val rfPrediction = rfModel.transform(testData)

  prediction.select ("label", "prediction").show()
  rfPrediction.select("label", "prediction").show()

  //Create the evaluator to evaluate the prediction
  val evaluator = new BinaryClassificationEvaluator()
    .setMetricName("areaUnderROC")
    .setRawPredictionCol("rawPrediction")
    .setLabelCol("label")

  //Evaluate the prediction with area under ROC
  val accuracy = evaluator.evaluate(rfPrediction)
  val eval = evaluator.evaluate(prediction)
  println("Test set areaunderROC/accuracy = " + eval)
  println("Test set Random Forest accuracy = " + accuracy)

  //Save the prediction into a csv
  //prediction.select("label", "prediction").write.format("csv").option("header","true").save("target/tmp/Resulst")
  spark.stop()
}
package wi

import breeze.linalg.max
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.log4j.Logger
import org.apache.log4j.Level

import scala.io.StdIn

object Main extends App {

  /**
    Function to create and save the models
  **/
  def createModel(): Unit = {
    val file = "data-students.json" // File data-student to train
    Logger.getLogger("org").setLevel(Level.ERROR) //Remove all the INFO prompt
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
    model.write.save("target/tmp/WILogisticRegression")
    rfModel.write.save("target/tmp/WIRandomForrest")
    spark.stop()
  }

  /**
    Function to load and use the models, then save the result as a CSV
  **/
  def useModel(path: String): Unit = {

    val dfJson = spark.read.json(path) //get dataFrame from the file

    Logger.getLogger("org").setLevel(Level.ERROR) //Remove all the INFO prompt

    //Load the models
    val model = LogisticRegressionModel.load("target/tmp/WILogisticRegression")
    val rfModel = RandomForestClassificationModel.load("target/tmp/WIRandomForrest")
    
    //Use the model to predict the label column
    val prediction = model.transform(testData)
    val rfPrediction = rfModel.transform(testData)

    prediction.select ("label", "prediction", "rawPrediction").show()
    rfPrediction.select("label", "prediction", "rawPrediction").show()

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
    prediction.select("label", "prediction").write.format("csv").option("header","true").save("target/tmp/ResultLR")
    rfPrediction.select("label", "prediction").write.format("csv").option("header","true").save("target/tmp/ResulstRF")
  }








  val file = "data-students.json" // File data-student to train
  Logger.getLogger("org").setLevel(Level.ERROR) //Remove all the INFO prompt
  val spark = SparkSession.builder.appName("Wi_App").config("spark.master", "local").getOrCreate() //Create spark session
  val dfJson = spark.read.json(file) //get dataFrame from the file
  val dfCleaned = Cleaner.prepareDF(dfJson) //Clean dataFrame
  dfCleaned.show()
  val splits = dfCleaned.randomSplit(Array(0.7, 0.3)) //split the dataFrame into training and test part
  var (trainingData, testData) = (splits(0), splits(1))

  //Create RandomForrest
  val rf = new RandomForestClassifier()
    .setFeaturesCol("features")
    .setLabelCol("label")

  //Create both model with the trianing dataFrame
  val rfModel = rf.fit(trainingData)

  //Save the models 
  //model.write.save("target/tmp/WILogisticRegression")
  //rfModel.write.save("target/tmp/WIRandomForrest")

  //Use the model to predict the label column
  val rfPrediction = rfModel.transform(testData)

  rfPrediction.select("label", "prediction", "rawPrediction").show()

  //Create the evaluator to evaluate the prediction
  val evaluator = new BinaryClassificationEvaluator()
    .setMetricName("areaUnderROC")
    .setRawPredictionCol("rawPrediction")
    .setLabelCol("label")

  //Evaluate the prediction with area under ROC
  val accuracy = evaluator.evaluate(rfPrediction)
  println("Test set Random Forest accuracy = " + accuracy)

  //Prepare the dataframe to be save as a CSV
  val predictionWithID = rfPrediction.withColumn("rowId1", monotonically_increasing_id())
  val testDataWithID = testData.withColumn("rowId2", monotonically_increasing_id())
  val predictedLabel = predictionWithID.select("prediction","rowId1")
  val lastDF = predictedLabel.join(testDataWithID, predictedLabel("rowId1")===testDataWithID("rowId2"))
  val dFforCSV = lastDF.drop("rowId1").drop("rowId2").drop("features")
  val ndFforCSV = Cleaner.prediction(dFforCSV)
  val finalDF = ndFforCSV.withColumnRenamed("prediction", "Label")
  finalDF.show()

  //Save the prediction into a csv
  //prediction.select("label", "prediction").write.format("csv").option("header","true").save("target/tmp/ResultLR")
  //dFforCSV.write.format("csv").option("header","true").save("target/tmp/ResultRF")

  //Load the models
  //val lrModel = LogisticRegressionModel.load("target/tmp/WILogisticRegression")
  //val rfModel = RandomForestClassificationModel.load("target/tmp/WIRandomForrest")

  spark.stop()
}
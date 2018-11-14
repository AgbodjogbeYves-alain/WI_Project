package wi

import breeze.linalg.max
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
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
    println("Loading data")
    val dfJson = spark.read.json(file) //get dataFrame from the file
    println("Preparing data")
    val dfCleaned = Cleaner.prepareDF(dfJson) //Clean dataFrame
    println("Creating the model")

    //split the dataFrame into training and test part
    val splits = dfCleaned.randomSplit(Array(0.7, 0.3)) 
    var (trainingData, testData) = (splits(0), splits(1))

    //Create RandomForrest
    val rf = new RandomForestClassifier()
      .setFeaturesCol("features")
      .setLabelCol("label")

    //Create model with the trianing dataFrame
    val rfModel = rf.fit(trainingData)
    println("Model created")

    //Save the model
    rfModel.write.save("model/WIRandomForestModel")
    println("Model saved")

    spark.stop()
  }

  /**
    Function to load and use the models, then save the result as a CSV
  **/
  def useModel(path: String): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR) //Remove all the INFO prompt
    val spark = SparkSession.builder.appName("Wi_App").config("spark.master", "local").getOrCreate() //Create spark session
    
    println("Loading data")
    val dfJson = spark.read.json(path) //get dataFrame from the file

    println("Preparing data")
    val dfCleaned = Cleaner.prepareDF(dfJson) //Clean dataFrame

    println("Loading the model")
    //Load the model
    val rfModel = RandomForestClassificationModel.load("model/WIRandomForrestModel")
    
    println("Predicting value with the model")
    //Use the model to predict the label column
    val rfPrediction = rfModel.transform(dfCleaned)

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
    val testDataWithID = dfCleaned.withColumn("rowId2", monotonically_increasing_id())
    val predictedLabel = predictionWithID.select("prediction","rowId1")
    val lastDF = predictedLabel.join(testDataWithID, predictedLabel("rowId1")===testDataWithID("rowId2"))
    val dFforCSV = lastDF.drop("rowId1").drop("rowId2").drop("features")
    val ndFforCSV = Cleaner.prediction(dFforCSV)
    //val finalDF = ndFforCSV.withColumnRenamed("prediction", "Label")
    val finalDF = Cleaner.sizeforCSV(ndFforCSV)

    finalDF.printSchema()
    finalDF.show()
    println("Saving the results as a CSV")
    //Save the prediction into a csv
    finalDF.write.format("csv").option("header","true").save("target/tmp/ResultRF")

    println("CSV available in target/tmp/ResultsRF")
    spark.stop()
  }

  if(args.isEmpty){
    println("You must enter the path of the file to predict in parameter")
  }else{
    val param = args
    useModel(param(0))
  }

}
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
  val nDataFrame = Cleaner.timestamp(dfJson)
  val nnDataFrame = Cleaner.impid(nDataFrame)



  spark.stop()
}
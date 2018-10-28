package wi

import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils

object Main extends App {
  val file = "data-students.json" // Should be some file on your system
  val spark = SparkSession.builder.appName("Wi_App").config("spark.master", "local").getOrCreate()
  val dfJson = spark.read.json(file)
  val df = Cleaner.prepareDF(dfJson)
  df.show()
  spark.stop()
}
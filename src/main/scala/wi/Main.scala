package wi

import org.apache.spark.sql.SparkSession

object Main extends App {
  val file = "data-students.json" // Should be some file on your system
  val spark = SparkSession.builder.appName("Wi_App").config("spark.master", "local").getOrCreate()
  val dfJson = spark.read.json(file)
  dfJson.printSchema()
  val nDataFrame = Cleaner.timestamp(dfJson)
  val nnDataFrame = Cleaner.impid(nDataFrame)
  spark.stop()
}
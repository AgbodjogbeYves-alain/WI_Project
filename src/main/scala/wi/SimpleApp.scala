package wi

import org.apache.spark.sql.SparkSession

object SimpleApp extends App {
  val file = "data-students.json" // Should be some file on your system
  val spark = SparkSession.builder.appName("Simple Application").config("spark.master", "local").getOrCreate()
  val logData = spark.read.json(file).cache()
  logData.printSchema()
  spark.stop()
}
package example

import org.apache.spark.sql.SparkSession

object SimpleApp extends App {
  val file = "data-students.json" // Should be some file on your system
  val spark = SparkSession.builder.appName("Simple Application").getOrCreate()
  val logData = spark.read.json(file).cache()
  println(logData.flatMap(line => line.split(",")))
  spark.stop()
}
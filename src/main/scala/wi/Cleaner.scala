package wi

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.udf
import scala.collection.immutable.HashMap
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

object Cleaner {
  /**
    Unify os attribute 
    @param dataFrame DataFrame to change
  */
  def os(dataFrame: DataFrame): DataFrame = {
    val updater = udf((col: String) => {
      col match{
        case "android"|"Android" => "android"
        case "iOS" | "ios" => "iOS"
        case "Windows Phone OS" | "WindowsPhone" | "Windows Mobile OS" | "WindowsMobile" => "windows phone"
        case "null" => "null"
        case "other" | "Unknown" => "unknown"
        case "blackberry"| "Rim" => "blackberry"
        case "WebOS" => "webOS"
        case "Symbian" => "symbian"
        case "Bada" => "bada"
        case "windows" => "windows"
        case _ => "unknown"
      }
    })
    dataFrame.withColumn("os", updater(dataFrame("os")))
  }
  /**
    Return the most important interest for an user
    @param dataFrame DataFrame to change 
  */
  def interests(dataFrame: DataFrame): DataFrame = {
    val replacer = udf((col: String) => {
      val values = col.split(",")
      if (values.nonEmpty) values(0) else "null"
    })
    dataFrame.withColumn(Column.INTERESTS.toString, replacer(dataFrame(Column.INTERESTS.toString)))
  }

  /**
    Return a new dataframe without the timestamp column
    @param dataFrame DataFrame to change 
  */
  def timestamp(dataFrame: DataFrame): DataFrame = {
    val nDataFrame = dataFrame.drop("timestamp")
    return(nDataFrame)
  }

  /**
    Return a new dataframe without the impid column
    @param dataFrame DataFrame to change 
  */
  def impid(dataFrame: DataFrame): DataFrame = {
    val nDataFrame = dataFrame.drop("impid")
    return(nDataFrame)
  }

  /**
    Return a new dataframe with the given column filled with the given value
    @param dataFrame DataFrame to change 
    @param columnName name of the columns to change
    @param value String with which to fill
  */
  def fillWithString(dataFrame: DataFrame, columnName : String, value : String): DataFrame = {
    val ndf = dataFrame.na.fill(value, Seq(columnName))
    return ndf
  }

  def indextype(dataFrame: DataFrame): DataFrame = {
    val ndf = fillWithString(dataFrame, "type", "null")
    val indexer = new StringIndexer()
      .setInputCol("type")
      .setOutputCol("indextype")
    val indexed = indexer.fit(ndf).transform(ndf)
    indexed.drop("type")
    return indexed
  }
  
}
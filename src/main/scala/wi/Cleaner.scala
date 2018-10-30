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
    Return a new dataframe with the given column filled with the given String value
    @param dataFrame DataFrame to change 
    @param columnName name of the columns to change
    @param value String with which to fill
  */
  def fillWithString(dataFrame: DataFrame, columnName : String, value : String): DataFrame = {
    val ndf = dataFrame.na.fill(value, Seq(columnName))
    return ndf
  }

  /**
    Return a new dataframe with the given column filled with the given Double value
    @param dataFrame DataFrame to change 
    @param columnName name of the columns to change
    @param value Double with which to fill
  */
  def fillWithDouble(dataFrame: DataFrame, columnName : String, value : Double): DataFrame = {
    val ndf = dataFrame.na.fill(value, Seq(columnName))
    return ndf
  }

  /**
    Return a new dataframe with the given column filled with the given Boolean value
    @param dataFrame DataFrame to change 
    @param columnName name of the columns to change
    @param value Boolean with which to fill
  */
  def fillWithBoolean(dataFrame: DataFrame, columnName : String, value : Boolean): DataFrame = {
    val ndf = dataFrame.na.fill(value, Seq(columnName))
    return ndf
  }

  /**
    Return a new dataframe with indexed field
    @param dataFrame DataFrame to change 
    @param column name of the column to index
  */
  def toIndex(dataFrame: DataFrame, column: String): DataFrame = {
    val indexer = new StringIndexer()
      .setInputCol(column)
      .setOutputCol("index" + column)
    val indexed = indexer.fit(dataFrame).transform(dataFrame)
    val nindexed = indexed.drop(column)
    return nindexed
  }

  def toVector(dataFrame: DataFrame, column: String): DataFrame = {
    new VectorAssembler()
      .setInputCols(Array(column))
      .setOutputCol("vector" + column)
      .transform(dataFrame)
  }

  def fillDF(dataFrame: DataFrame): DataFrame = {
    var filldf = dataFrame
    filldf = fillWithString(filldf, Column.APP_OR_SITE.toString, "null")
    filldf = fillWithDouble(filldf, Column.BID_FLOOR.toString, 0.0)
    filldf = fillWithString(filldf, Column.CITY.toString, "null")
    filldf = fillWithString(filldf, Column.EXCHANGE.toString, "null")
    filldf = fillWithString(filldf, Column.INTERESTS.toString, "null")
    filldf = fillWithBoolean(filldf, Column.LABEL.toString, false)
    filldf = fillWithString(filldf, Column.MEDIA.toString, "null")
    filldf = fillWithString(filldf, Column.NETWORK.toString, "null")
    filldf = fillWithString(filldf, Column.OS.toString, "null")
    filldf = os(filldf)
    filldf = fillWithString(filldf, Column.PUBLISHER.toString, "null")
    filldf = fillWithString(filldf, Column.TYPE.toString, "null")
    filldf = fillWithString(filldf, Column.USER.toString, "null")
    return filldf
  }
  
  def stringIndexerDF(dataFrame: DataFrame): DataFrame = {
    var indexedDF = dataFrame
    indexedDF = toIndex(indexedDF, Column.APP_OR_SITE.toString)
    indexedDF = toIndex(indexedDF, Column.CITY.toString)
    indexedDF = toIndex(indexedDF, Column.EXCHANGE.toString)
    indexedDF = toIndex(indexedDF, Column.MEDIA.toString)
    indexedDF = toIndex(indexedDF, Column.NETWORK.toString)
    indexedDF = toIndex(indexedDF, Column.OS.toString)
    indexedDF = toIndex(indexedDF, Column.PUBLISHER.toString)
    indexedDF = toIndex(indexedDF, Column.TYPE.toString)
    return indexedDF
  }

  def prepareDF(dataFrame: DataFrame): DataFrame ={
    var ndf = dataFrame
    ndf = fillDF(ndf)
    ndf = stringIndexerDF(ndf)
    ndf = impid(ndf)
    ndf = interests(ndf)
    return ndf
  }
}
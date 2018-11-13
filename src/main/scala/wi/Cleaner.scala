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
    transform boolean to int
    @param b Boolean to change
  */
  def boolToInt(b:Boolean) = if(b) 1 else 0
  val boolToInt_udf = udf(boolToInt _)
  /**
    transform label attribute to int
    @param dataFrame DataFrame to change
  */
  def label(dataFrame: DataFrame): DataFrame = {
    dataFrame.withColumn(Column.LABEL.toString, boolToInt_udf(dataFrame(Column.LABEL.toString)))
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
    Return the bidfloor of the user with 2 decimals
    @param dataFrame DataFrame to change 
  */
  def bidFloor(dataFrame: DataFrame): DataFrame = {
    val replacer = udf((col: Double) => {
        BigDecimal(col).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
    })
    dataFrame.withColumn(Column.BID_FLOOR.toString, replacer(dataFrame(Column.BID_FLOOR.toString)))
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
    Return a new dataframe with the given column element rounded to the nearest int
    @param dataFrame DataFrame to change 
    @param columnName name of the columns to change
  */
  def roundDouble(dataFrame: DataFrame, columnName : String): DataFrame = {
    val replacer = udf((col: Int) => {
        col.toInt.toDouble
    })
    dataFrame.withColumn(columnName, replacer(dataFrame(columnName)))
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
    val renamedindexed = nindexed.withColumnRenamed("index" + column, column)
    return renamedindexed
  }


  /**
    Return a new dataframe where VectorAssembler 
    @param dataFrame DataFrame to change 
  */
  def toVector(dataFrame: DataFrame): DataFrame = {
    val assembleur = new VectorAssembler()
      .setInputCols(Array(Column.APP_OR_SITE.toString, Column.BID_FLOOR.toString, Column.EXCHANGE.toString, Column.MEDIA.toString, Column.OS.toString, Column.TYPE.toString))
      .setOutputCol("features")
    val assembled = assembleur.transform(dataFrame)
    return assembled
  }

  /**
    Return a new dataframe with all the column filled with corresponding value
    @param dataFrame DataFrame to change 
  */
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
  
  /**
    Return a new dataframe with column string indexed
    @param dataFrame DataFrame to change 
  */
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
    indexedDF = toIndex(indexedDF, Column.INTERESTS.toString)
    return indexedDF
  }

  /**
    Return a new dataframe with all the prepare method to prepare the data
    @param dataFrame DataFrame to change 
  */
  def prepareDF(dataFrame: DataFrame): DataFrame ={
    var ndf = dataFrame
    ndf = fillDF(ndf)
    ndf = impid(ndf)
    ndf = interests(ndf)
    ndf = label(ndf)
    ndf = stringIndexerDF(ndf)
    ndf = toVector(ndf)
    return ndf
  }
}
package wi

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.udf
import scala.collection.immutable.HashMap

object Cleaner {
  /**
    Unify os attribute 
    @param dataFrame DataFrame to change
  */
  def os(dataFrame: DataFrame): DataFrame = {
    val osMap = HashMap(
      "android"->OS.ANDROID,
      "iOS"->OS.IOS,
      "Windows Phone OS"->OS.WINDOWS_PHONE,
      "null"->OS.NULL,
      "other"->OS.UNKNOWN,
      "Unknown"->OS.UNKNOWN,
      "blackberry"->OS.BLACKBERRY,
      "WebOS"->OS.WEBOS,
      "WindowsPhone"->OS.WINDOWS_PHONE,
      "Windows Mobile OS"->OS.WINDOWS_PHONE,
      "WindowsMobile"->OS.WINDOWS_PHONE,
      "Android"->OS.ANDROID,
      "Symbian"->OS.SYMBIAN,
      "Rim"->OS.BLACKBERRY,
      "ios"->OS.IOS,
      "Bada"->OS.BADA,
      "windows"->OS.WINDOWS
    )
    val updater = udf((col: String) => osMap(col).toString)
    dataFrame.withColumn(Column.OS.toString, updater(dataFrame(Column.OS.toString)))
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
}
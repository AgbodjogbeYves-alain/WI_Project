package wi

object Column extends Enumeration {
  val APP_OR_SITE: Column.Value = Value("appOrSite")
  val BID_FLOOR: Column.Value = Value("bidfloor")
  val CITY: Column.Value = Value("city")
  val EXCHANGE: Column.Value = Value("exchange")
  val IMPRESSION_ID: Column.Value = Value("impid")
  val INTERESTS: Column.Value = Value("interests")
  val LABEL: Column.Value = Value("label")
  val MEDIA: Column.Value = Value("media")
  val NETWORK: Column.Value = Value("network")
  val OS: Column.Value = Value("os")
  val PUBLISHER: Column.Value = Value("publisher")
  val SIZE: Column.Value = Value("size")
  val TIMESTAMP: Column.Value = Value("timestamp")
  val TYPE: Column.Value = Value("type")
  val USER: Column.Value = Value("user")
}
import Dependencies._

val sparkVersion = "2.3.0"

lazy val root = (project in file(".")).
  settings(
    inThisBuild(List(
      organization := "com.example",
      scalaVersion := "2.12.7",
      version      := "0.1.0-SNAPSHOT"
    )),
    name := "wi_project",
    libraryDependencies ++= Seq(
      scalaTest % Test,
        "org.apache.spark" %% "spark-core",
        "org.apache.spark" %% "spark-sql",
        "org.apache.spark" %% "spark-mllib"
    )
  )

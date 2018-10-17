import Dependencies._

lazy val root = (project in file(".")).
  settings(
    inThisBuild(List(
      organization := "com.example",
      scalaVersion := "2.12.7",
      version      := "0.1.0-SNAPSHOT"
    )),
    name := "wi_project",
    libraryDependencies += scalaTest % Test
    libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.3.2"
  )
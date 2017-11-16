/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

import ReleaseTransformations._
import sbtrelease.Vcs

import scala.sys.process.Process

scalaVersion in ThisBuild := "2.12.4"
crossScalaVersions in ThisBuild := Seq("2.11.11", "2.12.4")

organization in ThisBuild := "org.platanios"

// In order to update the snapshots more frequently, the Coursier "Time-To-Live" (TTL) option can be modified. This can
// be done by modifying the "COURSIER_TTL" environment variable. Its value is parsed using
// 'scala.concurrent.duration.Duration', so that things like "24 hours", "5 min", "10s", or "0s", are fine, and it also
// accepts infinity ("Inf") as a duration. It defaults to 24 hours, meaning that the snapshot artifacts are updated
// every 24 hours.
resolvers in ThisBuild += Resolver.sonatypeRepo("snapshots")

val tensorFlowForScalaVersion = "0.1.0-SNAPSHOT"

autoCompilerPlugins in ThisBuild := true

scalacOptions in ThisBuild ++= Seq(
  "-deprecation",
  "-encoding", "UTF-8",
  "-feature",
  "-language:existentials",
  "-language:higherKinds",
  "-language:implicitConversions",
  "-unchecked",
  // "-Xfatal-warnings",
  // "-Xlog-implicits",
  "-Yno-adapted-args",
  // "-Ywarn-dead-code",
  // "-Ywarn-numeric-widen",
  // "-Ywarn-value-discard",
  "-Xfuture",
  "-P:splain:all",
  "-P:splain:infix",
  "-P:splain:foundreq",
  "-P:splain:implicits",
  "-P:splain:color",
  "-P:splain:tree"
  // "-P:splain:boundsimplicits:false"
)

lazy val loggingSettings = Seq(
  libraryDependencies ++= Seq(
    "com.typesafe.scala-logging" %% "scala-logging"   % "3.7.2",
    "ch.qos.logback"             %  "logback-classic" % "1.2.3")
)

lazy val commonSettings = loggingSettings ++ Seq(
  // Plugin that prints better implicit resolution errors.
  addCompilerPlugin("io.tryp"  % "splain" % "0.2.7" cross CrossVersion.patch)
)

lazy val testSettings = Seq(
  libraryDependencies ++= Seq(
    "junit"         %  "junit" %   "4.12",
    "org.scalactic" %% "scalactic" % "3.0.4",
    "org.scalatest" %% "scalatest" % "3.0.4" % "test"),
  logBuffered in Test := false,
  fork in test := false,
  testForkedParallel in Test := false,
  parallelExecution in Test := false,
  testOptions in Test += Tests.Argument(TestFrameworks.ScalaTest, "-oDF")
)

lazy val mt = (project in file("./mt"))
    .settings(moduleName := "symphony-mt", name := "Symphony Machine Translation")
    .settings(commonSettings)
    .settings(publishSettings)
    .settings(testSettings)
    .settings(
      libraryDependencies ++= Seq(
        "org.platanios" %% "tensorflow" % tensorFlowForScalaVersion, // classifier "darwin-cpu-x86_64"
        "org.platanios" %% "tensorflow-data" % tensorFlowForScalaVersion
      )
    )

lazy val noPublishSettings = Seq(
  publish := Unit,
  publishLocal := Unit,
  publishArtifact := false,
  skip in publish := true,
  releaseProcess := Nil
)

val deletedPublishedSnapshots = taskKey[Unit]("Delete published snapshots.")

lazy val publishSettings = Seq(
  publishArtifact := true,
  homepage := Some(url("https://github.com/eaplatanios/tensorflow_scala")),
  licenses := Seq("Apache License 2.0" -> url("https://www.apache.org/licenses/LICENSE-2.0.txt")),
  scmInfo := Some(ScmInfo(url("https://github.com/eaplatanios/symphony-mt"),
                          "scm:git:git@github.com:eaplatanios/symphony-mt.git")),
  developers := List(
    Developer(
      id="eaplatanios",
      name="Emmanouil Antonios Platanios",
      email="e.a.platanios@gmail.com",
      url=url("http://platanios.org/"))
  ),
  autoAPIMappings := true,
  apiURL := Some(url("http://eaplatanios.github.io/symphony-mt/api/")),
  releaseCrossBuild := true,
  releaseTagName := {
    val buildVersionValue = (version in ThisBuild).value
    val versionValue = version.value
    s"v${if (releaseUseGlobalVersion.value) buildVersionValue else versionValue}"
  },
  releaseVersionBump := sbtrelease.Version.Bump.Next,
  releaseUseGlobalVersion := true,
  releasePublishArtifactsAction := PgpKeys.publishSigned.value,
  releaseVcs := Vcs.detect(baseDirectory.value),
  releaseVcsSign := true,
  releaseIgnoreUntrackedFiles := true,
  useGpg := true,  // Bouncy Castle has bugs with sub-keys, so we use gpg instead
  pgpPassphrase := sys.env.get("PGP_PASSWORD").map(_.toArray),
  pgpPublicRing := file("~/.gnupg/pubring.gpg"),
  pgpSecretRing := file("~/.gnupg/secring.gpg"),
  publishMavenStyle := true,
  // publishArtifact in Test := false,
  pomIncludeRepository := Function.const(false),
  publishTo := Some(
    if (isSnapshot.value)
      Opts.resolver.sonatypeSnapshots
    else
      Opts.resolver.sonatypeStaging
  ),
  releaseProcess := Seq[ReleaseStep](
    checkSnapshotDependencies,
    inquireVersions,
    runClean,
    runTest,
    setReleaseVersion,
    commitReleaseVersion,
    tagRelease,
    publishArtifacts,
    setNextVersion,
    commitNextVersion,
    releaseStepCommand("sonatypeReleaseAll"),
    pushChanges
  ),
  // For Travis CI - see http://www.cakesolutions.net/teamblogs/publishing-artefacts-to-oss-sonatype-nexus-using-sbt-and-travis-ci
  credentials ++= (for {
    username <- Option(System.getenv().get("SONATYPE_USERNAME"))
    password <- Option(System.getenv().get("SONATYPE_PASSWORD"))
  } yield Credentials("Sonatype Nexus Repository Manager", "oss.sonatype.org", username, password)).toSeq,
  deletedPublishedSnapshots := {
    Process(
      "curl" :: "--request" :: "DELETE" :: "--write" :: "%{http_code} %{url_effective}\\n" ::
          "--user" :: s"${System.getenv().get("SONATYPE_USERNAME")}:${System.getenv().get("SONATYPE_PASSWORD")}" ::
          "--output" :: "/dev/null" :: "--silent" ::
          s"${Opts.resolver.sonatypeSnapshots.root}/${organization.value.replace(".", "/")}/" :: Nil) ! streams.value.log
  }
)

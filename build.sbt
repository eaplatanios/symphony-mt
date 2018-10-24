/* Copyright 2017-18, Emmanouil Antonios Platanios. All Rights Reserved.
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

scalaVersion in ThisBuild := "2.12.7"
crossScalaVersions in ThisBuild := Seq("2.11.12", "2.12.7")

organization in ThisBuild := "org.platanios"

// In order to update the snapshots more frequently, the Coursier "Time-To-Live" (TTL) option can be modified. This can
// be done by modifying the "COURSIER_TTL" environment variable. Its value is parsed using
// 'scala.concurrent.duration.Duration', so that things like "24 hours", "5 min", "10s", or "0s", are fine, and it also
// accepts infinity ("Inf") as a duration. It defaults to 24 hours, meaning that the snapshot artifacts are updated
// every 24 hours.
resolvers in ThisBuild += Resolver.sonatypeRepo("snapshots")

val tensorFlowForScalaVersion = "0.4.0-SNAPSHOT"

autoCompilerPlugins in ThisBuild := true

scalacOptions in ThisBuild ++= Seq(
  "-deprecation",
  "-encoding", "UTF-8",
  "-feature",
  "-language:existentials",
  "-language:higherKinds",
  "-language:implicitConversions",
  "-unchecked",
  "-Yno-adapted-args",
  // "-Ystatistics:typer",
  "-Xfuture")

val scalacProfilingEnabled: SettingKey[Boolean] =
  settingKey[Boolean]("Flag specifying whether to enable profiling for the Scala compiler.")

scalacProfilingEnabled in ThisBuild := false

lazy val loggingSettings = Seq(
  libraryDependencies ++= Seq(
    "com.typesafe.scala-logging" %% "scala-logging"   % "3.9.0",
    "ch.qos.logback"             %  "logback-classic" % "1.2.3"))

lazy val commonSettings = loggingSettings ++ Seq(
  // Plugin that prints better implicit resolution errors.
  addCompilerPlugin("io.tryp"  % "splain" % "0.3.3" cross CrossVersion.patch)
)

lazy val testSettings = Seq(
  libraryDependencies ++= Seq(
    "junit"         %  "junit"     % "4.12",
    "org.scalactic" %% "scalactic" % "3.0.4",
    "org.scalatest" %% "scalatest" % "3.0.4" % "test"),
  logBuffered in Test := false,
  fork in test := false,
  testForkedParallel in Test := false,
  parallelExecution in Test := false,
  testOptions in Test += Tests.Argument(TestFrameworks.ScalaTest, "-oDF"))

lazy val tensorFlowSettings = Seq(
  libraryDependencies += "org.platanios" %% "tensorflow" % tensorFlowForScalaVersion, // classifier "darwin-cpu-x86_64",
)

lazy val all = (project in file("."))
    .aggregate(mt, experiments, docs)
    .dependsOn(mt, experiments, docs)
    .settings(moduleName := "symphony", name := "Symphony")
    .settings(commonSettings)
    .settings(publishSettings)
    .settings(
      assemblyJarName in assembly := s"symphony-mt-${version.value}.jar",
      mainClass in assembly := Some("org.platanios.symphony.mt.experiments.Experiment"),
      test in assembly := {},
      sourcesInBase := false,
      unmanagedSourceDirectories in Compile := Nil,
      unmanagedSourceDirectories in Test := Nil,
      unmanagedResourceDirectories in Compile := Nil,
      unmanagedResourceDirectories in Test := Nil,
      publishArtifact := true)

lazy val mt = (project in file("./mt"))
    .settings(moduleName := "symphony-mt", name := "Symphony Machine Translation")
    .settings(commonSettings)
    .settings(testSettings)
    .settings(tensorFlowSettings)
    .settings(publishSettings)
    .settings(
      libraryDependencies ++= Seq(
        "com.github.pathikrit" %% "better-files" % "3.4.0",
        "org.apache.commons" % "commons-compress" % "1.16.1"),
        // Scalac Profiling Settings
      libraryDependencies ++= {
        if (scalacProfilingEnabled.value)
          Seq(compilerPlugin("ch.epfl.scala" %% "scalac-profiling" % "1.0.0"))
        else
          Seq.empty
      },
      scalacOptions ++= {
        if (scalacProfilingEnabled.value) {
          Seq(
            "-Ystatistics:typer",
            // Scala profiler plugin options
            "-P:scalac-profiling:no-profiledb",
            "-P:scalac-profiling:show-profiles",
            "-P:scalac-profiling:show-concrete-implicit-tparams")
        } else {
          Seq.empty
        }
      },
      unmanagedResourceDirectories in Compile += baseDirectory.value / "lib",
      unmanagedResourceDirectories in Test += baseDirectory.value / "lib",
      unmanagedJars in Compile ++= Seq(
        baseDirectory.value / "lib" / "meteor-1.5.jar",
        baseDirectory.value / "lib" / "tercom-0.10.0.jar"))

lazy val experiments = (project in file("./experiments"))
    .dependsOn(mt)
    .settings(moduleName := "symphony-mt-experiments", name := "Symphony Machine Translation Experiments")
    .settings(commonSettings)
    .settings(testSettings)
    .settings(publishSettings)
    .settings(
      mainClass in assembly := Some("org.platanios.symphony.mt.experiments.Experiment"),
      libraryDependencies ++= Seq(
        "com.github.pathikrit" %% "better-files" % "3.4.0",
        "com.github.scopt" %% "scopt" % "3.7.0",
        "com.hierynomus" % "sshj" % "0.24.0",
        "com.jcraft" % "jzlib" % "1.1.3",
        "io.circe" %% "circe-core" % "0.9.1",
        "io.circe" %% "circe-generic" % "0.9.1",
        "io.circe" %% "circe-parser" % "0.9.1"))

val MT = config("mt")
val Experiments = config("experiments")

lazy val docs = (project in file("docs"))
    .dependsOn(mt)
    .enablePlugins(SiteScaladocPlugin, ParadoxPlugin, ParadoxMaterialThemePlugin, GhpagesPlugin)
    .settings(moduleName := "symphony-docs", name := "Symphony Documentation")
    .settings(
      SiteScaladocPlugin.scaladocSettings(MT, mappings in (Compile, packageDoc) in mt, "api/mt"),
      SiteScaladocPlugin.scaladocSettings(Experiments, mappings in (Compile, packageDoc) in experiments, "api/experiments"),
      ghpagesNoJekyll := true,
      siteSubdirName in SiteScaladoc := "api/latest",
      siteSourceDirectory := (target in (Compile, paradox)).value,
      makeSite := makeSite.dependsOn(paradox in Compile).value,
      paradoxMaterialTheme := ParadoxMaterialTheme(),
      paradoxProperties += ("material.theme.version" -> (version in paradoxMaterialTheme).value),
      paradoxProperties ++= paradoxMaterialTheme.value.paradoxProperties,
      mappings in makeSite ++= (mappings in (Compile, paradoxMaterialTheme)).value,
      mappings in makeSite ++= Seq(
        file("LICENSE") -> "LICENSE"),
      // file("src/assets/favicon.ico") -> "favicon.ico"),
      scmInfo := Some(ScmInfo(
        url("https://github.com/eaplatanios/symphony-mt"),
        "git@github.com:eaplatanios/symphony-mt.git")),
      git.remoteRepo := scmInfo.value.get.connection,
      paradoxMaterialTheme in Compile ~= {
        _.withColor("red", "red")
            .withRepository(uri("https://github.com/eaplatanios/symphony-mt"))
            .withSocial(
              uri("https://github.com/eaplatanios"),
              uri("https://twitter.com/eaplatanios"))
            .withLanguage(java.util.Locale.ENGLISH)
            .withFont("Roboto", "Source Code Pro")
      })

lazy val noPublishSettings = Seq(
  publish := Unit,
  publishLocal := Unit,
  publishArtifact := false,
  skip in publish := true,
  releaseProcess := Nil)

val deletedPublishedSnapshots = taskKey[Unit]("Delete published snapshots.")

lazy val publishSettings = Seq(
  publishArtifact := true,
  homepage := Some(url("https://github.com/eaplatanios/symphony-mt")),
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
    releaseStepCommandAndRemaining("+publishSigned"),
    setNextVersion,
    commitNextVersion,
    releaseStepCommand("sonatypeReleaseAll"),
    pushChanges
  ),
  // The following 2 lines are needed to get around this: https://github.com/sbt/sbt/issues/4275
  publishConfiguration := publishConfiguration.value.withOverwrite(true),
  publishLocalConfiguration := publishLocalConfiguration.value.withOverwrite(true),
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
  })

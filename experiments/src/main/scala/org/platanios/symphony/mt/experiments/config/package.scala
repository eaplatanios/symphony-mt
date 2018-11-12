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

package org.platanios.symphony.mt.experiments

import com.typesafe.config.Config

import scala.reflect.runtime.universe._

/**
  * @author Emmanouil Antonios Platanios
  */
package object config {
  implicit class ConfigWithDefaults(config: Config) {
    def get[T: TypeTag](path: String): T = {
      val value = typeOf[T] match {
        case t if t =:= typeOf[Config] => config.getConfig(path)
        case t if t =:= typeOf[Boolean] => config.getBoolean(path)
        case t if t =:= typeOf[Int] => config.getInt(path)
        case t if t =:= typeOf[Long] => config.getLong(path)
        case t if t =:= typeOf[Float] => config.getDouble(path).toFloat
        case t if t =:= typeOf[Double] => config.getDouble(path)
        case t if t =:= typeOf[String] => config.getString(path)
      }
      value.asInstanceOf[T]
    }

    def getOption[T: TypeTag](path: String): Option[T] = {
      if (config.hasPath(path)) {
        val value = typeOf[T] match {
          case t if t =:= typeOf[Config] => config.getConfig(path)
          case t if t =:= typeOf[Boolean] => config.getBoolean(path)
          case t if t =:= typeOf[Int] => config.getInt(path)
          case t if t =:= typeOf[Long] => config.getLong(path)
          case t if t =:= typeOf[Float] => config.getDouble(path).toFloat
          case t if t =:= typeOf[Double] => config.getDouble(path)
          case t if t =:= typeOf[String] => config.getString(path)
        }
        Some(value.asInstanceOf[T])
      } else {
        None
      }
    }

    def get[T: TypeTag](path: String, default: T): T = {
      getOption[T](path).getOrElse(default)
    }
  }
}

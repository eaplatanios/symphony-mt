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

package org.platanios.symphony.mt.utilities

import scala.collection.Searching._
import scala.collection.mutable

/** Efficient streaming histogram implementation based on an algorithm described in
  * [A Streaming Parallel Decision Tree Algorithm](http://www.jmlr.org/papers/volume11/ben-haim10a/ben-haim10a.pdf).
  * The histogram consumes numeric samples and maintains a running approximation of the samples distribution using the
  * specified maximum number of bins.
  *
  * This implementation is based on this
  * [Java/Clojure implementation by Adam Ashenfelter](https://github.com/bigmlcom/histogram).
  *
  * @param  maxNumBins Maximum number of bins used.
  *
  * @author Emmanouil Antonios Platanios
  */
class Histogram(val maxNumBins: Int) {
  import Histogram._

  protected val binReservoir: BinReservoir = {
    if (maxNumBins > Histogram.RESERVOIR_THRESHOLD)
      new TreeBinReservoir(maxNumBins)
    else
      new ArrayBinReservoir(maxNumBins)
  }

  protected var _minimum: Option[Double] = None
  protected var _maximum: Option[Double] = None

  protected var pointToSum: Option[mutable.TreeMap[Double, Double]] = None
  protected var sumToBin  : Option[mutable.TreeMap[Double, Bin]]    = None

  /** Boolean indicating whether the gaps between bins are weighted by the number of samples in the bins. */
  @inline def useCountWeightedGaps: Boolean = {
    binReservoir.useNumSamplesWeightedGaps
  }

  /** Number of samples seen after which the bins are fixed. This makes insertions faster after that number of samples
    * have been seen. */
  @inline def freezeThreshold: Option[Long] = {
    binReservoir.freezeThreshold
  }

  /** Number of samples that have been added to this histogram so far. */
  @inline def numSamples: Long = {
    binReservoir.numSamples
  }

  /** Minimum value that has been inserted into this histogram. */
  @inline def minimum: Option[Double] = {
    _minimum
  }

  /** Maximum value that has been inserted into this histogram. */
  @inline def maximum: Option[Double] = {
    _maximum
  }

  /** Collections of bins that form this histogram. */
  @inline def bins: Seq[Bin] = {
    binReservoir.bins
  }

  /** Approximate count of the number of times `value` has been observed. */
  def count(value: Double): Double = {
    if (value < _minimum.get || value > _maximum.get) {
      0.0
    } else if (value == _minimum.get && value == _maximum.get) {
      Double.PositiveInfinity
    } else {
      binReservoir.get(value) match {
        case Some(_) =>
          val lower = Math.nextAfter(value, Double.NegativeInfinity)
          val higher = Math.nextAfter(value, Double.PositiveInfinity)
          val lowerDensity = pdf(lower)
          val higherDensity = pdf(higher)
          (lowerDensity + higherDensity) / 2.0
        case None =>
          val floorBin = binReservoir.floor(value).getOrElse(Bin(_minimum.get, numSamples = 0L))
          val ceilBin = binReservoir.ceil(value).getOrElse(Bin(_maximum.get, numSamples = 0L))

          // We compute the density starting from the sum:
          //   s = p + (1/2 + r - r^2/2)*i + r^2/2*i1
          //   r = (value - m) / (m1 - m)
          //   s_dx = i - (i1 - i) * r

          val m = floorBin.mean
          val m1 = ceilBin.mean
          val r = (value - m) / (m1 - m)
          val i = floorBin._numSamples.toDouble
          val i1 = ceilBin._numSamples.toDouble

          (i + (i1 - i) * r) / (m1 - m)
      }
    }
  }

  /** Approximate number of observations with value less than `value`. */
  def sum(value: Double): Double = {
    if (binReservoir.numSamples == 0) {
      0.0
    } else {
      if (value < _minimum.get) {
        0.0
      } else if (value > _maximum.get) {
        binReservoir.numSamples.toDouble
      } else if (binReservoir.last.exists(_.mean == value)) {
        binReservoir.numSamples - (binReservoir.last.get._numSamples.toDouble / 2.0)
      } else {
        binReservoir.get(value) match {
          case Some(bin) =>
            getPointToSum(bin.mean)
          case None =>
            val (binI, previousNumSamples) = binReservoir.floor(value) match {
              case Some(bin) => (bin, getPointToSum(bin.mean))
              case None => (
                  Bin(mean = _minimum.get, numSamples = 0L),
                  binReservoir.head.map(_._numSamples.toDouble).getOrElse(0.0) / 2.0)
            }

            val binI1 = binReservoir.ceil(value) match {
              case Some(bin) => bin
              case None => Bin(mean = _maximum.get, numSamples = 0L)
            }

            // We derive the sum in terms of p, r, i, and i1, starting from the Ben-Haim paper:
            //   m = i + (i1 - i) * r
            //   s = p + i/2 + (m + i) * r/2
            //   p' = p + i/2 (the previous value includes i/2)
            //   s = p' + (i + (i1 - i) * r + i) * r/2
            //   s = p' + (i + r*i1 - r*i + i) * r/2
            //   s = p' + r/2*i + r^2/2*i1 - r^2/2*i + r/2*i
            //   s = p' + r/2*i + r/2*i - r^2/2*i + r^2/2*i1
            //   s = p' + r*i - r^2/2*i + r^2/2*i1
            //   s = p' + (r - r^2/2)*i + r^2/2*i1

            val p = previousNumSamples
            val r = (value - binI.mean) / (binI1.mean - binI.mean)
            val i = binI._numSamples.toDouble
            val i1 = binI1._numSamples.toDouble
            val rSquaredHalf = 0.5 * r * r

            p + i * (r - rSquaredHalf) + i1 * rSquaredHalf
        }
      }
    }
  }

  /** Approximate probability density function. */
  def pdf(value: Double): Double = {
    count(value) / binReservoir.numSamples.toDouble
  }

  /** Approximate cumulative density function. */
  def cdf(value: Double): Double = {
    sum(value) / binReservoir.numSamples.toDouble
  }

  /** Inserts the provided value into this histogram.
    *
    * @param  value Value to insert.
    * @return This histogram, after the insertion has been processed.
    */
  def insert(value: Double): Unit = {
    insertBin(Bin(mean = value, numSamples = 1))
  }

  /** Inserts the provided bin into this histogram.
    *
    * @param  bin Bin to insert.
    * @return This histogram, after the insertion has been processed.
    */
  def insertBin(bin: Bin): Histogram = {
    if (_minimum.isEmpty || _minimum.get > bin.mean)
      _minimum = Some(bin.mean)
    if (_maximum.isEmpty || _maximum.get < bin.mean)
      _maximum = Some(bin.mean)
    clearCacheMaps()
    binReservoir.add(bin)
    binReservoir.merge()
    this
  }

  protected def getPointToSum: mutable.TreeMap[Double, Double] = {
    if (pointToSum.isEmpty)
      refreshCacheMaps()
    pointToSum.get
  }

  protected def getSumToBin: mutable.TreeMap[Double, Bin] = {
    if (sumToBin.isEmpty)
      refreshCacheMaps()
    sumToBin.get
  }

  protected def clearCacheMaps(): Unit = {
    pointToSum = None
    sumToBin = None
  }

  protected def refreshCacheMaps(): Unit = {
    pointToSum = Some(mutable.TreeMap.empty[Double, Double])
    sumToBin = Some(mutable.TreeMap.empty[Double, Bin])

    val minBin = Bin(mean = _minimum.get, numSamples = 0L)
    val maxBin = Bin(mean = _maximum.get, numSamples = 0L)

    pointToSum.get.put(_minimum.get, 0.0)
    sumToBin.get.put(0.0, minBin)
    sumToBin.get.put(numSamples.toDouble, maxBin)

    var sum = 0.0
    var lastBin = minBin
    binReservoir.bins.foreach(bin => {
      sum += (bin._numSamples + lastBin._numSamples).toDouble / 2.0
      pointToSum.get.put(bin.mean, sum)
      sumToBin.get.put(sum, bin)
      lastBin = bin
    })
    sum += lastBin._numSamples.toDouble / 2.0
    pointToSum.get.put(_maximum.get, sum)
  }
}

object Histogram {
  private[Histogram] val RESERVOIR_THRESHOLD: Long = 256L

  def apply(maxNumBins: Int): Histogram = {
    new Histogram(maxNumBins)
  }

  /** Represents a histogram bin.
    *
    * @param  mean Mean value of samples placed in this histogram bin.
    */
  case class Bin(mean: Double) extends Ordered[Bin] {
    /** Number of samples placed in this histogram bin. */
    private[Histogram] var _numSamples: Long = 0L

    def numSamples: Long = {
      _numSamples
    }

    def weight: Double = {
      mean * _numSamples
    }

    def combine(other: Bin): Bin = {
      val numSamples = _numSamples + other._numSamples
      val combinedBin = Bin(mean = (weight + other.weight) / numSamples.toDouble)
      combinedBin._numSamples = numSamples
      combinedBin
    }

    override def compare(that: Bin): Int = {
      Ordering.Double.compare(mean, that.mean)
    }
  }

  object Bin {
    def apply(mean: Double, numSamples: Long): Bin = {
      val bin = Bin(mean)
      bin._numSamples = numSamples
      bin
    }
  }

  /** Data structure for managing histogram bins (i.e., insertion of bins, merging of bins, etc.). */
  trait BinReservoir {
    val maxNumBins               : Int
    val useNumSamplesWeightedGaps: Boolean
    val freezeThreshold          : Option[Long]

    protected var _numSamples: Long = 0L

    @inline def isFrozen: Boolean = {
      freezeThreshold.exists(_ < _numSamples)
    }

    @inline def numSamples: Long = {
      _numSamples
    }

    def add(bin: Bin): Unit
    def head: Option[Bin]
    def last: Option[Bin]
    def get(value: Double): Option[Bin]
    def floor(value: Double): Option[Bin]
    def ceil(value: Double): Option[Bin]
    def bins: Seq[Bin]

    def merge(): Unit

    def gapWeight(previous: Bin, next: Bin): Double = {
      if (!useNumSamplesWeightedGaps)
        next.mean - previous.mean
      else
        (next.mean - previous.mean) * math.log(math.E + math.min(previous._numSamples, next._numSamples).toDouble)
    }
  }

  /** This bin reservoir implements bin operations (insertions, merges, etc.), using an underlying array buffer. It is
    * best used for histograms with a small (i.e., `<= 256`) number of bins. It has O(N) insertion performance with
    * respect to the number of bins in the histogram. For histograms with more bins, the `TreeBinReservoir` class offers
    * better performance.
    *
    * @param  maxNumBins                Maximum number of bins.
    * @param  useNumSamplesWeightedGaps Boolean value indicating whether to weigh the gaps between bins by the number of
    *                                   samples in those bins.
    * @param  freezeThreshold           Optional threshold specifying the number of samples seen after which the bins
    *                                   are fixed. This makes insertions faster after that number of samples have been
    *                                   seen.
    */
  class ArrayBinReservoir(
      override val maxNumBins: Int,
      override val useNumSamplesWeightedGaps: Boolean = false,
      override val freezeThreshold: Option[Long] = None
  ) extends BinReservoir {
    protected val _bins: mutable.ArrayBuffer[Bin] = mutable.ArrayBuffer.empty[Bin]

    override def add(bin: Bin): Unit = {
      _numSamples += bin._numSamples
      _bins.search(bin) match {
        case Found(index) =>
          _bins(index)._numSamples += bin._numSamples
        case InsertionPoint(index) if !isFrozen || _bins.size != maxNumBins =>
          _bins.insert(index, bin)
        case InsertionPoint(index) =>
          val previousIndex = index - 1
          val previousDistance = if (previousIndex >= 0) bin.mean - _bins(previousIndex).mean else Double.MaxValue
          val nextDistance = if (index < _bins.size) _bins(index).mean - bin.mean else Double.MaxValue
          if (previousDistance < nextDistance)
            _bins(previousIndex)._numSamples += bin._numSamples
          else
            _bins(index)._numSamples += bin._numSamples
      }
    }

    override def head: Option[Bin] = {
      _bins.headOption
    }

    override def last: Option[Bin] = {
      _bins.lastOption
    }

    override def get(p: Double): Option[Bin] = {
      _bins.search(Bin(p)) match {
        case Found(index) => Some(_bins(index))
        case _ => None
      }
    }

    override def floor(p: Double): Option[Bin] = {
      _bins.search(Bin(p)) match {
        case Found(index) => Some(_bins(index))
        case InsertionPoint(index) if index > 0 => Some(_bins(index - 1))
        case _ => None
      }
    }

    override def ceil(p: Double): Option[Bin] = {
      _bins.search(Bin(p)) match {
        case Found(index) => Some(_bins(index))
        case InsertionPoint(index) if index < _bins.size => Some(_bins(index))
        case _ => None
      }
    }

    override def bins: Seq[Bin] = {
      _bins
    }

    override def merge(): Unit = {
      while (_bins.size > maxNumBins) {
        var minGap = Double.MaxValue
        var minGapIndex = -1
        for (i <- 0 until (_bins.size - 1)) {
          val gap = gapWeight(_bins(i), _bins(i + 1))
          if (minGap > gap) {
            minGap = gap
            minGapIndex = i
          }
        }
        val previousBin = _bins(minGapIndex)
        val nextBin = _bins.remove(minGapIndex + 1)
        _bins.update(minGapIndex, previousBin.combine(nextBin))
      }
    }
  }

  /** This bin reservoir implements bin operations (insertions, merges, etc.), using an underlying tree map. It is
    * best used for histograms with a large (i.e., `> 256`) number of bins. It has O(log(N)) insertion performance with
    * respect to the number of bins in the histogram. For histograms with fewer bins, the `ArrayBinReservoir` class
    * offers better performance.
    *
    * @param  maxNumBins                Maximum number of bins.
    * @param  useNumSamplesWeightedGaps Boolean value indicating whether to weigh the gaps between bins by the number of
    *                                   samples in those bins.
    * @param  freezeThreshold           Optional threshold specifying the number of samples seen after which the bins
    *                                   are fixed. This makes insertions faster after that number of samples have been
    *                                   seen.
    */
  class TreeBinReservoir(
      override val maxNumBins: Int,
      override val useNumSamplesWeightedGaps: Boolean = false,
      override val freezeThreshold: Option[Long] = None
  ) extends BinReservoir {
    import TreeBinReservoir.Gap

    protected val _bins        : mutable.TreeMap[Double, Bin] = mutable.TreeMap.empty[Double, Bin]
    protected val _gaps        : mutable.TreeSet[Gap]         = mutable.TreeSet.empty[Gap]
    protected val _valuesToGaps: mutable.HashMap[Double, Gap] = mutable.HashMap.empty[Double, Gap]

    override def add(bin: Bin): Unit = {
      _numSamples += bin._numSamples
      if (isFrozen && _bins.size == maxNumBins) {
        var floorDifference = Double.MaxValue
        val floorBin = this.floor(bin.mean)
        floorBin.foreach(b => floorDifference = math.abs(b.mean - bin.mean))
        var ceilDifference = Double.MaxValue
        val ceilBin = this.ceil(bin.mean)
        ceilBin.foreach(b => ceilDifference = math.abs(b.mean - bin.mean))
        if (floorDifference <= ceilDifference)
          floorBin.get._numSamples += bin.numSamples
        else
          ceilBin.get._numSamples += bin.numSamples
      } else {
        get(bin.mean) match {
          case Some(existingBin) =>
            existingBin._numSamples += bin.numSamples
            if (useNumSamplesWeightedGaps)
              updateGaps(existingBin)
          case None =>
            updateGaps(bin)
            _bins.put(bin.mean, bin)
        }
      }
    }

    override def head: Option[Bin] = {
      _bins.headOption.map(_._2)
    }

    override def last: Option[Bin] = {
      _bins.lastOption.map(_._2)
    }

    override def get(value: Double): Option[Bin] = {
      _bins.get(value)
    }

    override def floor(value: Double): Option[Bin] = {
      _bins.to(value).lastOption.map(_._2)
    }

    override def ceil(value: Double): Option[Bin] = {
      _bins.from(value).headOption.map(_._2)
    }

    override def bins: Seq[Bin] = {
      _bins.values.toSeq
    }

    override def merge(): Unit = {
      while (_bins.size > maxNumBins) {
        val minGap = _gaps.head
        _gaps.remove(minGap)

        _valuesToGaps.get(minGap.endBin.mean) match {
          case Some(followingGap) => _gaps.remove(followingGap)
          case None => ()
        }

        _bins.remove(minGap.startBin.mean)
        _bins.remove(minGap.endBin.mean)
        _valuesToGaps.remove(minGap.startBin.mean)
        _valuesToGaps.remove(minGap.endBin.mean)

        val newBin = minGap.startBin.combine(minGap.endBin)
        updateGaps(newBin)
        _bins.put(newBin.mean, newBin)
      }
    }

    protected def updateGaps(newBin: Bin): Unit = {
      floor(newBin.mean).foreach(updateGaps(_, newBin))
      ceil(newBin.mean).foreach(updateGaps(newBin, _))
    }

    protected def updateGaps(previousBin: Bin, nextBin: Bin): Unit = {
      val newGap = Gap(previousBin, nextBin, gapWeight(previousBin, nextBin))
      _valuesToGaps.get(previousBin.mean) match {
        case Some(previousGap) => _gaps.remove(previousGap)
        case None => ()
      }
      _valuesToGaps.put(previousBin.mean, newGap)
      _gaps.add(newGap)
    }
  }

  object TreeBinReservoir {
    case class Gap(startBin: Bin, endBin: Bin, space: Double) extends Ordered[Gap] {
      override def compare(that: Gap): Int = {
        var result = Ordering.Double.compare(space, that.space)
        if (result == 0)
          result = startBin.compare(that.startBin)
        result
      }
    }
  }
}

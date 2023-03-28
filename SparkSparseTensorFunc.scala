//spark-shell --conf spark.ui.port=4041 --name boris-explore --executor-memory 14G --num-executors 16 --driver-memory 48g --executor-cores 2

import spark.implicits._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import breeze.linalg.CSCMatrix
import breeze.linalg.svd
import scala.math.sqrt
import scala.util.Random._
import scala.collection.mutable.ArrayBuffer

import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema


class EnrichedWithToTuple[A](elements: Seq[A]) {
  def toTuple2 = elements match { case Seq(a, b) => (a, b) }
  def toTuple3 = elements match { case Seq(a, b, c) => (a, b, c) }
  def toTuple4 = elements match { case Seq(a, b, c, d) => (a, b, c, d) }
  def toTuple5 = elements match { case Seq(a, b, c, d, e) => (a, b, c, d, e) }
  def toTuple6 = elements match { case Seq(a, b, c, d, e, f) => (a, b, c, d, e, f) }
  def toTuple7 = elements match { case Seq(a, b, c, d, e, f, g) => (a, b, c, d, e, f, g) }
  def toTuple8 = elements match { case Seq(a, b, c, d, e, f, g, h) => (a, b, c, d, e, f, g, h) }
  def toTuple9 = elements match { case Seq(a, b, c, d, e, f, g, h, i) => (a, b, c, d, e, f, g, h, i) }
  def toTuple10 = elements match { case Seq(a, b, c, d, e, f, g, h, i, j) => (a, b, c, d, e, f, g, h, i, j) }
  def toTuple11 = elements match { case Seq(a, b, c, d, e, f, g, h, i, j, k) => (a, b, c, d, e, f, g, h, i, j, k) }
  def toTuple12 = elements match { case Seq(a, b, c, d, e, f, g, h, i, j, k, l) => (a, b, c, d, e, f, g, h, i, j, k, l) }
  def toTuple13 = elements match { case Seq(a, b, c, d, e, f, g, h, i, j, k, l, m) => (a, b, c, d, e, f, g, h, i, j, k, l, m) }
}

implicit def enrichWithToTuple[A](elements: Seq[A]) = new EnrichedWithToTuple(elements)

import shapeless.syntax.std.tuple._

def toCSCMatrix(dfMult : DataFrame, maxd : Int = 0) : CSCMatrix[Double] = {
  val rddMult1 = dfMult.rdd.
    map(r => (r.asInstanceOf[Row].getInt(0), r.asInstanceOf[Row].getInt(1), r.asInstanceOf[Row].getDouble(2))).
    collect
  val nnz = rddMult1.size
  val maxr = (for ((r,_,_) <- rddMult1) yield r).toSeq.max
  val maxc = (for ((_,c,_) <- rddMult1) yield c).toSeq.max
  val builder = new CSCMatrix.Builder[Double](maxr+1, maxc+1, nnz)

  for ((r,c,d) <- rddMult1) {
    builder.add(r, c, d)
  }

  builder.result
}

def sparseVstack(df1 : DataFrame, df2 : DataFrame) : DataFrame = {
  val col0 = List("MATRIX_ID") ++ df1.columns
  val df11 = df1.withColumn("MATRIX_ID", lit(0)).select(col0.head, col0.tail:_*)
  val df22 = df2.withColumn("MATRIX_ID", lit(1)).select(col0.head, col0.tail:_*)
  df11.union(df22)
}



def unfold(df : DataFrame, mode : String, data_column : String = "count") : DataFrame = {
  val excludeSet = Set(data_column, mode)
  val otherModes = for ((a,b) <- df.columns.zipWithIndex if !excludeSet.contains(a)) yield (a,b)
  val modeCol = for ((a,b) <- df.columns.zipWithIndex if a == mode) yield (a,b)
  val dataCol = for ((a,b) <- df.columns.zipWithIndex if a == data_column) yield (a,b)
  val rddFlat = df.rdd.map(r => (for ((a,b) <- otherModes) yield r.getInt(b), r.getInt(modeCol(0)._2), r.getDouble(dataCol(0)._2)) ).
    toDF((for ((a,_) <- otherModes) yield a).mkString("+"), mode, data_column)
  rddFlat
}

def fold(df : DataFrame, modes : Seq[String]) : DataFrame = {
  /* Please read the comment below and provied appropriate Tuple2, Tuple3, etc, depending on the number of tensor dimensions*/
  val cols = df.columns
  val foldedCols = cols(0).split("\\+")
  val colNames = foldedCols.toSeq ++ cols.slice(1,3).toSeq
  modes.size match {
    case 4 => df.rdd.map(r => (r.asInstanceOf[Row].getSeq[Int](0) ++ Seq(r.asInstanceOf[Row].getInt(1))).toTuple3 ++ Tuple1(r.asInstanceOf[Row].getDouble(2))).toDF(colNames: _*).select(modes.head, modes.tail: _*)
    case 10 => df.rdd.map(r => (r.asInstanceOf[Row].getSeq[Int](0) ++ Seq(r.asInstanceOf[Row].getInt(1))).toTuple9 ++ Tuple1(r.asInstanceOf[Row].getDouble(2))).toDF(colNames: _*).select(modes.head, modes.tail: _*)
    case 11 => df.rdd.map(r => (r.asInstanceOf[Row].getSeq[Int](0) ++ Seq(r.asInstanceOf[Row].getInt(1))).toTuple10 ++ Tuple1(r.asInstanceOf[Row].getDouble(2))).toDF(colNames: _*).select(modes.head, modes.tail: _*)
    case 12 => df.rdd.map(r => (r.asInstanceOf[Row].getSeq[Int](0) ++ Seq(r.asInstanceOf[Row].getInt(1))).toTuple11 ++ Tuple1(r.asInstanceOf[Row].getDouble(2))).toDF(colNames: _*).select(modes.head, modes.tail: _*)
    case _ => null
  }
}


def sparseMultiply(df1 : DataFrame, df2 : DataFrame) : DataFrame = {

  val M_ = df1.schema.fields(0).dataType match {
    case ArrayType(_,_) => {
      df1.schema.fields(1).dataType match {
        case IntegerType => df1.rdd.map ((t) => (Seq(t.asInstanceOf[Row].getInt(1)), (t.asInstanceOf[Row].getSeq[Int](0), t.asInstanceOf[Row].getDouble (2) ) ) )
        case ArrayType(_,_) => df1.rdd.map ((t) => (t.asInstanceOf[Row].getSeq[Int](1), (t.asInstanceOf[Row].getSeq[Int](0), t.asInstanceOf[Row].getDouble (2) ) ) )
      }
    }
    case IntegerType => {
      df1.schema.fields(1).dataType match {
        case IntegerType => df1.rdd.map ((t) => (Seq(t.asInstanceOf[Row].getInt(1)), (Seq(t.asInstanceOf[Row].getInt(0)), t.asInstanceOf[Row].getDouble (2) ) ) )
        case ArrayType(_,_) => df1.rdd.map ((t) => (t.asInstanceOf[Row].getSeq[Int](1), (Seq(t.asInstanceOf[Row].getInt(0)), t.asInstanceOf[Row].getDouble (2) ) ) )
      }
    }
    case _ => null
  }

  val N_ = df2.schema.fields(0).dataType match {
    case ArrayType(_,_) => {
      df2.schema.fields(1).dataType match {
        case IntegerType => df2.rdd.map ((t) => (Seq(t.asInstanceOf[Row].getInt(1)), (t.asInstanceOf[Row].getSeq[Int](0), t.asInstanceOf[Row].getDouble (2) ) ) )
        case ArrayType(_,_) => df2.rdd.map ((t) => (t.asInstanceOf[Row].getSeq[Int](1), (t.asInstanceOf[Row].getSeq[Int](0), t.asInstanceOf[Row].getDouble (2) ) ) )
      }
    }
    case IntegerType => {
      df2.schema.fields(1).dataType match {
        case IntegerType => df2.rdd.map ((t) => (Seq(t.asInstanceOf[Row].getInt(1)), (Seq(t.asInstanceOf[Row].getInt(0)), t.asInstanceOf[Row].getDouble (2) ) ) )
        case ArrayType(_,_) => df2.rdd.map ((t) => (t.asInstanceOf[Row].getSeq[Int](1), (Seq(t.asInstanceOf[Row].getInt(0)), t.asInstanceOf[Row].getDouble (2) ) ) )
      }
    }
    case _ => null
  }

  val productEntries = M_.join(N_).repartition(10000).map({ case (_, ((i, v), (k, w))) => ((i, k), (v * w)) }).reduceByKey(_ + _)

  productEntries.map(r => (r._1._1, r._1._2, r._2)).toDF(df1.columns(0), df2.columns(0), "product")
}




def multi_mode_dot(core : DataFrame, factors : Seq[DataFrame], skip : Int = -1, persist : Boolean = true, coreSize : Int = 300000 ) : DataFrame = {
  val modeNames = core.columns.slice(0, core.columns.size-1)
  var newCore = core
  for (((df,mode),i) <- (factors zip modeNames).zipWithIndex) {
    if (i != skip) {
      println(s"Unfolding ${mode}")
      val dfU = unfold(newCore, mode = mode, data_column = "product")
      val dfM = sparseMultiply(df.withColumnRenamed("_1", mode), dfU)
      val mode0Cols = dfM.columns
      newCore = fold(dfM.select(col(mode0Cols(1)), col(mode0Cols(0))(0).alias(mode0Cols(0)), col(mode0Cols(2))), core.columns)
      if (persist) {
        println("Persisting core...")
        newCore.repartition(32).write.mode("overwrite").parquet(f"CORE_${i}")
        println("Reading core...")
        newCore = spark.read.parquet(f"CORE_${i}").
          withColumn("abs_product", abs('product)).
          withColumn("rownum", row_number().over(Window.orderBy(desc("abs_product")))).
          filter(col("rownum") < 500000).drop("rownum").drop("abs_product")
        val coreSize = newCore.count
        println(f"Core size ${coreSize}")
      }
    }
  }
  newCore
}

def transposeMatrix(df : DataFrame) : DataFrame = {
  df.select(col("_2").alias("_1"), col("_1").alias("_2"), col("product"))
}

/*
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                                             Sparse Distributed Tucker
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 */



val rndCore = for {
  x <- 0 until 2
  y <- 0 until 3
  z <- 0 until 4
} yield (x,y,z,nextDouble)

val dfRndCore = sc.parallelize(rndCore).toDF("_1", "_2", "_3", "product")

/*Random factors shape (3,2), (4,3), (5,4) TRANSPOSED!!! */
val f0 = for {
  x <- 0 until 2
  y <- 0 until 3
} yield (x,y,nextDouble)

val dfF0 = sc.parallelize(f0).toDF("_1", "_2", "product")

val f1 = for {
  x <- 0 until 3
  y <- 0 until 4
} yield (x,y,nextDouble)

val dfF1 = sc.parallelize(f1).toDF("_1", "_2", "product")

val f2 = for {
  x <- 0 until 4
  y <- 0 until 5
} yield (x,y,nextDouble)

val dfF2 = sc.parallelize(f2).toDF("_1", "_2", "product")

val dfTuckerTensor = multi_mode_dot(dfRndCore, for (df <- List(dfF0, dfF1, dfF2)) yield transposeMatrix(df))

dfTuckerTensor.agg(max("_1"), max("_2"), max("_3")).show

def getRandomFactors(dfT : DataFrame, desiredFactors : Seq[Int]) : List[DataFrame] = {
  val aggCols = for (c <- dfT.columns) yield max(c)+1
  val factorShapes = dfT.agg(aggCols.head, aggCols.tail:_*).
    rdd.map(r => for (i <- 0 until r.asInstanceOf[Row].length-1) yield r.asInstanceOf[Row].getInt(i)).collect()(0)
  val seqRanks = factorShapes.toSeq zip desiredFactors
  val seqFactors = for (r <- seqRanks)
    yield sc.parallelize(
      for {
        x <- 0 until r._2
        y <- 0 until r._1
      } yield (x,y,nextDouble)).toDF("_1", "_2", "product")
  seqFactors.toList
}


def tuckerFactorization(dfTuckerTensor : DataFrame, factors : List[DataFrame], N : Int = 10, coreSize : Int = 300000) : List[DataFrame] = {
  val factList = ArrayBuffer(factors:_*)
  val factorShapes = for (f <- factList) yield f.agg(max("_1"), max("_2")).rdd.map(r => (r.asInstanceOf[Row].getInt(0) + 1, r.asInstanceOf[Row].getInt(1) + 1)).collect()(0)
  for (iter <- 0 until N) {
    println(f"Iteration $iter")
    for (i <- 0 to dfTuckerTensor.columns.size - 2) {
      println(f"Factor $i")
      val Y = multi_mode_dot(dfTuckerTensor, factList, skip = i, persist = true, coreSize = coreSize)
      val unfolded = unfold(Y, dfTuckerTensor.columns(i), "product")
      val un = unfolded.select(array(unfolded.columns(1)).alias(unfolded.columns(1)), col(unfolded.columns(0)), col(unfolded.columns(2)))
      val mult = sparseMultiply(un, un)
      val M = mult.rdd.map(r => (r.asInstanceOf[Row].getSeq[Int](0)(0), r.asInstanceOf[Row].getSeq[Int](1)(0), r.asInstanceOf[Row].getDouble(2))).toDF("_1", "_2", "product")
      val sm = toCSCMatrix(M)
      val svd.SVD(u, _, _) = svd(sm.toDense)
      println("Completed SVD")
      val newFactor = for {
        k <- 0 until List(u.cols, factorShapes(i)._1).min
        j <- 0 until u.rows
      } yield (k, j, u.valueAt(j, k))
      val dfNewFactor = sc.parallelize(newFactor).toDF("_1", "_2", "product").filter(abs('product) > 0)
      dfNewFactor.write.mode("overwrite").parquet(f"FACTOR_${i}")
      println("Created New Factor")
      factList(i) = spark.read.parquet(f"FACTOR_${i}")
    }
  }
  factList.toList
}

val tuckerFactors = tuckerFactorization(dfTuckerTensor, List(dfF0, dfF1, dfF2))

val G = multi_mode_dot(dfTuckerTensor, tuckerFactors)

val tuckerFactorsT = for (df <- tuckerFactors) yield transposeMatrix(df)

val tensorRecon = multi_mode_dot(G, tuckerFactorsT)
val tensorRecon = multi_mode_dot(dfRndCore, List(dfF0, dfF1, dfF2))

val dfJ = dfTuckerTensor.join(tensorRecon.withColumnRenamed("product", "product_rec"), Seq("_1", "_2", "_3"))

val dfD = dfJ.withColumn("diff", 'product - 'product_rec)

dfD.withColumn("diff", pow('diff, 2)).agg(mean("diff")).show


/*
             Test with canned data - multi_mode_dot is correct!
 */

val dfTT = multi_mode_dot(tuckerCore, tuckerFactors)

dfTT.orderBy("_1", "_2", "_3").show

dfTT.filter((col("_1") === lit(2)) && (col("_2") === lit(3)) && (col("_3") === lit(4) )).show

val tuckerFactors = tuckerFactorization(tuckerTensor, randomFactors)
val G = multi_mode_dot(tuckerTensor, tuckerFactors)
G.orderBy("_1", "_2", "_3").show(false)
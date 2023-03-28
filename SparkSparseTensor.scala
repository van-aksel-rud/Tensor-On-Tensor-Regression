//spark-shell --conf spark.kryoserializer.buffer.max=128m --conf spark.ui.port=4041 --name boris-explore --executor-memory 14G --num-executors 16 --driver-memory 48g --executor-cores 2

import spark.implicits._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import breeze.linalg.CSCMatrix
import breeze.linalg.svd
import breeze.linalg.inv
import scala.math.sqrt
import scala.util.Random._
import scala.collection.mutable.ArrayBuffer

import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema

val dataRoot = "abfs://some_container@some_domain,..."

/*

Distributed sparse Tucker tensor factorization using Spark dataframes

Sparse tensor representation using spark dataframe:
Col1 | Col2 | Col3 | ... | hr_id | product
-----+------+------+-----+-------+---------

The above dataframe is a spark dataframe with columns Col1 to ColN representing tensor modes
The data in the columns does not have to be numerical, it could be strings or arbitrary data types
Matrix multiplication and flattening operations would work the same regardless of the data types

Important convention: The target regression column is the last column called product (DoubleType())
The column before last called hr_id in this code is free dimension of the tensor - this allows to represent multiple tensor valued inputs. We regress along this dimension. This can be thought as discrete time.

The tensor regression output is therefore itself tensor valued
*/

val experiment = 31

val dfTsTensor = spark.read.parquet(dataRoot)

def spMult(df1 : DataFrame, df2 : DataFrame) : DataFrame = {
  val df11 = df1.select(col(df1.columns(0)).alias("inner"), col(df1.columns(1)).alias("_1"), col(df1.columns(2)).alias("product"))
  val df22 = df2.select(col(df1.columns(0)).alias("inner"), col(df1.columns(1)).alias("_2"), col(df1.columns(2)).alias("product1"))
  df11.join(df22.hint("broadcast"), Seq("inner")).
    withColumn("product", 'product * 'product1).
    groupBy("_1", "_2").
    agg(sum('product).alias("product"))
}

def multi_mode_dot_offset(core : DataFrame, factors : Seq[DataFrame], skip : Int, corePrecision : Double = 0.5) : DataFrame = {

  val modeNames = core.columns.slice(0, core.columns.size-1)
  val modesAndFactors = for (((df,mode),i) <- (factors zip modeNames).zipWithIndex) yield (df, mode, i)

  val k = skip - 1
  var newCore = if (k < 0) core else spark.read.parquet(f"CORE_$k")
  for ((df, mode, i) <- modesAndFactors) {
    if (i > skip)  {
      println(s"Unfolding ${mode}")
      val dfU = unfold(newCore, mode = mode, data_column = "product")
      val offset = df.filter(col("_2") === 0).drop("_2").withColumnRenamed("_1", mode).withColumnRenamed("product", "bias")
      val dfReg = df.filter(col("_2") > 0).withColumn("_2", col("_2")-1)

      val dfr = dfReg.withColumnRenamed("_1", "outer").withColumnRenamed("_2", mode).withColumnRenamed("product", "product1")
      val dfJ = dfU.join(dfr.hint("broadcast"), Seq(mode))
      val dfMultRes = dfJ.withColumn("product", 'product * 'product1).groupBy(dfJ.columns(1), "outer").
        agg(sum('product).alias("product")).withColumnRenamed("outer", mode).
        withColumn("mag", abs('product)).filter('mag > corePrecision).drop("mag")
      val dfMultOffset = dfMultRes.join(offset.hint("broadcast"), Seq(mode)).withColumn("product", 'product + 'bias).
        drop("bias").select(dfU.columns.head, dfU.columns.tail:_*)
      newCore = fold(dfMultOffset, core.columns)
      println("Persisting core...")
      newCore.write.mode("overwrite").parquet(f"CORE_${i}")
      println("Reading core...")
      newCore = spark.read.parquet(f"CORE_${i}")
      val coreSize = newCore.count
      println(f"Core size ${coreSize}")
    }
  }
  newCore
}

def regressTensor(dfX : DataFrame, dfBlist : List[DataFrame], precision : Double = 1.0) : DataFrame = {
  var dfCore = dfX
  val columns = dfX.columns.reverse.drop(2).reverse
  for (((mode,dfB),k) <- (columns zip dfBlist).zipWithIndex) {
    println(s"Mode ${mode}")
    val dfU = unfold(dfCore, mode=mode, data_column = "product")
    val offset = dfB.filter(col("_2") === 0).drop("_2").withColumnRenamed("_1", mode).withColumnRenamed("product", "bias")
    val dfReg = dfB.filter(col("_2") > 0).withColumn("_2", col("_2")-1)
    val dfMult = dfU.join(dfReg.select(col("_2").alias(mode), col("_1").alias("outer"), col("product").alias("product1")).hint("broadcast"), Seq(mode)).
      drop(mode).withColumnRenamed("outer", mode).withColumn("product", 'product * 'product1).groupBy(dfU.columns(0), mode).
      agg(sum('product).alias("product"))
    val dfMultOffset = dfMult.join(offset.hint("broadcast"), Seq(mode)).withColumn("product", 'product + 'bias).
      drop("bias").select(dfU.columns.head, dfU.columns.tail:_*)
    dfCore = fold(dfMultOffset, dfX.columns).filter(abs('product) > precision)
    if (k % 2 == 0) {
      dfCore.write.mode("overwrite").parquet("REG_TMP_0")
      dfCore = spark.read.parquet("REG_TMP_0")
      val cnt = dfCore.count
      println(s"Core size ${cnt}")
    } else {
      dfCore.write.mode("overwrite").parquet("REG_TMP_1")
      dfCore = spark.read.parquet("REG_TMP_1")
      val cnt = dfCore.count
      println(s"Core size ${cnt}")
    }
  }
  dfCore
}


def getRegressionFactorShapes(dfSource : DataFrame, dfTarget : DataFrame, precision : Double = 1e-6) : List[DataFrame] = {

  // The B matrix contains both the offset vector and the regression coefficients
  // The output is a dataframe with first column corresponding to Y space, second to X
  val sourceCol = for (c <- dfSource.columns) yield max(c)
  val sourceShape = dfSource.select(sourceCol:_*).collect()(0)
  val targetCol = for (c <- dfTarget.columns) yield max(c)
  val targetShape = dfTarget.select(targetCol:_*).collect()(0)
  val ss = for (i <- 0 until sourceShape.size - 2) yield sourceShape.getInt(i) + 1 // for offset vector
  val ts = for (i <- 0 until targetShape.size - 2) yield targetShape.getInt(i)

  val ret = ArrayBuffer[DataFrame]()
  for ((a,b) <- ss zip ts) {
    val res = for {
      k <- 0 to b
      j <- 0 to a
    } yield (k, j, nextDouble())
    ret += sc.parallelize(res).toDF("_1", "_2", "product").filter(abs('product) > precision)
  }
  ret.toList
}

def getSvdRegressionFactorShapes(dfSource : DataFrame, dfTarget : DataFrame, precision : Double = 1e-6) : List[DataFrame] = {

  // The B matrix contains both the offset vector and the regression coefficients
  // The output is a dataframe with first column corresponding to Y space, second to X
  val targetCol = for (c <- dfTarget.columns) yield max(c)
  val targetShape = dfTarget.select(targetCol:_*).collect()(0)
  val ts = for (i <- 0 until targetShape.size - 2) yield targetShape.getInt(i)

  val ret = ArrayBuffer[DataFrame]()
  for (i <- 0 until ts.size) {
    val dfU = unfold(dfSource, dfSource.columns(i), data_column = "product")
    val SQ = spMult(dfU, dfU)
    val svd.SVD(u, _, _) = svd(toCSCMatrix(SQ).toDense)
    val res = for {
      k <- 0 until u.rows
      j <- 0 until u.cols
    } yield (k, j+1, u.valueAt(k,j).abs)
    val Binit = sc.parallelize(res).toDF("_1", "_2", "product").filter(col("_1") <= ts(i)).filter('product > precision).
     union(sc.parallelize(for (cc <- 0 to ts(i)) yield (cc,0,0.0)).toDF)
    ret += Binit
  }
  ret.toList
}

def modeDot(dfLagT : DataFrame, dfFactor : DataFrame, skip : Int, corePrecision : Double = 1e-6) : DataFrame =
{
  val modeNames = dfLagT.columns.slice(0, dfLagT.columns.size - 1)
  val mode = modeNames(skip)
  val k = skip - 1
  var newCore = if (k < 0) dfLagT else spark.read.parquet(f"CORE_$k")
  println(s"Unfolding ${mode}")
  val dfU = unfold(newCore, mode = mode, data_column = "product")
  val offset = dfFactor.filter(col("_2") === 0).drop("_2").withColumnRenamed("_1", mode).withColumnRenamed("product", "bias")
  val dfReg = dfFactor.filter(col("_2") > 0).withColumn("_2", col("_2")-1)
  val dfr = dfReg.withColumnRenamed("_1", "outer").withColumnRenamed("_2", mode).withColumnRenamed("product", "product1")
  val dfJ = dfU.join(dfr.hint("broadcast"), Seq(mode))
  val dfMultRes = dfJ.withColumn("product", 'product * 'product1).groupBy(dfJ.columns(1), "outer").
    agg(sum('product).alias("product")).withColumnRenamed("outer", mode).
    filter(abs('product) > corePrecision)
  val dfMultOffset = dfMultRes.join(offset.hint("broadcast"), Seq(mode)).withColumn("product", 'product + 'bias).
    drop("bias").select(dfU.columns.head, dfU.columns.tail:_*)
  newCore = fold(dfMultOffset, dfLagT.columns)
  println("Persisting core...")
  newCore.write.mode("overwrite").parquet(f"CORE_${skip}")
  spark.read.parquet(f"CORE_${skip}")
}

/*
def regressMatrix(val X_bar_k, val Y_k, val dfB) = {
  val offset = dfB.filter(col("_2") === 0).drop("_2").withColumnRenamed("_1", mode).withColumnRenamed("product", "bias")
  val dfReg = dfB.filter(col("_2") > 0).withColumn("_2", col("_2")-1)
  val dfM = spMult(transposeMatrix(dfReg), X_bar_k.select(col(X_bar_k.columns(1)).alias("_1"), col(X_bar_k.columns(0)).alias("_2"), col("product")))
  val dfMO = dfM.join(offset, Seq("_1")).withColumn("product", 'product + 'bias).drop("bias")
  val dfJY = Y_k.select(col(Y_k.columns(0)).alias("_2"), col(Y_k.columns(1)).alias("_1"), col("product").alias("org_product")).join(dfMO, Seq("_1", "_2"))
}
*/

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// Tensor on Tensor regression ///////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

val aggCols = for (c <- dfTsTensor.columns) yield (lit(1) + max(c)).alias(c)
dfTsTensor.agg(aggCols.head, aggCols.tail:_*).show

val newCols = for ((c,i) <- dfTsTensor.columns zip (Stream from 1)) yield col(c).alias(f"_${i}")

val dfT = dfTsTensor.select(newCols:_*).withColumnRenamed("_11", "product")

val dfSource = dfT
// Source dataframe introduces 3 previous hours as additional observations

val newColumns = dfSource.columns.reverse.drop(1).reverse ++ List("_11", "product")
val dfLag0 = dfSource.withColumn("_11", col("_10")).
  withColumn("_10", lit(0)).select(newColumns.head, newColumns.tail:_*)
val dfLag1 = dfLag0.withColumn("_11", col("_11") + 1).withColumn("_10", lit(1))
val dfLag2 = dfLag0.withColumn("_11", col("_11") + 2).withColumn("_10", lit(2))
val dfLag3 = dfLag0.withColumn("_11", col("_11") + 3).withColumn("_10", lit(3))

val dfLagT = dfLag0.union(dfLag1.union(dfLag2.union(dfLag3)))

dfLagT.write.mode("overwrite").parquet(dataRoot + f"PROC/ALL_LAGS_${experiment}")
val dfLagT = spark.read.parquet(dataRoot + f"PROC/ALL_LAGS_${experiment}")

val dfLag4 = dfLagT.groupBy("_11", "_10").count.groupBy("_11").agg(collect_set("_10").alias("lags")).withColumn("numLags", size('lags)).filter('numLags === 4).select("_11")

// Hours that have 3 prior observations
val dfX = dfLagT.join(dfLag4, "_11").select(dfLagT.columns.head, dfLagT.columns.tail:_*).filter(col("_10") < 3)
val dfY = dfLagT.join(dfLag4, "_11").select(dfLagT.columns.head, dfLagT.columns.tail:_*).filter(col("_10") === 3).
  withColumn("_7", lit(0)).withColumn("_8", lit(0)).withColumn("_9", lit(0)).withColumn("_10", lit(0))

dfX.write.mode("overwrite").parquet(dataRoot + f"PROC/LAG4_X_${experiment}")
dfY.write.mode("overwrite").parquet(dataRoot + f"PROC/LAG4_Y_${experiment}")

val dfXO = spark.read.parquet(dataRoot + f"PROC/LAG4_X_${experiment}").filter('_11 > 4500)
val dfYO = spark.read.parquet(dataRoot + f"PROC/LAG4_Y_${experiment}").filter('_11 > 4500)
val minHr = dfXO.agg(min("_11")).collect()(0).getInt(0)
val dfX = dfXO.withColumn("_11", col("_11") - lit(minHr))
val dfY = dfYO.withColumn("_11", col("_11") - lit(minHr))

val dfX = spark.read.csv("sine-core/sine-core.txt").select(col("_c0").alias("_1").cast(IntegerType),
  col("_c1").alias("_2").cast(IntegerType),
  col("_c2").alias("_3").cast(IntegerType),
  col("_c3").alias("product").cast(DoubleType)
)

val dfY = spark.read.csv("sine-core/sine-target.txt").select(col("_c0").alias("_1").cast(IntegerType),
  col("_c1").alias("_2").cast(IntegerType),
  col("_c2").alias("_3").cast(IntegerType),
  col("_c3").alias("product").cast(DoubleType)
)

val regFactors = getSvdRegressionFactorShapes(dfX, dfY, precision = 1e-3)

val maxCol = for (f <- dfX.columns.reverse.drop(1).reverse) yield max(f)
val xdim = dfX.agg(maxCol.head, maxCol.tail:_*).collect()(0)
val ydim = dfY.agg(maxCol.head, maxCol.tail:_*).collect()(0)
val factorSizes = for (c <- 0 to dfX.columns.size-2) yield xdim.getInt(c)+1
val yfactorSizes = for (c <- 0 to dfX.columns.size-2) yield ydim.getInt(c)+1
var bList = ArrayBuffer(regFactors:_*)

val i = 0
val corePrecision = 0.0
val N = 10
val runid = 3

for (iter <- 0 until N) {
  println(f"***************** Iteration $iter")
  for (i <- 0 until regFactors.size) {
    println(f"+++ Factor $i")
    val X_bar = multi_mode_dot_offset(dfX, bList, skip = i, corePrecision = corePrecision)
    val mode = dfX.columns(i)
    val X_bar_k = unfold(X_bar, mode, "product")
    val Y_k = unfold(dfY, mode, "product")
    val YX = spMult(Y_k, X_bar_k)

    val XX = spMult(X_bar_k, X_bar_k).persist(StorageLevel.MEMORY_ONLY).
      withColumn("_1", col("_1") + 1).
      withColumn("_2", col("_2") + 1)

    val firstRow = X_bar_k.groupBy(mode).agg(sum('product).alias("product")).withColumn("_1", col(mode)+1).persist(StorageLevel.MEMORY_ONLY)
    val countRows = yfactorSizes.zipWithIndex.filter(_._2 != i).map(_._1).reduce(_*_)
    val factorSize = factorSizes(i)
    val countCell = sc.parallelize(Seq((0, 0, countRows.toDouble))).toDF("_1", "_2", "product")
    val dfReg = sc.parallelize(0 to factorSize).toDF.select('value, 'value, lit(0.001))

    val dfAug = firstRow.select(col("_1"), lit(0).alias("_2"), col("product"))
    val dfAugAug = dfAug.union(transposeMatrix(dfAug)).union(countCell).union(XX)

    val Ma = dfAugAug.union(dfReg).groupBy("_1", "_2").agg(sum('product))
    val cscM = toCSCMatrix(Ma)

    try {
      var Minv = inv(cscM.toDense)
      val MinvCoord = for {
        k <- 0 until Minv.cols
        j <- 0 until Minv.rows
      } yield (k, j, Minv.valueAt(k, j))

      val XXinv = sc.parallelize(MinvCoord).toDF("_1","_2", "product").filter(abs('product) > 0)

      val YcolSum = Y_k.groupBy(mode).agg(sum('product).alias("product")).
        select(col(mode).alias("_1"), lit(0).alias("_2"), col("product"))

      val XYaug = YX.withColumn("_2", col("_2") + 1).union(YcolSum)

      val dfB = spMult(transposeMatrix(XYaug), XXinv)
      dfB.write.mode("overwrite").parquet(f"BMATRIX_${iter}_${runid}_$i")
      bList(i) = spark.read.parquet(f"BMATRIX_${iter}_${runid}_$i")
      println("Created new factor")

      /*Write core for current factor*/
      modeDot(dfX, bList(i), i, corePrecision)
      val dummy = XX.unpersist()
      val dummy2 = firstRow.unpersist()

    } catch {
      case t: Throwable => {
        println("Got exception during INV")
        println(cscM)
        println(cscM.toDense.toString(50, 1000))
        throw t
      }
    }

  }

  for (i <- 0 until regFactors.size) {
    val df = spark.read.parquet(f"BMATRIX_S$i")
    df.write.mode("overwrite").parquet(f"BMATRIX_SAVE_${iter}_${runid}_$i")
  }

  val dfYreg = regressTensor(dfX, bList.toList, precision = corePrecision)
  val dfYregPos = dfYreg.filter('product >= 1)
  val dfEval = dfY.join(dfYregPos.withColumnRenamed("product", "pred"), dfY.columns.reverse.drop(1).reverse.toSeq).filter('pred > 0)
  val dfD = dfEval.withColumn("diff", pow('product - 'pred,2))
  val yMean = dfD.agg(mean("product")).collect()(0).asInstanceOf[Row].getDouble(0)
  val dfR2 = dfD.withColumn("SStot", pow('product - yMean, 2)).agg(sum('diff).alias("SSres"), sum('SStot).alias("SStot")).collect()
  val ssRes = dfR2(0).asInstanceOf[Row].getDouble(0)
  val ssTot = dfR2(0).asInstanceOf[Row].getDouble(1)
  val r2Score = 1 - ssRes / ssTot
  println(f"R2 score ${r2Score}")
  val dfScore = sc.parallelize(Seq(r2Score)).toDF
  dfScore.write.mode("overwrite").parquet(f"SCORE_${iter}_${runid}_$i")
}

val iter = 0
for (i <- 0 until regFactors.size) {
  val df = spark.read.parquet(f"BMATRIX_S$i")
  df.write.mode("overwrite").parquet(f"BMATRIX_SAVE_${iter}_${runid}_$i")
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

val dfYreg = regressTensor(dfX, bList.toList)
val dfEval = dfY.join(dfYreg.withColumnRenamed("product", "pred"), dfY.columns.reverse.drop(1).reverse.toSeq).filter('pred > 0)


val dfD = dfEval.withColumn("diff", pow('product - 'pred,2))
val yMean = dfD.agg(mean("product")).collect()(0).asInstanceOf[Row].getDouble(0)
val dfR2 = dfD.withColumn("SStot", pow('product - yMean, 2)).agg(sum('diff).alias("SSres"), sum('SStot).alias("SStot")).collect()
val ssRes = dfR2(0).asInstanceOf[Row].getDouble(0)
val ssTot = dfR2(0).asInstanceOf[Row].getDouble(1)
val r2Score = 1 - ssRes / ssTot
//package com.yoyi.models.xgboost
//
//import com.yoyi.utils.SparkUtil
//import ml.dmlc.xgboost4j.scala.spark.{XGBoost, XGBoostClassificationModel, XGBoostModel}
//import org.apache.spark
//import org.apache.spark.sql.functions._
//import org.apache.spark.ml.linalg.{DenseVector => MlDenseVector, SparseVector => MlSparseVector, Vector => MlVector, Vectors => MlVectors}
//
//
///**
//  * @author weiwei.jiang
//  * @date 2022-11-2022/11/3-15:15
//  */
//object TestXgb {
//  def testXgb4j()={
//    val sparkSession = SparkUtil.initSparkSession("")
//    val datas = sparkSession.emptyDataFrame
//    val xgbModel: XGBoostModel = ml.dmlc.xgboost4j.scala.spark.XGBoost.
//      trainWithDataFrame(datas,Map("a"->"b"),round = 10,nWorkers = 10)
//
//    //解析dump的树结构，构建xgb森林
//    //这里直接解析xgb4j训练的结果
//    //也可以解析python xgboost dump的树结构
//    val gbtModel: GBTModel = XGBoostModel.loadBoosters(xgbModel.booster.getModelDump(), GBTModel.Classification)
//
//    //将模型广播出去，进行预测
//    val gbtModelBr = sparkSession.sparkContext.broadcast(gbtModel)
//
//    val rawPredictUDF = udf { features: Any =>
//      val vector = features.asInstanceOf[MlVector] match {
//        case d: MlDenseVector => Vectors.dense(d.values.map(_.toFloat))
//        case s: MlSparseVector => Vectors.sparse(s.indices, s.values.map(_.toFloat))
//      }
//      val prediction = gbtModelBr.value.predict(vector).toDouble
//      MlVectors.dense(Array(-prediction, prediction))
//    }
//
//    val gbtPredictions = datas.withColumn("rawPrediction", rawPredictUDF(col("features")))
//
//  }
//
//  def testXgb4Py()={
//    val modelPath = "./sex_model.raw"
//    XGBoostModel.load(modelPath,GBTModel.Classification)
//  }
//
//  def main(args: Array[String]): Unit = {
//    testXgb4Py()
//
//  }
//}

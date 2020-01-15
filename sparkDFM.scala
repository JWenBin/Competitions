import com.alibaba.fastjson.{JSON, JSONArray}
import com.bpml.{ALSJ, BLAS, BoundedPriorityQueue, MyAverage}
import utils.OtherUtil.initSpark
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.commons.io.IOUtils
import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.Tensor
import java.io.FileInputStream
import java.io.IOException

import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.hadoop.mapred.TextInputFormat
import spire.std.byte

import Array._
object sparkDFM{
  def main(args: Array[String]): Unit ={
    val spark = initSpark("Normalize")
    indexValue(spark,1000)
    return
	}

  def indexValue(spark:SparkSession, topN:Int): Unit ={
    val normPlayLog="file:///F:/python/data/cs2"
    val fDict="file:///F:/python/data/feature_dict3" // batch_size 1024 batch=10 feature=10 训练模型
    val dfmModel="F:/python/model/deepFM2J2/" // batch_size 1024 loss<0.1训练模型
    val saveRec="F:/python/data/0920m/rec"
    import spark.implicits._
    val featureDict=
      //spark.sparkContext.textFile(fDict)
      spark.sparkContext.hadoopFile(fDict,classOf[TextInputFormat],classOf[LongWritable],classOf[Text],1)
        .map(p => new String(p._2.getBytes, 0, p._2.getLength, "GBK"))
      .collect()
      .head
    val featureJson=JSON.parseObject(featureDict)
    println(featureJson.keySet())

    val country=spark.sparkContext.broadcast( featureJson.getJSONObject("country") )
    val itemID=spark.sparkContext.broadcast( featureJson.getJSONObject("itemID") )
    val uId=spark.sparkContext.broadcast( featureJson.getJSONObject("uId") )
    val penalty=spark.sparkContext.broadcast( featureJson.get("penalty") )
    val genre=spark.sparkContext.broadcast( featureJson.getJSONObject("genre") )
    val artistid=spark.sparkContext.broadcast( featureJson.getJSONObject("artistid") )
    val ua=spark.sparkContext.broadcast( featureJson.getJSONObject("ua") )
    val appVersion=spark.sparkContext.broadcast( featureJson.getJSONObject("appversion") )
    val singertype=spark.sparkContext.broadcast( featureJson.getJSONObject("singertype") )
    val region=spark.sparkContext.broadcast( featureJson.getJSONObject("region") )

    println(spark.read.format("com.databricks.spark.csv").option("header", "true").load(normPlayLog).schema)
    val itemIndex=spark.read.format("com.databricks.spark.csv").option("header", "true").load(normPlayLog)
        .select("itemID","penalty","genre","artistid","singertype","region").dropDuplicates("itemID","genre","artistid").rdd
      .mapPartitions(it=> {
      val itemIDJson=itemID.value
      val penaltyJson =penalty.value.toString.toInt
      val genreJson=genre.value
      val artistidJson=artistid.value
      val sexJson=singertype.value
      val regionJson=region.value
      it.map(r => { //"itemID","penalty","genre","artistid" penalty itemID singertype region
        ( itemIDJson.get(r.getString(0).toInt).toString.toInt,penaltyJson, genreJson.getIntValue(r.getString(2)), artistidJson.get(r.getString(3).toInt).toString.toInt, r.getString(1).toFloat,r.getString(0).toInt, sexJson.getIntValue(r.getString(4)), regionJson.getIntValue(r.getString(5))     )
      })
    })
    val item= spark.sparkContext.broadcast(itemIndex.collect())

      val recDF=spark.read.load("F:/python/data/0920m/tmp/userIndexI"+i).rdd.repartition(21)
        .mapPartitions(it => {
          val countryJson = country.value
          val uIdJson = uId.value
          val uaJson =ua.value
          val appJson =appVersion.value
          val itemInfo = item.value //"itemID","penalty","genre","artistid" penalty itemID singertype region

          val graph = new Graph()
          //导入图
          val graphBytes = IOUtils.toByteArray(new FileInputStream(dfmModel + "model.pb"))
          graph.importGraphDef(graphBytes);
          //根据图建立Session
          val session = new Session(graph)
          var count=0
          it.map(r => {
            var country = 0
            var appIndex = 0
            var uaIndex=0
            try {
              appIndex = appJson.get(r.getString(3).toInt).toString.toInt
              uaIndex = uaJson.get(r.getString(2)).toString.toInt
              country = countryJson.get(r.getString(0)).toString.toInt
            } catch {
              case e: Exception => {
                e.printStackTrace
              }
                println(r)
                country = countryJson.get("nan").toString.toInt
            }
            val uid = r.getString(1).toInt
            val uidIndex = uIdJson.get(uid).toString.toInt

            val index_value_itemid: Array[(Array[Int], Array[Float], Int)] = itemInfo.map(item => {//"itemID","penalty","genre","artistid" penalty itemID singertype region
              val itemID = item._6
              // uId  artistid  itemID  country  penalty      ua  appversion   genre  singertype  region
              (Array(uidIndex, item._4,item._1, country,  item._2,uaIndex,appIndex,item._3, item._7,item._8), Array(1, 1, 1,1, item._5, 1, 1, 1, 1, 1), itemID)
            })
            (uid, getScore(index_value_itemid, topN, session))
          })
        }).toDF("uid", "rec")
      recDF.write.mode("Overwrite").save(saveRec+i)
      //System.gc() //执行Full gc进行垃圾回收
      spark.read.load(saveRec+i).selectExpr("count(1)").show(false)
    
  }

  def getScore(index_value_itemid: Array[(Array[Int], Array[Float], Int)] ,topN:Int ,session:Session):  List[(Int, Float)] = {
    var arr= ofDim[Float](index_value_itemid.length,1)
    val feat_index: Tensor[_] = Tensor.create(index_value_itemid.map(_._1) )
    val feat_value= Tensor.create(index_value_itemid.map(_._2) )

    //相当于TensorFlow Python中的sess.run
    val z = session.runner()
      .feed("feat_index", feat_index)
      .feed("feat_value", feat_value)
      .fetch("add_out")
      .run().get(0)
    // z.floatValue() 改成：z.copyTo(arr).toList(0)(0)；
    val scoreArr=z.copyTo(arr)
    val itemArr=index_value_itemid.map(_._3)
    val pq = new BoundedPriorityQueue[(Int, Float)](topN)(Ordering.by( _._2))
    for(  i <- 0 until index_value_itemid.length){
      pq += itemArr(i) -> scoreArr(i).head
    }
    pq.toList.sortBy(- _._2)

  }

}

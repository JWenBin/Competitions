import pandas as pd
import tensorflow as tf
import numpy as np
import time
pd.set_option('display.max_columns', 100)  # 设置最大显示列数

TRAIN_FILE = "D:\\data\\input\\tl\\trainData.csv"
TEST_FILE = "D:\\data\\input\\tl\\testData.csv"
model_path="d:/data/input/tl/model/dfm/"
fdict_path = "d:/data/input/tl/feature_dict"

# target|clickDate|uid|newsid|pos|app_version|device_vendor|netmodel|osversion|lng|lat|device_version|date|hour|minute|level|personidentification|followscore|personalscore|gender|delaySecond
NUMERIC_COLS = ["date","hour","minute"]
IGNORE_COLS = ["lng","lat",  "target","clickDate" ,"followscore","personalscore","delaySecond"  ,"deviceid","trguid","guid"]
# NUMERIC_COLS = ["lng","lat","date","hour","minute",  "level","followscore","personalscore","gender","delaySecond"]
# IGNORE_COLS = ["lng","lat",  "target","clickDate"  ,"followscore","personalscore"]
train_num=10000000
"""模型参数"""
dfm_params = {
    "decay": 0.9,
    "learn_rate_step":13117,  #喂入多少轮BATCH_SIZE后，更新一次学习率，一般设为：总样本数/BATCH_SIZE 13117 8238
    "threads":6,
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 10,
    "dropout_fm": [1.0, 1.0],
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layer_activation": tf.nn.relu,
    "epoch": 20,
    "batch_size": 1024,
    "learning_rate": 0.01,
    # "learning_rate":0.001,  0.1-1 0.01-10 0.002-10 0.001-10(然loss从2变3了，似乎是个别异常数据拉高的)
    "optimizer": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "eval_metric": 'gini_norm',
    "random_seed": 3
}

dfTrain = pd.read_csv(TRAIN_FILE)#.head(10)
dfTest = pd.read_csv(TEST_FILE)#.head(10)
dfDict= pd.concat([dfTest, dfTrain], ignore_index=True)
print(dfTrain.head(5))

data_rows=len(dfTrain)

feature_dict = {}
total_feature = 0
for col in dfDict.columns:
    if col in IGNORE_COLS:
        continue
    elif col in NUMERIC_COLS:
        feature_dict[col] = total_feature
        total_feature += 1
    else:
        unique_val = dfDict[col].unique()
        feature_dict[col] = dict(zip(unique_val,range(total_feature,len(unique_val) + total_feature)))
        total_feature += len(unique_val)
print(total_feature)
# with open(fdict_path, 'w+') as f:
#                 f.write(str(feature_dict) )

def norm_itemID(feat_dict,item_df):
    f_index=[]
    f_value=[]
    for i in item_df.index:
        item=item_df.iloc[i,:]
        itemID=item[0]
        penalty=item[1]
        genre = item[2]
        artistid = item[3]
        singertype= item[4]
        region= item[5]
        # f_value.append([1, 1, 1, penalty, 1, 1])
        # f_index.append( [ feat_dict['itemID'][itemID], 0,0, feat_dict['penalty'], feat_dict['genre'][genre], feat_dict['artistid'][artistid] ] )
        f_value.append([1,1, 1, 1, penalty, 1,1, 1, 1, 1])
        f_index.append( [0, feat_dict['artistid'][artistid],feat_dict['itemID'][itemID], 0, feat_dict['penalty'], 0,0,  feat_dict['genre'][genre], feat_dict['singertype'][singertype], feat_dict['region'][region] ] )
    return f_index,f_value

def norm_f_index(feat_dict, uid_country, f_index):
    #  itemID  country  uId   penalty  genre  artistid
    # uId  artistid  itemID  country  penalty  ua  appversion  genre singertype  region
    uid=uid_country[0]
    country=uid_country[1]
    ua=uid_country[2]
    appversion=uid_country[3]
    user_index=feat_dict['uId'][uid]
    country_index=feat_dict['country'][country]
    # 要是能直接改两列的值为用户信息就好了
    for v in f_index:
        # v[1]=country_index
        # v[2]=user_index
        v[3]=country_index
        v[0]=user_index
        v[5]=ua
        v[6]=appversion
    return f_index

# f_index,f_value=norm_itemID(feature_dict,  item_df)

"""
对训练集进行转化，可截取前一半训练，后一半测试，但id都要记录到字典
"""
train_y = dfTrain[['target']]#.values.tolist()
train_feature_index = dfTrain.copy()
train_feature_value = dfTrain.copy()
# train_y = dfTrain[['target']].tail(train_num).values.tolist()
# train_feature_index = dfTrain.tail(train_num).copy()
# train_feature_value = dfTrain.tail(train_num).copy()

for col in train_feature_index.columns:
    if col in IGNORE_COLS:
        train_feature_index.drop(col,axis=1,inplace=True)
        train_feature_value.drop(col,axis=1,inplace=True)
        continue
    elif col in NUMERIC_COLS:
        train_feature_index[col] = feature_dict[col]
    else:
        train_feature_index[col] = train_feature_index[col].map(feature_dict[col])
        train_feature_value[col] = 1
print(train_feature_index.head(5))
print(train_feature_value.head(5))
print(train_feature_index.tail(5))
print(train_feature_value.tail(5))

dfm_params['feature_size'] = total_feature  # 254
dfm_params['field_size'] = len(train_feature_index.columns)  # 37
print(len(train_feature_index.columns))

def get_batch(train_feature_index,train_feature_value,train_y, batch_size):
    with tf.device('/cpu:0'):
        print(len(train_feature_index),len(train_y))
        input_queue = tf.train.slice_input_producer([train_feature_index,train_feature_value,train_y],num_epochs=dfm_params['epoch'], shuffle=True )
        i_batch, v_batch, label_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=dfm_params['threads'], capacity=32, allow_smaller_final_batch=False)
        print(v_batch)
        return i_batch, v_batch,label_batch
def data_generator(train_feature_index,train_feature_value,train_y):
    # dataset = np.array( train_feature_index )
    print(type(train_feature_index),type(train_feature_value),type(train_y))
    dataset = np.array( pd.concat([train_feature_index, train_feature_value,train_y],axis=1) )
    for d in dataset:
        # print(d)
        # print(d[0:14])
        # print(d[14:28])
        # print(d[28])
        yield d[0:14],d[14:28],d[28]

dataset = tf.data.Dataset.from_generator( lambda:data_generator(train_feature_index,train_feature_value,train_y), ( tf.int32, tf.float32, tf.float32), (tf.TensorShape([14,]), tf.TensorShape([14,]), tf.TensorShape([]) ))
dataset = dataset.repeat(dfm_params['epoch'])
dataset = dataset.batch(dfm_params['batch_size'])
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()

"""开始建立模型"""
feat_index = tf.placeholder(tf.int32, shape=[None, None], name='feat_index')
feat_value = tf.placeholder(tf.float32, shape=[None, None], name='feat_value')

label = tf.placeholder(tf.float32, shape=[None, 1], name='label')

"""建立weights"""
weights = dict()

# embeddings
weights['feature_embeddings'] = tf.Variable(
    tf.random_normal([dfm_params['feature_size'], dfm_params['embedding_size']], 0.0, 0.01),
    name='feature_embeddings')
weights['feature_bias'] = tf.Variable(tf.random_normal([dfm_params['feature_size'], 1], 0.0, 1.0), name='feature_bias')

# deep layers
num_layer = len(dfm_params['deep_layers'])
input_size = dfm_params['field_size'] * dfm_params['embedding_size']
glorot = np.sqrt(2.0 / (input_size + dfm_params['deep_layers'][0]))

weights['layer_0'] = tf.Variable(
    np.random.normal(loc=0, scale=glorot, size=(input_size, dfm_params['deep_layers'][0])), dtype=np.float32
)
weights['bias_0'] = tf.Variable(
    np.random.normal(loc=0, scale=glorot, size=(1, dfm_params['deep_layers'][0])), dtype=np.float32
)

for i in range(1, num_layer):
    glorot = np.sqrt(2.0 / (dfm_params['deep_layers'][i - 1] + dfm_params['deep_layers'][i]))
    weights["layer_%d" % i] = tf.Variable(
        np.random.normal(loc=0, scale=glorot, size=(dfm_params['deep_layers'][i - 1], dfm_params['deep_layers'][i])),
        dtype=np.float32)  # layers[i-1] * layers[i]
    weights["bias_%d" % i] = tf.Variable(
        np.random.normal(loc=0, scale=glorot, size=(1, dfm_params['deep_layers'][i])),
        dtype=np.float32)  # 1 * layer[i]

# final concat projection layer

if dfm_params['use_fm'] and dfm_params['use_deep']:
    input_size = dfm_params['field_size'] + dfm_params['embedding_size'] + dfm_params['deep_layers'][-1]
elif dfm_params['use_fm']:
    input_size = dfm_params['field_size'] + dfm_params['embedding_size']
elif dfm_params['use_deep']:
    input_size = dfm_params['deep_layers'][-1]

glorot = np.sqrt(2.0 / (input_size + 1))
weights['concat_projection'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                                           dtype=np.float32, name='concat_projection')
weights['concat_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32, name='concat_bias')

"""embedding"""
embeddings = tf.nn.embedding_lookup(weights['feature_embeddings'], feat_index)

reshaped_feat_value = tf.reshape(feat_value, shape=[-1, dfm_params['field_size'], 1])

embeddings = tf.multiply(embeddings, reshaped_feat_value)

"""fm part"""
fm_first_order = tf.nn.embedding_lookup(weights['feature_bias'], feat_index)
fm_first_order = tf.reduce_sum(tf.multiply(fm_first_order, reshaped_feat_value), 2)

summed_features_emb = tf.reduce_sum(embeddings, 1)
summed_features_emb_square = tf.square(summed_features_emb)

squared_features_emb = tf.square(embeddings)
squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)

fm_second_order = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)

"""deep part"""
y_deep = tf.reshape(embeddings, shape=[-1, dfm_params['field_size'] * dfm_params['embedding_size']])

for i in range(0, len(dfm_params['deep_layers'])):
    y_deep = tf.add(tf.matmul(y_deep, weights["layer_%d" % i]), weights["bias_%d" % i])
    y_deep = tf.nn.relu(y_deep)

"""final layer"""
if dfm_params['use_fm'] and dfm_params['use_deep']:
    concat_input = tf.concat([fm_first_order, fm_second_order, y_deep], axis=1)
elif dfm_params['use_fm']:
    concat_input = tf.concat([fm_first_order, fm_second_order], axis=1)
elif dfm_params['use_deep']:
    concat_input = y_deep

# item_index = tf.slice(feat_index,[0,0],[label.shape], name='item_index')
out = tf.nn.sigmoid(tf.add(tf.matmul(concat_input, weights['concat_projection']), weights['concat_bias']) ,name="out")
outpre = tf.concat([out, label],1 ,name="outpre")

"""loss and optimizer"""
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(dfm_params['learning_rate'], global_step,dfm_params['learn_rate_step'], dfm_params['decay'], staircase=True)
loss = tf.losses.log_loss(tf.reshape(label, (-1, 1)), out)
optimizer = tf.train.AdamOptimizer(learning_rate=dfm_params['learning_rate'], beta1=0.9, beta2=0.999,epsilon=1e-8).minimize(loss,global_step=global_step)

# i_batch, v_batch, label_batch = get_batch(train_feature_index,train_feature_value,train_y.values.tolist(), dfm_params['batch_size'])

def getAuc(predict_playNum):
    p=[]
    n=[]
    count=0
    for i in predict_playNum.index:
        row = predict_playNum.iloc[i, :]
        if row[1]>=1:
            p.append(row[0])
        else:
            n.append(row[0])
    for rowp in p:
        for rown in n:
            if rowp>rown:
                count=count+1
    return count / (len(p)*len(n))

"""train"""
gpu_config = tf.ConfigProto(device_count={'GPU': 0, 'CPU': 6} )
# gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config=tf.ConfigProto(gpu_options=gpu_options)
with tf.Session(config=config) as sess:
    writer = tf.summary.FileWriter("D:\\Anaconda3\\Scripts\\logs", sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # 开启协调器
    coord = tf.train.Coordinator()
    # 使用start_queue_runners 启动队列填充
    threads = tf.train.start_queue_runners(sess, coord)
    batch= int(data_rows / dfm_params['batch_size'])
    print(data_rows , dfm_params['batch_size'],batch)
    batch_x_epochs = 0 # global_step
    end_loss=0
    tf.train.Saver().restore(sess, save_path=model_path)
    print('strat training... ',time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    try:
        while not coord.should_stop(): # batch= int(data_rows / dfm_params['batch_size']) 9000
            # 获取训练用的每一个batch中batch_size个样本和标签

            # i, v, l = sess.run([i_batch, v_batch, label_batch])
            # predict_label = sess.run([outpre], feed_dict={feat_value: v, feat_index: i, label: l})
            # recDf=pd.DataFrame(predict_label[0],columns=['score','playNum'])
            # print(getAuc(recDf))
            # recDf.to_csv(rec_path+'dfmRec'+ str(batch_x_epochs), index=False, header=True)
            # # if batch_x_epochs % 100 == 0 and batch_x_epochs!=0:

            # i, v, l = sess.run([i_batch, v_batch, label_batch])
            i, v, l = sess.run(one_element)
            # print('index: ',i)
            # print( 'value: ',v)
            # print('label: ', l)
            # print(i.shape)
            sess.run(optimizer, feed_dict={feat_index: i, feat_value: v, label: l.reshape(dfm_params['batch_size'],1) })
            train_loss = loss.eval({feat_index: i, feat_value: v, label: l.reshape(dfm_params['batch_size'],1)})
            end_loss=train_loss
            # if batch_x_epochs % 200 == 0 and batch_x_epochs!=0:
            if batch_x_epochs % 600 == 0:
                learning_rate_val = sess.run(learning_rate)
                global_step_val = sess.run(global_step)
                # print('label',l )
                print("batch_x_epochs %d,  Training loss %g,  global_step %g,  learning_rate %g,  at %s" % (batch_x_epochs, train_loss, global_step_val,learning_rate_val, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) ))
                tf.train.Saver().save(sess, save_path=model_path)
                frozen_graph_def = tf.graph_util.convert_variables_to_constants(
                    sess,
                    sess.graph_def,
                    ["feat_index", "feat_value", "out", "outpre"])
                # 保存图为pb文件
                with open(model_path + 'model.pb', 'wb') as f:
                    f.write(frozen_graph_def.SerializeToString())
            batch_x_epochs = batch_x_epochs + 1

    except tf.errors.OutOfRangeError:  # num_epochs 次数用完会抛出此异常
        print("---Train end---", batch_x_epochs, end_loss)
        tf.train.Saver().save(sess, save_path=model_path)
    finally:
        # 协调器coord发出所有线程终止信号
        coord.request_stop()
        print('---Programm end---')
    coord.join(threads)  # 把开启的线程加入主线程，等待threads结束





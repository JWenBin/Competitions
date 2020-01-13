import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from deepctr.models import DeepFM
from deepctr.inputs import SparseFeat, VarLenSparseFeat,get_feature_names
from deepctr.layers import custom_objects
from tensorflow.python.keras.models import  save_model,load_model
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
import os
pd.set_option('display.max_columns', 100)  # 设置最大显示列数
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr.inputs import SparseFeat, DenseFeat, get_feature_names
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "-1"cpu模式
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config = tf.ConfigProto(gpu_options=gpu_options)
set_session(tf.Session(config=config))

def split(x):
    # print(x)
    key_ans = x.split('|')
    # print(key_ans)
    for key in key_ans:
        # print(key)
        if key not in key2index:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            key2index[key] = len(key2index) + 1
    return list(map(lambda x: key2index[x], key_ans))
def splita(x):
    key_ans = x.split('|')
    for key in key_ans:
        if key not in key2indexa:
            key2indexa[key] = len(key2indexa) + 1
    return list(map(lambda x: key2indexa[x], key_ans))
def splito(x):
    key_ans = x.split('|')
    for key in key_ans:
        if key not in key2indexo:
            key2indexo[key] = len(key2indexo) + 1
    return list(map(lambda x: key2indexo[x], key_ans))
# def key_len(key_name):
#     key_list = list(map(split, data[key_name].values))
#     key_length = np.array(list(map(len, key_list)))
#     return max(key_length)

if __name__ == "__main__":
    input_dir='D:\\data\\input\\tl\\'
    embedding_dim=10
    data = pd.read_csv(input_dir+"89.csv").head(10000)#,header=None,names=["target", "uid", "newsid", "pos", "app_version", "device_vendor", "netmodel", "osversion", "device_version", "date","hour","minute", "tag","level","personidentification","followscore","personalscore","gender"])
    # ("target", "uid", "newsid", "pos", "app_version", "device_vendor", "netmodel", "osversion", "device_version", "date","hour","minute", "tag","level","personidentification","followscore","personalscore","gender")
    sparse_features = [ 'uid', "newsid", "pos", "app_version", "device_vendor", "netmodel", "osversion", "device_version","gender", 'lng', 'lat' ] # 经纬度（地图）是二维，不适合直接做1维数值特征
    dense_features = [ "date","hour","minute", "level","personidentification","followscore","personalscore"]
    target = ['target']
    print(data.head(2))
    data[sparse_features] = data[sparse_features].fillna('str', )
    data[dense_features] = data[dense_features].fillna(0, )
    data['lng']=data[ 'lng'].astype('int').astype('str')
    data['lat']=data[ 'lat'].astype('int').astype('str')

    # 1.Label Encoding for sparse features,and process sequence features
    for feat in sparse_features:
        print('feat:',feat)
        lbe = LabelEncoder()
        if feat == 'osversion' or feat == 'gender':
            data[feat] = lbe.fit_transform(data[feat].astype('str'))
        else:
            data[feat] = lbe.fit_transform(data[feat])
    # preprocess the sequence feature
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    key2index = {}
    key2indexa = {}
    key2indexo = {}
    tag_list = list(map(split, data["tag"].values))
    app_list = list(map(splita, data["applist"].values))
    outtag_list = list(map(splito, data["outtag"].values))

    max_len = {'applist': max( np.array(list(map(len, list(map(split, data["applist"].values)) ))) ), 'tag': max( np.array(list(map(len, list(map(split, data["tag"].values)) ))) ), 'outtag': max( np.array(list(map(len, list(map(split, data["outtag"].values)) ))) ) }

    # Notice : padding=`post`
    tag_list = pad_sequences(tag_list, maxlen=max_len['tag'], padding='post', )
    app_list = pad_sequences(app_list, maxlen=max_len['applist'], padding='post', )
    outtag_list = pad_sequences(outtag_list, maxlen=max_len['outtag'], padding='post', )
    key2index_len = {'applist': len(key2indexa), 'tag': len(key2index), 'outtag': len(key2indexo)}

    # 2.count #unique features for each sparse field and generate feature config for sequence feature
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(),embedding_dim=embedding_dim)
                        for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]

    use_weighted_sequence = True
    if use_weighted_sequence:
        # varlen_feature_columns = [VarLenSparseFeat('%s' % i, maxlen= max_len[i], vocabulary_size=key2index_len[i]
        #      + 1,embedding_dim=embedding_dim, combiner='mean',weight_name=None ) for i in ['applist', 'tag', 'outtag']]
        varlen_feature_columns = [VarLenSparseFeat('%s' % i, maxlen= max_len[i], vocabulary_size=key2index_len[i]
             + 1,embedding_dim=embedding_dim, combiner='mean',weight_name=None ) for i in [ 'tag', 'outtag']]
    else:
        varlen_feature_columns = [VarLenSparseFeat('tag', maxlen=max_len,vocabulary_size= len(
            key2index) + 1,embedding_dim=embedding_dim, combiner='mean',weight_name=None)]  # Notice : value 0 is for padding for sequence input feature

    linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns+dnn_feature_columns)

    print('fixlen_feature_columns:',fixlen_feature_columns)
    print('varlen_feature_columns:',varlen_feature_columns)
    print('feature_names:',feature_names)
    # 3.generate input data for model
    # model_input = data[feature_names]
    model_input = {name:data[name] for name in feature_names}
    model_input["tag"] = tag_list
    model_input["tag_weight"] =  np.random.randn(data.shape[0],max_len['tag'],1)
    # model_input["applist"] = app_list
    # model_input["applist_weight"] =  np.random.randn(data.shape[0],max_len['applist'],1)
    model_input["outtag"] = outtag_list
    model_input["outtag_weight"] =  np.random.randn(data.shape[0],max_len['outtag'],1)
    # model_input['target'] =data['target']
    # model_input=pd.DataFrame(model_input)
    # del data

    # train, test = train_test_split(model_input, test_size=0.2)
    # train_model_input = {name: train[name] for name in feature_names}
    # test_model_input = {name: test[name] for name in feature_names}
    # del model_input

    # 4.Define Model,compile and train
    model = DeepFM(linear_feature_columns,dnn_feature_columns,task='binary')
    checkpoint_path = input_dir+'model\\dfmCtr\\' + "cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    # model = DeepFM(linear_feature_columns, dnn_feature_columns,
    #                dnn_hidden_units=(256, 256, 256), l2_reg_linear=0.001,
    #                l2_reg_embedding=0.001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
    #                dnn_dropout=0.5, dnn_activation='relu', dnn_use_bn=True, task='binary')
    try:
        model.load_weights(checkpoint_path);
        print('load weights')
    except:
        pass
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics={'output_acc': 'accuracy', 'output_auc':'auc'})
    # model.compile("adam", "mse", metrics=['mse'], )
    # checkpoit = ModelCheckpoint(filepath=os.path.join("D:\\data\\input\\tl\model\\dfm2", 'model-{epoch:02d}.h5'))
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    history = model.fit(model_input, data[target].values, batch_size=64, epochs=5, verbose=20, validation_split=0.2,  callbacks=[cp_callback])
    # history = model.fit(model_input, data[target].values, batch_size=1024, epochs=20, verbose=2, validation_split=0.2,  callbacks=[checkpoint_path, tensorboard])
    # history = model.fit(train_model_input, train[target],
    #                     batch_size=1024, epochs=5, verbose=2, shuffle=True,
    #                     callbacks=[cp_callback],
    #                     validation_data=(test_model_input, test[target]))
    model.save("dfm.h5")

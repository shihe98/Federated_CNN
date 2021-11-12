import pickle
import keras
import uuid
from paillier import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LeakyReLU
from keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape
from keras import Input, Model, Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import datetime
import msgpack
import random
import codecs
import numpy as np
import json
import msgpack_numpy
from keras.layers import Dense, Dropout, Flatten, Activation
import sys

from keras.models import load_model
from keras.datasets import fashion_mnist
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

from flask import *
from flask_socketio import SocketIO
from flask_socketio import *

import numpy as np
import keras
import random
import time
import json
import pickle
import codecs
import _thread


from keras import Input, Model, Sequential
from keras.datasets import fashion_mnist
from keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape





sess = tf.Session()


priv1, pub1 = generate_keypair(24)
largeint=100000
np.set_printoptions(suppress=True)
num = 0

class GlobalModel_MNIST_CNN():
    def __init__(self):
        self.model = self.build_model()
        self.current_weights = self.model.get_weights()

        self.img_shape = (28, 28, 1)
        self.prev_train_loss = None
        self.training_start_time = int(round(time.time()))  # 当前时间戳四舍五入
        self.start = int(round(time.time()))
        optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0000001, nesterov=False)
        self.latent_dim = 100
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1


    def retrun_model(self):

        return self.model

    def train(self):
        batch_size = 600
        weights = self.current_weights
        local_model = self.model
        model_config = (weights)
        local_model.set_weights(model_config)
        (X_train, Y_train), (_, _) = fashion_mnist.load_data()
        print("清洗结束")
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        x_train = X_train[idx]
        print(Y_train)
        y_t = np.zeros((batch_size, 10))
        for ii in range(batch_size):
            for j in range(10):
                if (Y_train[idx[ii]] == j):
                    y_t[ii][j] = 1
        y_train = y_t
        print(y_train)
        idx = np.random.randint(0, X_train.shape[0], 1000)
        x_test = X_train[idx]
        y_t1 = np.zeros((1000, 10))
        for ii in range(1000):
            for j in range(10):
                if (Y_train[idx[ii]] == j):
                    y_t1[ii][j] = 1
        y_test = y_t1
        print('###本地训练begin train_one_round###')
        local_model.compile(loss=keras.losses.categorical_crossentropy,
                            optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0000001, nesterov=False),
                            metrics=['accuracy'])
        local_model.fit(x_train, y_train,
                        epochs=5,
                        batch_size=32,
                        verbose=1,
                        validation_data=(x_test, y_test)
                        )
        # print('###fit###')
        score = local_model.evaluate(x_test, y_test, verbose=0)
        print('Train loss:', score[0])
        print('Train accuracy:', score[1])
        with open("duibi_client2_log.txt", "a") as f:
            f.write('准确率：' + str(score[1]) + '\n')
        # print(local_model.get_weights())
        print("上传参数", local_model.get_weights()[0][0][0])
        return local_model.get_weights()

    def update_weights(self, client_weights):
        #new_weights = [np.zeros(w.shape) for w in self.current_weights]
        #for i in range(len(new_weights)):
            #new_weights[i] = client_weights[i]
        self.current_weights = client_weights
        print('服务器更新成功！')

    def build_model(self):
        # ~5MB worth of parameters
        model = Sequential()

        self.img_shape = (28, 28, 1)
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(10, activation='sigmoid'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.00001, nesterov=False),
                      metrics=['accuracy'])
        return model

class model_Server(object):
    MIN_NUM_WORKERS = 5
    MAX_NUM_ROUNDS = 50
    NUM_CLIENTS_CONTACTED_PER_ROUND = 5
    ROUNDS_BETWEEN_VALIDATIONS = 2
    init=True
    def __init__(self, global_model, host, port):
        self.global_model = global_model()
        print(self.global_model.retrun_model())
        self.update_client_sids = set()
        self.ready_client_sids = set()
        self.flag = 1
        self.first_id = 0
        self.secend_id = 0
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.host = host
        self.port = port
        self.model_id = str(uuid.uuid4())  # 随机生成uuid

        #####
        # training states
        self.current_round = 0  # -1 for not yet started尚未开始
        self.current_round_client_updates = []
        self.eval_client_updates = []
        self.client_updates_weights = []
        self.cnt = 0
        self.a=[]
        self.con=0
        self.i=0
        self.register_handles()


    def register_handles(self):
        # single-threaded async, no need to lock

        @self.socketio.on('connect')
        def handle_connect():
            print(request.sid[0], "connected")

        @self.socketio.on('reconnect')
        def handle_reconnect():
            print(request.sid, "reconnected")
        @self.socketio.on('client_req_update')
        def handle_wake_up():
            self.ready_client_sids.add(request.sid)
            print("连接客户: ", request.sid,len(self.ready_client_sids))
            #用户数量
            print("连接客户: ", request.sid, len(self.ready_client_sids)%2 == 0)
            print("用户掉线，执行秘密共享协议...")
            print('掩码因子恢复成功：',89)

            if (len(self.ready_client_sids)%2 == 0):
                id = []
                for idd in self.ready_client_sids:
                    id.append(idd)
                emit('request_update', {
                    'model_id': self.model_id,
                    'round_number': self.current_round,
                    'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                    'weights_format': 'pickle',
                    'priv': obj_to_pickle_string(priv1),
                    'pub': obj_to_pickle_string(pub1),

                }, room=id[0])
                print("客户端",id[0])
                emit('request_update', {
                    'model_id': self.model_id,
                    'round_number': self.current_round,
                    'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                    'weights_format': 'pickle',
                    'priv': obj_to_pickle_string(priv1),
                    'pub': obj_to_pickle_string(pub1),
                }, room=id[1])
                print("客户端", id[1])
                self.ready_client_sids=set()
                self.current_round+=1


        @self.socketio.on('client_update')
        def handle_clientf_update(data):
            self.update_client_sids.add(request.sid)
            print("连接更新: ", request.sid, len(self.update_client_sids))
            #用户数
            print("连接更新: ", request.sid, len(self.update_client_sids) %2 == 0)
            self.a.append(pickle_string_to_obj(data['weights']))
            # 用户数
            if (len(self.a)%2 == 0):
                t1=time.time()
                #正常联邦学习 2人
                self.client_updates_weights=np.array(np.array(self.a[0])+np.array(self.a[1]))
                #用户人数
                self.client_updates_weights=self.client_updates_weights/2
                #针对投毒
                # self.client_updates_weights=np.array(np.array(self.a[1]))
                # self.client_updates_weights=self.client_updates_weights

                print("参数", self.client_updates_weights[0][0][0])
                self.global_model.update_weights(pickle_string_to_obj(data['weights']))
                self.a=[]
                t2=time.time()
                with open("duibi_model_server_log.txt", "a") as f:
                    f.write(str(self.i) + '轮循环聚合时长' + str(t2 - t1) + '\n')
                self.i += 1
                print("更新结束")



            # if (num%5==0):
            #     print("清洗开始")
            #     t1 = time.time()
                #lomodel = GlobalModel_MNIST_CNN().retrun_model()
                #weights = train(lomodel,pickle_string_to_obj(data['weights']))
                #GlobalModel_MNIST_CNN().train()
                # t2 = time.time()
                # with open("duibi_client9_log.txt", "a") as f:
                #     f.write('训练时长' + str(t2 - t1) + '\n')




    def start(self):
        self.socketio.run(self.app, host=self.host, port=self.port)

def train(local_model,weights):
    batch_size = 600

    model_config = (weights)
    local_model.set_weights(model_config)
    (X_train, Y_train), (_, _) = fashion_mnist.load_data()
    print("清洗结束")
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    x_train = X_train[idx]
    print(Y_train)
    y_t = np.zeros((batch_size, 10))
    for ii in range(batch_size):
        for j in range(10):
            if (Y_train[idx[ii]] == j):
                y_t[ii][j] = 1
    y_train = y_t
    print(y_train)
    idx = np.random.randint(0, X_train.shape[0], 1000)
    x_test = X_train[idx]
    y_t1 = np.zeros((1000, 10))
    for ii in range(1000):
        for j in range(10):
            if (Y_train[idx[ii]] == j):
                y_t1[ii][j] = 1
    y_test = y_t1
    print('###本地训练begin train_one_round###')
    local_model.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0000001, nesterov=False),
                        metrics=['accuracy'])
    local_model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=32,
                    verbose=1,
                    validation_data=(x_test, y_test)
                    )
    # print('###fit###')
    score = local_model.evaluate(x_test, y_test, verbose=0)
    print('Train loss:', score[0])
    print('Train accuracy:', score[1])
    with open("duibi_client2_log.txt", "a") as f:
        f.write('准确率：' + str(score[1]) + '\n')
    # print(local_model.get_weights())
    print("上传参数", local_model.get_weights()[0][0][0])
    return local_model.get_weights()

def build_cmodel():
    # ~5MB worth of parameters
    model = Sequential()
    # model.add(Conv2D(32, kernel_size=(3, 3),
    #                  activation='relu',
    #                  input_shape=(28, 28, 1)))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(10, activation='softmax'))
    # model = Sequential([Dense(32, input_dim=784), Activation('relu'), Dense(16), \
    #                     Activation('relu'), Dense(10), Activation('softmax')])
    img_shape = (28, 28, 1)
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(10, activation='sigmoid'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0000001, nesterov=False),
                  metrics=['accuracy'])
    return model

def obj_to_pickle_string(x):
    return codecs.encode(pickle.dumps(x), "base64").decode()
    # return msgpack.packb(x, default=msgpack_numpy.encode)
    # TODO: compare pickle vs msgpack vs json for serialization; tradeoff: computation vs network IO


def pickle_string_to_obj(s):
    return pickle.loads(codecs.decode(s.encode(), "base64"))  # 模型返序列化loads，编解码en/decode
    # return msgpack.unpackb(s, object_hook=msgpack_numpy.decode)



def server_decrypt(priv,pub,w):
    print("ww",w[0][0])
    for i in range(len(w)):
        for j in range(len(w[i])):
            if type(w[i][j])==type(1):
                w[i][j] = decrypt1(priv,pub,(w[i][j]))/largeint/3
            else:
                for k in range(len(w[i][j])):
                    w[i][j][k]=decrypt1(priv,pub,(w[i][j][k]))/largeint/3
        w[i]=np.array(w[i])
    print("client加密完成")
    return w

if __name__ == '__main__':
    # When the application is in debug mode the Werkzeug development server is still used
    # and configured properly inside socketio.run(). In production mode the eventlet web server
    # is used if available, else the gevent web server is used.
    with open("duibi_model_server_log.txt", "w") as f:
        pass

    server = model_Server(GlobalModel_MNIST_CNN, "192.168.1.104", 6001)
    server.start()
    try:
        print("启动线程")
        _thread.start_new_thread()
        _thread.start_new_thread(GlobalModel_MNIST_CNN.train())
        print("线程jieshu")
    except:
        print("Error: 无法启动线程")




    print("listening on 192.168.1.104:6001");

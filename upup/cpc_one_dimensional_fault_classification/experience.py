# from typing import Counter
from collections import Counter
from keras.backend.theano_backend import reset_uids
from useful_tools import check_label, within_the_class_distance, plot_with_labels, MatHandler, network_encoder, DANN_MatHandler, TEST_MatHandler, plot_with_labels_DANN

from sklearn.manifold import TSNE

from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from  matplotlib.colors import  rgb2hex
from sklearn.cluster import KMeans
from keras.utils import to_categorical

import keras
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

from useful_tools.data_util import MatHandler
from useful_tools import SortedGenerator_Mat_8192
from useful_tools import network_cpc_Mat
from os.path import join, basename, dirname, exists
import keras
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

import matplotlib.pyplot as plt
import numpy as np


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay




def TSNE_Mat(
    code_size,
    signal_size=1024,
    cluster_num = 4
):
    """
    TSNE降维训练好的模型并且画聚类图和类内距离和类间距离
    """

    # 模型
    encoder_input = keras.layers.Input([signal_size, 1])
    encoder_output = network_encoder(encoder_input, code_size)
    encoder_model = keras.models.Model(encoder_input, encoder_output, name='encoder')
    encoder_model.summary()
    encoder_model.load_weights('./models/16_encoder_0-1.h5')

    # 生成数据
    mathandler = TEST_MatHandler()
    data = mathandler.X_val
    label = mathandler.y_val

    # 预测模型
    output = encoder_model.predict(data)

    # 清理没用的指针
    K.clear_session()

    # tsne降维
    tsne = TSNE(perplexity=10, n_components=2, init='random', n_iter=5000)

    low_dim_embs = tsne.fit_transform(output)
    labels = label

    # 画图
    # DONE
    plot_with_labels(low_dim_embs, labels, clusters_number=cluster_num)



def DANN_TSNE_Mat(
        code_size,
        signal_size=1024,
        cluster_num=4
    ):
    """
    TSNE降维训练好的模型并且画聚类图和类内距离和类间距离
    """

    # 模型
    encoder_input = keras.layers.Input([signal_size, 1])
    encoder_output = network_encoder(encoder_input, code_size)
    encoder_model = keras.models.Model(encoder_input, encoder_output, name='encoder')
    encoder_model.summary()
    encoder_model.load_weights('./models/16_encoder_0-1.h5')

    # 生成数据
    mathandler = DANN_MatHandler()
    data = mathandler.X_val
    label = mathandler.y_val

    # 预测模型
    output = encoder_model.predict(data)

    # 清理没用的指针
    K.clear_session()

    # tsne降维
    tsne = TSNE(perplexity=10, n_components=2, init='random', n_iter=5000)
    low_dim_embs = tsne.fit_transform(output)
    labels = label

    # 画图
    # DONE
    plot_with_labels_DANN(low_dim_embs, labels, clusters_number=cluster_num)

##########################画混淆矩阵+计算正确率
def check_10_cluster(code_size, signal_size=1024, dimension=16):
    """
    PCA
    多次
    k_means后数据后计算轮廓系数
    """

    #模型
    encoder_input = keras.layers.Input([signal_size, 1])
    encoder_output = network_encoder(encoder_input, code_size)
    encoder_model = keras.models.Model(encoder_input, encoder_output, name='encoder')
    encoder_model.summary()
    encoder_model.load_weights(r'.\models\16_encoder_0-1.h5')
    # 生成数据                                        
    mathandler = MatHandler()
    data = mathandler.X_val
    label = mathandler.y_val
    # 预测模型
    data = encoder_model.predict(data)

    pca_data = data

    cluster_num = 4

    ################# k_means
    kmeans = KMeans(n_clusters=cluster_num)
    result_list = kmeans.fit_predict(pca_data)
    #################

    confusion, list_Counter = check_label(label ,result_list, result_num=cluster_num)
    confusion_Matrix(confusion = confusion*100)
    print(confusion)

    print(list_Counter)
    max_accuracy = counter_max_accuracy(list_Counter, confusion, result_num=cluster_num)
    print("max_accuracy:"+str(max_accuracy))
    return "suc"


def confusion_Matrix(confusion, epochs, acc, step):
    """
    画混淆矩阵
    """

    plt.clf()

    confusion = np.array(confusion[:,:], dtype=np.int32)
    from matplotlib import rcParams
    config = {
        "font.family":'Times New Roman',  # 设置字体类型
        "font.size": 20,
    }
    font = {'family' : 'Times New Roman',
            'weight' : 'normal',    
            'size'   : 30,
    }
    rcParams.update(config)

    # 热度图，后面是指定的颜色块，可设置其他的不同颜色
    plt.imshow(confusion, cmap=plt.cm.Blues)

    indices = range(len(confusion))
    # # 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表

    plt.xticks(indices, range(4))
    plt.yticks(indices, range(4))

    cb = plt.colorbar()
    cb.set_label('percentage(%)')
    
    plt.xlabel('real class', font)
    plt.ylabel('predict class', font)

    # 显示数据
    for first_index in range(len(confusion)):    #第几行
        for second_index in range(len(confusion[first_index])):    #第几列
            plt.text(second_index, first_index, confusion[first_index][second_index])

    plt.savefig('./picture/confusion/' + str(step) + '_'+ str(acc) +'_replace_confuse_'+ str(epochs) +'.png')


def counter_max_accuracy(list_Counter, confusion, result_num):
    # 计算每个分类总的准确率
    sum_right_number = 0
    sum = 0
    for i in range(result_num):
        temp_class_number = []
        class_num = 0.0
        for j in range(result_num):
            temp_class_number.append(list_Counter[i][class_num])
            sum += list_Counter[i][class_num]
            class_num += 1
        print("第"+str(i)+"类：最大的数"+str(max(temp_class_number)))
        sum_right_number += max(temp_class_number)
    
    return sum_right_number/sum
##########################

def check_label_classifier(label, predict_label, result_num):

    predict_label = predict_label.argmax(axis=1).astype(float)
    list_Counter = []
    confusion = np.zeros((4, 4))

    print(label)
    print(predict_label)
    print(label.shape)
    print(predict_label.shape)
    print(type(label[1]))
    print(type(predict_label[1]))

    for i in range(result_num):
        list_Counter.append(Counter(label[predict_label == i]))



    for i in range(result_num):
        print("第" + str(i) + "类：")
        sum = 0
        class_num = 0.0
        for j in range(result_num):
            sum += list_Counter[i][class_num]
            class_num += 1
        
        class_num = 0.0
        for j in range(result_num):
            if list_Counter[i][class_num] != 0:
                print("类" + str(class_num) + "的百分比：" + str(list_Counter[i][class_num]/sum))
                confusion[i][int(class_num)] = list_Counter[i][class_num]/sum
            class_num += 1

    confusion = np.around(confusion, 2)

    return confusion


def oneD_test_accuracy(
        code_size,
        epochs,
        step,
        signal_size=1024,
        lr=1e-3,
):
    # 模型
    encoder_input = keras.layers.Input([signal_size, 1])
    encoder_output = network_encoder(encoder_input, code_size)
    encoder_model = keras.models.Model(encoder_input, encoder_output, name='encoder')
    encoder_model.summary()
    print('Step is :' + step)
    if step == 'one':
        encoder_model.load_weights('./models/16_encoder_0-1.h5')
    elif step == 'two':
        encoder_model.load_weights('./models/new_DANN_models/encoder_new_DANN_20_epochs_16_batchs.h5')
    elif step == 'three':
        encoder_model.load_weights('./models/replace_models/replace_encoder.h5')
    else:
        raise ValueError("step input error")

    for layer in encoder_model.layers:
        print(layer.trainable)
        layer.trainable = False

    x = keras.layers.Dense(units=8, activation='sigmoid', name='Dense_1')(encoder_output)
    classifier_output = keras.layers.Dense(units=4, activation='softmax', name='Dense_2')(x)

    classifier = keras.models.Model(encoder_input, classifier_output, name='classifier')
    classifier.summary()

    mathandler = MatHandler()
    data = mathandler.X_train
    label = mathandler.y_train
    label = to_categorical(label, num_classes=4)

    classifier.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.Adam(lr=lr),
        metrics=['accuracy']
    )
    classifier.fit(data, label, batch_size=32, epochs=epochs, shuffle=True, verbose=1, validation_split=0.1)
    #classifier.load_weights('./models/classifier_model5_29.h5')

    mathandler1 = TEST_MatHandler()
    test_data = mathandler1.X_val
    test_label = mathandler1.y_val
    tmp_label = test_label
    test_label = to_categorical(tmp_label, num_classes=4)

    score = classifier.evaluate(test_data, test_label)
    predict_label = classifier.predict(test_data)

    # confusion = check_label_classifier(tmp_label, predict_label=predict_label, result_num=4)
    # confusion_Matrix(confusion * 100, epochs, acc=score[1], step=step)
    # print(confusion)
    predict_label = predict_label.argmax(axis=1).astype(float)
    cm = confusion_matrix(tmp_label, predict_label, normalize='true')
    labels = ['0', '1', '2', '3']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    print(cm)
    disp.plot()
    disp.plot(cmap='Blues')  # 蓝色
    plt.savefig('./picture/confusionmax/one/' + '_' + str(score[1]) + '--' + str(epochs) + '--' + str(score[0]) + '.png')
    plt.show()

    print("test loss:", score[0])
    print('test accuracy:', score[1])








def check_label_classifier_now(label, predict_label, result_num):

    predict_label = predict_label.argmax(axis=1).astype(float)
    list_Counter = []

    print(label)
    print(predict_label)
    print(label.shape)
    print(predict_label.shape)
    print(type(label[1]))
    print(type(predict_label[1]))

    for i in range(result_num):
        list_Counter.append(Counter(label[predict_label == i]))

def Classfier(
    code_size,
    epochs,
    batch_size,
    output_dir,
    step,
    signal_size=1024,
    lr=1e-3,
    num_classes=4,
    ):
    '''
    进行测试
    '''
    # 加载数据
    mathandler = MatHandler()
    data = mathandler.X_train
    label = mathandler.y_train
    label = keras.utils.to_categorical(label, num_classes=4)

    # 模型
    encoder_input = keras.layers.Input([signal_size, 1])
    encoder_output = network_encoder(encoder_input, code_size)
    encoder_model = keras.models.Model(encoder_input, encoder_output, name='encoder')
    encoder_model.summary()
    print('Step is :' + step)
    if step == 'one':
        encoder_model.load_weights('./models/16_encoder_0-1.h5')
    elif step == 'two':
        encoder_model.load_weights('./models/new_DANN_models/encoder_new_DANN_20_epochs_16_batchs.h5')
    elif step == 'three':
        encoder_model.load_weights('./models/replace_models/replace_encoder.h5')
    else:
        raise ValueError("step input error")

    # for layer in encoder_model.layers:
    #     layer.trainable = False

    x = keras.layers.Dense(units=8, activation='sigmoid', name='Dense_1')(encoder_output)
    classifier_output = keras.layers.Dense(units=4, activation='softmax', name='Dense_2')(x)  # 输出分类结果

    classifier = keras.models.Model(inputs=encoder_input, outputs=classifier_output,name='classifier')
    classifier.compile(
        optimizer=keras.optimizers.Adam(lr=lr),
        loss = 'categorical_crossentropy',
        metrics=['accuracy']
    )
    for layer in classifier.layers:
        layer.trainable = True

    for layer in classifier.layers:
        if (layer.name == 'dense_2'):
            break
        layer.trainable = False

    classifier.summary()
    classifier.fit(data, label, batch_size=32, epochs=epochs, shuffle=True, verbose=1, validation_split=0.1)
    classifier.save(join(output_dir, 'classifier_model_0-1.h5'))

    #加载测试数据
    mathandler = DANN_MatHandler()
    test_data = mathandler.X_val
    test_label = mathandler.y_val
    tmp_label = test_label
    test_label = keras.utils.to_categorical(test_label, num_classes=4)

    score = classifier.evaluate(test_data, test_label)
    predict_label = classifier.predict(test_data)
    # predict_label = keras.utils.to_categorical(predict_label, num_classes=4)


    # tmp_label = np.argmax(tmp_label,axis=0)
    # predict_label = np.argmax(predict_label,axis=0)
    predict_label = predict_label.argmax(axis=1).astype(float)


    cm = confusion_matrix(tmp_label, predict_label , normalize='true')
    labels=['0','1','2','3']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    print(cm)
    disp.plot(cmap='Blues')  # 蓝色
    plt.savefig('./picture/confusionmax/two/' + '_' + str(score[1]) + '--' + str(epochs) + '--' + str(score[0]) + '.png')
    plt.show()

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # tmp_label = np.argmax(tmp_label, axis=0)
    # predict_label = np.argmax(predict_label, axis=0)
    # cm = confusion_matrix(tmp_label, predict_label, normalize='true')
    # labels = ['0', '1', '2', '3']
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    # print(cm)
    # disp.plot()

    K.clear_session()



if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    #加载模型+画图+计算类内距离+类间距离
    TSNE_Mat(
        code_size=16,
        signal_size=1024,
        cluster_num=4
    )

    # DANN_TSNE_Mat(
    #     code_size=16,
    #     signal_size=1024,
    #     cluster_num=4
    # )

    # Step_three_train_Dense(
    #     code_size=16,
    #     epochs=200,
    #     step='one',
    #     signal_size=1024,
    #     lr=1e-3
    # )


    oneD_test_accuracy(
        code_size=16,
        epochs=100,
        step='one',
        signal_size=1024,
        lr=1e-3,
    )

    # Classfier(
    #     epochs = 100,
    #     step = 'one',
    #     batch_size = 32,
    #     code_size = 16,
    #     output_dir = 'models',
    #     signal_size = 1024,
    #     lr = 1e-3
    # )


    # classfier(
    #     epochs=100,
    #     batch_size=32,
    #     output_dir='models',
    #     code_size=16,
    #     lr=1e-3,
    #     terms=4,
    #     predict_terms=4,
    #     signal_size=1024,
    # )
    # test_accuracy(
    #     code_size=16,
    #     epochs=32,
    #     signal_size=1024,
    #     lr=1e-3,
    # )
    #画混淆矩阵+计算正确率
    # check_10_cluster(
    #     code_size=16,
    #     signal_size=1024,
    #     dimension=4
    # )


    print("suc")



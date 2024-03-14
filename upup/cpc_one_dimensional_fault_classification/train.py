from useful_tools.data_util import MatHandler
from useful_tools import SortedGenerator_Mat_8192
from useful_tools import network_cpc_Mat
from os.path import join, basename, dirname, exists
import keras
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


def train_model_old_cpc(epochs, batch_size, output_dir, code_size, lr=1e-4, terms=4, predict_terms=4, signal_size=1024):
    """
    训练
    """

    # 设置显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    KTF.set_session(sess)

    # 数据集迭代器
    train_data = SortedGenerator_Mat_8192(batch_size=batch_size, subset='train', terms=terms,
                                       positive_samples=1, predict_terms=predict_terms,
                                       signal_size=signal_size)

    validation_data = SortedGenerator_Mat_8192(batch_size=batch_size, subset='valid', terms=terms,
                                            positive_samples=1, predict_terms=predict_terms,
                                            signal_size=signal_size)

    # 构建模型
    model = network_cpc_Mat(image_shape=signal_size, terms=terms, predict_terms=predict_terms,
                        code_size=code_size, learning_rate=lr)

    callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/3, patience=2, min_lr=1e-4)]

    # 训练模型
    model.fit_generator(
        generator=train_data,
        steps_per_epoch=len(train_data),
        validation_data=validation_data,
        validation_steps=len(validation_data),
        epochs=epochs,
        verbose=1,
        callbacks=callbacks
    )

    # 保存总模型
    model.save(join(output_dir, '16_cpc_0-1.h5'))

    # 保存encoder模型
    encoder = model.layers[1].layer
    encoder.save(join(output_dir, '16_encoder_0-1.h5'))


if __name__ == "__main__":

    # 训练模型
    train_model_old_cpc(
        epochs=10,
        batch_size=32,
        output_dir='models',
        code_size=16,
        lr=1e-3,
        terms=4,
        predict_terms=4,
        signal_size=1024
    )

    print("suc")


import tensorflow as tf
from  tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import cv2

# 屏蔽waring信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""------------------加载数据---------------------"""
# 载入数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
# 改变数据格式，为了能够输入卷积层
trX = trX.reshape(-1, 28, 28, 1)  # -1表示不考虑输入图片的数量,1表示单通道
teX = teX.reshape(-1, 28, 28, 1)

"""------------------构建模型---------------------"""
# 定义输入输出的数据容器
X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

# 定义和初始化权重、dropout参数
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

w1 = init_weights([3, 3, 1, 32])        # 3X3的卷积核，获得32个特征
w2 = init_weights([3, 3, 32, 64])       # 3X3的卷积核，获得64个特征
w3 = init_weights([3, 3, 64, 128])      # 3X3的卷积核，获得128个特征
w4 = init_weights([128 * 4 * 4, 625])   # 从卷积层到全连层
w_o = init_weights([625, 10])           # 从全连层到输出层


# 定义模型
def create_model(X, w1, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    # 第一组卷积层和pooling层
    conv1 = tf.nn.conv2d(X, w1, strides=[1, 1, 1, 1], padding='SAME')
    conv1_out = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(conv1_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    pool1_out = tf.nn.dropout(pool1, p_keep_conv)

    # 第二组卷积层和pooling层
    conv2 = tf.nn.conv2d(pool1_out, w2, strides=[1, 1, 1, 1], padding='SAME')
    conv2_out = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(conv2_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    pool2_out = tf.nn.dropout(pool2, p_keep_conv)

    # 第三组卷积层和pooling层
    conv3 = tf.nn.conv2d(pool2_out, w3, strides=[1, 1, 1, 1], padding='SAME')
    conv3_out = tf.nn.relu(conv3)
    pool3 = tf.nn.max_pool(conv3_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    pool3 = tf.reshape(pool3, [-1, w4.get_shape().as_list()[0]])  # 转化成一维的向量
    pool3_out = tf.nn.dropout(pool3, p_keep_conv)

    # 全连层
    fully_layer = tf.matmul(pool3_out, w4)
    fully_layer_out = tf.nn.relu(fully_layer)
    fully_layer_out = tf.nn.dropout(fully_layer_out, p_keep_hidden)

    # 输出层
    out = tf.matmul(fully_layer_out, w_o)

    return out

def recog_number(digit):
        
    model = create_model(X, w1, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

    # 定义代价函数、训练方法、预测操作
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    predict_op = tf.argmax(model, 1,name="predict")

    # 定义一个saver
    saver=tf.train.Saver()

    # 定义存储路径
    ckpt_dir="./ckpt_dir"

    """------------------训练模型---------------------"""
    train_batch_size = 128  # 训练集的mini_batch_size=128
    test_batch_size = 256   # 测试集中调用的batch_size=256
    epoches = 5  # 迭代周期
    with tf.Session() as sess:
        # 初始化所有变量
        tf.global_variables_initializer().run()
        """-----加载模型，用导入的图片进行测试--------"""
        # 载入图片
        
        dst = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_CUBIC)
        dst = dst.reshape(1, 28, 28, 1)
        # 载入模型
        saver.restore(sess,ckpt_dir+"/model.ckpt-9")  # 从第十次的结果中恢复

        # 进行预测
        predict_result = sess.run(predict_op, feed_dict={X: dst,
                                                        p_keep_conv: 1.0,
                                                        p_keep_hidden: 1.0})
        print("你导入的图片是：",predict_result[0])
    return predict_result[0]
import tensorflow as tf
from tensorflow.keras import Sequential, layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


class TrainLeNet:
    """
    构建LeNet-5卷积网络，并进行训练
    """
    def __init__(self):
        """
        @param onehot 实例化独热码
        @param train_file_path 训练集保存路径
        @param test_file_path 测试集保存路径
        """
        self.onehot = OneHotEncoder()
        self.train_file_path = r'F:\Democray\PythonFiles\Digital-Recognition\dataset\csv\mnist_train.csv'
        self.test_file_path = r'F:\Democray\PythonFiles\Digital-Recognition\dataset\csv\mnist_test.csv'


    def LeNet(self):
        """
        构建LeNet-5卷积网络
        """
        model = Sequential([
            layers.Conv2D(6, 5, padding='same', input_shape=(28,28,1), activation='relu'),
            layers.MaxPool2D(pool_size=(2,2), strides=2),
            layers.Conv2D(16, 5, padding='valid', activation='relu'),
            layers.MaxPool2D(pool_size=(2,2), strides=2),
            layers.Conv2D(120, 5, padding='valid', activation='relu'),
            layers.Flatten(),
            layers.Dense(84, activation='relu'),
            layers.Dense(10, activation='sigmoid')
        ])        
        return model


    def train_model(self):
        """
        载入csv格式数据集
        进行训练
        """
        # 加载训练集
        with open(self.train_file_path,encoding = 'utf-8') as f:
            train_data = np.loadtxt(f,delimiter = ",")
            # 归一化
            train_img_normalized = train_data[:,1:]/255.0
            train_img = np.round(train_img_normalized, 2)
            trian_label = train_data[:,0]
            # 将label转化为独热码
            train_label_onehot = self.onehot.fit_transform(trian_label.reshape((-1,1)))
            y_train = train_label_onehot.toarray()
            x_train = train_img.reshape((-1,28,28,1))
            print('x_train shape:\t', x_train.shape)
            print('y_train shape:\t', y_train.shape)

        # 加载测试集
        with open(self.test_file_path,encoding = 'utf-8') as f:
            test_data = np.loadtxt(f,delimiter = ",")
            # 归一化
            test_img_normalized = test_data[:,1:]/255.0
            test_img = np.round(test_img_normalized, 2)
            test_label = test_data[:,0]
            # 将label转化为独热码
            test_label_onehot = self.onehot.fit_transform(test_label.reshape((-1,1)))
            y_test = test_label_onehot.toarray()
            x_test = test_img.reshape((-1,28,28,1))
            print('x_test shape:\t', x_test.shape)
            print('y_test shape:\t', y_test.shape)

        model = self.LeNet()
        # 预览模型
        model.summary()
        # 编译模型
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                    loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
        # 训练模型
        history = model.fit(x=x_train, y=y_train, validation_data=(x_test,y_test), batch_size=50, epochs=30)
        # 保存模型
        model.save('./model/mnist_LeNet_model.h5')
        return history


    def show_img(self):
        """
        画出csv格式数据集的内容
        检查图片中的数字与csv格式数据集是第一列数字否一致
        """
        with open(self.test_file_path,encoding = 'utf-8') as f:
            test_data = np.loadtxt(f,delimiter = ",")
            label = test_data[:,0]
            print(label[16])
            img = test_data[:,1:]
            img_x = img[16,:]
            img_array = np.asfarray(img_x).reshape((28,28))
            plt.imshow(img_array,interpolation = 'nearest')
            plt.show()

        # data_file = open('./mnist_train_20.csv')
        # data_file = open("./dataset/csv/mnist_test_30.csv") # open("文件路径") 该函数用于打开.csv文件，并分配给data_file变量方便使用
        # data_list = data_file.readlines() # readlines()函数用于读取.csv文件并将其读入到data_list变量中
        # data_file.close() # 关闭.csv文件，为了防止之后的处理中不小心对原始.csv文件进行修改
        # len(data_list) # len(变量名)用于检测读取的文件长度

        # all_values = data_list[0].split(',') # split()函数将第49条数据进行拆分，以‘，’为分界点进行拆分
        # image_array = np.asfarray(all_values[1:]).reshape((28,28)) # asfarray()函数将all_values中的后784个数字进行重新排列
        # # reshape()函数可以对数组进行整型，使其成为28×28的二维数组，asfarry()函数可以使其成为矩阵。
        # plt.imshow(image_array, interpolation = 'nearest')  # imshow()函数可以将28×28的矩阵中的数值当做像素值，使其形成图片
        # plt.show()


if __name__ == '__main__':
    tl = TrainLeNet()
    history = tl.train_model()

    loss = history.history['loss']
    accuracy = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_accuracy = history.history['val_accuracy']

    plt.figure(1)
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model Loss Curve')
    plt.legend(['train_loss', 'val_loss'], loc='upper right')
    plt.savefig('./plot/loss_curve.jpg')
    plt.figure(2)
    plt.plot(accuracy)
    plt.plot(val_accuracy)
    plt.title('Model Acc Curve')
    plt.legend(['train_acc', 'val_acc'], loc='lower right')
    plt.savefig('./plot/acc_curve.jpg')
    plt.show()


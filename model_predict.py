import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def predict():
    """
    使用训练好的model对数据集进行预测
    检验model的准确性
    """
    model = tf.keras.models.load_model('./mnist_LeNet_model.h5')
    print(model.summary())
    # 加载数据集
    file_path = r'F:\Democray\PythonFiles\Digital-Recognition\dataset\csv\mnist_test.csv'
    with open(file_path,encoding = 'utf-8') as f:
        predict_data = np.loadtxt(f,delimiter = ",")
        predict_img = predict_data[5,1:]
        predict_img = predict_img.reshape(1,28,28,1)
        print(predict_img.shape)
        img_array = np.asfarray(predict_img)
        print(img_array.shape)
    # 进行预测    
    proporition = model.predict(img_array)
    proporition = tf.nn.softmax(proporition[0]).numpy()
    result = proporition.argmax()
    print('\nproporition: ', proporition)
    print('\n=======================')
    print('result: ', result)
    print('=======================')
    # 画图
    plt.imshow(img_array.reshape(28,28),interpolation = 'nearest')
    plt.show()


if __name__ == '__main__':
    predict()
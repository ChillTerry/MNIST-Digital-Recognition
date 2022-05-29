
def convert(imgFilePath, labelFilePath, outputFilePath, n):
    """
    读取dataset/raw文件夹下的MNIST数据集
    将其转换为csv格式
    """
    img = open(imgFilePath, 'rb')
    label = open(labelFilePath, 'rb')
    output = open(outputFilePath, 'w')
    
    img.read(16)
    label.read(8)
    images = []
    for i in range(n):
        image = [ord(label.read(1))]
        for j in range(28*28):
            image.append(ord(img.read(1)))
        images.append(image)

    for image in images:
        output.write(','.join(str(pixel) for pixel in image)+'\n')

    img.close()
    label.close()
    output.close()


if __name__ == '__main__':
    # convert('./dataset/raw/train-images.idx3-ubyte', './dataset/raw/train-labels.idx1-ubyte', 
    #         './dataset/csv/mnist_train.csv', 60000)

    # convert('./dataset/raw/t10k-images.idx3-ubyte', './dataset/raw/t10k-labels.idx1-ubyte', 
    #         './dataset/csv/mnist_test.csv', 10000)

    convert('./dataset/raw/train-images.idx3-ubyte', './dataset/raw/train-labels.idx1-ubyte', 
             './dataset/csv/mnist_train_200.csv', 200)

    convert('./dataset/raw/t10k-images.idx3-ubyte', './dataset/raw/t10k-labels.idx1-ubyte', 
            './dataset/csv/mnist_test_70.csv', 70)

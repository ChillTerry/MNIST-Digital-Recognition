# MNIST-Digital-Recognition

## preproccess.py
This script is used to convert the raw MNIST dataset into csv formate.
(the full MNIST trianing dataset has 60000 samples and testing dataset has 10000 samples. In order to squeeze the time to quickly see how good the model is, i just select 200 training dataset and 70 testing dataset at the begining. So when you want to train the full dataset, you need to be aware of it, and use the upper two codes)

![image](https://user-images.githubusercontent.com/64240681/170854795-5e9c601d-bbf8-4af5-bb60-8e7986b80bef.png)

## train_model.py
In this script i build a LeNet-5 nearual network and generate the `mnist_LeNet_model.h5` 
The model is good which i just train it 30 epochs but get 99% accuracy

![acc_curve](https://user-images.githubusercontent.com/64240681/170854856-78a61f1b-56a0-4b54-b17d-4a2eb825f21e.jpg)

## model_predict.py
Using the script to check your model's capability.

![image](https://user-images.githubusercontent.com/64240681/170855001-6076deb8-df64-4d1c-b83e-c708afd7bbb6.png)

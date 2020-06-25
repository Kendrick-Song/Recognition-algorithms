'''获取数据集'''
from sklearn import datasets

faces = datasets.fetch_olivetti_faces()
# print(faces.images.shape)


'''查看数据集'''
# from matplotlib import pyplot as plt
# i = 0
# plt.figure(figsize=(20, 20))
# for img in faces.images:
#     #总共400张图，把图像分割成20X20
#     plt.subplot(20, 20, i+1)
#     plt.imshow(img, cmap="gray")
#     #关闭x，y轴显示
#     plt.xticks([])
#     plt.yticks([])
#     plt.xlabel(faces.target[i])
#     i = i + 1
# plt.show()


'''特征数据和标签定义'''
#人脸数据
X = faces.images
#人脸对应的标签
y = faces.target
X = X.reshape(400, 64, 64, 1)


'''随机分隔30%数据作为测试集'''
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


'''构建模型'''
import  tensorflow.keras as keras

model = keras.Sequential()
# 第一层卷积，卷积的数量为128，卷积的高和宽是3x3，激活函数使用relu
model.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu', input_shape=(64, 64, 1)))
# 第二层卷积
model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
#把多维数组压缩成一维，里面的操作可以简单理解为reshape，方便后面Dense使用
model.add(keras.layers.Flatten())
#对应cnn的全链接层，可以简单理解为把上面的小图汇集起来，进行分类
model.add(keras.layers.Dense(40, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


'''训练'''
model.fit(X_train, y_train, epochs=5)
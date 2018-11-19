import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#y_data는 원핫인코딩 상태 0,1,2라고 하면 1인것이 그값
x_data = [[1,2,1,1], [2,1,3,2], [3,1,4,4], [4,1,5,5,], [1,7,5,5], [1,2,5,6,], [1,6,6,6], [1,7,7,7,]]
y_data = [[0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,1,0], [0,1,0], [1,0,0], [1,0,0]]

X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 3])
nb_classes = 3 #분류클래스의 갯수

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b) #소프트맥스 함수로 총합이 1이 되게함
cost = tf.reduce_mean(-tf.reduce_sum(Y  * tf.log(hypothesis), axis=1)) #CSE함수
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
            print(sess.run(hypothesis, feed_dict={X: [[1,2,1,1]]})) #[[0 0 1]]
            #학습이 진행될수록 Y와 비슷해지는 모습을 관측할수 있음
            #H(x)가 0.8에 상회
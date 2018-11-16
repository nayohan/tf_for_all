#기존 리니어 리그레션이 수학에 가깝다면 로지스틱분류는 Y가 X에 비례하는 사항이 아닌 분류를 위한 솔루션으로
#시그모이드함수와 그에따른 코스트펑션이 변하는게 특징
import os
import tensorflow as tf
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

#x1 = 공부한시간 ,x2 = 비디오갯수, y = 성공 실패
xy = np.loadtxt('data_03_diabetes.csv', dtype=np.float32, delimiter=',')
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#가설에 시그모이드함수 추가 / cost에 새로운 함수 / train에 optimizer 병합
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

#로지스틱분류의 경우 가설이 1이면 정답인건데 0.5이상부터 맞다고 바이너리 분류해줌
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
#정확도를 Y와 predicted를 비교해서 정답을 맞췄는지 확인
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 1000 == 0:
            print(step, "cost:", cost_val)

    #잘됬는지 가설,코스트,정확도를 보자
    h, c, a = sess.run([hypothesis, cost, accuracy], feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCost:", c, "\nAccuracy:", a)
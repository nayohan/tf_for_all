"""Recap"""
#Hypothesis H(x) = Wx +b0
#Cost function cost(W,b) = 1/m * 시그마(예측값H(x)- 실제값(y))제곱
#Gradient descent algorithm  W := W - Cost미분값

"""다변수 선형회귀(Multi_variable Linear regression)"""
# x1, x2, x3, ... xn , y
#H(x1,x2,x3) = w1x2 + w2x2 + w3x3 +b
#cost(W,b) = 1/m * 시그마(H(x1,x2,x3) - y)제곱

"""행렬(Matrix)"""
# 수학적으로 => w1x1 + w2x2 + w3x3 + ... + wnxn
#                (w1)
# (x1, x2, x3) *  (w2) = (x1w1 + x2w2 + x3w3)
#                (w3)
# 행렬 표기 -> H(X) = XW

"""Hypothesis using matrix"""
# | x1 | x2 | x3 |  y  |
# | 73 | 80 | 75 | 152 |   <-인스턴스               | x11w1 + x12w2 + x13w3 |
# | 89 | 91 | 90 | 180 |   <-인스턴스   w1          | x21w1 + x22w2 + x23w3 |
# | 93 | 88 | 93 | 185 |        *       w2      =   | x31w1 + x32w2 + x33w3 |
# | 96 | 98 |100 | 196 |                w3          | x41w1 + x42w2 + x43w3 |
# | 73 | 66 | 70 | 142 |                            | x51w1 + x52w2 + x53w3 |
#                  [5,3]        *     [3.1]     =   [5,1]
# [인스턴스갯수, x변수갯수] * [x변수갯수, y변수갯수] = [x인스턴스갯수, y변수갯수]

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

x1_data = [73, 89, 93, 96, 73]
x2_data = [80, 91, 88, 98, 66]
x3_data = [75, 90, 93, 100, 70]
y_data = [152, 180, 185, 196, 142]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b
cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(40000): #hy_val이 y값과 비슷해지는지 보자
    cost_val, hy_val, train_val = sess.run([cost, hypothesis, train], feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, y: y_data})
    if step % 1000 == 0:
        print(step, cost_val, hy_val, train_val)
sess.close()
#이렇게 완성을했다. 다음은 더 최적화해보자.
#cost를 최소화 해보자 #cost값이 0에 가까울수록 잘 훈련된 모델
#가장 cost값이 작은 것을 찾기위해 w와 b에 값을 차례로 다음 이 값들을 토대로 그래프를 그려본다
#y축: cost x축:w

"""경사하강법 (Gradient descent algorithm)"""
#주어진 cost함수로 가장 적절한 w,b를 찾아줌
#그려진 그래프에서 경사가 있는 쪽으로 계속내려가면서 가장 작은 값을 찾는 방법
#아무데서나 시작해서 w,b값을 +,-방향으로 조금씩바꾸고 기존 cost값보다 작으면 더작은값으로 이동함

"""W := W - α * 미분(cost(비용)함수)"""
#간단히 그래프가 2차원그래프라고 하면 1사분면에서 미분값은 + 2사부면이면 -가 나온다.
#기존W가 2사분면에 있으면, 미분값은 -가되고, -- => +가 되서 1사분면쪽으로 W가 움직이게 된다.
#α는 러닝계수(learning_rate)

"""단점"""
#2차원 즉 가중치(W)만 있는 곳에서는 완벽하지만,
#편향(b)가 추가되서 3차원공간에 밥그릇같은 홀에 빠져버리면 나오지 못함
#즉 W,b가 둘다 있는 경우가 대다수므로, cost함수의 모양이 convex함수 모양인지 확인해야함

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

x_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

hypothesis = W * X
cost = tf.reduce_mean(tf.square(hypothesis - Y))

learning_rate = 0.01
gradient = tf.reduce_mean((W * X - Y) * X) * 2 #cost함수 미분
descent = W - (learning_rate * gradient)
update = W.assign(descent)
""" 위와 같은내용 텐서플로우에서 제공하는 API
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
"""

sess = tf.Session()
sess.run(tf.global_variables_initializer()) #초기화괄호조심
for step in range(2000):
    cost_val, W_val, _ = sess.run([cost, W, update], feed_dict={X: x_data, Y: y_data})
    if step % 100 == 0:
        print(step, cost_val, W_val)
sess.close()
















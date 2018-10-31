import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

W = tf.Variable(tf.random_normal([1]), name='weight') #자동으로 변경
b = tf.Variable(tf.random_normal([1]), name='bias')

#플레이스 홀더로 선형회귀값을 넣어보자
x = tf.placeholder(tf.float32, shape=[None]) #형태가 없고 나중에
y = tf.placeholder(tf.float32, shape=[None]) #세션 실행할때 값넣음

hypothesis = W * x + b

cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer()) #초기화 필수

for step in range(2000): #세션을 한번에 실행해서 변수에 저장후 출력 feed_dict로 값 삽입
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={x: [1, 2, 3], y: [4, 5, 6]})
    if step % 100 == 0:
        print(step, cost_val, W_val, b_val)
#잘된다

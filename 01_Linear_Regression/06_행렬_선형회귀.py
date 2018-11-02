# 다변수 선형회귀에서 직접 다 정의해서 하면 힘들다.
# 행렬을 이용해 선형회귀를 학습해보자
#주어진데이터는 05와 같다.
# | x1 | x2 | x3 |  y  |
# | 73 | 80 | 75 | 152 |                            | x11w1 + x12w2 + x13w3 |
# | 89 | 91 | 90 | 180 |                w1          | x21w1 + x22w2 + x23w3 |
# | 93 | 88 | 93 | 185 |        *       w2      =   | x31w1 + x32w2 + x33w3 |
# | 96 | 98 |100 | 196 |                w3          | x41w1 + x42w2 + x43w3 |
# | 73 | 66 | 70 | 142 |                            | x51w1 + x52w2 + x53w3 |
#                  [5,3]        *     [3.1]     =   [5,1]
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

#데이터들의 차원들을 잘 맞춰주어야 함!
#[5,3] * [3,1] = [5,1]  =====>  X * W = Y
x_data = [[73, 80, 75], [89, 91, 90], [93, 88, 93], [96, 98, 100], [73, 66, 70]]
y_data = [[152], [180], [185], [196], [142]]
X = tf.placeholder(tf.float32, shape=[None, 3]) #[5,3]
Y = tf.placeholder(tf.float32, shape=[None, 1]) #[5,1]

W = tf.Variable(tf.random_normal([3, 1])) #[3,1] X와Y 원소하나씩 가져옴
b = tf.Variable(tf.random_normal([1])) #[1] 행렬전부에 더할거라 원소하나면 됨

hypothesis = tf.matmul(X, W) + b #matmul은 행렬곱시켜줌
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(40000): #학습률을 더올리면 정답과 더가까워진다. 시간이 오래걸릴뿐
    cost_val, hy_val, train_val = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 1000 == 0:
        print(step, cost_val, hy_val, train_val)
sess.close()
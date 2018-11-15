# 넘파이로 데이터를 한번에 불러오면 메모리에 문제가 있다
# 텐서플로우에 내장되어 있는 큐러너로 차근차근 불러와보자
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

#파일네임큐에 쪼개져있는 모든파일을 불러온다
filename_queue = tf.train.string_input_producer(
    ['data_01_test_score.csv', 'data_02_test_score.csv'], shuffle=None, name='filename_queue')
#리더로 파일네임큐를 읽는다
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
#레코드형식과 디코드를 한다
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)
#배치에 큐로 만든다
train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=25)

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(tf.random_normal([1]))
hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#큐를 실행
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(40000):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    hyp_val, cost_val, _ = sess.run([hypothesis, cost, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 1000 == 0:
        print(step, "Cost:", cost_val)

coord.request_stop()
coord.join(threads)
#8분 실화냐 왜이럴까

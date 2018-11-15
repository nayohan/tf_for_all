import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np                  #넘파이 배열,행렬계산에 편리함
import matplotlib.pyplot as plt     #그래프 그려주는 라이브러리

"""Slicing"""
nums = [0, 1, 2, 3, 4]  #"[0,1,2,3,4]"
print(nums[2:4]) #index 2 to 4   "[2,3]" (exclusive)
print(nums[2:])  #index 2 to end "[2,3,4]"
print(nums[:2]) #start to index2 "[0,1]" (exclusive)
print(nums[:])   #whole list     "[0,1,2,3,4]"
print(nums[:-1]) #same as [:4]    "[0,1,2,3]"

"""2차원 Slicing"""
#2차원에서도 적용이 가능하다. 행과 열을 따로 각각 생각해보자
b = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
print(b[:, -1])     #[3, 6, 9, 12]ㅋ
print(b, -1)        #[10, 11, 12]
print(b[[-1], :])     #[10, 11, 12]
print(b[-1, ...])   #[10, 11, 12]
print(b[0:2, :])    #[[1, 2, 3], [4, 5, 6]]

"""데이터 불러오는 기존방식"""
# x_data = [[73, 80, 75], [89, 91, 90], [93, 88, 93], [96, 98, 100], [73, 66, 70]]
# y_data = [[152], [180], [185], [196], [142]]

"""파일로 불러오는 방식"""
#넘파이의 loadtxt를 사용하고, delimiter로 각 데이터를 구분 dtpye으로 데이터의 형태 입력
xy = np.loadtxt('data_01_test_score.csv', dtype=np.float32, delimiter=',')
x_data = xy[:, 0:3] #슬라이싱이 여기서 쓰임
y_data = xy[:, [-1]] #[]를 해야 [y]로 [인스턴스갯수,1]의 행렬이 생성됨

X = tf.placeholder(tf.float32, shape=([None, 3]))
Y = tf.placeholder(tf.float32, shape=([None, 1]))

#컴퓨터가 건드릴 변수
W = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(tf.random_normal([1]))

#그래프
hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

#학습횟수, 그래플 보여줄 Variable (x, cost_val)
iters_num = 40000
x_show = np.arange(0, iters_num, 1) #np.arange(start, stop, step)
cost_val = np.arange(0, iters_num, 1)  # int32, (20000,)

#세션실행
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(iters_num):
    cost_val[step], hyp_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y:y_data})
    #if step % 10000 == 0:   #그래프로 대체
        #print(step, cost_val[step], hyp_val)

#학습은 다됬으니 맞추는지 보자
print("Your score will be ", sess.run(hypothesis, feed_dict={X: [[100, 100, 100]]}))
print("Other score will be ", sess.run(hypothesis, feed_dict={X: [[10, 10, 10], [20, 20, 20]]}))
sess.close()

#그래프로도 보여줌
plt.ylim(1, 15)
plt.plot(x_show, cost_val)
plt.show()
#20초
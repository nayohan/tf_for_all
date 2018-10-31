import os #텐서플로우GPU를 쓰면 나오는 오류제거
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

"""Linear Regression (선형회귀)"""
#x 예측을하기위한 기본적인 자료
#y 결과값

"""Hypothesis function(가설)"""
#선형회귀 데이터로 만들어낸 함수
#1차 방정식 형태의 가설 : H(x)= Wx + b

"""cost function(손실)(loss)"""
#실제값과 얼마나 큰차이가 있는지 알려주는 함수
#cost : (H(x) - y)^2
#cost(W,b) `=` {1} over {m} sum _{i=1} ^{m} (H(x ^{(i)} )-y ^{(i)} ) ^{2}

"""Hypothesis 와 cost 구현해보자"""
#1. 그래프 빌드 -> 2. 세션런 -> 3.결과 업데이트
#주어진 xy 데이터
x_train = [1, 2, 3]
y_train = [4, 5, 6]

#variable : W,b -> 텐서플로우학습과정에서 기계가 변경시키는 값
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#Hypothesis 노드 정의
hypothesis = W * x_train + b

#cost 노드 정의
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

#cost를 minimize(GD) 경사하강법으로 추후설명
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(cost)

#세션런
sess = tf.Session()
sess.run(tf.global_variables_initializer()) #초기화

for step in range(40000):
    sess.run(train) #학습시작
    if step % 500 == 0: #결과물 출력
        print(step, sess.run(cost), sess.run(W), sess.run(b))
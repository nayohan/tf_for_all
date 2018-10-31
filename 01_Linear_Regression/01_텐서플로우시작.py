import os #텐서플로우GPU를 쓰면 나오는 오류제거
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
print(tf.__version__)

"""헬로 텐서플로우 출력하기"""
hello = tf.constant("Hello, Tensorflow!") #노드 생성
sess = tf.Session()                       #세션을 만들어두고
print(sess.run(hello))                    #세션에서 노드를 실행해주는 방식
#출력시 b'Hello, Tensorflow'라고 나오는데 바이트 세션이란는 뜻

"""노드"""
node1 = tf.constant(3.0, tf.float32) #상수노드 생성
node2 = tf.constant(2.0) #위와 동일하게
node3 = tf.add(node1, node2)
#노드를 출력하면 그래프의 요소(텐서)라고 나옴
print("node1 : ", node1)
print("node2 : ", node2)
print("node3 : ", node3)

#결과값을 나오게하려면 세션을 만들고, sess.run으로 그래프의 노드를 실행시켜야함
sess = tf.Session()
print(sess.run(node3))
#텐서플로우 전체 구조 그래프를 빌드 -> 세션 런 -> 그래프 결과를 출력

"""플레이스홀더"""
#세션을 만들고 실행시킬때 값을 넣어주면서 그래프를 실행시킬때 사용
#형태가 자유롭고, 값을 따로 집어넣을 수 있음
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = tf.add(a, b)

#그래프 정의 -> 세션실행(피드 딕트로 값 삽입) -> 결과 도출
sess = tf.Session()
print(sess.run(adder_node, feed_dict={a: 2, b: 3.5}))

#텐서 Ranks, Shapes, and Types 세종류
#랭크     0= 스칼라 1= 벡터 2=매트릭스 3=3차원 텐서 n=n차원 텐서
#셰이프   [] 안에 몇개의 요소를 가지는지
#타입     tf.float32,64 tf.int32,64

"""
lambda 와 partial 설명
"""
# lambda
f = []
for i in range(5):
    f.append(lambda : print(i))

print(i)  # lambda 실행

# 실행되는 시점에서 i = 4 이기 때문에 4 가 나왔다.
f[0]()
f[1]()
f[2]()
f[3]()
f[4]()

# python 에서는 함수 바깥의 t 출력할 수 있다.
t = 1
def f():
    t2 = 2
    def g():
        print('init t = ', t)
        print('t2 = ', t2)
    g()  # 함수 바깥의 파라미터 실행시킬 수 있다.
    t2 = 11  # 바뀌지 않음 ! why? f() 안의 변수 2개인 t2 중에서 제일 먼저 찾은 것을 리턴한다.
f()
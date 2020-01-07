"""
파라미터 최적화 알고리즘
4) adam (Adaptive Moment estermate)
1. adam 개념 설명
Adagrad + Momentum 이 결합
학습률 변화 + 속도(momentum) 개념 도입

2. 알고리즘 설명
W : 파라미터
t (timestamp) : 반복할 때마다 증가하는 숫자.
- 시간이 지날 때마다 t 증가
- update 메소드가 호출될 때마다 +1
m (momentum) : 첫번째 모멘텀
v : 두번째 모멘텀
beta1, beta2 : m , v 변화에 사용되는 상수
lr : 학습률(learning rate)

m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad * grad
m_hat = m / (1 - beta1 ** t)
v_hat = v / (1 - beta2 ** t)
W = W - lr * m_hat / sqrt(v_hat) (m 이 gradient 라고 생각한다면, 이 식은 adagrad 와 비슷하다)
"""

class Adam:
    def __init__(self, ):
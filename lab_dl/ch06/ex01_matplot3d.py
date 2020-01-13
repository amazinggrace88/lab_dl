"""
그래프 그리기
- 3d 그래프
- 등고선 그래프
"""
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # 3D 그래프를 그리기 위해 반드시 import 해야 하는 패키지
import numpy as np


def fn(x, y):
    """f(x, y) = (1/20) * x**2 + y**2"""
    return x**2 / 20 + y**2
    # 두 개의 편미분이 존재
    # partial f/partial x = x/10
    # partial f/partial y = 2*y


def fn_derivative(x, y):
    return x / 10, 2 * y  # python 에서는 두 가지 리턴 가능


if __name__ == '__main__':
    # 1) 3 demension graph
    # x 좌표 만들기
    x = np.linspace(-10, 10, 100)
    # y 좌표 만들기
    y = np.linspace(-10, 10, 100)
    # 3 차원 그래프를 그리기 위하여 (x, y) 의 쌍으로 이루어진 데이터가 필요
    X, Y = np.meshgrid(x, y)
    Z = fn(X, Y)  # (x, y, z) 완성
    fig = plt.figure()
    ax = plt.axes(projection='3d')  # projection : mpl_toolkits.mplot3d 필요
    ax.contour3D(X, Y, Z, 1000, cmap='OrRd')  # Z 축은 등고선(contour)를 쌓아나가는 의미로 1000개나 100개 등 숫자를 조절할 수 있다.
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # 2) 등고선(contour) 그래프
    plt.contour(X, Y, Z, 50, cmap='PuRd')  # 50 : 등고선의 갯수
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')  # 보는 방향(위) 설정
    plt.show()

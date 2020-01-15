"""
im2col 함수를 사용한 4차원 데이터 convolution 구현
"""
import numpy as np
from lab_dl.lab_dl.common.util import im2col


if __name__ == '__main__':
    np.random.seed(115)

    # p.238 그림 7-11 참조
    # 가상의 이미지 데이터 1개 생성
    # (n, c, h, w) = (이미지 갯수, color-depth, height, width) 4차원
    x = np.random.randint(10, size=(1, 3, 7, 7))  # rgb 3개, h=7, w=7 인 이미지 1장
    print(x, ', shape:', x.shape)  # 큰 배열 안에 원소 1개 - 이미지 1장
    
    # (3, 5, 5) 크기의 필터 1개 생성
    # (fn, c, fh, fw) = (필터 갯수, color-dept, filter height, filter width)
    # x 와 w 의 c 는 같다.
    w = np.random.randint(5, size=(1, 3, 5, 5))
    print(w, ', shape:', w.shape)

    # 필터 적용 : stride=1, padding=0으로 적용하면서 convolution(합성곱) 연산한다고 가정
    #           filter_h = ((h - fh + 2*p)/s) + 1 계산하여 직접 parameter 넣어준다.
    #           filter_w = ((w - fw + 2*p)/s) + 1
    # 이미지 데이터 x 를 함수 im2col 에 전달해보자.
    x_col = im2col(x, filter_h=5, filter_w=5, stride=1, pad=0)
    print('x_col:', x_col.shape)  # (9, 75)
    """
    (9, 75) = (oh*ow, c*fh*fw)
    oh*ow 의 의미 -> x 에서 w 가 움직이는 범위 (=이미지 1장을 펼친 범위)
    75 의 의미 -> dot 연산을 시키기 위해 w 의 shape (c, fh, fw)를 c*fh*fw=75 로 펼침
    """
    # 4차원 배열인 필터 w를 2차원 배열로 변환
    w_col = w.reshape(1, -1)  # row 의 갯수가 1, 모든 원소들은 column 으로 변환
    print(w_col, ', shape:', w_col.shape)
    w_col = w_col.T
    print(w_col, ', shape:', w_col.shape)

    # 2차원으로 변환된 이미지와 필터를 dot 연산
    out = x_col.dot(w_col)
    print('out:', out.shape)

    # dot product 의 결과를 (n, oh, ow, fn) 형태로 reshape
    out = out.reshape(1, 3, 3, -1)  # -1 : 남는 원소 그대로 배열
    print('out:', out.shape)
    """
    reshape 의 축의 순서에 유의할 것
    (1, 3, 3, 1) = (n, oh, ow, color-depth) -> color depth 가 앞으로 가야 정상적 형태
    """
    out = out.transpose(0, 3, 1, 2)  # c 인덱스 3번을 인덱스 1번으로 옮긴다.
    print(out, 'out:', out.shape)

    # 실습
    # p.238 그림 7-12을 p.244 그림 7-19 처럼 만들기
    # 가상으로 만들었던 이미지 데이터 x, 2차원으로 변환한 x_col 사용
    # 1. w 생성
    # (3, 5, 5) 필터를 10개 생성 --> w.shape=(10, 3, 5, 5)=(fn, c, fh, fw)
    w = np.random.randint(5, size=(10, 3, 5, 5))
    print(w, ', shape:', w.shape)
    # 2. w 변형
    # --> 갯수는 놔두되 (3, 5, 5) 만 펼친다. (갯수는 놔두어야 한다) 즉, (fn, c*fh*fw)
    w_col = w.reshape(10, -1)
    print(w_col, ', shape:', w_col.shape)   # shape: (10, 75)
    # 3. dot with w.T
    # x_col (9, 75) @ w.T (75, 10)
    out = x_col.dot(w_col.T)
    print('out:', out.shape)  # (9, 10)
    # 4. dot 연산의 결과를 변형(reshape)
    # --> (n, oh, ow, fn) : x 이미지 갯수 1개
    out = out.reshape(1, 3, 3, 10)
    print('out:', out.shape)
    # 5. transpose
    # --> reshape 축 순서를 바꿀 것 (n 이 앞으로)
    out = out.transpose(0, 3, 1, 2)
    print('out:', out.shape)

    print()
    # 실습 2
    # p.239 그림 7-13, p.244 그림 7-19 참조
    # 1. (3, 7, 7) 이미지 12개 난수 생성 -> (n, c, h, w) = (12, 3, 7, 7)
    x = np.random.randint(10, size=(12, 3, 7, 7))
    print('x : ', x.shape)
    # 2. (3, 5, 5) shape의 필터 10개 난수로 생성 -> (fn, c, fh, fw) = (10, 3, 5, 5)
    w = np.random.randint(5, size=(10, 3, 5, 5))
    print('w : ', w.shape)
    # stride=1, padding=0일 때, output height, output width =?
    # oh = ((7 - 5 + 2*0) / 1) + 1
    # ow = ((7 - 5 + 2*0) / 1) + 1
    # 3. 이미지 데이터 x 를 im2col 함수를 사용하여 x_col 로 변환 -> (108, 75)=(n*oh*ow, c*fh*fw)
    x_col = im2col(x, filter_h=5, filter_w=5, stride=1, pad=0)
    print('x_col :', x_col.shape)
    """
    n 개의 데이터는 x_col.shape=(n*oh*ow, c*fh*fw) --> n이 추가된다.
    """
    # 4. 필터 w를 x_col과 dot 연산을 할 수 있도록 reshape & transpose: w_col=(fn, c*fh*fw)=(10, 75)
    w_col = w.reshape(10, -1)
    print('w_col : ', w_col.shape)
    # x_col @ w_col
    w_col = w_col.T
    out = x_col.dot(w_col)
    print('out : ', out.shape)  # (108, 75)*(75, 10)=(108, 10)
    # @ 연산의 결과를 reshape & transpose
    out = out.reshape(12, 3, 3, 10)  # (n*oh*ow, fn)-> reshape (n, oh, ow, fn)
    print('out : ', out.shape)
    out = out.transpose(0, 3, 1, 2)  # (n이미지갯수, fn필터갯수, oh결과높이, ow결과가로)
    print('out : ', out.shape)

    # tensorflow 와는 다른 방식 - 채널이 가장 마지막에 있다. channel이 맨 뒤에 가야하는 경우
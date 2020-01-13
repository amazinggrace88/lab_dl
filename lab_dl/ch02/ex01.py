"""
퍼셉트론(percenptron) 
 다수의 신호를 입력받아서 하나의 신호를 출력
 일종의 함수, (x_1(가중치 w_1), x_2(가중치 w_2)을 입력받아 y(출력은 1개)를 출력)
 함수 파라미터, 리턴값과 유사
"""
"""
논리 회로의 종류
 컴퓨터가 사용하는 소자는 전기 흐름(1), 흐르지 않는 상태(0)
 하드디스크의 소자들이 반도체 (전기 흐르는 물질-도체, 전기가 흐르지 않는 물질-부도체) 로 이루어져 있다.
 
AND
: 두 입력이 모두 1 일때만 1을 출력
NAND
: 두 입력이 모두 1이 아닐때만 1을 출력 (AND의 반대)
OR
: 입력 중 하나 이상이 1이면 1을 출력
XOR
: 배타적 논리합, x_1, x_2 중 한쪽이 1일 때만 1을 출력
"""
def and_gate(x1, x2):
    w1, w2 = 1, 1  # 가중치 1
    bias = -1
    y = x1 * w1 + x2 * w2 + bias
    if y > 0:
        return 1
    else:
        return 0


def nand_gate(x1, x2):
    w1, w2 = 1, 1
    bias = -1
    y = x1 * w1 + x2 * w2 + bias
    if y > 0:
        return 0
    else:
        return 1


def or_gate(x1, x2):
    w1, w2 = 1, 1
    bias = 1
    y = x1 * w1 + x2 * w2 + bias
    if y >= 2:
        return 1
    else:
        return 0


def xor_gate(x1, x2):
    """Exclusive OR : 배타적 OR는 선형관계식(y = x1 * w1 + x2 * w2 + bias) 하나만 이용해서는 만들 수 없음
    NAND, OR, AND를 조합해야 가능"""
    # 하나의 퍼셉트론으로는 구현이 불가~
    # NAND 교집합 OR
    z1 = nand_gate(x1, x2)
    z2 = or_gate(x1, x2)
    return and_gate(z1, z2)  # forward propagation(순방향 전파)


if __name__ == '__main__':
    for x1 in (0, 1):
        for x2 in (0, 1):
            print(f'AND({x1}, {x2}) -> {and_gate(x1, x2)}')
    for x1 in (0, 1):
        for x2 in (0, 1):
            print(f'NAND({x1}, {x2}) -> {nand_gate(x1, x2)}')
    for x1 in (0, 1):
        for x2 in (0, 1):
            print(f'OR({x1}, {x2}) -> {or_gate(x1, x2)}')
    for x1 in (0, 1):
        for x2 in (0, 1):
            print(f'XOR({x1}, {x2}) -> {xor_gate(x1, x2)}')



"""
pickle 데이터 타입
"""
# dict, list 를 파일에 저장하기 위한 방법
# binary 로 저장하고, 어디서 끊는지 정보는 또 파일에 있어야 한다.
# serialize(직렬화) : pickling - 객체 타입을 파일에 저장 (직렬 serial) - 데이터를 쭉 펼쳐놓겠다~ 텍스트가 아니라 알아볼 수 없도록!
#                    서버에서 HTML 파일을 보낼 때에도 직렬화하여 보낸다. 
# deserialize(역직렬화) : unpickling - 파일에 있는 내용을 객체 타입으로 변환 (병렬 parallel)
#                       서버에서 파일을 받으면 역직렬화 해야 함
# pickle 은 파이썬만의 특징!

# 1. 직렬화로 배열 저장
import pickle  # 모듈 pickle 사용하기~

arr = [1, 100, 'A', 3.141592]
with open('array.pickle', mode='wb') as f:  # w : write, b : binary - 데이터를 binary 타입으로 write 한다~
    pickle.dump(arr, f)  # object 을 file 에 저장 - serialization 직렬화


# 2. 역직렬화로 파일에서 배열 꺼내기
# 파일 -> 객체 deserialization (역직렬화)
with open('array.pickle', mode='rb') as f:  # r : read, b : binary - binary 타입 데이터를 read 한다~
    data = pickle.load(f)  # file 에서 load 되어 리스트 형태로 변환 끝냄 - deserialization 역직렬화
print(data)


# 3. 실습
data = {
    'name': '오쌤',
    'age': 16,
    'k1': [1, 2.0, 'AB'],
    'k2': {'tel': '010-0000-0000', 'email': 'jake@test.com'}
}
# data 를 data.pkl 파일에 저장
with open('data.pkl', mode='wb') as f:
    pickle.dump(data, f)

# data.pkl 을 읽고 dict 객체 복원
with open('data.pkl', mode='rb') as f:
    data = pickle.load(f)
print(data)

"""pickle 기능!
csv 파일 - text 파일이므로 문자열, 숫자열 그대로 보이므로 크기가 바이너리보다 크다.
우리는 df를 만들어서 분석하다가 다 못끝냈을 때, 파일을 다시 열면 csv 파일 다시 여는데만 시간이 더 오래 걸린다.
처음 만들때 df 객체 자체를 pickle 로 저장하면, pickle 여는 시간이 훨씬 적게 걸린다.
"""
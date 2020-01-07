"""
ex11.py 에서 저장한 pickle 파일을 읽고, 파라미터(가중치/편향 행렬)들을 출력
"""
import pickle

with open('params.pickle', 'rb') as file:  # b 명시, 파일이 폴더에 꼭 있어야 함
    params = pickle.load(file)
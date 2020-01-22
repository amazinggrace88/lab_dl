"""
현재 버전보다 설치된 버전이 낮은 패키지를 찾을 때
pip list --outdated
설치된 패키지를 업데이트할 때
pip install --upgrade 패키지1, 패키지2,..
tensorflow 설치
pip install tensorflow (GPU 사용하지 않는 버전)
pip install tensorflow-gpu (GPU 사용하는 버전)
keras 설치
pip install keras
"""
import tensorflow as tf
import keras

print('Tensorflow Version : ', tf.__version__)
print('Keras Version : ', keras.__version__)

# Using TensorFlow backend. 의미 : 케라스가 backand 에서 tensorflow 를 사용한다
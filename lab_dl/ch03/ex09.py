"""
PIL 패키지와 numpy 패키지를 이용하면,
이미지 파일(jpg, png, bmp, ...)의 픽셀 정보를 numpy.ndarray 형식으로 변환하거나
numpy.ndarray 형식의 이미지 픽셀 정보를 이미지 파일로 저장할 수 있습니다.

- 이미지 다운, 변경하기
"""
import numpy as np
from PIL import Image


def image_to_pixel(image_file):
    """이미지 파일 이름(경로)를 파라미터로 전달받아서,
    numpy.ndarray에 픽셀 정보를 저장해서 리턴. - height, width, .. - 가로, 세로 다르다. """
    img = Image.open(image_file, mode='r')  # fp : file pointer, 이미지 파일 오픈
    print(type(img))
    # <class 'PIL.JpegImagePlugin.JpegImageFile'> ... imagefile 이름 나옴
    pixels = np.array(img)  # 이미지 파일 객체를 numpy.ndarray 형식으로 변환
    # color : 1bit(gray scale), 24bit(RGB), 32bit(RGBA : RGB + 불투명도(255가 불투명하다는 의미, 0 불투명이 제일 적음, 투명))
    print('pixels shape : ', pixels.shape)
    return pixels


def pixel_to_image(pixel, image_file):
    """numpy.ndarray 형식의 이미지 픽셀 정보와, 저장할 파일 이름을 파라미터로
    전달받아서, 이미지 파일을 저장"""
    img = Image.fromarray(pixel)  # ndarray 타입의 데이터를 이미지로 변환
    print(type(img))  # Image 클래스
    img.show()  # 이미지 뷰어를 사용해서 이미지 보기
    img.save(image_file)


if __name__ == '__main__':
    # image_to_pixel(), pixel_to_image() 함수 테스트
    pixels_1 = image_to_pixel('ex09_pangsoo.jpg')
    # pixels shape :  (1080 세로, 1565 가로, 3 층) -> 3층(RGB) * 8 = 24 bit color 이미지 표현
    pixels_2 = image_to_pixel('christmas.png')
    # pixels shape :  (512, 512, 4)

    pixel_to_image(pixels_1, 'test1.jpg')  # ndarray 형식의 배열들을 jpg 로 저장
    pixel_to_image(pixels_2, 'test2.png')  # ndarray 형식의 배열들을 png 로 저장


# Air_Mouse: 손동작 인식을 통한 비접촉식 커서 조작 시스템

## 프로젝트 소개
- MediaPipe와 OpenCV 라이브러리를 결합하여 손동작 인식 기능을 활용한 비접촉식 커서 조작 시스템을 구현
- 사용자는 손동작만으로 컴퓨터 화면 상의 커서를 자유롭게 조종
- 물리적인 마우스 없이도 다양한 작업을 수행
- LSTM 과 MediaPipe 를 이용한 moving 제스처 인식 및 moving 제스처에 맞는 작업 수행


**간단한 프리뷰**

https://github.com/hje1072/Air_Mouse-/assets/71210590/f22e8780-0fb0-43f0-bd66-cce858fece75


## 파일 소개
간단한 파일 소개

**Air_Mouse.py**

손 동작을 인식해서 마우스 조종

**create_dataset.py**

훈련시킬 무빙제스처 만들기 

**train.ipynb**

무빙제스처 모델학습

## 필수 조건
이 프로젝트를 실행하기 위해 필요한 도구와 라이브러리 목록입니다.

- Python 3.8+
- OpenCV
- MediaPipe
- pyautogui
- pycaw
- tensorflow 

# Air_Mouse: 손동작 인식을 통한 비접촉식 커서 조작 시스템

## 프로젝트 소개
- MediaPipe와 OpenCV 라이브러리를 결합하여 손동작 인식 기능을 활용한 비접촉식 커서 조작 시스템을 구현
- 사용자는 손동작만으로 컴퓨터 화면 상의 커서를 자유롭게 조종
- 물리적인 마우스 없이도 다양한 작업을 수행
- LSTM 과 MediaPipe 를 이용한 moving 제스처 인식 및 moving 제스처에 맞는 작업 수행


**간단한 프리뷰**

https://github.com/hje1072/Air_Mouse-/assets/71210590/f22e8780-0fb0-43f0-bd66-cce858fece75


## 구현된 기능들


* 움직이기

손가락의 검지의 방향으로 마우스를 움직인다.

https://github.com/hje1072/Air_Mouse-/assets/71210590/b0387a14-95f0-412e-811e-b8fc778be23d


* 클릭

엄지손가락을 접는 제스처로 한번 클릭한다.



https://github.com/hje1072/Air_Mouse-/assets/71210590/85121344-f07f-439c-a79f-d140eaf0f5bc



* 더블클릭

엄지와 중지를 대고있는 제스처를 통해 더블클릭한다.



https://github.com/hje1072/Air_Mouse-/assets/71210590/e0217e67-0f84-4790-8872-2f2b9c1f17b6


* 마우스 속도 조절

엄지와 약지를 대고있는 제스처를 통해 마우스 속도를 조절하는 시퀀스에 들어간다.   
이 시퀀스동안 카메라와 새끼손가락 끝과의 z거리를 가지고 가까우면 빠르게, 멀면 느리게 속도를 조절한다.   
엄지와 검지를 모으는 ok 사인을 통해 시퀀스를 종료시킨다.




https://github.com/hje1072/Air_Mouse-/assets/71210590/859718c6-a00b-43b7-aa48-171cc64bf35e


* 시스템 음량 조절

검지와 새끼손가락만 핀 rock노래를 할때 많이 하는 제스처를 통해 소리조절 시퀀스에 들어간다.
이 시퀀스동안 카메라와 새끼손가락 끝과의 z거리를 측정. 가까우면 최대음량, 멀면 최소음량으로 설정한다.
엄지와 검지를 모으는 ok사인을 통해 스퀀스를 종료시킨다.




https://github.com/hje1072/Air_Mouse-/assets/71210590/bc04fc53-baf6-4506-b328-67797620f6d6




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

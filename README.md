# Air_Mouse: 손동작 인식을 통한 비접촉식 커서 조작 시스템

![A_photo_of_a_delicious_late-night_snack_spread _Th](https://github.com/hje1072/Air_Mouse-/assets/71210590/0989d3f1-3f2a-4b60-86c0-0f423d1673de)

 오늘도 하루를 마치고 맛있게 먹는 야식!!! 너무나도 기분이 좋은 나머지 손으로 덥석덥석 집어 먹은 당신입니다.    
야식을 먹는동안 눈도 호강시켜주기 위해 컴퓨터를 키는 당신.. 그러나 문제가 발생하는데..


![A_photo_of_a_person's_hands_that_are_dirty_from_ea](https://github.com/hje1072/Air_Mouse-/assets/71210590/c2fab9f1-e8fb-4c60-9610-4be458f65c8d)

 이런!! 손이 너무 더러워서 유튜브를 보고싶지만 마우스와 키보드를 만지기 너무 껄끄러운 상황이군요!!!
 이럴때에 아이언맨처럼 공중에 제스처를 취하는 것으로 마우스와 키보드를 조절할 수있다면 좋을텐데 말이죠.. 

![image](https://github.com/hje1072/Air_Mouse-/assets/71210590/f125b54b-58f3-406f-9494-7a4cfd73b6b5)

 앗 그래요!! 한번 마우스를 직접 만지지않고 조절하는 프로그램을 만들어놓은다면 어떨까요!! 

 


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



* 정지 및 확인

모든 손가락을 핀 제스처는 마우스 움직임을 정지시킨다. 정지상태를 1초 유지하면 무빙제스처를 확인하는 시퀀스로 들어간다.   
엄지와 검지를 모으는 ok사인 제스처는 무빙제스처시퀀스를 강제로 종료시킬 수도 있다.




https://github.com/hje1072/Air_Mouse-/assets/71210590/d0fa72a4-e7f5-4591-b5f5-438a38591e67


* 유튜브 숏츠넘기기

유튜브 숏츠를 많이 보는 시대라 넣은 기능. 손가락을 모두 핀채로 1초유지하면 삐 소리와함께 무빙제스처 체크 시퀀스에 들어간다.   
이때 손을 위 아래로 까닥까닥거리면 숏츠를 다음 숏츠로 넘길 수가 있다. 숏츠를 보고있지 않을 경우 일반적으로 페이지를 아래방향으로 조금 스크롤하는 기능을 수행한다.   



https://github.com/hje1072/Air_Mouse-/assets/71210590/2c0d62d4-4bc2-45b3-9628-a09fb75181e2



* 뒤로가기

window 운영체제 기준으로 페이지를 뒤로가기한다. 손가락을 모두 핀채로 1초 유지하면 삐 소리와 함께 무빙제스처 체크 시퀀스에 들어간다.   
이때, 손을 좌우로 계속 뒤집어 주는 제스처를 취하면 페이즈를 이전페이지로 뒤로간다. 




https://github.com/hje1072/Air_Mouse-/assets/71210590/174fb7fc-48c9-464e-adaa-29da05460dac



* 프로그램 종료

프로그램을 종료한다. 손가락을 모두 핀채로 1초 유지하면 삐 소리와 함께 무빙제스처 체크 시퀀스에 들어간다.   
이때, 손을 인사하듯이 흔들어주면 프로그램이 종료된다.


https://github.com/hje1072/Air_Mouse-/assets/71210590/87d4969c-9b80-47f8-a7f7-e1e9098f58da



## 파일 소개
간단한 파일 소개

**Air_Mouse.py**

손 동작을 인식해서 마우스 조종

**create_dataset.py**

훈련시킬 무빙제스처 만들기 (만일 무빙제스처 인식이 잘 안 된다면 다시 DATA셋을 만들어 모델을 학습시키자.)   
(원한다면 자신만의 제스처를 만들어도 좋다.)   

[https://github.com/kairess/gesture-recognition] 

**train.ipynb**

무빙제스처 모델학습

[https://github.com/kairess/gesture-recognition] 

## 필수 조건
이 프로젝트를 실행하기 위해 필요한 도구와 라이브러리 목록입니다.

- Python 3.8+
- OpenCV
- MediaPipe
- pyautogui
- pycaw
- tensorflow 



## 참고 자료들

- https://github.com/kairess/Rock-Paper-Scissors-Machine

- https://github.com/kairess/gesture-recognition

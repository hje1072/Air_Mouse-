import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

#키보드 마우스조종용도
import pyautogui

max_num_hands = 1
gesture = {
    0:'direction', 1:'direction', 2:'double click', 3:'three', 4:'click', 5:'menu',
    6:'volume', 7:'direction', 8:'spiderman', 9:'direction', 10:'ok',
}

'''
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
}


'''

#제스쳐 모델 넣어주기
actions = ['push', 'hand_shake', 'spin']
seq_length = 30

#model = load_model('models/model.keras')

#ok 제스쳐 마우스 속도 
#volume 으로서 z 이용해서 z가 가까워지면 느리게
#z가 멀어지면 빠르게/.


#one 마우스 움직이기 

#two 마우스  더블클릭

#four 마우스 클릭이자 

#three 는 볼륨 체크.

#


'''movig gesture 관리하기.'''

#모든 동작은 menu 상태에서 동작.

#whleel 손 흔들기 hand_shake : 숏츠 넘기기


#recovery 뒤로가기. spin : 이전으로 가기

#스페이스바. 쿵쿵 누르기  push : 스페이스바.




#이벤트 중첩 실행 방지용으로 쓸예정.
event_gesture = { 2:'더블클릭' , 4 : '클릭', 5 :'대기상태', 3 : '스피드조절',  10:'제스쳐확인.'} 

click_gesture = {4 : 'click'}
buffer = True

buffer_delay = 0


#하고싶은 이벤트가 맞는지 체크.
event_check = 0
prev = 9999999


#스피드 체크용으로 쓸 예정.
mutual_exclusive = False

speed = 300

#스피드 조절용
def calculate_speed(n):
    if n < 1:
        return 100
    elif n >= 10:
        return 500
    else:
        # 선형 그래프 식으로 speed 계산
        speed = 100 + (n - 1) * (500 - 100) / (10 - 1)
        return speed


# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)



#knn 방식을 이용해서 제스쳐들을 나누기.
# Gesture recognition model
file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)



#비디오 영상 불러오기.
cap = cv2.VideoCapture(0)


while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            
            idx = int(results[0][0]) #지금 무슨 모양인지 알려줌 
                                 
            '''
            # Draw gesture result
            if idx in rps_gesture.keys():
                cv2.putText(img, text=rps_gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

            '''
           # print(idx)
            
            #마우스 조종을 위한 데이터 뽑기
            검지_시작 = res.landmark[8]
            검지_끝 = res.landmark[5]
            
            #print((검지_시작.x - 검지_끝.x))
            
            #print(검지_끝.z)
            
            if (idx not in event_gesture) and (mutual_exclusive == False) :    
                
                #마우스 조종 파트.
                
                pyautogui.move((검지_시작.x - 검지_끝.x)*speed, (검지_시작.y - 검지_끝.y)*speed)
                
                
                buffer = True #연타방지용
                buffer_delay = 0
            
            else :
                
                #0.3초동안 그 동작을 유지하면 그 제스쳐에 해당하는 기능 수행
                if idx == prev and idx != 5:
                    event_check += 1
                    
                    if event_check >= 20 :
                        event_check = 0
                        event_key = idx
                else :
                    
                    #0.5초 안넘으면 대기 
                    event_check = 0
                    event_key = 5
                    prev = idx
                
                
                
                
                #볼륨 조절
                if (event_key == 3) or mutual_exclusive :
                    mutual_exclusive = True
                    
                    
                    
                    #print(  abs(검지_끝.z * 100) )
                    speed = calculate_speed(abs(검지_끝.z * 100))
                    cv2.putText(img, f'Now Speed => {speed}', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                    
                    if  event_key == 10 :  mutual_exclusive = False

                
                
                #원클릭
                elif (event_key == 4) and buffer :
                    pyautogui.click()
                    #pyautogui.scroll(-300)
                    
                    buffer = False #연타방지용
                
                #더블클릭
                elif (event_key == 2) and buffer :
                    pyautogui.click()
                    pyautogui.click()
                
                    buffer = False #연타방지용
                
                
                
                
                else : #event 메뉴 대기 화면 주기적으로 buffer를 True로 변경해줌.
                    
                    cv2.putText(img, 'Waiting for the event...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                    
                    buffer_delay += 1
                    
                    
                    if buffer_delay >= 30 : 
                        buffer = True
                        buffer_delay = 0
                        
            
            
            
            
            
            
            # 무슨 제스쳐를 하고있는지 체크용.
            cv2.putText(img, text=gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Game', img)
    if cv2.waitKey(1) == ord('q'):
        break
    
    
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model


#키보드 마우스조종용도
import pyautogui

max_num_hands = 1
gesture = {
    0:'direction', 1:'direction', 2:'double click', 3:'three', 4:'click', 5:'menu',
    6:'movig_gesture', 7:'direction', 8:'volume', 9:'direction', 10:'ok',
}

'''
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
}


'''

#Q 종료시퀀스용
Q = False

#제스쳐 모델 넣어주기
actions = ['push', 'hand_shake', 'spin']
seq_length = 30

model = load_model('models/model.keras')



#ok 제스쳐 마우스 속도 
#volume 으로서 z 이용해서 z가 가까워지면 느리게
#z가 멀어지면 빠르게/.


#one 마우스 움직이기 

#two 마우스  더블클릭

#four 마우스 클릭이자 

#three 는 스피드 체크.

#8 : volume 볼륨조절


'''movig gesture 관리하기.'''
gesture_check = False
gesture_buffer = 0



prev_action = '?'
this_action = '?'
seq = []
action_seq = []


#moving 제스처 들어갈 때 알려주는 비프음.
import winsound as sd
def beepsound():
    fr = 2000    # range : 37 ~ 32767
    du = 1000     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)


#모든 동작은 menu 상태에서 동작.

#whleel 손 흔들기 hand_shake : 숏츠 넘기기


#recovery 뒤로가기. spin : 이전으로 가기

#스페이스바. 쿵쿵 누르기  push : 스페이스바.




#이벤트 중첩 실행 방지용으로 쓸예정.
event_gesture = { 2:'더블클릭' , 4 : '클릭', 5 :'대기상태', 3 : '스피드조절', 6 : 'moving제스쳐' ,8 : '음향조절' , 10:'제스쳐확인.'} 

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


#볼륨 조절용
mutual_exclusive2 = False

def calculate_volume(n):
    if n <= 1:
        return 0.00
    elif n >= 10:
        return 1.0
    else:
        # 선형 그래프 식으로 speed 계산
        speed = 1 + (n - 1) * (100 - 1) / (10 - 1)
        # 1 ~ 100 사이의 값을 0.00 ~ 1.0 사이 값으로 변환
        new_speed = (speed - 1) / (100 - 1) * (1.0 - 0.00) + 0.00
        return new_speed

#윈도우 기반 볼륨 조절
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# [0.01, 1.0] 사이값을 던지면 그걸로 음량 정해줌.
def set_volume(level):
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volume.SetMasterVolumeLevelScalar(level, None)


def get_current_volume():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))

    current_volume = volume.GetMasterVolumeLevelScalar()
    return current_volume


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
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
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
            
            idx = int(results[0][0]) #지금 무슨 모양인지 체크.
                                 
            
            
            
            #제스처 인식 
            
            if gesture_check :
            
                d = np.concatenate([joint.flatten(), angle])
    
                seq.append(d)
    
                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
    
                if len(seq) < seq_length:
                    continue
    
                input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
    
                y_pred = model.predict(input_data).squeeze()
    
                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]
    
                if conf < 0.9:
                    continue
    
                action = actions[i_pred]
                action_seq.append(action)
    
                if len(action_seq) < 3:
                    continue
    
                this_action = '?'
                if action_seq[-1] == action_seq[-2] == action_seq[-3] :
                    this_action = action
                
                
                
                
            
            
            
            
           # print(idx)
            
            #마우스 조종을 위한 데이터 뽑기
            검지_시작 = res.landmark[8]
            검지_끝 = res.landmark[5]
            
            #print((검지_시작.x - 검지_끝.x))
            
            #print(검지_끝.z)
            
            if (idx not in event_gesture) and (mutual_exclusive + mutual_exclusive2 + gesture_check == False) :    
                
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
                
                
                
                #moving 제스처 확인용
                if idx == 5 and ( mutual_exclusive + mutual_exclusive2 + gesture_check == False) :
                    
                    gesture_buffer += 1
                    if gesture_buffer >= 60 :
                        gesture_buffer = 0
                        gesture_check = True
                        beepsound()
                        
                        
                        #제스쳐 인식효과 높이기위해 파라미터 초기화
                        prev_action = '?'
                        this_action = '?'
                        seq = []
                        action_seq = []

                    
                else : gesture_buffer = 0 if gesture_check == False else gesture_buffer
                    
                
                if gesture_check :
                    
                    if event_key == 10 :
                        gesture_buffer = 0
                        gesture_check = False
                    
                    else :
                    
                    
                        print(f"{gesture_buffer}")
                        #똑같은 제스처 1초 유지시 이벤트 발동.
                        if this_action == prev_action :
                            gesture_buffer += 1
                            
                            if gesture_buffer >= 30 :
                                gesture_buffer = 0
                                gesture_check = False
                                
                        else : 
                            gesture_buffer = 0
                            prev_action = this_action
                            
                        
                        #이벤트에 맞는 행동 이후 나가기. 
                        #각각에 맞는 이벤트 넣어주기.
                        #TODO
                        if gesture_check == False :
                            
                            #숏츠 내리기
                            if this_action == "push" : 
                                print("숏츠넘기기")
                                pyautogui.scroll(-500)
                                
                            
                            #프로그램 종료
                            elif this_action == "hand_shake" : 
                                print("프로그램 종료")
                                Q = True
                                
                            
                            #뒤로돌아가기 크롬기반으로 돌아가게 설정.
                            elif this_action == "spin" : 
                                print("크롬 백")
                                
                                pyautogui.keyDown('alt') # alt 누르기
                                pyautogui.press('left') # 왼쪽화살표
                                pyautogui.keyUp('alt') # alt 떼기
                                
                        
                        
                        cv2.putText(img, f'{this_action}', org=(20, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255), thickness=4)

                
                
                
                #볼륨 조절
                elif (event_key == 3) or mutual_exclusive :
                    mutual_exclusive = True
                    
                    
                    
                    #print(  abs(검지_끝.z * 100) )
                    speed = calculate_speed(abs(res.landmark[20].z * 100)) #새끼손가락의 z위치
                    cv2.putText(img, f'Now Speed => {speed}', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                    
                    if  event_key == 10 :  mutual_exclusive = False

                elif (event_key == 8) or mutual_exclusive2 :
                    
                    mutual_exclusive2 = True
                    
                    #print(  abs(검지_끝.z * 100) ) 
                    
                    v = calculate_volume(abs(res.landmark[20].z * 100))
                    
                    
                    cv2.putText(img, f'Now Volume => {v}', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                    
                    
                    
                    if event_key == 10 :
                        set_volume(v)
                        mutual_exclusive2 = False

                
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
    if cv2.waitKey(1) == ord('q') or Q :
        break
    
    
cv2.destroyAllWindows()
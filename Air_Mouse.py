import cv2
import mediapipe as mp
import numpy as np

#키보드 마우스조종용도
import pyautogui

max_num_hands = 1
gesture = {
    0:'direction', 1:'direction', 2:'double click', 3:'three', 4:'click', 5:'menu',
    6:'six', 7:'direction', 8:'spiderman', 9:'direction', 10:'volume',
}

'''
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
}


'''

#ok 제스쳐 마우스 속도 
#volume 으로서 z 이용해서 z가 가까워지면 느리게
#z가 멀어지면 빠르게/.


#one 마우스 움직이기 

#two 마우스  더블클릭

#four 마우스 클릭

#three 는 애매함.

#


'''movig gesture 관리하기.'''

#모든 동작은 menu 상태에서 동작.

#whleel 손 흔들기


#recovery 뒤로가기.

#스페이스바. 쿵쿵 누르기




#클릭버트용도로 쓸예정.
click_set_gesture = { 3:'doubleClick' , 4 : 'click', 5 :'click_set',  9:'yeah'} #잠시 빼놓음

click_gesture = {4 : 'click'}
click_pressure = True




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
            print(idx)
            
            #마우스 조종을 위한 데이터 뽑기
            검지_시작 = res.landmark[8]
            검지_끝 = res.landmark[5]
            
            print((검지_시작.x - 검지_끝.x))
            
            
            if idx not in click_set_gesture :    
                
                pyautogui.move((검지_시작.x - 검지_끝.x)*300, (검지_시작.y - 검지_끝.y)*300)
                click_pressure = True #연타방지용
                
            
            else :
                
                if (idx in click_gesture) and click_pressure :
                    #pyautogui.click()
                    pyautogui.scroll(-300)
                    
                    
                    
                    click_pressure = False #연타방지용
                
               
            
            
            
            
            
            
            # 무슨 제스쳐를 하고있는지 체크용.
            cv2.putText(img, text=gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Game', img)
    if cv2.waitKey(1) == ord('q'):
        break
    
    
cv2.destroyAllWindows()
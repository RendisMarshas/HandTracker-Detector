import cv2
import mediapipe as mp 
import time
from random import randrange

#To capture video from webcam
cap = cv2.VideoCapture(0)


mpHands = mp.solutions.hands
#By default parameters its automatically False, that allows to detect and track hand
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Frame rate statistics
pTime = 0
cTime = 0


while True:
    #Returns current frame
    success, img = cap.read()
    #Image converter to diffrent color, we  need to to this becouse line 12 object only use this color
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #hands.process is a feature that process and give as results
    results = hands.process(imgRGB)

    #To connect dots on hand
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            #to know id location in frame
            for id, lm in enumerate(handLms.landmark):
                #chanels, hight, wight 
                h, w, c= img.shape
                #to find position of the center
                cx, cy = int(lm.x*w), int(lm.y*h)
                #Detecting landmark wich is 0, benefits of this is that you can tra k spesific landmark and get its id information 
                if id ==8:
                    cv2.circle(img, (cx, cy), 10, (0, randrange(255), 0), cv2.FILLED)


            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Count screen frame rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_COMPLEX,2,(0,randrange(226),0),3)

    cv2.imshow("Hand detector app", img)
    key = cv2.waitKey(1)

    #Stops program if you press key Q
    if key==81 or key==113:
        break


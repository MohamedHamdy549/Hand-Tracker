import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)   # capture Frame

mpHands = mp.solutions.hands    # import hand tracking model
Hands = mpHands.Hands()     # initializes hand tracking model
mpDraw = mp.solutions.drawing_utils     # module that draw landmarks on hand

while True:
    success, img = cap.read()   # read returns (boolean , array)
    img = cv2.flip(img, 1)  # Flip img
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # chang from BGR => RGB

    results = Hands.process(imgRGB)
    lm_list = []
    c = 0
    if results.multi_hand_landmarks:  # check for hands points
        for hand_idx, points in enumerate(results.multi_hand_landmarks):  # returns hand_index, each_hand_points
            for id, plm in enumerate(points.landmark):  # returns point_index, each_point_landmark
                h, w, c = img.shape  # h: high, w: width, c: channels
                cx = int(plm.x * w)
                cy = int(plm.y * h)

                if hand_idx == 1:  # Adjust id for the second hand
                    id += 21  # Start the ID from 21 for the second hand

                lm_list.append([id, cx, cy])  # position of point
                # print (lm_list)
                mpDraw.draw_landmarks(img, points, mpHands.HAND_CONNECTIONS)

                tips = [4, 8, 12, 16, 20]
                x = 21
                if hand_idx == 1:
                    tips.extend([25, 29, 33, 37, 41])
                    x = 42
                # print(tips)

                if id in tips:
                    cv2.circle(img, (cx, cy), radius=8, color=(0, 255, 0), thickness=-1)  # draw_circles_onPoints

        if len(lm_list) >= x:
            fingers = []

            for tip in tips:
                # Check if thumb and determine hand orientation
                if tip == 4 or tip == 25:  # thumb
                    if (lm_list[8][1] > lm_list[20][1] and tip == 4) or (lm_list[8][1] < lm_list[20][1] and tip == 25):
                        if lm_list[tip][1] > lm_list[tip - 2][1]:
                            fingers.append(1)
                    else:
                        if lm_list[tip][1] < lm_list[tip - 2][1]:
                            fingers.append(1)
                else:  # Other fingers
                    if lm_list[tip][2] < lm_list[tip - 2][2]:
                        fingers.append(1)

            numFingers = fingers.count(1)
            cv2.putText(img, f'{numFingers}', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    cv2.imshow('Hand Tracker', img)  # show the frame

    if cv2.waitKey(5) & 0xff == 27:  # End when esc pressed
        break

cap.release()
cv2.destroyAllWindows()
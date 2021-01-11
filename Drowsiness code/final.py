from tkinter import *
from PIL import Image,ImageTk
gw = '#134e86'
g="#0a2845#"
gg ="white"
global R1
R1 = Tk()

R1.geometry('712x712')
R1.title('Drowsy detection')

R1.resizable(width = FALSE ,height= FALSE)
Image_open = Image.open("cover.jpg")
image = ImageTk.PhotoImage(Image_open)
sigup = Label(R1,image=image,bg=gg)
sigup.place(x=0,y=0,bordermode="outside")

def main():
    import cv2
    import imutils
    from scipy.spatial import distance as dist
    from imutils.video import VideoStream
    from imutils import face_utils
    from threading import Thread
    import numpy as np
    import playsound
    import argparse

    import time
    import dlib

    import pygame
    def sound_alarm(path):
    # play an alarm sound
            pygame.mixer.init()
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()

    def eye_aspect_ratio(eye):
            # compute the euclidean distances between the two sets of
            # vertical eye landmarks (x, y)-coordinates
            A = dist.euclidean(eye[1], eye[5])
            B = dist.euclidean(eye[2], eye[4])

            # compute the euclidean distance between the horizontal
            # eye landmark (x, y)-coordinates
            C = dist.euclidean(eye[0], eye[3])

            # compute the eye aspect ratio
            ear = (A + B) / (2.0 * C)

            # return the eye aspect ratio
            return ear
     
    # construct the argument parse and parse the arguments

    def mouth_aspect_ratio(mouth):

        A = dist.euclidean(mouth[2], mouth[10])
        B = dist.euclidean(mouth[4], mouth[8])
        C= dist.euclidean(mouth[3], mouth[9])
        D = dist.euclidean(mouth[0], mouth[6])
        mar = (A+B+C) / (3.0 * D)
        return mar

    '''
    ap = argparse.ArgumentParser()
    ap.add_argument("-p","--shape-predictor", required=True,
                                    help="path alarm .WAV file")
    ap.add_argument("-a", "--alarm", type=str, default="",
                                    help="path alarm .WAV file")
    ap.add_argument("-w", "--webcam", type=int, default=0,
                                    help="index of webcam on system")
    args = vars(ap.parse_args())'''


    # define two constants, one for the eye aspect ratio to indicate
    # blink and then a second constant for the number of consecutive
    # frames the eye must be below the threshold for to set off the
    # alarm
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 48
    MOUTH_AR_THRESH = 0.6
    MOUTH_AR_CONSEC_FRAMES = 30


    # initialize the frame counter as well as a boolean used to
    # indicate if the alarm is going off
    COUNTER = 0
    ALARM_ON = False

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


    # start the video stream thread
    print("[INFO] starting video stream thread...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    # loop over frames from the video stream
    while True:
            # grab the frame from the threaded video file stream, resize
            # it, and convert it to grayscale
            # channels)
            frame = vs.read()
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

            # detect faces in the grayscale frame
            rects = detector(gray, 0)

            # loop over the face detections
            for rect in rects:
                    # determine the facial landmarks for the face region, then
                    # convert the facial landmark (x, y)-coordinates to a NumPy
                    # array
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    # extract the left and right eye coordinates, then use the
                    # coordinates to compute the eye aspect ratio for both eyes
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)

                    # average the eye aspect ratio together for both eyes
                    ear = (leftEAR + rightEAR) / 2.0
                    mouth=shape[mStart:mEnd]
                    mouthAR =mouth_aspect_ratio(mouth)

                    mar = mouthAR
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                    mouthHull = cv2.convexHull(mouth)
                    cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)


                
                    # check to see if the eye aspect ratio is below the blink
                    # threshold, and if so, increment the blink frame counter
                    if (ear < EYE_AR_THRESH) or (mar > MOUTH_AR_THRESH):
                            COUNTER += 1

                            # if the eyes were closed for a sufficient number of
                            # then sound the alarm
                            if (COUNTER >= EYE_AR_CONSEC_FRAMES) or (COUNTER >= MOUTH_AR_CONSEC_FRAMES):
                                    # if the alarm is not on, turn it on
                                    if not ALARM_ON:
                                            ALARM_ON = True

                                            # check to see if an alarm file was supplied,
                                            # and if so, start a thread to have the alarm
                                            # sound played in the background
                                            
                                            t = Thread(target=sound_alarm,args=('alarm.wav',))
                                            t.deamon = True
                                            t.start()

                                    # draw an alarm on the frame
                                    cv2.putText(frame, "Drowsiness ALERT!", (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # otherwise, the eye aspect ratio is not below the blink
                    # threshold, so reset the counter and alarm
                    else:
                            COUNTER = 0
                            ALARM_ON = False

                    # draw the computed eye aspect ratio on the frame to help
                    # with debugging and setting the correct eye aspect ratio
                    # thresholds and frame counters
                    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    '''
                    if mar > MOUTH_AR_THRESH:
                        COUNTER += 1


                        if COUNTER >= MOUTH_AR_CONSEC_FRAMES:
                            # if the alarm is not on, turn it on
                            if not ALARM_ON:
                                ALARM_ON = True

                                if args["alarm"] != "":
                                    t = Thread(target=sound_alarm,
                                               args=(args["alarm"],))
                                    t.deamon = True
                                    t.start()

                            # draw an alarm on the frame
                            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    else:
                        COUNTER = 0
                        ALARM_ON = False'''

                    cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
     
            # show the frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
     
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                    break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    
def head(): 
    import cv2
    import numpy as np

    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('nodcontrol.avi',fourcc, 20.0, (640,480))

    #dinstance function
    def distance(x,y):
        import math
        return math.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2) 
        
    #capture source video
    cap = cv2.VideoCapture(0)

    #params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    #path to face cascde
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    #function to get coordinates
    def get_coords(p1):
        try: return int(p1[0][0][0]), int(p1[0][0][1])
        except: return int(p1[0][0]), int(p1[0][1])

    #define font and text color
    font = cv2.FONT_HERSHEY_SIMPLEX


    #define movement threshodls
    max_head_movement = 20
    movement_threshold = 50
    gesture_threshold = 175
    

    #find the face in the image
    face_found = False
    frame_num = 0
    while frame_num < 30:
        # Take first frame and find corners in it
        frame_num += 1
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            face_found = True
        cv2.imshow('image',frame)
        #out.write(frame)
        cv2.waitKey(1)
    face_center = x+w/2, y+h/3
    p0 = np.array([[face_center]], np.float32)

    gesture = False
    x_movement = 0
    y_movement = 0
    gesture_show = 60 #number of frames a gesture is shown

    while True:    
        ret,frame = cap.read()
        old_gray = frame_gray.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        cv2.circle(frame, get_coords(p1), 4, (0,0,255), -1)
        cv2.circle(frame, get_coords(p0), 4, (255,0,0))
        
        #get the xy coordinates for points p0 and p1
        a,b = get_coords(p0), get_coords(p1)
        x_movement += abs(a[0]-b[0])
        y_movement += abs(a[1]-b[1])
        
        text = 'x_movement: ' + str(x_movement)
        if not gesture: cv2.putText(frame,text,(50,50), font, 0.8,(0,0,255),2)
        text = 'y_movement: ' + str(y_movement)
        if not gesture: cv2.putText(frame,text,(50,100), font, 0.8,(0,0,255),2)

        if x_movement > gesture_threshold:
            gesture = 'NOT PAYING ATTENTION'
        if y_movement > gesture_threshold:
            gesture = 'NOT PAYING ATTENTION'
        if gesture and gesture_show > 0:
            cv2.putText(frame,'Status: ' + gesture,(50,50), font, 1.2,(0,0,255),3)
            gesture_show -=1
        if gesture_show == 0:
            gesture = False
            x_movement = 0
            y_movement = 0
            gesture_show = 60 #number of frames a gesture is shown
            
        #print distance(get_coords(p0), get_coords(p1))
        p0 = p1

        cv2.imshow('image',frame)
        #out.write(frame)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    cap.release()


    
loginbt = Button(R1,text = "Detection",width=15,height=2,bg="lightblue",fg="black",font="5",relief=RAISED,overrelief=RIDGE,command=main)


signUpbt = Button(R1,text = "Head Bend",width=15,height=2,bg="lightblue",fg="black",font="5",relief=RAISED,overrelief=RIDGE,command=head)
loginbt.place(x =200 ,y=350)
signUpbt.place( x =400,y=350)

R1.mainloop()

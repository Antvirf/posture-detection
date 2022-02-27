import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time

def notify(title, text, sound, sound_only=False):
    if sound_only:
        os.system("afplay /System/Library/Sounds/Submarine.aiff")
    else:
        os.system("""
            osascript -e 'display notification "{}" with title "{}" sound name "{}"'""".format(text, title, sound))

def say_stuff(stuff):
    os.system(
    """say "{}" """.format(stuff)
    )

def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

haar_cascade_face = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
haar_cascade_upperbody = cv2.CascadeClassifier('data/haarcascades/haarcascade_upperbody.xml')

def detect_upperbody(cascade, test_image, scaleFactor = 1.2):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()

    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    gray_image = np.array(gray_image, dtype='uint8')

    # Applying the haar classifier to detect faces
    shoulders_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=4)

    area = 0
    for (x, y, w, h) in shoulders_rect:
        area = (w/100.0) * (h/100.0)
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (255, 0, 0), 3)
    return image_copy, area

def detect_faces(cascade, test_image, scaleFactor = 1.1, threshold=False):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()

    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    gray_image = np.array(gray_image, dtype='uint8')

    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)

    area = 0
    for (x, y, w, h) in faces_rect:
        area = (w/100.0) * (h/100.0)
        if threshold and area > threshold:
            cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 0, 255), 4)
        else:
            cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
    return image_copy, area

cap = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #do face detection
    shoulders_detect, area = detect_upperbody(haar_cascade_upperbody, frame)

    # Display the resulting frame
    cv2.imshow('frame',shoulders_detect)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



if __name__ == '__main__':
    # Start camera capture
    cap = cv2.VideoCapture(0)

    # New capturing code
    # First loop
    frame_rate = 3 # Higher FPS for the slouch threshold detection to minimise impact of potential errors/outliers
    prev, capture_time = 0, 0
    start_time = time.time()
    thresholds_array = []
    threshold_area = 0
    consecutive_breaches = 0
    calibrated = False

    try:
        print("Program started, please slouch for 5 seconds to calibrate...")
        while True:
            time_elapsed = time.time() - prev
            time_elapsed_total = time.time() - start_time
            res, frame = cap.read()

            face_detect, area = detect_faces(haar_cascade_face, frame)

            # Check whether calibration is complete and should switch to monitoring
            if time_elapsed_total >5 and not calibrated:
                # Compute average
                print("Calibration complete, slouch monitoring active.")
                threshold_area = sum(thresholds_array)/len(thresholds_array)/1.0
                calibrated = True

            # Append thresholds array if calibration is ongoing
            if time_elapsed > 1./frame_rate:
                prev = time.time()
                if not calibrated:
                    # Calibrate
                    thresholds_array.append(area)

                else:
                    # Monitor
                    if area > threshold_area:
                        print('Breach found, consecutive:', consecutive_breaches)
                        consecutive_breaches +=1
                    else:
                        consecutive_breaches = 0
                    
                    if consecutive_breaches > frame_rate * 3: # Threshold set to 3 seconds
                        notify("Posture Alert", "Stop slouching", "submarine")
            
            # If you need to show image - but breaks the other logic
            #cv2.waitKey(1)
            


        
        while False:
            time_elapsed = time.time() - prev
            time_elapsed_total = time.time() - start_time
            res, frame = cap.read()

            if time_elapsed > 1./frame_rate:
                prev = time.time()
                face_detect, area = detect_faces(haar_cascade_face, frame, threshold_area)

                if area > threshold_area:
                    consecutive_breaches += 1
                else:
                    consecutive_breaches = 0

                if consecutive_breaches > 5:
                    notify("Posture Alert", "Stop slouching", "submarine")
                    #say_stuff('h')

                # Show the image
                cv2.imshow('frame',face_detect)

    except KeyboardInterrupt:
        # When everything done, release the capture
        print("\nRequested to stop, stopping")
        cap.release()
        cv2.destroyAllWindows()

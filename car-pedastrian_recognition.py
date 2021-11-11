import cv2
 
car_tracker = cv2.CascadeClassifier('cars.xml')
pedestrian_tracker = cv2.CascadeClassifier('haarcascade_fullbody.xml')
# choose an image to detect car

source = cv2.VideoCapture(0)


while True:  
        
    successful_frame_read, frame = source.read()   
    

    # Must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cars = car_tracker.detectMultiScale(grayscaled_img)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_img)


    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x ,y), (x+w, y+h) ,(0 ,0, 255), 2)
        
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x ,y), (x+w, y+h) ,(0 ,255, 0), 2)


    cv2.imshow(' car detector', frame)
    key = cv2.waitKey(1)
    
       # Stop if Q is pressed
    if key==81 or key==113:
        break 



print('code completed')

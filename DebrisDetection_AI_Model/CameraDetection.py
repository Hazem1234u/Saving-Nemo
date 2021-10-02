import operator
import cv2
from tensorflow.keras.models import load_model

model=load_model("C:/Users/shrou/Desktop/model68.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)

# Category dictionary
categories={0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}

while True:
    _, frame = cap.read()
    
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Coordinates of the ROI
    x1 = int(0.1*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.1*frame.shape[1])
    
    # Drawing the ROI
    cv2.rectangle(frame, (100,50), (500, 350), (255,0,0) ,1)
    # The increment/decrement by 1 is to compensate for the bounding box
    #cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    
    # Extracting the ROI
    roi = frame[50:350, 100:500]
    #roi = frame[y1:y2, x1:x2]

    
    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (64, 64))
    
    #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("test", test_image)
    
    # Batch of 1
    result = model.predict(test_image.reshape(1, 64, 64, 3))
    prediction = {'cardboard': result[0][0], 
                  'glass': result[0][1], 
                  'metal': result[0][2],
                  'paper': result[0][3],
                  'plastic': result[0][4],
                  'trash': result[0][5]
                  }
    
    # Sorting based on top prediction
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    
    # Displaying the predictions
    cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)    
    cv2.imshow("Frame", frame)
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
        
cap.release()
cv2.destroyAllWindows()
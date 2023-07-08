import cv2

recog = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recog1 = cv2.CascadeClassifier("haarcascade_smile.xml")

img = cv2.VideoCapture("test.jpg") 

res,pix = img.read() #Read the image and store in pix

gray = cv2.cvtColor(pix,cv2.COLOR_BGR2GRAY) #Convert the image to gray

faces = recog.detectMultiScale(gray,1.3,6) #Detects objects of different faces in the input image
smile = recog1.detectMultiScale(gray,1.8,20) #Detects objects of different smiles in the input image


for (x,y,w,h) in faces:
    cv2.rectangle(pix,(x,y),(x+w,y+h),(238,180,34),2) #Draws a rectangle on the faces
for (x,y,w,h) in smile:
    cv2.rectangle(pix,(x,y),(x+w,y+h),(148,0,211),2) #Draws a rectangle on the smiles
    
    
cv2.imshow("My image",pix)
cv2.waitKey(0)
img.release()
cv2.destroyAllWindows()




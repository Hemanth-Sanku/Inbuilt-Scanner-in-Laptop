#Importing Necessary Libraries
import cv2
import numpy as np
import mapper
import img2pdf 
from PIL import Image, ImageEnhance
import os



#Capturing Photo Uising Webcam
cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "test_img.jpg".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        break



#Reading Img and Enhansing
image=cv2.imread("test_img.jpg")   #read in the image
image=cv2.resize(image,(1300,800)) #resizing because opencv does not work well with bigger images
orig=image.copy()

#GrayScale Filter
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  #RGB To Gray Scale
cv2.imshow("Title",gray)

#Canny Edge detection
edged=cv2.Canny(gray,30,50)  #30 MinThreshold and 50 is the MaxThreshold
cv2.imshow("Canny",edged)

contours,hierarchy=cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)  #retrieve the contours as a list, with simple apprximation model
contours=sorted(contours,key=cv2.contourArea,reverse=True)

#the loop extracts the boundary contours of the page
for c in contours:
    p=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,0.02*p,True)

    if len(approx)==4:
        target=approx
        break



#Mapping
approx=mapper.mapp(target) #find endpoints of the sheet

pts=np.float32([[0,0],[800,0],[800,800],[0,800]])  #map to 800*800 target window

op=cv2.getPerspectiveTransform(approx,pts)  #get the top or bird eye view effect
dst=cv2.warpPerspective(orig,op,(800,800))
cv2.imshow("Scanned",dst)



#saving the scanned img
img_name = "scanned_org.jpg".format(img_counter)
cv2.imwrite(img_name,dst)


#Improving Brightness
#read the image
im = Image.open("scanned_org.jpg")

#image brightness enhancer
enhancer = ImageEnhance.Brightness(im)

factor = 1.5 #brightens the image
im_output = enhancer.enhance(factor)
im_output.save('scanned_brightened.jpg')



#Improving Sharpness
#read the image
im = Image.open("scanned_brightened.jpg")

#image brightness enhancer
enhancer = ImageEnhance.Sharpness(im)

factor = 1 #brightens the image
im_output = enhancer.enhance(factor)
im_output.save('scanned_sharpened.jpg')



#Converting Scanned img to pdf
# storing image path 
img_path = "D:/K L University/OneDrive - K L University/Academics/Project/Code/2/scanned_sharpened.jpg"
  
# storing pdf path 
pdf_path = "D:/K L University/OneDrive - K L University/Academics/Project/Code/2/Output/scanned_Doc.pdf"
  
# opening image 
image = Image.open(img_path) 
  
# converting into bytes using img2pdf 
pdf_bytes = img2pdf.convert(image.filename) 
  
# opening or creating pdf file 
file = open(pdf_path, "wb") 
  
# writing pdf files with chunks 
file.write(pdf_bytes) 
  
# closing image file 
image.close() 
  
# closing pdf file 
file.close()
# output 
print("Successfully made pdf file") 



#END
# press Esc to close
cv2.waitKey(0)
cv2.destroyAllWindows()

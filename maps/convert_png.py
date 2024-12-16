import cv2 # Import OpenCV
   
# read the image file
img = cv2.imread('both_cones.png')
   
ret, bw_img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
   
# converting to its binary form
bw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  
# Display and save image
cv2.imwrite("both_cones.png", bw)
cv2.destroyAllWindows()

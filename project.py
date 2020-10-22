import numpy as np
import cv2
import tensorflow as tf
pr=tf.keras.models.load_model("gauri.h5")

img = np.ones([300,300,3],dtype='uint8')*255
windowName = 'Mouse Demo'
cv2.namedWindow(windowName)
status=False
def demo(event,x,y,flags,param):
    global status
    if event == cv2.EVENT_LBUTTONDOWN:
        status=True
    if event==cv2.EVENT_MOUSEMOVE:
        if status==True:
                cv2.circle(img,(x,y),5,(0,0,0),10)
    if event == cv2.EVENT_LBUTTONUP:
        status=False
       


cv2.setMouseCallback(windowName,demo)

while True:
    cv2.imshow(windowName,img)
    if cv2.waitKey(1) == ord('q'):
        break
    elif cv2.waitKey(1) == ord('c'):
        img[:,:] = 255
    elif cv2.waitKey(1) == ord('p'):
        out = img[:,:]
        out_resized = cv2.resize(out, (28, 28))
        out_resized = cv2.cvtColor(out_resized, cv2.COLOR_BGR2GRAY)
        img_arr = np.array(out_resized)
        img_arr = img_arr.flatten()
        img_arr.reshape(784, -1)
        # print("\n\n\n", img_arr.shape)
        # out_resize=cv2.resize(out,(28,28)).reshape(1,28,28,-1)
        print(pr.predict_classes(img_arr[np.newaxis, :]))
cv2.destroyAllWindows()

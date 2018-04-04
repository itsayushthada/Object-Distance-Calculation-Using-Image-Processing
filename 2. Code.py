'''REQUIRED PACKAGES'''
import cv2
import numpy as np

'''MACROS'''
default_bounds = False
bounds_offset = 20
open_kernel = 5
close_kernel = 30
object_color = [255, 0, 0]
intensity_calibaration = 1
display_all_contours = False
window_name = "Camera Capture"
display_width = 680
display_height = 320
square_object_detector = False
video_feed = False
Video_location = "\video\fun.mp4"

'''OBJECT COLOUR RANGE GENERATOR'''
def hsv_bounds(a):
    if default_bounds == True: 
        lower_bound = np.array([33,80,40])
        upper_bound = np.array([102,255,255])
    else:
        color = np.uint8([[a]])
        hsv_color = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
        if(hsv_color[:,:,0] <= bounds_offset):
            lower_bound = np.array(np.uint8([0, 80, 40]))
        else:
            lower_bound = np.array(np.uint8([np.squeeze(hsv_color[:,:,0])-bounds_offset, 80, 40]))
        
        if(hsv_color[:,:,0] <= 180-bounds_offset):        
            upper_bound = np.array(np.uint8([np.squeeze(hsv_color[:,:,0])+bounds_offset, 255, 255]))
        else:
            upper_bound = np.array(np.uint8([180, 255, 255]))
        
    return lower_bound, upper_bound

'''OBJECT DETECTION, TRACING AND DISTANCE CALCULATION'''
def main():
    if video_feed == False:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(Video_location)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    lower_bound, upper_bound = hsv_bounds(object_color)
    kernel_open=np.ones((open_kernel, open_kernel))
    kernel_close=np.ones((close_kernel, close_kernel))

    while(True):
        ret, frame = cap.read()
        ret = cap.set(4,display_width)
        ret = cap.set(5,display_height)

        HSVframe = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(HSVframe,lower_bound,upper_bound)
        mask_open = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel_open) #Good for Salt Noise(High frequency)
        mask_close = cv2.morphologyEx(mask_open,cv2.MORPH_CLOSE,kernel_close) #Good for Pepper Noise (Low Frequency)

        mask_final = mask_open
        im2, contours, hierarchy = cv2.findContours(mask_final.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
        max_area, max_index, i = 0,0,0	
        for cnt in contours:
            a= cv2.contourArea(cnt)
            if a > max_area:
                max_area = a
                max_index = i
            i +=1

        if max_area != 0:
            if display_all_contours == False: 
                x,y,w,h = np.array(cv2.boundingRect(contours[max_index]))
                if square_object_detector == True:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),2)
                cv2.drawContours(frame, contours[max_index], -1, (0,0,0),3)
            else:
                for cnt in contours:
                    x,y,w,h = np.array(cv2.boundingRect(cnt))
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),2)
                    cv2.drawContours(frame, cnt, -1, (0,0,0),3)

            
            dist = 1402.45 * (max_area**(-0.467)) * intensity_calibaration
            print("Distance is "+str(dist)+" cms.")
        
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

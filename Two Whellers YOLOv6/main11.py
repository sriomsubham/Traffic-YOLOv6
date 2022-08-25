import cv2 as cv
import numpy as np
import time
from YOLOv6 import inference

def main():

    cap1 = cv.VideoCapture("Test1.mp4") # path of the first inferenced video
    cap2 = cv.VideoCapture("Test2.mp4") # path of the second inferenced video

    perv_frame_time = 0  # for fps calculation

    out = cv.VideoWriter('Output_vedio.avi', cv.VideoWriter_fourcc(*'MJPG'), 60, (1200, 390)) # For saving the output vedio file
                        #(Output file name,   function to save ,             fps, frame size of output file)
    while True:
        ret1, frame1 = cap1.read()  # reading the video 1
        ret2, frame2 = cap2.read()  # reading the video 2
        if not ret1:
            frame1 = last1
        elif not ret2:
            frame2 = last2
        elif not (ret1 and ret2):
            break

        #cv.namedWindow(frame, cv.WINDOW_NORMAL)   #Create output window with freedom of dimenssions
        #cv.resizeWindow(frame, 900, 500)          #Resize output window to specified dimenssions
        frame1 = cv.resize(frame1, (600, 390))     #Resize output image to specified dimenssions
        frame2 = cv.resize(frame2, (600, 390))     #Resize output image to specified dimenssions

        frame1 = inference.main(frame1)
        frame2 = inference.main(frame2)

        '''
        # Print Frame Per Second (FPS) in each frame.
        font = cv.FONT_HERSHEY_COMPLEX   # for font type

        # formula for getting FPS #
        new_frame_time = time.time()
        fps = 1/(new_frame_time - perv_frame_time)
        perv_frame_time = new_frame_time
        # formula for getting FPS #

        cv.putText(frame1, f"FPS = {int(fps)}", (4, 15), font, 0.5, (20, 80, 255), 1)
        cv.putText(frame2, f"FPS = {int(fps)}", (4, 15), font, 0.5, (20, 80, 255), 1)
        '''
            
        # Combining Both vedios for size-by-side vewing
        hori = np.concatenate((frame1, frame2), axis=1)
        last1, last2 = frame1, frame2
        out.write(hori)            # for saving the vedio
        #cv.imshow("frames", hori)  # for displaying the vedio

        key = cv.waitKey(1) # wait 1 ms between each frame
        if (key == 27):
            break

    cap1.release()
    cap2.release()
    out.release()
    cv.destroyAllWindows()

main()

'''
This code reads two videos and displays them side by side using opencv and numpy

We can also change the size of the output window as well as output image/vedio. 
'''

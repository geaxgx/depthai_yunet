import cv2
import numpy as np
from FPS import now

landmark_color = [
        (255,   0,   0), # right eye
        (  0,   0, 255), # left eye
        (  0, 255,   0), # nose tip
        (255,   0, 255), # right mouth corner
        (  0, 255, 255)  # left mouth corner
    ]
box_color=(0, 255, 0) 
text_color=(0, 0, 255)

class YuNetRenderer:
    def __init__(self, 
                detector,
                output=None):

        self.detector = detector
        self.output = output

        self.show_fps = False

        if output is None:
            self.output = None
        else:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.output = cv2.VideoWriter(output,fourcc,self.detector.video_fps,(self.detector.img_w, self.detector.img_h)) 

    def draw_face(self, face):
        bbox = face[0:4].astype(np.int32)
        cv2.rectangle(self.frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)

        conf = face[-1]
        cv2.putText(self.frame, '{:.4f}'.format(conf), (bbox[0], bbox[1]+12), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color)

        landmarks = face[4:14].astype(np.int32).reshape((5,2))
        for idx, landmark in enumerate(landmarks):
            cv2.circle(self.frame, tuple(landmark), 2, landmark_color[idx], 2)


    def draw(self, frame, faces):
        self.frame = frame
        for face in faces:
            self.draw_face(face)
        
        return self.frame

    def exit(self):
        if self.output:
            self.output.release()
        
    def waitKey(self, delay=1):
        if self.show_fps:
                self.detector.fps.draw(self.frame, orig=(50,50), size=1, color=(240,180,100))
        cv2.imshow("YuNet", self.frame)
        if self.output:
            self.output.write(self.frame)
        key = cv2.waitKey(delay) 
        return key
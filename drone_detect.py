import numpy as np
import cv2
from core import Core
c = Core()



cap = cv2.VideoCapture(0)
c.set_model(c.get_model())
path = r'C:\Users\Rajnish\Desktop\geeksforgeeks.png'
while (True):
    ret, frame =cap.read()
    #image = c.load_image_by_path(frame)
    #drawing_image = c.get_drawing_image(image)
    #image1 = frame.copy()
    image1 = frame
    #image1 = cv2.imread(path)
    image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

    processed_image, scale = c.pre_process_image(image)


    boxes, scores, labels = c.predict_with_graph_loaded_model(processed_image, scale)

    #detections = c.draw_boxes_in_image(drawing_image, boxes, scores)
    detections = c.draw_boxes_in_image(image1, boxes, scores)

    #c.visualize(drawing_image)
    cv2.imshow('image', image1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

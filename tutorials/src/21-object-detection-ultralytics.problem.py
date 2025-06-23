import numpy as np
import cv2
from collections import defaultdict
from ultralytics import YOLO, SAM, ASSETS, FastSAM  # pip install ultralytics needed


# load model (download before running from https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt)
model = YOLO("./tutorials/data/models/yolov8n.pt")
model_sam = FastSAM('./tutorials.data/models/FastSAM-s.pt')
# capture webcam image
cap = cv2.VideoCapture(0)
clicked_point = None

def click_event(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = [x,y]

# get camera image parameters from get()
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
codec = int(cap.get(cv2.CAP_PROP_CODEC_PIXEL_FORMAT))
focus = int(cap.get(cv2.CAP_PROP_AUTOFOCUS))

print('Video properties:')
print('  Width = ' + str(width))
print('  Height = ' + str(height))
print('  Codec = ' + str(codec))
print('  Autofocus = ' + str(focus))

title = 'Video image'
cv2.namedWindow(title, cv2.WINDOW_FREERATIO)  # Note that window parameters have no effect on MacOS
cv2.setMouseCallback(title, click_event)
print('Press q to close the window.')

while True:
    ret, frame = cap.read()
    if (ret):
        # show the image
        img = frame.copy()
        # TODO detect objects in img
        # HINT use model.predict(source=img, stream=True, show=False)
        # and not this example: https://docs.ultralytics.com/modes/predict/#streaming-source-for-loop
        results = model.predict(source=img, stream=True, show=False)

        # # TODO parse results and draw bounding boxes on img
        for result in results:
            boxes = result.boxes
            names = result.names
            
            for i in range(len(boxes)):
                box = boxes[i]

                x1 = (int)(box.xyxy.numpy()[0][0])
                y1 = (int)(box.xyxy.numpy()[0][1])
                x2 = (int)(box.xyxy.numpy()[0][2])
                y2 = (int)(box.xyxy.numpy()[0][3])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

                name = names[(int)(box.cls.numpy()[0])]
                cv2.putText(img, name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                score = box.conf.numpy()[0]
                cv2.putText(img, str(score), (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        if clicked_point is not None:
            results_sam = model_sam(frame, device="cpu", retina_masks=True, imgsz=512, 
                                    points=[clicked_point], labels=[1], conf=0.4, iou=0.9)

            if results_sam and hasattr(results_sam[0], 'masks') and results_sam[0].masks is not None:
                alpha = 0.5
                overlay = img.copy()
                
                masks = results_sam[0].masks.data
                for mask in masks:
                    mask_np = mask.cpu().numpy().astype('uint8') * 255
                    overlay[mask_np == 255] = (0, 0, 255)
                    img = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        # show image with bounding boxes
        cv2.imshow(title, img)
        # press q to close the window
        if cv2.waitKey(10) == ord('q'):
            break
    else:
        print('Could not start video camera')
        break

cap.release()
cv2.destroyAllWindows()

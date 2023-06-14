import torch
import torchvision
import cv2
import os
import shutil
import numpy as np


# -------Model-------

# Load the weights from local
person_weights="C:\\Users\\lixin\\PycharmProjects\\yolov7\\yolov7x.pt"
# activity_weights="C:\\Users\\lixin\\PycharmProjects\\streamlit-yolov5-master1\\person_fire.pt"

# Load the YOLO models from Torch Hub and set parameters
model_person = torch.hub.load('WongKinYiu/yolov7', 'custom', person_weights)
model_person.conf = 0.45
model_person.nms = 0.8
# model_activity = torch.hub.load('ultralytics/yolov5', 'custom', activity_weights)
# model_activity.conf = 0.4

# Set the device to 'cuda' if available, otherwise use 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the model to evaluation mode
model_person.to(device).eval()
# model_activity.to(device).eval()

# Define the class labels for person detection
class_labels_person = model_person.module.names if hasattr(model_person, 'module') else model_person.names
# print("class_labels_person", class_labels_person)
# Define the class labels for person activity detection
# class_labels_activity = model_activity.module.names if hasattr(model_activity, 'module') else model_activity.names
# print("class_labels_activity", class_labels_activity)
# activity_colors=['','','','','','',(255,0,255), (0, 255,255), (255,255,0), (0,0,255)]



# -------Video-------

# Path to folder
#video_path = "C:\\Users\\lixin\\fighting dataset\\video\\violencedataset_CCTV\\violence video cleaned\\V6.mp4"
#output_path = "C:\\Users\\lixin\\fighting dataset\\video\\violencedataset_CCTV\\violence video cleaned\\6" \
           #   "_result\\"

source_folder = "C:\\Users\\lixin\\fighting dataset\\video\\Fightdataset_movie\\No fight"
destination_folder = "C:\\Users\\lixin\\fighting dataset\\video\\Fightdataset_movie\\No fight photos"

for filename in os.listdir(source_folder):
    # Get file name
    file_path = os.path.join(source_folder, filename)
    video_name = os.path.splitext(filename)[0]
    # Path to output
    target_folder = os.path.join(destination_folder, video_name)
    os.makedirs(target_folder, exist_ok=True)
    frame_interval = 4

    # Open the video file
    video = cv2.VideoCapture(file_path)
    # Get the video properties
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    # Create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(target_folder, fourcc, fps, (frame_width, frame_height))

    # Counter
    frame_count = 0
    image_count = 0
    box_count = 0

    # Function
    def extension(box1, box2):

        x11, y11, x12, y12 = map(int, box1)
        x21, y21, x22, y22 = map(int, box2)

        x1_ = np.minimum(x11, x21)
        y1_ = np.minimum(y11, y21)
        x2_ = np.maximum(x12, x22)
        y2_ = np.maximum(y12, y22)

        new_box = [x1_, y1_, x2_, y2_]

        return new_box
    def padding(box, pad_rate, width, height):

        x1_, y1_, x2_, y2_ = map(int, box)
        h_pad = int((y2_ - y1_) * pad_rate)
        w_pad = int((x2_ - x1_) * pad_rate)
        x1_new = max(0, x1_ - w_pad)
        y1_new = max(0, y1_ - h_pad)
        x2_new = min(x2_ + w_pad, width)
        y2_new = min(y2_ + h_pad, height)

        new_box_ = [x1_new, y1_new, x2_new, y2_new]

        return  new_box_
    def iou(box1, box2):

         x11, y11, x12, y12 = map(int, box1)
         x21, y21, x22, y22 = map(int, box2)

         xa = np.maximum(x11, np.transpose(x21))
         xb = np.minimum(x12, np.transpose(x22))
         ya = np.maximum(y11, np.transpose(y21))
         yb = np.minimum(y12, np.transpose(y22))

         area_inter = np.maximum(0, (xb - xa + 1)) * np.maximum(0, (yb - ya + 1))

         area_1 = (x12 - x11 + 1) * (y12 - y11 + 1)
         area_2 = (x22 - x21 + 1) * (y22 - y21 + 1)
         area_union = area_1 + np.transpose(area_2) - area_inter

         iou = area_inter / area_union
         return iou

    # Read the video frames
    print("New video reading starts!")
    print("╰(●’◡’●)╮ please be patient to wait (●′ω`●) QAQ ")

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print("No frame！")
            break
        frame_count += 1
        if frame_count % frame_interval == 0:

            image_count += 1

            # -------Person detection-------
            results_person = model_person(frame)
            # print("results_person", results_person.shape)
            results_person_info = results_person.xyxy[0]
            # print("results_person_info", results_person_info)
            results_person = results_person_info[results_person_info[:, 5] == 0]
            # print("results_person", results_person)
            # Get bounding box coordinates and class labels for person detection
            boxes_person = results_person[:, :4]
            labels_person = results_person[:, 5].cpu().numpy()

            # -------IOU Calculation-------
            boxes = []
            for i in range(len(boxes_person)):
                for j in range(i + 1, len(boxes_person)):
                    iou_value = iou(boxes_person[i], boxes_person[j])
                    if iou_value >= 0.05:
                        box_count += 1
                        extension_box = extension(boxes_person[i], boxes_person[j])
                        boxes.append(extension_box)
                        #padding_box = padding(extension_box, 0.2, frame.shape[1], frame.shape[0])

            # Crop and save the person region from the frame
            unique_data = set(tuple(sublist) for sublist in boxes)
            boxes_new = [list(t) for t in unique_data]
            for i in range(len(boxes_new)):
                x1, y1, x2, y2 = map(int, boxes_new[i])
                roi = frame[y1:y2, x1:x2]
                cropped_image = roi.copy()
                output_image_path = os.path.join(target_folder, f'box{box_count}.jpg')
                cv2.imwrite(output_image_path, cropped_image)

                        # Draw the boxes on the frame
                        #cv2.rectangle(frame, (x1, y1), (x2, y2), (10, 10, 128), 2, cv2.LINE_AA)
                        # Write the text on the frame
                        #cv2.putText(frame, class_labels_person[int(labels_person[i])], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 200, 20), 1  ,\
                        #    cv2.LINE_AA)
                        #cv2.putText(frame, 'Fighting？', (x1, y1 - 10),
                        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 128), 2, cv2.LINE_AA)
                        # print("frame", frame.shape)
                        # Crop the person region from the frame
                        #x1_new, y1_new, x2_new, y2_new = pad_person_box(x1, y1, x2, y2, 0.1, frame.shape[1], frame.shape[0])
                        #cv2.rectangle(frame, (x1_new, y1_new), (x2_new, y2_new), (255, 255, 0), 2, cv2.LINE_AA)
                        #person_img = frame[y1_new:y2_new, x1_new:x2_new,:]

            # Write the frame with bounding boxes to the output video
            #output_video.write(frame)
            #output_image_path = os.path.join(output_path, f'image{image_count}.jpg')
            #cv2.imwrite(output_image_path, frame)
            frame_count = 0

            # Display the frame with bounding boxes (optional)
            #cv2.imshow('Frame', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture, writer, and close the windows
    video.release()
    output_video.release()
    cv2.destroyAllWindows()
    print("Video processing completed!")
    print("-------------------------------------")

print("Mission completed!")
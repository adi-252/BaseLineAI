from ultralytics import YOLO;

model = YOLO("BaseLineAI/yolov8x.pt")

result = model.track('BaseLineAI/input_videos/input_video.mp4', save = True, project="prediction", conf=0.2)

# print(result)
# print("Boxes:")
# for box in result[0].boxes:
#     print(box)

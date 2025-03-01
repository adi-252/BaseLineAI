from ultralytics import YOLO;

model = YOLO("yolov8n.pt")

result = model.predict('input_videos/input_video.mp4', save = True)
print(result)

print("Boxes:")
for box in result[0].boxes:
    print(box)

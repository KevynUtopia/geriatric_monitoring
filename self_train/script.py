from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("/home/wzhangbu/elderlycare/weights/yolo11l-pose.pt", verbose=False)

# Train the model on COCO8
results = model.train(data="coco-pose.yaml", epochs=0, imgsz=640)
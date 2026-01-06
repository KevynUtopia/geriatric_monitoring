from ultralytics import YOLO

# Load a model
model = YOLO("/home/wzhangbu/elderlycare/weights/yolo11n-pose.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="config/coco8-pose.yaml", epochs=1, imgsz=640)

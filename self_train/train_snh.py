from ultralytics import YOLO

# Load a model
model = YOLO("/home/wzhangbu/elderlycare/weights/yolo11n-pose.pt")  # load a pretrained model (recommended for training)

# # find the path of the sourcecode of model.train()
# # Print the model source code path
# model_source = model.train.__code__.co_filename
# print(f"Model source code path: {model_source}")
# exit(0)

# Train the model
results = model.train(data="config/snh-pose.yaml",
                      epochs=100, imgsz=640, batch=64,
                      device=[0, 1], workers=16)

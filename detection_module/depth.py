from transformers import pipeline
from PIL import Image
import requests

cache_dir="/home/wzhangbu/elderlycare/weights"

# load pipe
pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-base-hf", cache_dir=cache_dir)

# load image
image = Image.open("images/test.png")

# inference
depth = pipe(image)["depth"]

print(type(depth))

# save image as rgb
depth.save("depth_base.png")
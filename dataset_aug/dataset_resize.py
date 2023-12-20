import os
import json
from PIL import Image
IMAGE_PATH = "D:/YOLOB/datasets/COCO/val2017"
ANNOT_PATH = "D:/YOLOB/datasets/COCO/annotations/instances_val2017.json"
lst = os.listdir(IMAGE_PATH)

for img in lst:
    im1 = Image.open(IMAGE_PATH + "/" + img)
    new_size = (416, 416)  # 새로운 크기 (가로, 세로)
    resized_image = im1.resize(new_size)
    resized_image.save(IMAGE_PATH + "/" + img)
with open (ANNOT_PATH, 'r') as f:
    json_data = json.load(f)
    for annot in json_data["annotations"]:
        x = 0
        y = 0
        for img in json_data["images"]:
            if img["id"] == annot["image_id"]:
                x = img["width"]
                y = img["height"]

        annot["bbox"] = [annot["bbox"][0] * (416/x), annot["bbox"][1] * (416/y), annot["bbox"][2] * (416/x), annot["bbox"][3] * (416/y)]
    with open(ANNOT_PATH, 'w') as file:
        json.dump(json_data, file)

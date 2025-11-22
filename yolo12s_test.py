from ultralytics import YOLO
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import glob
import random

model = YOLO(r'C:\Users\admin\Desktop\smoking detection\yolov12s\weights\best.pt')

test_images = glob.glob(r'C:\Users\admin\Desktop\smoking detection\test\images\*.jpg')  # Đổi path cho đúng

# Random image
random_image = random.choice(test_images)
print(f"Testing: {random_image}")

# Predict
results = model.predict(random_image, conf=0.25)

# Show
for result in results:
    img = result.plot()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(f'Detection: {len(result.boxes)} objects\n{random_image}')
    plt.show()

# model.predict(
#     source=r"C:\Users\admin\Desktop\smoking detection\test_vid\smoking2.mov",
#     save=True,
#     save_txt=False,
#     conf=0.25,
# )
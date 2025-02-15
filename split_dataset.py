import os
import random
import shutil

# Path ของโฟลเดอร์
image_folder = 'E:/pythonFILES/yolov11/datasets/images/train'
label_folder = 'E:/pythonFILES/yolov11/datasets/labels/train'

# โฟลเดอร์สำหรับ val
val_image_folder = 'E:/pythonFILES/yolov11/datasets/images/val'
val_label_folder = 'E:/pythonFILES/yolov11/datasets/labels/val'

# สร้างโฟลเดอร์ val ถ้ายังไม่มี
os.makedirs(val_image_folder, exist_ok=True)
os.makedirs(val_label_folder, exist_ok=True)

# รับรายการภาพทั้งหมดใน train
images = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]  # เปลี่ยนเป็นนามสกุลของภาพที่คุณใช้

# สุ่มเลือกภาพ 20% มาใส่ใน val
val_images = random.sample(images, k=int(len(images) * 0.2))

# คัดลอกภาพและไฟล์ label ไปยังโฟลเดอร์ val
for img in val_images:
    shutil.copy(os.path.join(image_folder, img), os.path.join(val_image_folder, img))
    label_file = img.replace('.jpg', '.txt')  # เปลี่ยน .jpg เป็น .txt
    shutil.copy(os.path.join(label_folder, label_file), os.path.join(val_label_folder, label_file))

print(f"Selected {len(val_images)} images for validation.")

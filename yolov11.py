import torch
from ultralytics import YOLO


def train_model():
    # ตรวจสอบว่า CUDA สามารถใช้ได้หรือไม่
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolo11n.pt").to(device)
    
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())

    # กำหนดการฝึกโมเดล
    model.train(
        data="dataset_guitar.yaml",  # ชื่อไฟล์ YAML สำหรับข้อมูล
        epochs=20,                   # จำนวน epochs
        imgsz=640,                   # ขนาดของภาพ (ปรับได้ตามต้องการ)
        batch=8,                    # ขนาด batch
        device=device,               # ใช้ device เป็น CUDA (GPU) หรือ CPU
        amp=False                     # ใช้ Automatic Mixed Precision (AMP) เพื่อประหยัดหน่วยความจำและเพิ่มประสิทธิภาพ
    )



if __name__ == "__main__":
    # หากเป็นระบบ Windows ควรใช้การตรวจสอบ main module
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    # เริ่มฝึกโมเดล
    train_model()

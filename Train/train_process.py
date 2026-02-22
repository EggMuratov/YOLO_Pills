from ultralytics import YOLO
import torch
from multiprocessing import freeze_support


model_name = "" #your model path
data_path = "" #your .yaml file
batch_size = 2
epochs = 70
model = YOLO(model_name + ".pt")


def main():
    print("_______________________TRAINING_______________________")

    results = model.train(data=data_path, epochs=epochs, batch=batch_size, imgsz=416, save_period=3,
                          device='0' if torch.cuda.is_available() else 'cpu', lr0=0.001,
                          name=f"{model_name}_train_stat", conf=0.5, iou=0.7)

    model.save(f"{model_name}_pills_detected.pt")


if __name__ == '__main__':
    freeze_support()
    main()

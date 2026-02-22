from ultralytics import YOLO
import torch

model_path = "" #your model path
image_path = "" #path for your image

model = YOLO(model_path)

results = model.predict(
    source=image_path,
    save=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    show_labels=True,
    show_conf=True,
    retina_masks=True,
)

for result in results:
    print(f"Обработка изображения: {result.path}")


    if result.boxes is not None:
        boxes = result.boxes

        print(f"\nНайдено объектов: {len(boxes)}")
        print("=" * 50)

        for i, box in enumerate(boxes):
            class_id = int(box.cls.item())
            confidence = box.conf.item()

            class_name = model.names[class_id]

            bbox = box.xyxy[0].tolist()
            x1, y1, x2, y2 = bbox

            print(f"Объект {i + 1}:")
            print(f"  Класс: {class_name} (ID: {class_id})")
            print(f"  Уверенность: {confidence:.2%}")

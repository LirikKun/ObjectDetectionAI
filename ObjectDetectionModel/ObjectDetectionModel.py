from ultralytics import YOLO
import os


class ObjectDetectionModel:
    def __init__(self, model: str = 'yolo12x-oiv7.pt', translated_labels_name: str = None) -> None:
        self._model_name = os.path.basename(model)

        module_dir = os.path.dirname(os.path.abspath(__file__))

        try:
            model_path = os.path.join(module_dir, 'models', self._model_name)
        except:
            model_path = model
        self._model = YOLO(model_path)

        if translated_labels_name is not None:
            translated_label_path = os.path.join(module_dir, 'translated_labels', translated_labels_name)
            self._translated_labels = dict()
            with open(translated_label_path, 'rt', encoding='utf-8') as translated_labels_file:
                for item in translated_labels_file:
                    key, value = item.strip().split(',')
                    self._translated_labels[key] = value
        else:
            self._translated_labels = None

    def predict(self, image_path: str):
        results = self._model(image_path)[0]

        labels = []
        for box in results.boxes:
            cls_id = int(box.cls[0])

            if self._translated_labels is not None:
                label = self._translated_labels[results.names[cls_id]]
            else:
                label = results.names[cls_id]

            labels.append(label)

        annotated_img = results.plot()

        return annotated_img, labels

    def train(self, epochs=100, batch=16, imgsz=640, workers=8, device=0):
        results = self._model.train(
            data="open-images-v7.yaml",
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            workers=workers,
            device=device,
            name=f'{self._model_name}-oiv7'
        )


if __name__ == '__main__':
    model = ObjectDetectionModel('yolov8x-oiv7.pt')
    model.predict(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_image.jpg'))



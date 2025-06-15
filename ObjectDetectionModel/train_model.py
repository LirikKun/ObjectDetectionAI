from ObjectDetectionModel import ObjectDetectionModel
import os

if __name__ == '__main__':
    model = ObjectDetectionModel()

    model.train(
        epochs=100,
        batch=16,
        imgsz=640,
        workers=8,
        device=0
    )
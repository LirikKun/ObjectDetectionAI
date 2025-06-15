import os
import cv2
import tempfile
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from ObjectDetectionModel.ObjectDetectionModel import ObjectDetectionModel

# Инициализируем модель
model = ObjectDetectionModel(
    model='yolo12x.pt',
    translated_labels_name='ru.csv'
)

# Обработчик изображений
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        file_path = tmp_file.name

    file = await photo.get_file()
    await file.download_to_drive(file_path)

    annotated_img, labels = model.predict(file_path)

    result_path = file_path.replace(".jpg", "_result.jpg")
    cv2.imwrite(result_path, annotated_img)

    await update.message.reply_photo(photo=open(result_path, 'rb'))

    if labels:
        unique_labels = sorted(set(labels))
        text = "Найдены объекты:\n" + "\n".join(f"- {label}" for label in unique_labels)
    else:
        text = "Объекты не найдены."
    await update.message.reply_text(text)

    os.remove(file_path)
    os.remove(result_path)

def main():
    with open('token', 'rt', encoding='utf-8') as token_file:
        token = token_file.readline().strip()
    app = ApplicationBuilder().token(token).build()

    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    print("Бот запущен...")
    app.run_polling()

if __name__ == '__main__':
    main()

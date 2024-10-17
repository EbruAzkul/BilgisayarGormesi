from ultralytics import YOLO
from PIL import Image

# modeli yükle
model = YOLO('yolov8n.pt')  # load a pretrained model

# resim dosyası üzerinde nesne tanıma
im1 = Image.open("barisayilmaz.jpeg")
sonuc = model.predict(source=im1, save=True)  # save ile resmi kaydeder
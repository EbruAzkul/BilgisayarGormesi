# pip install torch  # gpu kullanmak için
# pip install ultralytics # yolo kullanmak için
# pip install roboflow # roboflowa bağlanmak için

import torch
from ultralytics import YOLO
from roboflow import Roboflow

# CUDA ve GPU kullanılabilirliğini kontrol et
# bilgisayarınızda komut satırında nvidia-smi yazdığınızda indirebileceğiniz en son cuda sürümü çıkıyor.
# kullandığınız python versiyonu ile uyumunu kontrol edip uygun olan cudayı indirebilirsiniz
print("CUDA Version:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())

# YOLOv8 modelini başlat
model = YOLO('yolov8n.pt')  # YOLOv8 Nano modeli

# Roboflow API anahtarı ile projeyi indir
# Eğer hata verirse roboflowda kendiniz oturum açın. datanızı roboflowda etiketleyip derste göstediğimiz gibi aşağıdaki kodu elde edip değiştirin
# Kendiniz etiketlemek isterseniz diye kullandığım trafik lambası resimlerini, resimler klasörü içine ekledim


rf = Roboflow(api_key="imHCJ3m5FI110narMpFB")
project = rf.workspace("ebruazkul").project("bilgisayargormesikayisi")
version = project.version(5)
dataset = version.download("yolov8")

# Eğitim ve test işlemlerini ana kod bloğuna alıyoruz
# Modeli eğit
if __name__ == "__main__" :
    model.train(data=f"{dataset.location}/data.yaml", epochs=2)
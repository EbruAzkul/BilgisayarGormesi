import os
import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# YOLO dosyalarının yolları
yolo_classes = r"C:\\Users\\Lenovo\\PycharmProjects\\odev2\\models\\coco.names"
yolo_weights = r"C:\\Users\\Lenovo\\PycharmProjects\\odev2\\models\\yolov4.weights"
yolo_config = r"C:\\Users\\Lenovo\\PycharmProjects\\odev2\\models\\yolov4.cfg"


def load_yolo_model(weights, config):
    try:
        net = cv2.dnn.readNet(weights, config)
        print("Model başarıyla yüklendi.")
        return net
    except cv2.error as e:
        print(f"Hata: {e}")
        return None


def load_classes(class_file):
    with open(class_file, "r") as f:
        return [line.strip() for line in f.readlines()]


def fetch_image_urls(url):
    try:
        response = requests.get(url)
        html_content = response.text
        soup = BeautifulSoup(html_content, "html.parser")
        img_tags = soup.find_all("img")
        return [urljoin(url, img['src']) for img in img_tags if 'src' in img.attrs]
    except Exception as e:
        print(f"Görsel URL'leri alınamadı: {url}, Hata: {e}")
        return []


def download_image(image_url):
    try:
        image_response = requests.get(image_url)
        image = np.asarray(bytearray(image_response.content), dtype="uint8")
        return cv2.imdecode(image, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Görüntü indirilemedi: {image_url}, Hata: {e}")
        return None


def detect_person(net, image):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    outputs = net.forward(output_layers)
    boxes, confidences, class_ids = [], [], []
    human_class_index = classes.index("person")

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:
                center_x, center_y, w, h = (int(detection[0] * width),
                                            int(detection[1] * height),
                                            int(detection[2] * width),
                                            int(detection[3] * height))
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    person_detected = False

    for i in indices.flatten():
        box = boxes[i]
        x, y, w, h = box
        if class_ids[i] == human_class_index:
            person_detected = True
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return person_detected


def show_image(url, image):
    cv2.imshow(f"Sonuç: {url}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Model ve sınıfları yükleme
net = load_yolo_model(yolo_weights, yolo_config)
classes = load_classes(yolo_classes)

# URL'lerin bulunduğu dosyayı okuma
url_file = "dosya.txt"
with open(url_file, "r") as f:
    urls = [line.strip().replace('"', '') for line in f.readlines()]

# Sonuçları tutmak için bir liste
results = []

for url in urls:
    if not url.startswith("http"):
        print(f"Görsel bulunamadı: {url}")
        results.append((url, "Geçersiz URL"))
        continue

    img_urls = fetch_image_urls(url)

    if img_urls:
        image_url = img_urls[0]
        print(f"Görsel URL: {image_url}")
        image = download_image(image_url)

        if image is None:
            results.append((url, "Görsel indirilemedi"))
            continue

        if detect_person(net, image):
            results.append((url, "İnsan"))
        else:
            results.append((url, "İnsan Değil"))

        show_image(url, image)
    else:
        results.append((url, "Görsel bulunamadı"))

# Sonuçları yazdırma
for url, result in results:
    print(f"{url}: {result}")

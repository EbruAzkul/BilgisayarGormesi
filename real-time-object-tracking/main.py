import datetime
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Sabit tanımları
CONFIDENCE_THRESHOLD = 0.5
GREEN = (0, 255, 0)

# Webcam başlatma
video_cap = cv2.VideoCapture(0)

# YOLOv8 modelini yükleme
model = YOLO("yolov8n.pt")  # YOLOv8'in en hızlı modeli

# DeepSORT takipçisi başlatma
tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)

while True:
    # Kareyi oku
    ret, frame = video_cap.read()
    if not ret:
        print("Kamera akışı bulunamadı!")
        break

    # İşlem süresi hesaplaması için başlangıç zamanı
    start = datetime.datetime.now()

    # YOLO ile tespit yap
    detections = model(frame)[0]

    # Tespitleri liste olarak al: [xmin, ymin, xmax, ymax, confidence_score, class_id]
    detection_list = []
    for data in detections.boxes.data.tolist():
        confidence = data[4]
        if confidence < CONFIDENCE_THRESHOLD:
            continue  # Zayıf tespitleri atla

        xmin, ymin, xmax, ymax = map(int, data[:4])
        class_id = int(data[5])
        detection_list.append(([xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id))

    # Takip için DeepSORT'a tespitleri gönder
    tracks = tracker.update_tracks(detection_list, frame=frame)

    # Tespit edilen ve takip edilen nesneleri çiz
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()  # [left, top, right, bottom]
        cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])),
                      (int(ltrb[2]), int(ltrb[3])), GREEN, 2)
        cv2.putText(frame, f"ID: {track_id}", (int(ltrb[0]), int(ltrb[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # FPS hesapla ve ekrana yazdır
    end = datetime.datetime.now()
    total = (end - start).total_seconds()
    fps = f"FPS: {1 / total:.2f}"
    cv2.putText(frame, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Ekrana kareyi yansıt
    cv2.imshow("Webcam - Nesne Takibi", frame)

    # 'q' tuşuna basıldığında döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
video_cap.release()
cv2.destroyAllWindows()

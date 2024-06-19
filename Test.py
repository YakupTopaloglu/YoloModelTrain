import cv2
from ultralytics import YOLO

# Eğitilen modeli yükle
model_path = 'C:/Yedek/Masaüstü/Gamze/aiforG/best.pt'
model = YOLO(model_path)

# fotoğraf dosyasını yükle
image_path = 'C:/Yedek/Masaüstü/Gamze/aiforG/uzum_2.jpg'
cap = cv2.VideoCapture(image_path)

# Sınıf isimlerini yükle
class_names = model.names
print(class_names)
# Fotoğrafı işle
while cap.isOpened():
    ret, frame = cap.read()  # Bir sonraki kareyi al

    if not ret:
        break

    # Tahmin yap
    results = model.predict(frame)

    # Tahmin sonuçlarını işleme
    for result in results:
        boxes = result.boxes.xyxy  # Bounding box koordinatları
        scores = result.boxes.conf  # Güven skorları
        classes = result.boxes.cls  # Sınıf etiketleri

        # Her bir tahmin için
        for box, score, cls in zip(boxes, scores, classes):
            label = class_names[int(cls)]  # Sınıf etiketini sınıf ismine çevir
            x1, y1, x2, y2 = map(int, box)  # Koordinatları integer'a çevir
            
            # Bounding box ve etiket ekle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Görüntüyü göster
    cv2.imshow('Fotoğraf-Tasarım-Gamze', frame)

    if cv2.waitKey(1000000) & 0xFF == ord('q'):  # 'q' tuşuna basarak çıkış yap
        break

# Temizle
cap.release()
cv2.destroyAllWindows()

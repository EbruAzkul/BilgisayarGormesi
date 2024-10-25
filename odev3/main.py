import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntü dosyasını yükle
image_path = r"C:\\Users\\Lenovo\\Desktop\\Adsiz.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Görüntü dosyası bulunamadı: {image_path}")

# Sobel operatörleriyle yatay ve dikey türevleri hesapla
sobel_x = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)  # X yönü türevi
sobel_y = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)  # Y yönü türevi

# Negatif değerleri pozitif yapmak için mutlak değerlerini al
sobel_x = np.abs(sobel_x)
sobel_y = np.abs(sobel_y)

# Kenarları birleştir (L2 norm kullanarak daha hassas bir kenar görüntüsü)
combined_sobel = cv2.magnitude(sobel_x, sobel_y)

# Normalize ederek 0-255 aralığına getir
combined_sobel = cv2.normalize(combined_sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Sonuçları görselleştirme
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

# Orijinal görüntü
axs[0].imshow(image, cmap='gray')
axs[0].set_title("Orijinal Görüntü")
axs[0].axis("off")

# Sobel X sonucu
axs[1].imshow(sobel_x, cmap='gray')
axs[1].set_title("Yatay Türev (Sobel X)")
axs[1].axis("off")

# Sobel Y sonucu
axs[2].imshow(sobel_y, cmap='gray')
axs[2].set_title("Dikey Türev (Sobel Y)")
axs[2].axis("off")

# Birleştirilmiş türev görüntüsü
axs[3].imshow(combined_sobel, cmap='gray')
axs[3].set_title("Birleştirilmiş Türev (Sobel X + Y)")
axs[3].axis("off")

plt.show()

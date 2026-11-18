# 🫁 COVID-19 Akciğer Röntgeni Sınıflandırma (DenseNet121)

Bu proje, akciğer röntgeni (Chest X-Ray) görüntülerini kullanarak **COVID-19**, **Viral Pnömoni (Viral Pneumonia)** ve **Normal** vakaları sınıflandıran bir derin öğrenme (Transfer Learning) çalışmasıdır. Model, önceden ImageNet üzerinde eğitilmiş **DenseNet121** mimarisi üzerine kurulmuştur.

---

## 📋 İçindekiler

- [Proje Özeti](#-proje-özeti)
- [Veri Seti](#-veri-seti)
- [Model Mimarisi ve Eğitim](#-model-mimarisi-ve-eğitim)
- [Sonuçlar ve Performans](#-sonuçlar-ve-performans)
- [Kullanılan Teknolojiler](#-kullanılan-teknolojiler)
- [Kurulum ve Çalıştırma](#-kurulum-ve-çalıştırma)
- [Proje Yapısı](#-proje-yapısı)

---

## 🎯 Proje Özeti

Çalışma, tıbbi görüntü sınıflandırma problemlerinde transfer öğrenme (Transfer Learning) yaklaşımının başarısını göstermeyi amaçlamaktadır. Notebook içerisinde uçtan uca şu adımlar yer almaktadır:

1. Gerekli kütüphanelerin yüklenmesi
2. Veri yollarının belirlenmesi ve sınıf örneklerinin görselleştirilmesi
3. Veri ön işleme (Preprocessing) ve veri çoğaltma (Data Augmentation)
4. `DenseNet121` tabanlı sınıflandırma modelinin oluşturulması
5. Modelin eğitilmesi (98 epoch)
6. Eğitim/doğrulama grafiklerinin çizdirilmesi, test seti değerlendirmesi, sınıflandırma raporu ve karmaşıklık matrisi (Confusion Matrix)
7. Rastgele test görüntüleri üzerinde tahmin görselleştirmesi
8. Eğitilen modelin `.h5` formatında kaydedilmesi

---

## 📊 Veri Seti

Projede Kaggle üzerinde yer alan **[COVID-19 Image Dataset](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)** (`pranavraikokte/covid19-image-dataset`) kullanılmıştır.

- **Sınıflar (3 adet):** `Covid`, `Normal`, `Viral Pneumonia`
- **Veri Dağılımı:**
  - **Eğitim (Train):** 201 görüntü (eğitim klasörünün %80'i)
  - **Doğrulama (Validation):** 50 görüntü (eğitim klasörünün %20'si)
  - **Test:** 66 görüntü

### Veri Ön İşleme ve Çoğaltma (`ImageDataGenerator`)
- **Görüntü Boyutu:** `224 x 224` piksel
- **Batch Size:** `32`
- **Normalizasyon:** `rescale=1./255`
- **Veri Çoğaltma (yalnızca eğitim/doğrulama için):**
  - `rotation_range=20`
  - `zoom_range=0.15`
  - `horizontal_flip=True`

---

## 🧠 Model Mimarisi ve Eğitim

Model, **DenseNet121** (`weights='imagenet'`, `include_top=False`) temel alınarak ve temel model katmanları dondurularak (`trainable=False`) oluşturulmuştur:

```text
DenseNet121 (ImageNet ağırlıkları, dondurulmuş)
   └── GlobalAveragePooling2D()
         └── Dense(128, activation='relu')
               └── Dropout(0.3)
                     └── Dense(3, activation='softmax')
```

- **Optimizer:** `Adam`
- **Loss Function:** `categorical_crossentropy`
- **Metric:** `accuracy`
- **Epoch Sayısı:** `98`

---

## 📈 Sonuçlar ve Performans

Hiç görülmemiş **66 adet test görüntüsü** üzerinde elde edilen nihai sonuçlar:

- **Test Doğruluğu (Accuracy):** `%95.45`
- **Test Hatası (Loss):** `0.1856`

### Detaylı Sınıflandırma Raporu

| Sınıf | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Covid** | 1.00 | 0.96 | 0.98 | 26 |
| **Normal** | 0.95 | 0.90 | 0.92 | 20 |
| **Viral Pneumonia** | 0.91 | 1.00 | 0.95 | 20 |
| **Genel Doğruluk (Accuracy)** | | | **0.95** | **66** |
| **Macro Avg** | 0.95 | 0.95 | 0.95 | 66 |
| **Weighted Avg** | 0.96 | 0.95 | 0.95 | 66 |

> Model eğitimi sonunda ağırlıklar `covid19_densenet121_model.h5` dosyası olarak dışa aktarılmaktadır.

---

## 🛠 Kullanılan Teknolojiler

- **Python 3**
- **TensorFlow / Keras** (Notebook'ta `2.20.0` ile çalıştırılmıştır)
- **NumPy** & **Pandas**
- **Matplotlib** & **Seaborn**
- **Scikit-learn** (`classification_report`, `confusion_matrix`)

---

## 🚀 Kurulum ve Çalıştırma

### 1. Kaggle Üzerinde Çalıştırma (Önerilen)
Notebook doğrudan Kaggle dosya yapısına (`/kaggle/input/...`) göre ayarlanmıştır:
1. Kaggle'da yeni bir Notebook oluşturun veya `covid19-2.ipynb` dosyasını içe aktarın.
2. Veri seti olarak `pranavraikokte/covid19-image-dataset` ekleyin.
3. Hücreleri sırasıyla çalıştırın.

### 2. Yerel Ortamda Çalıştırma
```bash
# Gerekli kütüphaneleri yükleyin
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn jupyter

# Jupyter Notebook'u başlatın
jupyter notebook covid19-2.ipynb
```
> **Not:** Yerel ortamda çalıştırırken 2. aşamadaki `/kaggle/input` yolunu kendi bilgisayarınızdaki `train` ve `test` klasörlerinin yoluyla güncellemeyi unutmayın.

---

## 📁 Proje Yapısı

```text
covid19/
├── covid19-2.ipynb    # Veri hazırlama, DenseNet121 eğitimi, değerlendirme ve görselleştirme
└── README.md          # Proje dokümantasyonu
```

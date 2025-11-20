# 🌫️ Air Pollution Forecasting & Analysis App  
**LSTM / GRU Model + Flask Web App + Gemini AI Enriched Interpretation**

Bu proje, Pekin şehrinin PM2.5 hava kirliliği seviyelerini GRU tabanlı derin öğrenme modeli ile tahmin eden  
ve diğer şehirler (Ankara, İstanbul, İzmir vb.) için Gemini AI üzerinden gerçek zamanlı hava kalitesi analizi sunan bir web uygulamasıdır.

Uygulama, tamamen interaktif bir web arayüzü (index.html) ile çalışır ve hem ML tahmini hem de LLM destekli analiz üretir.

## 📁 Proje Yapısı


AirApp/
│
├── app.py # Flask API + ML Prediction + Gemini Analysis
├── templates/
│ └── index.html # Web arayüzü
│
├── best_pollution_lstm_model.h5 # GRU / LSTM derin öğrenme modeli
├── LSTM-Multivariate_pollution.csv # Veri seti
├── scaled_pollution.csv # Normalize edilmiş veri
├── .env # API keyler için önerilen dosya
└── README.md


---

## ⚙️ Kullanılan Teknolojiler

| Yapı | Açıklama |
|------|----------|
| **Flask** | Web sunucusu ve API |
| **TensorFlow / Keras** | GRU tabanlı derin öğrenme modeli |
| **Pandas / NumPy** | Veri işleme |
| **MinMaxScaler** | Normalizasyon |
| **Gemini 2.5 Flash API** | Gerçek zamanlı şehir analizi & ML tahmin yorumu |
| **HTML + TailwindCSS** | Web arayüzü |

---

# 🚀 Özellikler

### 🔮 **1. Pekin için ML tahmini**
- Model, 24 saatlik pencere (N=24) ile PM2.5 tahmini üretir.  
- 7 günlük tahmin – gerçek karşılaştırması yapılır.
- Tahmin sonuçları otomatik olarak Gemini AI tarafından yorumlanır.

### 🌍 **2. Diğer Şehirler için LLM-Gerçek Zamanlı Analiz**
- Ankara, İstanbul, İzmir vb. şehirlerde:
  - Gemini → Google Search + Güncel Hava Kalitesi + AQI çekilir.
  - Risk değerlendirmesi yapılır.
  - Halk sağlığı önerileri sunulur.

### 📊 **3. Kullanıcı Dostu Web Arayüzü**
- Tek sayfalık HTML/Tailwind UI
- “Tahmin & Analiz” butonu
- Sonuçlar: tablo, grafik, yorum bölümü

---

# 🧠 GRU Modeli (Özet)

Model, Pekin PM2.5 hava kalitesi çok değişkenli veri seti üzerinde eğitilmiştir:

**Kullanılan girdiler (multivariate):**
- temperature  
- pressure  
- humidity  
- wind speed  
- wind direction (LabelEncoding)  
- pollution (PM2.5)

**Model Çıkışı:**  
→ “Sonraki saat PM2.5 değeri”

**Gerçekleşen performans:**  
- RMSE ≈ **23.4 µg/m³**


# 🔧 Kurulum

## 1️⃣ Gerekli paketleri yükleyin

```bash
pip install flask tensorflow pandas numpy scikit-learn requests

Gemini API anahtarınızı .env dosyasına ekleyin
API_KEY=BURAYA_GEMINI_KEY

## 2 Uygulamayı çalıştırma
python app.py




import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

# Hata giderme için gerekli Keras importları
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import json
import requests
import io
import time # Hata durumunda bekleme (exponential backoff) için

# Flask uygulaması kurulumu
app = Flask(__name__)


# Model dosyasını doğru yola ayarlayın
MODEL_PATH = 'best_pollution_lstm_model.h5' 
DATASET_PATH = 'LSTM-Multivariate_pollution.csv'
N_HOURS = 24 
GERCEK_RMSE_MU_GM3 = 23.4 # Pekin modeli için sabit hata değeri

# Global değişkenler
gru_model = None
scaler = None
encoder = None
initial_data_scaled = None
initial_data_raw = None

API_KEY = 'API KEYİNİZİ GİRİNİZ'
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

# --- 1. VERİ HAZIRLAMA VE ÖN İŞLEME FONKSİYONLARI ---

def load_and_preprocess_data():
    """Veri setini yükler, ön işler ve ölçekleyiciyi (scaler) eğitir/kaydeder."""
    global scaler, encoder, initial_data_scaled, initial_data_raw
    
    if not os.path.exists(DATASET_PATH):
        print(f"HATA: Veri seti bulunamadı: {DATASET_PATH}")
        return None, None

    df = pd.read_csv(DATASET_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # NaN değerleri doldur
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Kategorik Sütun (wnd_dir)
    encoder = LabelEncoder()
    df['wnd_dir'] = encoder.fit_transform(df['wnd_dir'])
    initial_data_raw = df.copy()
    
    # Ölçekleme
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df)
    initial_data_scaled = pd.DataFrame(scaled_values, columns=df.columns, index=df.index)

    return initial_data_scaled, initial_data_raw

def denormalize_predictions(y_pred_scaled, target_feature='pollution'):
    """Ölçeklenmiş tahmini, orijinal birimine (µg/m³) geri dönüştürür."""
    # Ölçekleyici, tüm özelliklerle eğitildiği için bir "kukla" (dummy) dizi oluşturulmalıdır.
    dummy = np.zeros((len(y_pred_scaled), initial_data_raw.shape[1]))
    target_index = initial_data_raw.columns.get_loc(target_feature)
    dummy[:, target_index] = y_pred_scaled
    # Ters dönüşüm
    denormalized = scaler.inverse_transform(dummy)[:, target_index]
    return denormalized

def prepare_sequence(scaled_data, start_index, n_steps=N_HOURS):
    """Tahmin için çok değişkenli giriş dizisini hazırlar."""
    end_index = start_index + n_steps
    if end_index > len(scaled_data):
        return None
    sequence = scaled_data.iloc[start_index:end_index].values
    # LSTM/GRU modelinin beklediği şekil: (1, n_steps, n_features)
    return sequence.reshape(1, n_steps, scaled_data.shape[1])

# --- 2. LLM İLE ANALİZ FONKSİYONU ---

def call_gemini_api(payload, headers):
    """Gemini API çağrısını exponential backoff ile yönetir."""
    max_retries = 3
    delay = 2  # İlk bekleme süresi 2 saniye
    for attempt in range(max_retries):
        try:
            # API Key'in kod içine gömülmemesi için os.environ.get kullanılmalıdır.
            # Ancak bu projede basitlik adına direkt API_KEY değişkeni kullanılmıştır.
            response = requests.post(f"{API_URL}?key={API_KEY}", headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"API Hatası (Deneme {attempt + 1}): {e}. {delay} saniye bekleniyor...")
                time.sleep(delay)
                delay *= 2  # Gecikmeyi katlayarak artır
            else:
                raise e

def get_llm_analysis(city, context_data, is_ml_prediction, rmse_value=None):
    """Gemini'ye tahmini ve bağlamsal bilgileri içeren bir prompt gönderir."""
    
    # ------------------ PROMPT FARKLI SENARYOLARA GÖRE AYARLANIYOR ------------------
    if is_ml_prediction:
        # Senaryo 1: ML Modeli (Pekin) tahmini analizi
        df_analysis_str = context_data.to_string(index=True)
        user_prompt = f"""
        Aşağıda GRU derin öğrenme modelinin 24 saatlik girdi penceresi kullanılarak yaptığı bir haftalık PM2.5 tahmininin gerçek değerler (µg/m³) ile karşılaştırması verilmiştir.
        --- TEMEL BAŞARI KRİTERLERİ (Pekin Modeli) ---
        Modelin Ortalama Kök Ortalama Kare Hatası (RMSE): {rmse_value:.2f} µg/m³
        --- KARŞILAŞTIRMALI TAHMİN VERİSİ (µg/m³) ---
        {df_analysis_str}
        
        --- İSTENEN ANALİZ GÖREVLERİ ---
        1. Başarı Yorumu: {rmse_value:.2f} µg/m³ RMSE değerinin, Pekin'in hava kirliliği seviyeleri ve uluslararası standartlar (WHO 15 µg/m³) göz önüne alındığında halk sağlığı açısından ne anlama geldiğini yorumla.
        2. Zirve Analizi: Modelin en büyük hata yaptığı (Gerçek > Tahmin farkının en çok olduğu) saatleri/günleri belirle ve bu zirvelerin olası nedenlerini (meteorolojik/emisyon) açıkla.
        3. Eylem Önerileri: Bu tahmin zirvelerini azaltmak için yerel yönetimlere (halk sağlığı ve ulaşım) yönelik 3 somut, önleyici eylem önerisi sun.
        """
        system_instruction = f"""Sen, Pekin verileri üzerinde eğitilmiş bir GRU modelinin çıktılarını yorumlayan ve halk sağlığı/çevre yönetimine yönelik somut eylem önerileri sunan uzman bir Yapay Zeka Analistisin. Yanıtlarını Türkçe ve net bir dille sun. Cevaplarını 3 ana başlıkta (Başarı Yorumu, Zirve Analizi, Eylem Önerileri) yapılandır ve her başlık altına detaylı madde işaretli bir liste oluştur. Google Search kullanma."""
        tools = []
    
    else:
        # Senaryo 2: Gerçek Zamanlı Analiz (Ankara, İstanbul vb.)
        user_prompt = f"""
        Sen bir çevre danışmanısın. {city} şehrinin güncel hava kalitesi durumunu öğrenmek için Google Arama aracını kullan.
        
        --- İSTENEN ANALİZ GÖREVLERİ ---
        1. Güncel Durum: {city} şehrinin şu anki veya son 24 saatlik PM2.5 veya Hava Kalitesi İndeksi (AQI) değerini Google Search ile bul ve bu değeri WHO veya Avrupa standartlarıyla karşılaştır.
        2. Bağlamsal Yorum: {city}'nin kirlilik tipine ve mevsimsel özelliklerine göre (Türkiye'de kışın ısınma, yazın trafik gibi) bu güncel durumu yorumla.
        3. Eylem Önerileri: {city} yerel yönetimine ve vatandaşlarına yönelik 3 somut eylem önerisi sun (Halk sağlığı uyarısı, trafik kısıtlaması, maske önerisi gibi).
        """
        system_instruction = f"""Sen, Türkiye'deki {city} şehrinin güncel hava kalitesini analiz eden ve eylem önerileri sunan bir Yapay Zeka Danışmanısın. Yanıtlarını Türkçe, somut ve net bir dille sun. Cevaplarını 3 ana başlıkta (Güncel Durum, Bağlamsal Yorum, Eylem Önerileri) yapılandır ve Google Search kullanarak en güncel verilere dayan. """
        tools = [{"google_search": {} }]

    # -----------------------------------------------------------------------------

    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "tools": tools, 
    }
    
    headers = {'Content-Type': 'application/json'}
    
    try:
        result = call_gemini_api(payload, headers)
        
        candidate = result.get('candidates', [{}])[0]
        text = candidate.get('content', {}).get('parts', [{}])[0].get('text', 'Gemini analiz yanıtı alınamadı.')
        
        return text

    except requests.exceptions.RequestException as e:
        print(f"Gemini API Hatası: {e}")
        return f"Hata: Gemini API'ye erişilemiyor veya key/endpoint geçersiz. Detay: {e}"


# --- 3. FLASK YOLLARI (ROUTES) ---

@app.before_request
def initialize_app():
    """Uygulama başlamadan önce ML modelini ve veriyi yükler."""
    global gru_model, initial_data_scaled, initial_data_raw
    if gru_model is None:
        try:
            custom_objects = {
                'mse': MeanSquaredError,
                'mae': MeanAbsoluteError
            }
            # Model yükleme
            gru_model = load_model(MODEL_PATH, custom_objects=custom_objects)
            
            print(f"✅ Model başarıyla yüklendi: {MODEL_PATH}")
            initial_data_scaled, initial_data_raw = load_and_preprocess_data()
            if initial_data_scaled is None:
                raise Exception("Veri yüklenirken hata oluştu.")
            print("✅ Veri başarıyla yüklendi ve ölçeklendi.")
        except Exception as e:
            print(f"HATA: Model veya veri yüklenemedi. Detay: {e}")
            gru_model = False # Hata durumunu belirtmek için False yap

@app.route('/')
def index():
    """Ana sayfa - Tahmin ve Analiz butonları."""
    if gru_model is False:
        return render_template('index.html', error="Sunucu başlatılamadı. ML modeli veya veri seti eksik.")
    return render_template('index.html', error=None)

@app.route('/predict_and_analyze', methods=['POST'])
def predict_and_analyze():
    """Şehre göre Tahmin/Gerçek Zamanlı Analiz yapar."""
    data = request.get_json()
    city = data.get('city', 'Beijing')

    if city == 'Beijing':
        # ------------------- PEKİN: ML Tahmini Akışı -------------------
        if gru_model is False:
            return jsonify({'status': 'error', 'analysis': 'ML modeli yüklenemediği için Pekin tahmini yapılamaz.'})
        
        try:
            # Önceki akıştaki gibi test setinden bir haftalık veri çekilir
            test_data_len = len(initial_data_scaled) // 3
            test_start_index = len(initial_data_scaled) - test_data_len
            start_for_prediction = test_start_index + N_HOURS 
            sample_size = 7 * 24 
            
            predictions_list = []
            actual_list = []
            
            for i in range(sample_size):
                # T+1 tahmin etmek için T anındaki 24 saatlik diziyi hazırla
                sequence = prepare_sequence(initial_data_scaled, start_for_prediction + i, N_HOURS)
                if sequence is None: break
                
                # Tahmin
                y_pred_scaled = gru_model.predict(sequence, verbose=0)[0][0]
                y_actual_scaled = initial_data_scaled.iloc[start_for_prediction + i + N_HOURS].pollution
                
                predictions_list.append(y_pred_scaled)
                actual_list.append(y_actual_scaled)

            y_pred_denorm = denormalize_predictions(np.array(predictions_list))
            y_actual_denorm = denormalize_predictions(np.array(actual_list))
            
            df_analysis = pd.DataFrame({
                'Gerçek_PM25': y_actual_denorm.round(2),
                'Tahmin_PM25': y_pred_denorm.round(2)
            }, index=initial_data_raw.index[start_for_prediction + N_HOURS : start_for_prediction + N_HOURS + sample_size])

            gemini_analysis = get_llm_analysis(city, df_analysis.head(14), True, GERCEK_RMSE_MU_GM3)
            
            response_data = {
                'analysis': gemini_analysis,
                'rmse': f"{GERCEK_RMSE_MU_GM3:.2f} µg/m³",
                'sample_data': df_analysis.head(14).to_html(classes='table-auto w-full rounded-lg text-sm') 
            }
            return jsonify(response_data)
            
        except Exception as e:
            import traceback
            print(f"Flask Hata (Pekin Tahmini): {e}")
            traceback.print_exc()
            return jsonify({'status': 'error', 'analysis': f"Pekin tahmini sırasında kritik hata oluştu: {e}"})

    else:
        # ------------------- ANKARA/DİĞER: LLM Gerçek Zamanlı Analiz Akışı -------------------
        try:
            # LLM'den gerçek zamanlı analiz istenir (context_data = None)
            gemini_analysis = get_llm_analysis(city, None, False)
            
            response_data = {
                'analysis': gemini_analysis,
                'rmse': "Gerçek Zamanlı Arama",
                'sample_data': f'<p class="text-gray-600 p-4">Seçilen şehir ({city}) için ML modelimiz eğitilmemiştir. Analiz, Gemini AI tarafından güncel internet verileri kullanılarak yapılmıştır.</p>' 
            }
            return jsonify(response_data)

        except Exception as e:
            import traceback
            print(f"Flask Hata (Gerçek Zamanlı Analiz): {e}")
            traceback.print_exc()
            return jsonify({'status': 'error', 'analysis': f"Gerçek zamanlı analiz sırasında kritik hata oluştu: {e}"})

if __name__ == '__main__':
    initialize_app()
    if gru_model is not False:
        print("\n--- Flask Sunucusu Başlatılıyor ---\n")
        app.run(host='0.0.0.0', port=5000)
    else:
        print("\n!!! Uygulama başlatılamadı, detaylar yukarıdaki HATA kaydında. !!!")

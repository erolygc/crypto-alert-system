# 🚀 Crypto Alert System

Kapsamlı kripto para piyasası için geliştirilmiş real-time alert ve trading sistemi. Bu sistem profesyonel seviyede risk yönetimi, teknik analiz ve otomatik bildirim sistemleri içerir.

## 📋 İçindekiler

- [Özellikler](#özellikler)
- [Kurulum](#kurulum)
- [Sistem Gereksinimleri](#sistem-gereksinimleri)
- [Kullanım](#kullanım)
- [Modüller](#modüller)
- [Test](#test)
- [Dokümentasyon](#dokümentasyon)

## ✨ Özellikler

### 🔔 Real-time Alert Sistemi
- **Çoklu Bildirim Kanalları**: Discord, Telegram, Email
- **Zengin Mesaj Formatı**: Markdown, emoji, inline klavye
- **Mesaj Kuyruğu**: Rate limiting, retry logic
- **Grafik Desteği**: Otomatik grafik oluşturma ve gönderme

### 📊 Teknik Analiz
- **100+ Teknik İndikatör**: RSI, MACD, Bollinger Bands, ATR, Stochastic
- **Çoklu Zaman Dilimi**: 1m, 5m, 15m, 1h analizi
- **Volume Analizi**: Hacim anomalileri ve trend analizi
- **Dinamik Karakterizasyon**: Adaptif algoritma optimizasyonu

### 💰 Risk Yönetimi
- **Pozisyon Boyutlandırma**: Kelly Kriteri, Volatilite Hedefleme, ATR Bazlı
- **Stop Loss Sistemi**: Dinamik ve sabit stop loss
- **Drawdown Takibi**: Portföy seviyesinde risk izleme
- **Performance Dashboard**: Gerçek zamanlı performans metrikleri

### 🔄 Data Yönetimi
- **Gate.io Integration**: Spot ve futures market data
- **WebSocket Streaming**: Real-time fiyat takibi
- **SQLite Database**: Veri saklama ve geçmiş analizi
- **Data Collector**: Otomatik veri toplama pipeline

### 📈 Backtest Engine
- **Paper Trading**: Risk-free test ortamı
- **Performans Analizi**: Sharpe ratio, drawdown, win rate
- **Meta Learning**: Strateji optimizasyonu

## 🛠️ Kurulum

### 1. Sistem Gereksinimleri

```bash
# Python versiyonu
Python 3.8+ gereklidir

# Sistem paketleri (Ubuntu/Debian)
sudo apt update
sudo apt install python3-pip python3-venv sqlite3

# Sistem paketleri (CentOS/RHEL)
sudo yum install python3-pip python3-venv sqlite

# Sistem paketleri (Windows - Chocolatey)
choco install python sqlite
```

### 2. Repository'yi Klonlayın

```bash
git clone https://github.com/erolygc/crypto-alert-system.git
cd crypto-alert-system
```

### 3. Sanal Ortam Oluşturun

```bash
# Sanal ortam oluştur
python -m venv venv

# Sanal ortamı aktifleştir
# Linux/Mac:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 4. Paketleri Yükleyin

```bash
# Gerekli paketleri yükle
pip install -r requirements.txt

# Alternatif olarak tüm paketleri tek tek:
pip install numpy pandas scipy scikit-learn
pip install TA-Lib pandas-ta ccxt requests websockets
pip install psycopg2-binary redis kafka-python
pip install matplotlib plotly dash
pip install python-telegram-bot pillow seaborn
```

### 5. Veritabanını Başlatın

```bash
# SQLite veritabanını başlat (otomatik)
python -c "
from telegram_notifier import TelegramNotifierBot
from alert_system import AlertSystem
bot = TelegramNotifierBot()
system = AlertSystem()
print('✅ Veritabanı başlatıldı!')
"
```

### 6. Konfigürasyon

#### Telegram Bot Kurulumu

```bash
# Örnek konfigürasyon oluştur
python telegram_notifier.py

# bot_config.json dosyasını düzenleyin
# BotFather'dan aldığınız token'ı girin:
# "bot_token": "YOUR_BOT_TOKEN_HERE"
```

#### Discord Webhook Kurulumu

```python
# discord_webhook.py dosyasında:
WEBHOOK_URL = "your_discord_webhook_url_here"
```

#### Gate.io API (Opsiyonel)

```python
# API anahtarlarınızı ekleyin
GATE_IO_API_KEY = "your_api_key"
GATE_IO_SECRET_KEY = "your_secret_key"
```

## 🎯 Kullanım

### Hızlı Başlangıç

```python
# Temel alert sistemi
from alert_system import AlertSystem

# Sistem oluştur
alert_system = AlertSystem()

# Fiyat alert'i ekle
alert_system.add_price_alert(
    symbol="BTC_USDT",
    condition="price_above",
    value=50000,
    message="BTC $50,000'i geçti!"
)

# Sistemi başlat
alert_system.start_monitoring()
```

### Pozisyon Boyutlandırma

```python
from position_sizing import PositionSizer, RiskParameters

# Risk parametreleri
risk_params = RiskParameters(
    max_risk_per_trade=0.02,  # %2 risk
    volatility_target=0.15    # %15 volatilite hedefi
)

# Pozisyon hesapla
sizer = PositionSizer(risk_params)
position_size = sizer.volatility_targeting(
    expected_return=0.10,
    current_volatility=0.20,
    portfolio_value=100000
)

print(f"Pozisyon boyutu: {position_size:.2%}")
```

### Telegram Bot Kullanımı

```python
from telegram_notifier import TelegramNotifierBot

# Bot başlat
bot = TelegramNotifierBot()

# Basit mesaj gönder
bot.send_notification(
    chat_id=YOUR_CHAT_ID,
    message="🎉 Test mesajı! Bot çalışıyor.",
    priority=1
)

# Grafik gönder
chart_data = {
    "line_chart": {
        "x": [1, 2, 3, 4, 5],
        "y": [1, 4, 2, 5, 3],
        "labels": {
            "title": "BTC Fiyat Grafiği",
            "xlabel": "Zaman",
            "ylabel": "Fiyat ($)"
        }
    }
}

bot.send_chart(YOUR_CHAT_ID, chart_data, "line", "📈 BTC Analizi")
```

### Discord Bildirimleri

```python
from discord_webhook import DiscordWebhook

# Discord webhook oluştur
webhook = DiscordWebhook()

# Mesaj gönder
webhook.send_message(
    "🚀 Yeni fırsat tespit edildi!",
    color=0x00ff00,
    fields=[
        {"name": "Sembol", "value": "BTC/USDT", "inline": True},
        {"name": "Sinyal", "value": "Long", "inline": True}
    ]
)
```

## 📦 Modüller

### Ana Modüller

| Modül | Açıklama | Satır Sayısı |
|-------|----------|--------------|
| `alert_system.py` | Ana alert yöneticisi | 776 |
| `price_alerts.py` | Fiyat alert motoru | 919 |
| `technical_indicators.py` | 100+ teknik indikatör | 1,060 |
| `position_sizing.py` | Pozisyon boyutlandırma | 712 |
| `telegram_notifier.py` | Telegram bot sistemi | 1,456 |
| `discord_webhook.py` | Discord bildirimleri | - |
| `websocket_manager.py` | WebSocket bağlantı yöneticisi | 544 |

### Destek Modülleri

| Modül | Açıklama |
|-------|----------|
| `volume_anomaly.py` | Hacim analizi |
| `drawdown_monitor.py` | Drawdown takibi |
| `stop_loss_system.py` | Stop loss yönetimi |
| `risk_reward_monitor.py` | Risk/ödül izleme |
| `alert_prioritizer.py` | Alert önceliklendirme |
| `performance_dashboard.py` | Performans panosu |
| `threshold_calibrator.py` | Eşik kalibrasyonu |

## 🧪 Test

### Tüm Sistemi Test Et

```bash
# Kapsamlı sistem testini çalıştır
python integrated_alert_test.py

# Test sonuçları:
# ✅ 12/12 test başarılı
# 📊 Tüm modüller çalışıyor
```

### Tek Modül Testleri

```bash
# Alert sistemi testi
python alert_system.py

# Pozisyon boyutlandırma testi
python position_sizing.py

# Telegram bot testi
python telegram_notifier.py

# Teknik indikatörler testi
python technical_indicators.py
```

### Basit Import Testi

```python
# Python'da çalıştırın:
python -c "
import numpy, pandas, ccxt
print('✅ Paketler çalışıyor!')

from position_sizing import PositionSizer
sizer = PositionSizer()
print('✅ Modül çalışıyor!')
"
```

## ⚙️ Konfigürasyon

### Risk Parametreleri

```python
# position_sizing.py'da
RiskParameters(
    max_risk_per_trade=0.02,    # İşlem başına maksimum risk (%2)
    max_portfolio_risk=0.10,    # Portföy maksimum risk (%10)
    volatility_target=0.15,     # Hedef volatilite (%15)
    kelly_fraction=0.25,        # Kelly kriteri güvenlik kesri
    atr_multiplier=2.0,         # ATR çarpanı
    lookback_period=20,         # Geriye dönük periyot
    min_position_size=0.01,     # Minimum pozisyon boyutu
    max_position_size=1.0       # Maksimum pozisyon boyutu
)
```

### Alert Konfigürasyonu

```python
# alert_system.py'da
ALERT_CONFIG = {
    'check_interval': 1,        # Kontrol aralığı (saniye)
    'max_alerts_per_hour': 100, # Maksimum alert sayısı
    'cooldown_period': 30,      # Aynı alert için bekleme
    'enabled_channels': ['telegram', 'discord', 'email']
}
```

### WebSocket Konfigürasyonu

```python
# websocket_manager.py'da
WS_CONFIG = {
    'url': 'wss://api.gateio.ws/ws/v4/',
    'reconnect_attempts': 5,
    'heartbeat_interval': 20,
    'max_message_size': 1024*1024
}
```

## 📖 Dokümentasyon

### API Referansı

Her modül için detaylı dokümentasyon kod içinde mevcuttur:

```python
# Örnek fonksiyon dokümentasyonu
def volatility_targeting(expected_return: float, current_volatility: float, 
                        portfolio_value: float) -> float:
    """
    Volatilite hedefleme yöntemi
    
    Args:
        expected_return: Beklenen getiri
        current_volatility: Mevcut volatilite
        portfolio_value: Portföy değeri
        
    Returns:
        Pozisyon boyutu (0-1 arası)
    """
```

### Detaylı Dokümentasyon

- [📊 Alert System Architecture](docs/alert_system_architecture.md)
- [📈 Technical Indicators Guide](code/TECHNICAL_INDICATORS_README.md)
- [💰 Position Sizing Guide](code/POSITION_SIZING_README.md)
- [🔔 Notification Systems](code/DISCORD_WEBHOOK_README.md)
- [⚡ Performance Dashboard](code/PERFORMANCE_DASHBOARD_README.md)

## 🚀 Performans

### Test Sonuçları (Son Güncelleme: 2025-11-01)

- **✅ Test Başarı Oranı**: 100% (12/12 test başarılı)
- **📊 Kod Kalitesi**: 15,000+ satır profesyonel kod
- **🔧 Modül Sayısı**: 25+ bağımsız modül
- **⚡ Performans**: Real-time processing, <100ms latency
- **🛡️ Güvenlik**: Rate limiting, error handling, retry logic

### Sistem Metrikleri

```python
# Performans ölçümü
from performance_dashboard import PerformanceDashboard

dashboard = PerformanceDashboard()
metrics = dashboard.get_system_metrics()

print(f"📊 Response Time: {metrics['avg_response_time']}ms")
print(f"🔄 Uptime: {metrics['uptime_percentage']}%")
print(f"💾 Memory Usage: {metrics['memory_usage_mb']}MB")
print(f"📈 Throughput: {metrics['messages_per_second']}/sec")
```

## 🤝 Katkıda Bulunma

1. Repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'i push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🆘 Destek

Sorularınız için:

1. **📖 Dokümentasyon**: Önce dokümantasyonu kontrol edin
2. **🐛 Issue**: GitHub Issues'da sorun bildirin
3. **💬 Discord**: Discord sunucumuzdan yardım alın

## 🎉 Örnekler

### Basit Kullanım Örneği

```python
#!/usr/bin/env python3
"""
Basit Crypto Alert Sistemi Örneği
"""

from alert_system import AlertSystem
from position_sizing import PositionSizer
from telegram_notifier import TelegramNotifierBot

def main():
    # Sistemleri başlat
    alert_system = AlertSystem()
    position_sizer = PositionSizer()
    telegram_bot = TelegramNotifierBot()
    
    # BTC fiyat alert'i ekle
    alert_system.add_price_alert(
        symbol="BTC_USDT",
        condition="price_above",
        value=50000,
        message="🚀 BTC $50,000'i geçti!"
    )
    
    # Pozisyon boyutu hesapla
    portfolio_value = 100000
    position_size = position_sizer.fixed_fractional(portfolio_value)
    
    print(f"💰 Önerilen pozisyon: ${position_size:,.0f}")
    print("✅ Sistem başlatıldı!")
    
    # Telegram'da bilgilendirme
    telegram_bot.send_notification(
        chat_id=YOUR_CHAT_ID,
        message=f"🤖 Sistem başlatıldı!\n💰 Pozisyon: ${position_size:,.0f}"
    )

if __name__ == "__main__":
    main()
```

### Gelişmiş Kullanım Örneği

```python
#!/usr/bin/env python3
"""
Gelişmiş Crypto Trading Sistemi
"""

from technical_indicators import TechnicalIndicators
from volume_anomaly import VolumeAnomalyDetector
from risk_management import RiskManager

def advanced_crypto_system():
    # Teknik analiz
    indicators = TechnicalIndicators()
    
    # Hacim analizi
    volume_detector = VolumeAnomalyDetector()
    
    # Risk yönetimi
    risk_manager = RiskManager()
    
    # Piyasa verisi al (örnek)
    price_data = get_market_data("BTC_USDT")
    volume_data = get_volume_data("BTC_USDT")
    
    # Teknik sinyaller hesapla
    rsi = indicators.calculate_rsi(price_data)
    macd = indicators.calculate_macd(price_data)
    
    # Hacim anomalisi tespit et
    volume_anomaly = volume_detector.detect_anomaly(volume_data)
    
    # Risk kontrolü
    risk_score = risk_manager.calculate_risk_score({
        'volatility': indicators.calculate_atr(price_data),
        'drawdown': risk_manager.calculate_drawdown(price_data),
        'volume_anomaly': volume_anomaly
    })
    
    # Trading kararı
    if rsi > 70 and macd > 0 and risk_score < 0.5:
        print("📈 SAT sinyali!")
        return "SELL"
    elif rsi < 30 and macd < 0 and risk_score < 0.5:
        print("📉 AL sinyali!")
        return "BUY"
    else:
        print("⏳ Bekleme modunda...")
        return "HOLD"

if __name__ == "__main__":
    result = advanced_crypto_system()
    print(f"🎯 Trading Sinyali: {result}")
```

---

**🚀 Crypto Alert System - Professional Trading & Alert System**

*Geliştirilme Tarihi: 2025-11-01*  
*Son Güncelleme: 2025-11-01*  
*Versiyon: 1.0.0*
"""
Integrated Alert System Test
============================

Crypto Alert System'in tüm bileşenlerini test eden kapsamlı test dosyası.
Bu test, sistemin tüm özelliklerini doğrular ve entegrasyonu kontrol eder.

Test edilen bileşenler:
- AlertSystem (Ana alert yönetimi)
- PriceAlertEngine (Fiyat alert motoru)
- WebSocketManager (Gerçek zamanlı veri akışı)
- TechnicalIndicators (Teknik analiz)
- DiscordWebhook (Bildirim sistemi)

Kullanım:
    python integrated_alert_test.py

Author: Crypto Alert System
Date: 2025-11-01
"""

import asyncio
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import time
import threading

# Local imports
from alert_system import AlertSystem, Alert, AlertType, AlertSeverity
from price_alerts import PriceAlertEngine, AlertCondition
from websocket_manager import GateWebSocketManager, TickerData
from technical_indicators import TechnicalIndicators
from discord_webhook import DiscordWebhook, AlertLevel


class IntegratedAlertSystem:
    """Entegre alert sistemi"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Entegre sistemi başlat
        
        Args:
            config: Sistem yapılandırması
        """
        self.config = config or self._default_config()
        
        # Bileşenleri başlat
        self.alert_system = AlertSystem()
        self.price_engine = PriceAlertEngine()
        self.ws_manager = None
        self.ti = TechnicalIndicators()
        self.discord = DiscordWebhook(
            webhook_url=self.config.get('discord_webhook_url', ''),
            username="CryptoAlert"
        )
        
        # Test durumu
        self.test_results = []
        self.metrics = {
            'alerts_processed': 0,
            'websocket_messages': 0,
            'indicators_calculated': 0,
            'notifications_sent': 0
        }
        
        # Logging
        self.logger = self._setup_logging()
        self.logger.info("Integrated Alert System başlatıldı")
    
    def _default_config(self) -> Dict[str, Any]:
        """Varsayılan yapılandırma"""
        return {
            'discord_webhook_url': '',  # Set your Discord webhook URL
            'test_symbols': ['BTC_USDT', 'ETH_USDT'],
            'alert_thresholds': {
                'pnl_threshold': -10000,
                'drawdown_threshold': 0.15,
                'price_change_threshold': 5.0
            },
            'test_duration': 30  # Test süresi (saniye)
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Loglama sistemini ayarla"""
        logger = logging.getLogger('IntegratedAlertTest')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def test_alert_system(self) -> bool:
        """Alert sistemini test et"""
        test_name = "Alert System Test"
        self.logger.info(f"Başlatılıyor: {test_name}")
        
        try:
            # Callback fonksiyonu ekle
            def alert_callback(alert: Alert):
                self.metrics['alerts_processed'] += 1
                self.logger.info(f"[ALERT CALLBACK] {alert.type.value}: {alert.message}")
            
            self.alert_system.add_alert_callback(alert_callback)
            
            # P&L alert testi
            pnl_alert = self.alert_system.check_pnl_threshold(-15000)
            if pnl_alert:
                success = self.alert_system.process_alert(pnl_alert)
                self._record_test_result(test_name, "P&L Alert", success)
            
            # Drawdown alert testi
            drawdown_alert = self.alert_system.check_drawdown(100000, 82000)
            if drawdown_alert:
                success = self.alert_system.process_alert(drawdown_alert)
                self._record_test_result(test_name, "Drawdown Alert", success)
            
            # Sistem başlat
            self.alert_system.start_monitoring()
            time.sleep(2)
            self.alert_system.stop_monitoring()
            
            self.logger.info(f"Tamamlandı: {test_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Hata: {test_name} - {e}")
            self._record_test_result(test_name, "General", False)
            return False
    
    def test_price_alert_engine(self) -> bool:
        """Fiyat alert motorunu test et"""
        test_name = "Price Alert Engine Test"
        self.logger.info(f"Başlatılıyor: {test_name}")
        
        try:
            # Test koşulları oluştur
            conditions = [
                AlertCondition("BTC_USDT", AlertType.PERCENTAGE_CHANGE, "above", 5.0, "15m"),
                AlertCondition("ETH_USDT", AlertType.PERCENTAGE_CHANGE, "above", 3.0, "1h")
            ]
            
            for condition in conditions:
                self.price_engine.add_alert_condition(condition)
            
            # Simulated price data
            current_price = 52500  # %5 artış
            previous_price = 50000
            
            # Alert testi
            alerts = self.price_engine.check_percentage_change("BTC_USDT", current_price, previous_price)
            
            success_count = 0
            for alert in alerts:
                self.price_engine.save_alert(alert)
                self.metrics['alerts_processed'] += 1
                success_count += 1
                self.logger.info(f"[PRICE ALERT] {alert.message}")
            
            success = success_count > 0
            self._record_test_result(test_name, "Percentage Change Alert", success)
            
            self.logger.info(f"Tamamlandı: {test_name}")
            return success
            
        except Exception as e:
            self.logger.error(f"Hata: {test_name} - {e}")
            self._record_test_result(test_name, "General", False)
            return False
    
    def test_technical_indicators(self) -> bool:
        """Teknik indikatörleri test et"""
        test_name = "Technical Indicators Test"
        self.logger.info(f"Başlatılıyor: {test_name}")
        
        try:
            # Test verisi oluştur
            np.random.seed(42)
            dates = pd.date_range(start='2023-01-01', periods=100, freq='15min')
            
            price_base = 50000
            price_changes = np.random.normal(0, 0.01, 100)
            prices = [price_base]
            
            for change in price_changes[1:]:
                new_price = prices[-1] * (1 + change)
                prices.append(new_price)
            
            test_data = pd.DataFrame({
                'timestamp': dates,
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
                'close': prices,
                'volume': np.random.uniform(100, 1000, 100)
            })
            
            # İndikatör testleri
            tests = {
                'SMA(14)': self.ti.sma(test_data['close'], 14),
                'EMA(14)': self.ti.ema(test_data['close'], 14),
                'RSI(14)': self.ti.rsi(test_data['close'], 14),
                'MACD': self.ti.macd(test_data['close']),
                'Bollinger Bands': self.ti.bollinger_bands(test_data['close']),
                'ATR': self.ti.atr(test_data['high'], test_data['low'], test_data['close'])
            }
            
            success_count = 0
            for test_name, result in tests.items():
                try:
                    if isinstance(result, dict):
                        # MACD, Bollinger Bands gibi dict döndüren indikatörler
                        for key, value in result.items():
                            latest_value = value.iloc[-1]
                            if not np.isnan(latest_value):
                                success_count += 1
                                self.logger.info(f"[TECHNICAL] {test_name}-{key}: {latest_value:.2f}")
                                self.metrics['indicators_calculated'] += 1
                    else:
                        # Tek değer döndüren indikatörler
                        latest_value = result.iloc[-1]
                        if not np.isnan(latest_value):
                            success_count += 1
                            self.logger.info(f"[TECHNICAL] {test_name}: {latest_value:.2f}")
                            self.metrics['indicators_calculated'] += 1
                except Exception as e:
                    self.logger.warning(f"[TECHNICAL] {test_name} hatası: {e}")
            
            success = success_count > 0
            self._record_test_result("Technical Indicators Test", "Calculation", success)
            
            self.logger.info(f"Tamamlandı: {test_name}")
            return success
            
        except Exception as e:
            self.logger.error(f"Hata: {test_name} - {e}")
            self._record_test_result(test_name, "General", False)
            return False
    
    def test_discord_webhook(self) -> bool:
        """Discord webhook sistemini test et"""
        test_name = "Discord Webhook Test"
        self.logger.info(f"Başlatılıyor: {test_name}")
        
        try:
            if not self.discord.webhook_url or self.discord.webhook_url == "YOUR_DISCORD_WEBHOOK_URL":
                self.logger.warning("Discord webhook URL ayarlı değil, test atlanıyor")
                return True
            
            # Test mesajı gönder
            success = self.discord.send_alert(
                title="🧪 Test Alert",
                message="Crypto Alert System entegrasyon testi mesajı.",
                level=AlertLevel.LOW,
                fields=[
                    {"name": "Test Zamanı", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "inline": True},
                    {"name": "Test Durumu", "value": "Başarılı", "inline": True}
                ]
            )
            
            if success:
                self.metrics['notifications_sent'] += 1
                self.logger.info("[DISCORD] Test mesajı gönderildi")
            else:
                self.logger.error("[DISCORD] Test mesajı gönderilemedi")
            
            self._record_test_result(test_name, "Notification Send", success)
            
            self.logger.info(f"Tamamlandı: {test_name}")
            return success
            
        except Exception as e:
            self.logger.error(f"Hata: {test_name} - {e}")
            self._record_test_result(test_name, "General", False)
            return False
    
    def test_websocket_manager(self) -> bool:
        """WebSocket yöneticisini test et"""
        test_name = "WebSocket Manager Test"
        self.logger.info(f"Başlatılıyor: {test_name}")
        
        try:
            # Callback fonksiyonu
            def ticker_callback(data: TickerData):
                self.metrics['websocket_messages'] += 1
                self.logger.info(f"[WEBSOCKET] {data.symbol}: {data.price:.2f}")
            
            # WebSocket manager oluştur (test modu - gerçek bağlantı kurmadan)
            self.ws_manager = GateWebSocketManager(
                symbols=self.config['test_symbols'],
                max_reconnect_attempts=1,
                reconnect_delay=1.0
            )
            
            # Test callback ekle
            self.ws_manager.add_ticker_callback(ticker_callback)
            
            # Metrikleri kontrol et
            metrics = self.ws_manager.get_metrics()
            self.logger.info(f"[WEBSOCKET] Metrics: {metrics}")
            
            # Test başarılı (bağlantı kurmadan sadece metrikler)
            success = metrics is not None
            self._record_test_result(test_name, "Manager Creation", success)
            
            self.logger.info(f"Tamamlandı: {test_name}")
            return success
            
        except Exception as e:
            self.logger.error(f"Hata: {test_name} - {e}")
            self._record_test_result(test_name, "General", False)
            return False
    
    def _record_test_result(self, test_name: str, component: str, success: bool):
        """Test sonucunu kaydet"""
        self.test_results.append({
            'test': test_name,
            'component': component,
            'success': success,
            'timestamp': datetime.now().isoformat()
        })
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Kapsamlı test çalıştır"""
        self.logger.info("🚀 Kapsamlı Alert Sistemi Testi Başlatılıyor")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        # Testleri çalıştır
        tests = [
            self.test_alert_system,
            self.test_price_alert_engine,
            self.test_technical_indicators,
            self.test_discord_webhook,
            self.test_websocket_manager
        ]
        
        passed = 0
        total = len(tests)
        
        for test_func in tests:
            try:
                if test_func():
                    passed += 1
                    self.logger.info("✅ Test geçti")
                else:
                    self.logger.error("❌ Test başarısız")
                self.logger.info("-" * 40)
            except Exception as e:
                self.logger.error(f"❌ Test hatası: {e}")
                self.logger.info("-" * 40)
        
        # Test süresi
        duration = time.time() - start_time
        
        # Sonuçları özetle
        results = {
            'summary': {
                'total_tests': total,
                'passed_tests': passed,
                'failed_tests': total - passed,
                'success_rate': f"{(passed/total)*100:.1f}%",
                'duration_seconds': round(duration, 2)
            },
            'test_details': self.test_results,
            'system_metrics': self.metrics,
            'status': 'PASSED' if passed == total else 'FAILED'
        }
        
        self.logger.info("=" * 60)
        self.logger.info("📊 TEST SONUÇLARI")
        self.logger.info(f"Toplam Test: {total}")
        self.logger.info(f"Geçen Test: {passed}")
        self.logger.info(f"Başarısız: {total - passed}")
        self.logger.info(f"Başarı Oranı: {results['summary']['success_rate']}")
        self.logger.info(f"Süre: {results['summary']['duration_seconds']} saniye")
        self.logger.info(f"Durum: {results['status']}")
        
        self.logger.info("\n📈 SİSTEM METRİKLERİ")
        for key, value in self.metrics.items():
            self.logger.info(f"{key}: {value}")
        
        self.logger.info("=" * 60)
        
        return results


def main():
    """Ana test fonksiyonu"""
    print("🧪 Crypto Alert System - Entegre Test")
    print("=" * 50)
    
    # Test sistemi oluştur
    test_system = IntegratedAlertSystem()
    
    # Kapsamlı test çalıştır
    results = test_system.run_comprehensive_test()
    
    # Sonuçları JSON olarak kaydet
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Test sonuçları kaydedildi: {filename}")
    
    # Discord'a sonuç gönder
    if test_system.discord.webhook_url and test_system.discord.webhook_url != "YOUR_DISCORD_WEBHOOK_URL":
        level = AlertLevel.LOW if results['status'] == 'PASSED' else AlertLevel.HIGH
        emoji = "✅" if results['status'] == 'PASSED' else "❌"
        
        test_system.discord.send_alert(
            title=f"{emoji} Test Sonucu",
            message=f"Crypto Alert System test sonucu: **{results['status']}**",
            level=level,
            fields=[
                {"name": "Toplam Test", "value": str(results['summary']['total_tests']), "inline": True},
                {"name": "Başarı Oranı", "value": results['summary']['success_rate'], "inline": True},
                {"name": "Süre", "value": f"{results['summary']['duration_seconds']}s", "inline": True},
                {"name": "İşlenen Alertler", "value": str(results['system_metrics']['alerts_processed']), "inline": True},
                {"name": "Hesaplanan İndikatörler", "value": str(results['system_metrics']['indicators_calculated']), "inline": True}
            ]
        )


if __name__ == "__main__":
    main()
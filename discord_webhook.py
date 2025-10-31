"""
Discord Webhook Notification System
==================================

Discord webhook entegrasyonu için bildirim sistemi.
Rich embed formatı, file attachments, rate limiting destekler.

Kurulum:
    pip install requests

Kullanım:
    discord = DiscordWebhook(webhook_url="YOUR_WEBHOOK_URL")
    discord.send_alert("Alert title", "Alert message", level="high")
"""

import requests
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import time


class AlertLevel(Enum):
    """Alert seviyeleri"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DiscordEmbed:
    """Discord embed yapısı"""
    title: str
    description: str
    color: int
    timestamp: Optional[str] = None
    footer: Optional[Dict[str, str]] = None
    fields: Optional[List[Dict[str, Union[str, bool]]]] = None
    image: Optional[str] = None
    thumbnail: Optional[str] = None


class DiscordWebhook:
    """Discord webhook bildirim sistemi"""
    
    def __init__(self, webhook_url: str, username: str = "CryptoAlert", avatar_url: str = None):
        """
        Discord webhook'u başlat
        
        Args:
            webhook_url: Discord webhook URL'i
            username: Bot username
            avatar_url: Bot avatar URL'i
        """
        self.webhook_url = webhook_url
        self.username = username
        self.avatar_url = avatar_url
        self.logger = self._setup_logging()
        
        # Rate limiting
        self.last_sent_time = 0
        self.min_delay = 1.0  # Minimum 1 saniye bekleme
        
        # Color mapping
        self.color_map = {
            AlertLevel.LOW: 0x36a64f,      # Yeşil
            AlertLevel.MEDIUM: 0xffaa00,   # Sarı  
            AlertLevel.HIGH: 0xff6600,     # Turuncu
            AlertLevel.CRITICAL: 0xff0000  # Kırmızı
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Loglama sistemini ayarla"""
        logger = logging.getLogger('DiscordWebhook')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _rate_limit_check(self):
        """Rate limit kontrolü"""
        current_time = time.time()
        if current_time - self.last_sent_time < self.min_delay:
            time.sleep(self.min_delay - (current_time - self.last_sent_time))
        self.last_sent_time = time.time()
    
    def send_alert(self, title: str, message: str, level: Union[AlertLevel, str] = AlertLevel.MEDIUM,
                  fields: List[Dict[str, Union[str, bool]]] = None, 
                  image_url: str = None, thumbnail_url: str = None) -> bool:
        """
        Alert gönder
        
        Args:
            title: Alert başlığı
            message: Alert mesajı
            level: Alert seviyesi
            fields: Ek alanlar
            image_url: Ana resim URL'i
            thumbnail_url: Thumbnail URL'i
            
        Returns:
            bool: Başarılı gönderim durumu
        """
        try:
            # Level conversion
            if isinstance(level, str):
                level = AlertLevel(level.lower())
            
            self._rate_limit_check()
            
            # Embed oluştur
            embed = DiscordEmbed(
                title=title,
                description=message,
                color=self.color_map.get(level, self.color_map[AlertLevel.MEDIUM]),
                timestamp=datetime.now().isoformat(),
                fields=fields or [],
                image=image_url,
                thumbnail=thumbnail_url
            )
            
            return self._send_embed(embed)
            
        except Exception as e:
            self.logger.error(f"Alert gönderme hatası: {e}")
            return False
    
    def _send_embed(self, embed: DiscordEmbed) -> bool:
        """Embed gönder"""
        try:
            payload = {
                "username": self.username,
                "avatar_url": self.avatar_url,
                "embeds": [{
                    "title": embed.title,
                    "description": embed.description,
                    "color": embed.color,
                    "timestamp": embed.timestamp,
                    "footer": embed.footer,
                    "fields": embed.fields,
                    "image": {"url": embed.image} if embed.image else None,
                    "thumbnail": {"url": embed.thumbnail} if embed.thumbnail else None
                }]
            }
            
            # None değerleri temizle
            payload["embeds"][0] = {k: v for k, v in payload["embeds"][0].items() if v is not None}
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 204:
                self.logger.info(f"Discord alert gönderildi: {embed.title}")
                return True
            else:
                self.logger.error(f"Discord webhook hatası: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Embed gönderme hatası: {e}")
            return False
    
    def send_simple_message(self, content: str, username: str = None) -> bool:
        """Basit mesaj gönder"""
        try:
            self._rate_limit_check()
            
            payload = {
                "content": content,
                "username": username or self.username
            }
            
            if self.avatar_url:
                payload["avatar_url"] = self.avatar_url
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            return response.status_code == 204
            
        except Exception as e:
            self.logger.error(f"Simple message gönderme hatası: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Bağlantıyı test et"""
        return self.send_simple_message("🔔 Test mesajı - Crypto Alert Sistemi çalışıyor!")
    
    def send_system_status(self, status: str, details: str = None):
        """Sistem durumu bildirimi"""
        embed = DiscordEmbed(
            title="📊 Sistem Durumu",
            description=status,
            color=0x0066cc,  # Mavi
            timestamp=datetime.now().isoformat(),
            footer={"text": "Crypto Alert System"}
        )
        
        if details:
            embed.fields = [{"name": "Detaylar", "value": details, "inline": False}]
        
        return self._send_embed(embed)


# Kullanım örneği ve test
def test_discord_webhook():
    """Test fonksiyonu"""
    print("=== Discord Webhook Test ===")
    
    # Test webhook URL (gerçek URL ile değiştirin)
    webhook_url = "YOUR_DISCORD_WEBHOOK_URL"
    
    if webhook_url == "YOUR_DISCORD_WEBHOOK_URL":
        print("Lütfen geçerli Discord webhook URL'i ayarlayın!")
        return
    
    discord = DiscordWebhook(webhook_url)
    
    # Bağlantı testi
    print("Bağlantı testi...")
    if discord.test_connection():
        print("✅ Bağlantı başarılı!")
    else:
        print("❌ Bağlantı başarısız!")
        return
    
    # Alert testleri
    print("\n--- Alert Testleri ---")
    
    # Düşük seviye alert
    discord.send_alert(
        "ℹ️ Bilgilendirme",
        "Sistem normal şekilde çalışıyor.",
        level=AlertLevel.LOW,
        fields=[
            {"name": "Servis", "value": "WebSocket Manager", "inline": True},
            {"name": "Durum", "value": "Aktif", "inline": True}
        ]
    )
    
    # Yüksek seviye alert
    discord.send_alert(
        "🚨 Yüksek Öncelik Alert",
        "P&L eşiği aşıldı! Portföy değeri hedeflenen seviyenin altına düştü.",
        level=AlertLevel.HIGH,
        fields=[
            {"name": "Mevcut P&L", "value": "-12,500 USDT", "inline": True},
            {"name": "Eşik", "value": "-10,000 USDT", "inline": True},
            {"name": "Durum", "value": "Kritik", "inline": False}
        ]
    )
    
    # Kritik seviye alert
    discord.send_alert(
        "🔥 KRİTİK UYARI",
        "Maksimum drawdown limiti aşıldı! Acil müdahale gerekli.",
        level=AlertLevel.CRITICAL,
        fields=[
            {"name": "Mevcut Drawdown", "value": "18.5%", "inline": True},
            {"name": "Limit", "value": "15%", "inline": True},
            {"name": "Aksiyon", "value": "Pozisyon kapatma öneriliyor", "inline": False}
        ]
    )
    
    # Sistem durumu
    print("\n--- Sistem Durumu ---")
    discord.send_system_status(
        "Tüm servisler aktif",
        "WebSocket bağlantıları stabil\nAlert sistemi çalışıyor\nTeknik indikatörler güncel"
    )
    
    print("\nTest tamamlandı!")


if __name__ == "__main__":
    test_discord_webhook()
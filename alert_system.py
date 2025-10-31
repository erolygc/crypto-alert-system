"""
Akıllı Alert Sistemi
P&L threshold alerts, risk limit breaches, drawdown alerts, performance degradation warnings
Email/Slack/Discord notifications, alert escalation rules, alert history tracking
"""

import json
import logging
import smtplib
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sqlite3
import os
import threading
import time

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alert_system.log'),
        logging.StreamHandler()
    ]
)

class AlertSeverity(Enum):
    """Uyarı şiddet seviyeleri"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    """Alert türleri"""
    PNL_THRESHOLD = "pnl_threshold"
    RISK_LIMIT_BREACH = "risk_limit_breach"
    DRAWDOWN = "drawdown"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SYSTEM_ERROR = "system_error"

@dataclass
class Alert:
    """Alert veri yapısı"""
    id: str
    type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: datetime
    value: float
    threshold: float
    metadata: Dict[str, Any]
    acknowledged: bool = False
    escalation_level: int = 0

@dataclass
class NotificationConfig:
    """Bildirim yapılandırması"""
    email_enabled: bool = False
    slack_enabled: bool = False
    discord_enabled: bool = False
    
    # Email ayarları
    smtp_server: str = ""
    smtp_port: int = 587
    email_user: str = ""
    email_password: str = ""
    email_recipients: List[str] = None
    
    # Slack ayarları
    slack_webhook_url: str = ""
    slack_channel: str = ""
    
    # Discord ayarları
    discord_webhook_url: str = ""
    discord_username: str = "AlertBot"

@dataclass
class Thresholds:
    """Eşik değerler"""
    pnl_threshold: float = -10000.0  # Kar/Zarar eşiği
    drawdown_threshold: float = 0.15  # Max drawdown %15
    performance_degradation_threshold: float = 0.10  # %10 performans düşüşü
    risk_limit: float = 0.05  # %5 risk limiti

class AlertHistory:
    """Alert geçmişi yönetimi"""
    
    def __init__(self, db_path: str = "alert_history.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """SQLite veritabanını başlat"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                value REAL NOT NULL,
                threshold REAL NOT NULL,
                metadata TEXT,
                acknowledged INTEGER DEFAULT 0,
                escalation_level INTEGER DEFAULT 0,
                resolved_at TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_alert(self, alert: Alert):
        """Alert'i veritabanına kaydet"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO alerts 
            (id, type, severity, message, timestamp, value, threshold, metadata, acknowledged, escalation_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert.id, alert.type.value, alert.severity.value,
            alert.message, alert.timestamp.isoformat(),
            alert.value, alert.threshold,
            json.dumps(alert.metadata), alert.acknowledged, alert.escalation_level
        ))
        
        conn.commit()
        conn.close()

class AlertSystem:
    """Ana alert sistemi"""
    
    def __init__(self, config_path: str = "alert_config.json"):
        self.config_path = config_path
        self.thresholds = Thresholds()
        self.notification_config = NotificationConfig()
        self.load_config()
        
        # Bileşenler
        self.history = AlertHistory()
        
        # Performans takibi
        self.performance_window = deque(maxlen=100)
        
        # Aktif alertler
        self.active_alerts = {}
        self.alert_callbacks = []
        
        # Thread'i
        self.running = False
    
    def load_config(self):
        """Yapılandırma dosyasını yükle"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
        except Exception as e:
            logging.warning(f"Yapılandırma yükleme hatası: {e}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Alert callback'i ekle"""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self):
        """Monitoring'i başlat"""
        self.running = True
        logging.info("Alert sistemi başlatıldı")
    
    def stop_monitoring(self):
        """Monitoring'i durdur"""
        self.running = False
        logging.info("Alert sistemi durduruldu")
    
    def check_pnl_threshold(self, current_pnl: float) -> Optional[Alert]:
        """P&L eşiğini kontrol et"""
        threshold = self.thresholds.pnl_threshold
        
        if current_pnl <= threshold:
            alert = Alert(
                id=f"pnl_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type=AlertType.PNL_THRESHOLD,
                severity=AlertSeverity.CRITICAL if current_pnl < threshold * 0.5 else AlertSeverity.HIGH,
                message=f"P&L eşiği aşıldı! Mevcut: {current_pnl}, Eşik: {threshold}",
                timestamp=datetime.now(),
                value=current_pnl,
                threshold=threshold,
                metadata={"pnl": current_pnl, "threshold": threshold}
            )
            
            return alert
        
        return None

def main():
    """Ana test fonksiyonu"""
    print("=== Akıllı Alert Sistemi Test ===")
    
    # Alert sistemini başlat
    alert_system = AlertSystem()
    
    # Monitoring'i başlat
    alert_system.start_monitoring()
    
    # Test alertleri
    print("\n--- Test Alertleri ---")
    
    # P&L eşiği testi
    pnl_alert = alert_system.check_pnl_threshold(-15000)
    if pnl_alert:
        print(f"Alert oluşturuldu: {pnl_alert.message}")

if __name__ == "__main__":
    main()
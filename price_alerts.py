"""
Price Movement Alert Engine
===========================

Kapsamlı fiyat hareketi uyarı sistemi.

Özellikler:
- Percentage change alerts (1%, 2%, 5%, 10%)
- Support/Resistance level alerts  
- Breakout detection
- Gap detection
- Momentum alerts (speed, acceleration)
- Multi-timeframe analysis
- Alert conditions configuration
"""

import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd


class AlertType(Enum):
    """Alert türleri"""
    PERCENTAGE_CHANGE = "percentage_change"
    SUPPORT_RESISTANCE = "support_resistance"
    BREAKOUT = "breakout"
    GAP = "gap"
    MOMENTUM = "momentum"
    MULTI_TIMEFRAME = "multi_timeframe"


class AlertSeverity(Enum):
    """Alert şiddet seviyeleri"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AlertCondition:
    """Alert koşul yapısı"""
    symbol: str
    alert_type: AlertType
    condition: str  # 'above', 'below', 'crosses_above', 'crosses_below'
    value: float
    timeframe: str  # '1m', '5m', '15m', '1h', '4h', '1d'
    enabled: bool = True
    repeat: bool = False  # Allow repeating alerts
    cooldown_minutes: int = 5


@dataclass
class PriceAlert:
    """Price Alert veri yapısı"""
    id: str
    symbol: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: datetime
    current_price: float
    trigger_value: float
    percentage_change: float
    metadata: Dict[str, Any]
    acknowledged: bool = False


class TechnicalLevels:
    """Teknik seviye analizi"""
    
    def calculate_support_resistance(self, price_data: pd.DataFrame, 
                                   window: int = 20) -> Dict[str, float]:
        """Support ve resistance seviyelerini hesapla"""
        if len(price_data) < window:
            return {}
        
        # Son 'window' kadar veri al
        recent_data = price_data.tail(window)
        
        # Support seviyesi (local minimum)
        support_level = recent_data['low'].min()
        
        # Resistance seviyesi (local maximum)
        resistance_level = recent_data['high'].max()
        
        return {
            'support': support_level,
            'resistance': resistance_level
        }
    
    def detect_breakout(self, price: float, levels: Dict[str, float], 
                       tolerance: float = 0.001) -> Optional[str]:
        """Breakout tespiti"""
        if not levels:
            return None
        
        resistance = levels.get('resistance', 0)
        support = levels.get('support', float('inf'))
        
        if price > resistance * (1 + tolerance):
            return 'breakout_up'
        elif price < support * (1 - tolerance):
            return 'breakout_down'
        
        return None


class PriceAlertEngine:
    """Ana fiyat alert motoru"""
    
    def __init__(self, db_path: str = "price_alerts.db"):
        self.db_path = db_path
        self.conditions = []
        self.alert_history = []
        self.technical_levels = TechnicalLevels()
        self.init_database()
    
    def init_database(self):
        """Veritabanını başlat"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_alerts (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                current_price REAL NOT NULL,
                trigger_value REAL NOT NULL,
                percentage_change REAL NOT NULL,
                metadata TEXT,
                acknowledged INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_alert_condition(self, condition: AlertCondition) -> int:
        """Alert koşulu ekle"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alert_conditions 
            (symbol, alert_type, condition, value, timeframe, enabled, repeat, cooldown_minutes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            condition.symbol, condition.alert_type.value, condition.condition,
            condition.value, condition.timeframe, condition.enabled,
            condition.repeat, condition.cooldown_minutes, datetime.now().isoformat()
        ))
        
        conn.commit()
        condition_id = cursor.lastrowid
        conn.close()
        
        self.conditions.append(condition)
        return condition_id
    
    def check_percentage_change(self, symbol: str, current_price: float, 
                              previous_price: float) -> List[PriceAlert]:
        """Yüzdelik değişim kontrolü"""
        alerts = []
        
        if previous_price <= 0:
            return alerts
        
        change_percentage = ((current_price - previous_price) / previous_price) * 100
        
        # İlgili koşulları bul
        conditions = [c for c in self.conditions 
                     if c.symbol == symbol and c.alert_type == AlertType.PERCENTAGE_CHANGE]
        
        for condition in conditions:
            trigger_value = condition.value
            
            if condition.condition == 'above' and abs(change_percentage) >= trigger_value:
                alert = PriceAlert(
                    id=f"pct_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    symbol=symbol,
                    alert_type=AlertType.PERCENTAGE_CHANGE,
                    severity=self._determine_severity(abs(change_percentage), trigger_value),
                    message=f"{symbol} yüzdelik değişim: {change_percentage:.2f}% (Eşik: {trigger_value}%)",
                    timestamp=datetime.now(),
                    current_price=current_price,
                    trigger_value=trigger_value,
                    percentage_change=change_percentage,
                    metadata={'previous_price': previous_price}
                )
                alerts.append(alert)
        
        return alerts
    
    def _determine_severity(self, actual_value: float, threshold: float) -> AlertSeverity:
        """Alert şiddetini belirle"""
        ratio = actual_value / threshold
        
        if ratio >= 3.0:
            return AlertSeverity.CRITICAL
        elif ratio >= 2.0:
            return AlertSeverity.HIGH
        elif ratio >= 1.5:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    def save_alert(self, alert: PriceAlert):
        """Alert'i kaydet"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO price_alerts 
            (id, symbol, alert_type, severity, message, timestamp, current_price, trigger_value, percentage_change, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert.id, alert.symbol, alert.alert_type.value, alert.severity.value,
            alert.message, alert.timestamp.isoformat(),
            alert.current_price, alert.trigger_value, alert.percentage_change,
            json.dumps(alert.metadata)
        ))
        
        conn.commit()
        conn.close()
        
        self.alert_history.append(alert)


def main():
    """Test fonksiyonu"""
    print("=== Price Alert Engine Test ===")
    
    # Engine başlat
    engine = PriceAlertEngine()
    
    # Örnek koşul oluştur
    condition = AlertCondition("BTCUSDT", AlertType.PERCENTAGE_CHANGE, "above", 5.0, "15m")
    engine.add_alert_condition(condition)
    
    # Test verileri
    current_price = 52500  # %5 artış
    previous_price = 50000
    
    # Alert testi
    alerts = engine.check_percentage_change("BTCUSDT", current_price, previous_price)
    for alert in alerts:
        engine.save_alert(alert)
        print(f"Alert: {alert.message}")


if __name__ == "__main__":
    main()
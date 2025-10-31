#!/usr/bin/env python3
"""
Gate.io WebSocket Connection Manager
===================================

Bu modül Gate.io WebSocket API'si için robust bir bağlantı yöneticisi sağlar.
Özellikler:
- Otomatik yeniden bağlanma
- Çoklu sembol abonelik
- Gerçek zamanlı ticker verisi akışı
- Bağlantı sağlık izleme
- Hata yönetimi ve loglama

Kullanım:
    ws_manager = GateWebSocketManager()
    ws_manager.start()
    
    # Ticker verilerini dinle
    def on_ticker_data(data):
        print(f"Ticker: {data}")
    
    ws_manager.add_ticker_callback(on_ticker_data)

Yazar: AI Agent
Tarih: 2025-11-01
"""

import asyncio
import json
import logging
import time
import websockets
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import threading
from queue import Queue, Empty
import traceback


class ConnectionState(Enum):
    """WebSocket bağlantı durumları"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class TickerData:
    """Ticker verisi yapısı"""
    symbol: str
    price: float
    change_24h: float
    change_percent_24h: float
    high_24h: float
    low_24h: float
    volume_24h: float
    volume_quote_24h: float
    timestamp: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Dict'e dönüştür"""
        return asdict(self)


class GateWebSocketManager:
    """
    Gate.io WebSocket Bağlantı Yöneticisi
    
    Bu sınıf Gate.io WebSocket API'si ile güvenilir bir bağlantı sağlar
    ve çoklu sembol ticker verilerini gerçek zamanlı olarak alır.
    """
    
    def __init__(self, 
                 symbols: List[str] = None,
                 max_reconnect_attempts: int = 5,
                 reconnect_delay: float = 5.0,
                 ping_interval: float = 20.0,
                 timeout: float = 30.0):
        """
        Initialize the WebSocket manager
        
        Args:
            symbols: Ticker abonelik sembolleri (default: ['BTC_USDT', 'ETH_USDT', 'SOL_USDT'])
            max_reconnect_attempts: Maksimum yeniden bağlanma denemesi
            reconnect_delay: Yeniden bağlanma gecikmesi (saniye)
            ping_interval: Ping gönderme aralığı (saniye)
            timeout: WebSocket timeout süresi (saniye)
        """
        self.symbols = symbols or ['BTC_USDT', 'ETH_USDT', 'SOL_USDT']
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.ping_interval = ping_interval
        self.timeout = timeout
        
        # WebSocket ayarları
        self.base_url = "wss://api.gateio.ws/ws/v4/"
        self.connection_state = ConnectionState.DISCONNECTED
        self.ws = None
        self.reconnect_attempts = 0
        
        # Callback fonksiyonları
        self.ticker_callbacks: List[Callable[[TickerData], None]] = []
        self.status_callbacks: List[Callable[[ConnectionState], None]] = []
        
        # Threading ve queue
        self.loop = None
        self.running = False
        self.data_queue = Queue()
        self.heartbeat_thread = None
        
        # Logging
        self.logger = self._setup_logging()
        
        # Metrics
        self.metrics = {
            'connection_attempts': 0,
            'reconnections': 0,
            'messages_received': 0,
            'errors': 0,
            'last_connected': None,
            'uptime': 0
        }
        
        self.logger.info(f"GateWebSocketManager başlatıldı. Semboller: {self.symbols}")
    
    def _setup_logging(self) -> logging.Logger:
        """Loglama sistemini ayarla"""
        logger = logging.getLogger('GateWebSocketManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # File handler
            file_handler = logging.FileHandler('gateio_websocket.log')
            file_handler.setLevel(logging.DEBUG)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        
        return logger
    
    def add_ticker_callback(self, callback: Callable[[TickerData], None]):
        """Ticker veri callback fonksiyonu ekle"""
        if callback not in self.ticker_callbacks:
            self.ticker_callbacks.append(callback)
            self.logger.info(f"Yeni ticker callback eklendi: {callback.__name__}")
    
    def remove_ticker_callback(self, callback: Callable[[TickerData], None]):
        """Ticker veri callback fonksiyonunu kaldır"""
        if callback in self.ticker_callbacks:
            self.ticker_callbacks.remove(callback)
            self.logger.info(f"Ticker callback kaldırıldı: {callback.__name__}")
    
    def add_status_callback(self, callback: Callable[[ConnectionState], None]):
        """Durum değişiklik callback'i ekle"""
        if callback not in self.status_callbacks:
            self.status_callbacks.append(callback)
    
    def remove_status_callback(self, callback: Callable[[ConnectionState], None]):
        """Durum değişiklik callback'i kaldır"""
        if callback in self.status_callbacks:
            self.status_callbacks.remove(callback)
    
    def _notify_status_change(self, state: ConnectionState):
        """Durum değişikliklerini bildir"""
        old_state = self.connection_state
        self.connection_state = state
        
        self.logger.info(f"Bağlantı durumu değişti: {old_state.value} -> {state.value}")
        
        for callback in self.status_callbacks:
            try:
                callback(state)
            except Exception as e:
                self.logger.error(f"Status callback hatası: {e}")
                self.logger.error(traceback.format_exc())
    
    def start(self):
        """WebSocket bağlantısını başlat"""
        if self.running:
            self.logger.warning("WebSocket manager zaten çalışıyor")
            return
        
        self.running = True
        self.logger.info("GateWebSocketManager başlatılıyor...")
        
        # Event loop oluştur ve çalıştır
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self._connect_websocket())
        except KeyboardInterrupt:
            self.logger.info("Kullanıcı tarafından durduruldu")
        except Exception as e:
            self.logger.error(f"WebSocket manager hatası: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            self.stop()
    
    def stop(self):
        """WebSocket bağlantısını durdur"""
        if not self.running:
            return
        
        self.logger.info("GateWebSocketManager durduruluyor...")
        
        self.running = False
        self._notify_status_change(ConnectionState.DISCONNECTED)
        
        # Event loop'u kapat
        if self.loop:
            try:
                self.loop.close()
            except Exception as e:
                self.logger.error(f"Event loop kapatma hatası: {e}")
        
        self.logger.info("GateWebSocketManager durduruldu")
    
    def get_metrics(self) -> Dict[str, Any]:
        """WebSocket metriklerini al"""
        return {
            **self.metrics,
            'current_state': self.connection_state.value,
            'reconnect_attempts': self.reconnect_attempts,
            'is_running': self.running,
            'subscribed_symbols': self.symbols,
            'queue_size': self.data_queue.qsize()
        }
    
    def get_connection_state(self) -> ConnectionState:
        """Mevcut bağlantı durumunu al"""
        return self.connection_state
    
    def is_connected(self) -> bool:
        """Bağlantı durumunu kontrol et"""
        return self.connection_state == ConnectionState.CONNECTED
    
    def get_ticker_data(self, timeout: float = 1.0) -> Optional[TickerData]:
        """Queue'dan ticker verisi al (non-blocking)"""
        try:
            return self.data_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def __enter__(self):
        """Context manager girişi"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager çıkışı"""
        self.stop()


# Test ve örnek kullanım
def example_callback(ticker_data: TickerData):
    """Örnek ticker callback fonksiyonu"""
    print(f"[{ticker_data.symbol}] Fiyat: {ticker_data.price:.2f} USDT "
          f"(24h Değişim: {ticker_data.change_percent_24h:.2f}%) "
          f"Vol: {ticker_data.volume_24h:.2f}")


def status_callback(state: ConnectionState):
    """Örnek durum callback fonksiyonu"""
    print(f"Bağlantı durumu: {state.value}")


if __name__ == "__main__":
    print("Gate.io WebSocket Test Başlatılıyor...")
    
    # WebSocket manager oluştur
    ws_manager = GateWebSocketManager(
        symbols=['BTC_USDT', 'ETH_USDT', 'SOL_USDT'],
        max_reconnect_attempts=3,
        reconnect_delay=5.0
    )
    
    # Callback'leri ekle
    ws_manager.add_ticker_callback(example_callback)
    ws_manager.add_status_callback(status_callback)
    
    print("WebSocket manager oluşturuldu.")
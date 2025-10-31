#!/usr/bin/env python3
"""
Telegram Bot Notification System

Kapsamlı Telegram bot bildirim sistemi. Aşağıdaki özellikleri destekler:
- Bot token configuration
- Chat ID management
- Rich message formatting (Markdown)
- Inline keyboards
- Media support (charts, images)
- Bot commands (/start, /status, /config)
- Message queue management
- Retry logic for failed messages
- Rate limiting compliance

Kurulum:
    pip install python-telegram-bot pillow matplotlib seaborn requests

Kullanım:
    python telegram_notifier.py

Author: Trading Bot System
Date: 2025-11-01
"""

import asyncio
import logging
import json
import time
import sqlite3
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from collections import deque
import threading
import queue
import re

# Telegram Bot imports
from telegram import (
    Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup,
    ParseMode, InputMediaPhoto, InputMediaDocument, Message
)
from telegram.ext import (
    Updater, CommandHandler, CallbackQueryHandler,
    MessageHandler, Filters, CallbackContext
)
from telegram.error import (
    TelegramError, RetryAfter, NetworkError, TimedOut,
    BadRequest, Unauthorized, ChatMigrated, InvalidToken
)

# Media processing imports
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import io
import base64


@dataclass
class ChatConfig:
    """Chat yapılandırma bilgileri"""
    chat_id: int
    chat_name: str
    is_active: bool = True
    notification_level: str = "info"  # debug, info, warning, error, critical
    last_config_update: str = ""
    created_at: str = ""


@dataclass
class MessageQueueItem:
    """Mesaj kuyruğu öğesi"""
    message_id: str
    chat_id: int
    message_type: str  # text, photo, document, chart
    content: Union[str, bytes]
    parse_mode: Optional[str] = ParseMode.MARKDOWN
    reply_markup: Optional[InlineKeyboardMarkup] = None
    priority: int = 5  # 1-10 arası, 1 en yüksek
    retry_count: int = 0
    max_retries: int = 3
    created_at: str = ""
    scheduled_at: Optional[str] = None
    metadata: Optional[Dict] = None


class TelegramBotConfig:
    """Bot yapılandırma yöneticisi"""
    
    def __init__(self, config_file: str = "bot_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Yapılandırma dosyasını yükle"""
        default_config = {
            "bot_token": "",  # BotFather'dan alınan token
            "api_url": "https://api.telegram.org",
            "rate_limit": {
                "messages_per_second": 30,  # Telegram limiti
                "messages_per_minute": 1000,
                "messages_per_hour": 5000
            },
            "retry_config": {
                "max_retries": 3,
                "retry_delay": 1,  # saniye
                "exponential_backoff": True
            },
            "queue_config": {
                "max_queue_size": 1000,
                "worker_threads": 3,
                "queue_timeout": 300  # 5 dakika
            },
            "media_config": {
                "max_file_size": 50 * 1024 * 1024,  # 50MB
                "supported_formats": ["jpg", "jpeg", "png", "gif", "pdf", "doc", "docx"],
                "chart_dpi": 100,
                "chart_dpi_high": 200
            },
            "logging": {
                "level": "INFO",
                "file": "telegram_bot.log",
                "max_size": 10 * 1024 * 1024,
                "backup_count": 5
            },
            "database": {
                "path": "telegram_bot.db"
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # Default config ile merge
                    for key, value in default_config.items():
                        if key not in loaded_config:
                            loaded_config[key] = value
                    return loaded_config
            except Exception as e:
                logging.error(f"Config dosyası yüklenemedi: {e}")
        
        # Default config'i kaydet
        self._save_config(default_config)
        return default_config
    
    def _save_config(self, config: Dict):
        """Yapılandırma dosyasını kaydet"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Config dosyası kaydedilemedi: {e}")
    
    def update_config(self, key: str, value: Any):
        """Yapılandırma güncelle"""
        if key in self.config:
            self.config[key] = value
            self._save_config(self.config)
        else:
            raise KeyError(f"Bilinmeyen config key: {key}")
    
    def get(self, key: str, default=None):
        """Config değeri al"""
        return self.config.get(key, default)


class DatabaseManager:
    """Veritabanı yöneticisi"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Veritabanını başlat"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Chat konfigürasyonları tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_configs (
                chat_id INTEGER PRIMARY KEY,
                chat_name TEXT NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                notification_level TEXT DEFAULT 'info',
                last_config_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Mesaj kuyruğu tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS message_queue (
                message_id TEXT PRIMARY KEY,
                chat_id INTEGER NOT NULL,
                message_type TEXT NOT NULL,
                content TEXT,
                content_blob BLOB,
                parse_mode TEXT,
                reply_markup TEXT,
                priority INTEGER DEFAULT 5,
                retry_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                scheduled_at TIMESTAMP,
                sent_at TIMESTAMP,
                error_message TEXT,
                metadata TEXT,
                FOREIGN KEY (chat_id) REFERENCES chat_configs (chat_id)
            )
        ''')
        
        # Gönderilen mesajlar tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sent_messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                message_type TEXT NOT NULL,
                content TEXT,
                sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                message_uid TEXT,
                metadata TEXT,
                FOREIGN KEY (chat_id) REFERENCES chat_configs (chat_id)
            )
        ''')
        
        # Rate limiting tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rate_limits (
                date_hour TEXT PRIMARY KEY,
                message_count INTEGER DEFAULT 0,
                last_message_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_chat_config(self, config: ChatConfig):
        """Chat konfigürasyonu ekle"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO chat_configs 
            (chat_id, chat_name, is_active, notification_level, 
             last_config_update, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            config.chat_id, config.chat_name, config.is_active,
            config.notification_level, config.last_config_update,
            config.created_at
        ))
        
        conn.commit()
        conn.close()
    
    def get_chat_configs(self) -> List[ChatConfig]:
        """Tüm chat konfigürasyonlarını al"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM chat_configs WHERE is_active = 1')
        rows = cursor.fetchall()
        conn.close()
        
        configs = []
        for row in rows:
            configs.append(ChatConfig(
                chat_id=row[0], chat_name=row[1], is_active=bool(row[2]),
                notification_level=row[3], last_config_update=row[4],
                created_at=row[5]
            ))
        
        return configs
    
    def add_message_to_queue(self, item: MessageQueueItem):
        """Mesajı kuyruğa ekle"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Blob olarak kaydetmek için content'i encode et
        content_blob = None
        if isinstance(item.content, bytes):
            content_blob = item.content
        
        cursor.execute('''
            INSERT INTO message_queue 
            (message_id, chat_id, message_type, content, content_blob,
             parse_mode, reply_markup, priority, retry_count, max_retries,
             created_at, scheduled_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            item.message_id, item.chat_id, item.message_type, item.content,
            content_blob, item.parse_mode, json.dumps(item.reply_markup) if item.reply_markup else None,
            item.priority, item.retry_count, item.max_retries,
            item.created_at, item.scheduled_at, json.dumps(item.metadata) if item.metadata else None
        ))
        
        conn.commit()
        conn.close()
    
    def get_pending_messages(self, limit: int = 50) -> List[MessageQueueItem]:
        """Bekleyen mesajları al"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT message_id, chat_id, message_type, content, content_blob,
                   parse_mode, reply_markup, priority, retry_count, max_retries,
                   created_at, scheduled_at, metadata
            FROM message_queue 
            WHERE status = 'pending' 
            AND (scheduled_at IS NULL OR scheduled_at <= datetime('now'))
            ORDER BY priority ASC, created_at ASC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        items = []
        for row in rows:
            reply_markup = None
            if row[6]:  # reply_markup
                try:
                    reply_markup_data = json.loads(row[6])
                    reply_markup = InlineKeyboardMarkup(reply_markup_data.get('inline_keyboard', []))
                except:
                    pass
            
            items.append(MessageQueueItem(
                message_id=row[0], chat_id=row[1], message_type=row[2],
                content=row[3] if row[4] is None else row[4],
                parse_mode=row[5], reply_markup=reply_markup, priority=row[7],
                retry_count=row[8], max_retries=row[9], created_at=row[10],
                scheduled_at=row[11], metadata=json.loads(row[12]) if row[12] else None
            ))
        
        return items
    
    def update_message_status(self, message_id: str, status: str, 
                            error_message: str = None, sent_at: str = None):
        """Mesaj durumunu güncelle"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE message_queue 
            SET status = ?, error_message = ?, sent_at = ?
            WHERE message_id = ?
        ''', (status, error_message, sent_at, message_id))
        
        conn.commit()
        conn.close()
    
    def increment_retry_count(self, message_id: str):
        """Retry sayısını artır"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE message_queue 
            SET retry_count = retry_count + 1
            WHERE message_id = ?
        ''', (message_id,))
        
        conn.commit()
        conn.close()
    
    def record_rate_limit(self, date_hour: str, increment: int = 1):
        """Rate limit kaydı"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO rate_limits 
            (date_hour, message_count, last_message_at)
            VALUES (?, 
                    COALESCE((SELECT message_count FROM rate_limits WHERE date_hour = ?), 0) + ?,
                    CURRENT_TIMESTAMP)
        ''', (date_hour, date_hour, increment))
        
        conn.commit()
        conn.close()
    
    def get_rate_limit_count(self, date_hour: str) -> int:
        """Belirli saat için mesaj sayısını al"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT message_count FROM rate_limits WHERE date_hour = ?', (date_hour,))
        row = cursor.fetchone()
        conn.close()
        
        return row[0] if row else 0


class RateLimiter:
    """Rate limiting yöneticisi"""
    
    def __init__(self, db_manager: DatabaseManager, rate_config: Dict):
        self.db_manager = db_manager
        self.rate_config = rate_config
        self.message_timestamps = deque(maxlen=1000)  # Son 1000 mesajın zaman damgaları
        
    def can_send_message(self) -> bool:
        """Mesaj gönderilebilir mi kontrol et"""
        now = datetime.now()
        current_hour = now.strftime("%Y-%m-%d-%H")
        
        # Veritabanından saatlik kontrol
        hourly_count = self.db_manager.get_rate_limit_count(current_hour)
        if hourly_count >= self.rate_config.get("messages_per_hour", 5000):
            return False
        
        # In-memory dakikalık kontrol
        one_minute_ago = now - timedelta(minutes=1)
        recent_messages = [
            ts for ts in self.message_timestamps 
            if ts > one_minute_ago
        ]
        
        if len(recent_messages) >= self.rate_config.get("messages_per_minute", 1000):
            return False
        
        # In-memory saniyelik kontrol
        one_second_ago = now - timedelta(seconds=1)
        very_recent = [
            ts for ts in self.message_timestamps 
            if ts > one_second_ago
        ]
        
        return len(very_recent) < self.rate_config.get("messages_per_second", 30)
    
    def record_message(self):
        """Mesaj gönderimi kaydet"""
        now = datetime.now()
        current_hour = now.strftime("%Y-%m-%d-%H")
        
        self.message_timestamps.append(now)
        self.db_manager.record_rate_limit(current_hour, 1)
        
        # Rate limit'e yaklaşıyorsa logla
        hourly_count = self.db_manager.get_rate_limit_count(current_hour)
        if hourly_count > self.rate_config.get("messages_per_hour", 5000) * 0.9:
            logging.warning(f"Rate limit'e yaklaşılıyor: {hourly_count}/{self.rate_config.get('messages_per_hour', 5000)}")


class MediaProcessor:
    """Medya işleme yöneticisi"""
    
    def __init__(self, media_config: Dict):
        self.media_config = media_config
        self.temp_dir = "temp_media"
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def create_chart(self, chart_data: Dict, chart_type: str = "line") -> bytes:
        """Grafik oluştur ve byte array olarak döndür"""
        try:
            # Matplotlib konfigürasyonu
            plt.style.use('seaborn-v0_8')
            fig, ax = plt.subplots(figsize=(12, 8), dpi=self.media_config.get("chart_dpi", 100))
            
            # Grafik tipine göre işlem
            if chart_type == "line" and "line_chart" in chart_data:
                self._create_line_chart(ax, chart_data["line_chart"])
            elif chart_type == "bar" and "bar_chart" in chart_data:
                self._create_bar_chart(ax, chart_data["bar_chart"])
            elif chart_type == "pie" and "pie_chart" in chart_data:
                self._create_pie_chart(ax, chart_data["pie_chart"])
            elif chart_type == "heatmap" and "heatmap_data" in chart_data:
                self._create_heatmap(ax, chart_data["heatmap_data"])
            else:
                raise ValueError(f"Desteklenmeyen grafik tipi: {chart_type}")
            
            plt.tight_layout()
            
            # Byte array olarak kaydet
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            plt.close(fig)
            
            return buffer.getvalue()
        
        except Exception as e:
            logging.error(f"Grafik oluşturulamadı: {e}")
            raise
    
    def _create_line_chart(self, ax, data: Dict):
        """Çizgi grafik oluştur"""
        x_data = data.get("x", [])
        y_data = data.get("y", [])
        labels = data.get("labels", {})
        
        ax.plot(x_data, y_data, linewidth=2, alpha=0.8)
        ax.set_title(labels.get("title", "Çizgi Grafik"), fontsize=16, fontweight='bold')
        ax.set_xlabel(labels.get("xlabel", "X Ekseni"), fontsize=12)
        ax.set_ylabel(labels.get("ylabel", "Y Ekseni"), fontsize=12)
        ax.grid(True, alpha=0.3)
    
    def _create_bar_chart(self, ax, data: Dict):
        """Bar grafik oluştur"""
        categories = data.get("categories", [])
        values = data.get("values", [])
        labels = data.get("labels", {})
        
        bars = ax.bar(categories, values, alpha=0.8, color='skyblue', edgecolor='navy')
        ax.set_title(labels.get("title", "Bar Grafik"), fontsize=16, fontweight='bold')
        ax.set_xlabel(labels.get("xlabel", "Kategoriler"), fontsize=12)
        ax.set_ylabel(labels.get("ylabel", "Değerler"), fontsize=12)
        
        # Değerleri bar üzerinde göster
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}', ha='center', va='bottom')
    
    def _create_pie_chart(self, ax, data: Dict):
        """Pasta grafik oluştur"""
        labels = data.get("labels", [])
        sizes = data.get("sizes", [])
        colors = data.get("colors", None)
        labels_dict = data.get("labels_config", {})
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                        autopct='%1.1f%%', startangle=90)
        ax.set_title(labels_dict.get("title", "Pasta Grafik"), 
                    fontsize=16, fontweight='bold')
        
        # Yazıları daha büyük yap
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    def _create_heatmap(self, ax, data: Dict):
        """Heatmap oluştur"""
        matrix = data.get("matrix", [])
        row_labels = data.get("row_labels", [])
        col_labels = data.get("col_labels", [])
        labels = data.get("labels", {})
        
        im = ax.imshow(matrix, cmap='RdYlBu_r', aspect='auto')
        
        # Label'ları ayarla
        ax.set_xticks(range(len(col_labels)))
        ax.set_yticks(range(len(row_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)
        
        ax.set_title(labels.get("title", "Heatmap"), fontsize=16, fontweight='bold')
        
        # Colorbar ekle
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(labels.get("colorbar_label", "Değer"))
        
        # Değerleri hücrelerde göster
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = ax.text(j, i, f'{matrix[i][j]:.1f}',
                              ha="center", va="center", color="black")
    
    def create_text_image(self, text: str, image_type: str = "alert") -> bytes:
        """Metin içeren resim oluştur"""
        try:
            # Resim boyutu ve arkaplan
            if image_type == "alert":
                width, height = 800, 400
                bg_color = (255, 100, 100)  # Kırmızı
                text_color = (255, 255, 255)
            elif image_type == "success":
                width, height = 800, 400
                bg_color = (100, 255, 100)  # Yeşil
                text_color = (0, 0, 0)
            elif image_type == "info":
                width, height = 800, 400
                bg_color = (100, 100, 255)  # Mavi
                text_color = (255, 255, 255)
            else:
                width, height = 800, 400
                bg_color = (200, 200, 200)
                text_color = (0, 0, 0)
            
            # Resim oluştur
            image = Image.new('RGB', (width, height), bg_color)
            draw = ImageDraw.Draw(image)
            
            # Font ayarla (sistem fontu kullan)
            try:
                font_size = 36
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # Metni ortalı yaz
            lines = text.split('\n')
            line_height = 50
            start_y = (height - len(lines) * line_height) // 2
            
            for i, line in enumerate(lines):
                # Metrin genişliğini hesapla
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                
                # X pozisyonunu ortala
                x = (width - text_width) // 2
                y = start_y + i * line_height
                
                draw.text((x, y), line, fill=text_color, font=font)
            
            # Byte array olarak kaydet
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            return buffer.getvalue()
        
        except Exception as e:
            logging.error(f"Metin resmi oluşturulamadı: {e}")
            raise
    
    def validate_file_size(self, file_path: str) -> bool:
        """Dosya boyutu kontrol et"""
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            max_size = self.media_config.get("max_file_size", 50 * 1024 * 1024)
            return size <= max_size
        return False
    
    def validate_file_format(self, file_path: str) -> bool:
        """Dosya formatı kontrol et"""
        if os.path.exists(file_path):
            _, ext = os.path.splitext(file_path.lower())
            ext = ext[1:]  # '.' karakterini kaldır
            supported_formats = self.media_config.get("supported_formats", [])
            return ext in supported_formats
        return False


class MessageQueue:
    """Mesaj kuyruğu yöneticisi"""
    
    def __init__(self, db_manager: DatabaseManager, queue_config: Dict):
        self.db_manager = db_manager
        self.queue_config = queue_config
        self.queue = queue.Queue(maxsize=queue_config.get("max_queue_size", 1000))
        self.running = False
        self.workers = []
        self.bot = None
        self.rate_limiter = None
        self.retry_config = queue_config.get("retry_config", {})
        
    def set_dependencies(self, bot: Bot, rate_limiter: RateLimiter):
        """Bağımlılıkları ayarla"""
        self.bot = bot
        self.rate_limiter = rate_limiter
    
    def start(self):
        """Kuyruk işleyicisini başlat"""
        if self.running:
            return
        
        self.running = True
        worker_count = self.queue_config.get("worker_threads", 3)
        
        for i in range(worker_count):
            worker = threading.Thread(target=self._worker, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)
        
        logging.info(f"Message queue başlatıldı. {worker_count} worker thread.")
    
    def stop(self):
        """Kuyruk işleyicisini durdur"""
        self.running = False
        
        # Worker'lara shutdown sinyali gönder
        for _ in range(len(self.workers)):
            self.queue.put(None)
        
        # Worker'ların bitmesini bekle
        for worker in self.workers:
            worker.join(timeout=5)
        
        logging.info("Message queue durduruldu.")
    
    def add_message(self, item: MessageQueueItem) -> bool:
        """Kuyruğa mesaj ekle"""
        try:
            self.queue.put(item, timeout=1)
            # Veritabanına da ekle
            self.db_manager.add_message_to_queue(item)
            logging.debug(f"Mesaj kuyruğa eklendi: {item.message_id}")
            return True
        except queue.Full:
            logging.error("Mesaj kuyruğu dolu!")
            return False
        except Exception as e:
            logging.error(f"Mesaj kuyruğa eklenemedi: {e}")
            return False
    
    def _worker(self, worker_id: int):
        """Worker thread"""
        logging.info(f"Worker {worker_id} başladı.")
        
        while self.running:
            try:
                # Veritabanından bekleyen mesajları al
                pending_items = self.db_manager.get_pending_messages(limit=10)
                
                for item in pending_items:
                    self._process_message(item)
                    
                    # Rate limiting kontrolü
                    if not self.rate_limiter.can_send_message():
                        logging.warning("Rate limit aşıldı, bekleme...")
                        time.sleep(60)  # 1 dakika bekle
                
                # Kısa bekle
                time.sleep(1)
                
            except Exception as e:
                logging.error(f"Worker {worker_id} hatası: {e}")
                time.sleep(5)
        
        logging.info(f"Worker {worker_id} durdu.")
    
    def _process_message(self, item: MessageQueueItem):
        """Mesaj işle"""
        try:
            if not self.rate_limiter.can_send_message():
                logging.warning("Rate limit nedeniyle mesaj gönderilemedi.")
                return
            
            # Retry logic ile gönder
            success = self._send_with_retry(item)
            
            if success:
                self.db_manager.update_message_status(
                    item.message_id, 'sent', sent_at=datetime.now().isoformat()
                )
                self.rate_limiter.record_message()
                logging.info(f"Mesaj başarıyla gönderildi: {item.message_id}")
            else:
                # Retry limit aşıldı
                if item.retry_count >= item.max_retries:
                    self.db_manager.update_message_status(
                        item.message_id, 'failed', 
                        error_message="Max retry limit aşıldı"
                    )
                    logging.error(f"Mesaj gönderilemedi (max retry): {item.message_id}")
                
        except Exception as e:
            logging.error(f"Mesaj işlenemedi: {item.message_id}, Hata: {e}")
    
    def _send_with_retry(self, item: MessageQueueItem) -> bool:
        """Retry logic ile mesaj gönder"""
        retry_delay = self.retry_config.get("retry_delay", 1)
        max_retries = item.max_retries
        
        for attempt in range(max_retries + 1):
            try:
                # Mesaj tipine göre gönder
                if item.message_type == "text":
                    self.bot.send_message(
                        chat_id=item.chat_id,
                        text=item.content,
                        parse_mode=item.parse_mode,
                        reply_markup=item.reply_markup
                    )
                elif item.message_type == "photo":
                    self.bot.send_photo(
                        chat_id=item.chat_id,
                        photo=item.content if isinstance(item.content, str) else io.BytesIO(item.content),
                        caption=item.metadata.get("caption", "") if item.metadata else "",
                        parse_mode=item.parse_mode,
                        reply_markup=item.reply_markup
                    )
                elif item.message_type == "document":
                    self.bot.send_document(
                        chat_id=item.chat_id,
                        document=item.content if isinstance(item.content, str) else io.BytesIO(item.content),
                        caption=item.metadata.get("caption", "") if item.metadata else "",
                        parse_mode=item.parse_mode,
                        reply_markup=item.reply_markup
                    )
                elif item.message_type == "chart":
                    self.bot.send_photo(
                        chat_id=item.chat_id,
                        photo=io.BytesIO(item.content),
                        caption=item.metadata.get("caption", "") if item.metadata else "",
                        parse_mode=item.parse_mode,
                        reply_markup=item.reply_markup
                    )
                
                return True
                
            except RetryAfter as e:
                # Rate limit bekleme
                logging.warning(f"RetryAfter hatası, {e.retry_after} saniye bekleme...")
                time.sleep(e.retry_after + 1)
                
            except (NetworkError, TimedOut) as e:
                # Geçici ağ hataları
                if attempt < max_retries:
                    delay = retry_delay * (2 ** attempt)  # Exponential backoff
                    logging.warning(f"Ağ hatası, {delay} saniye sonra tekrar deneniyor...")
                    time.sleep(delay)
                else:
                    raise
                    
            except (BadRequest, Unauthorized, ChatMigrated) as e:
                # Kalıcı hatalar
                logging.error(f"Kalıcı hata: {e}")
                raise
                
            except TelegramError as e:
                # Diğer Telegram hataları
                if attempt < max_retries:
                    delay = retry_delay * (2 ** attempt)
                    logging.warning(f"Telegram hatası, {delay} saniye sonra tekrar deneniyor: {e}")
                    time.sleep(delay)
                else:
                    raise
        
        # Tüm denemeler başarısız
        self.db_manager.increment_retry_count(item.message_id)
        return False


class TelegramNotifierBot:
    """Ana Telegram bot sınıfı"""
    
    def __init__(self, config_file: str = "bot_config.json"):
        # Yapılandırmaları yükle
        self.config = TelegramBotConfig(config_file)
        
        # Veritabanı yöneticisi
        self.db_manager = DatabaseManager(self.config.get("database", {}).get("path", "telegram_bot.db"))
        
        # Rate limiter
        rate_config = self.config.get("rate_limit", {})
        self.rate_limiter = RateLimiter(self.db_manager, rate_config)
        
        # Media processor
        media_config = self.config.get("media_config", {})
        self.media_processor = MediaProcessor(media_config)
        
        # Message queue
        queue_config = self.config.get("queue_config", {})
        self.message_queue = MessageQueue(self.db_manager, queue_config)
        
        # Bot ve updater
        self.bot = None
        self.updater = None
        
        # Yapılandırmaları kaydet
        self._save_initial_config()
    
    def _save_initial_config(self):
        """Başlangıç yapılandırmasını kaydet"""
        # Örnek chat konfigürasyonu ekle (test için)
        sample_config = ChatConfig(
            chat_id=-123456789,  # Test chat ID'si
            chat_name="Test Chat",
            notification_level="info",
            last_config_update=datetime.now().isoformat(),
            created_at=datetime.now().isoformat()
        )
        
        # Sadece eğer veritabanında yoksa ekle
        existing_configs = self.db_manager.get_chat_configs()
        if not any(config.chat_id == sample_config.chat_id for config in existing_configs):
            self.db_manager.add_chat_config(sample_config)
    
    def initialize_bot(self):
        """Bot'u başlat"""
        bot_token = self.config.get("bot_token")
        if not bot_token:
            raise ValueError("Bot token bulunamadı! Lütfen bot_config.json dosyasında bot_token değerini ayarlayın.")
        
        # Bot'u oluştur
        self.bot = Bot(token=bot_token)
        self.updater = Updater(bot=self.bot, use_context=True)
        
        # Bağımlılıkları ayarla
        self.message_queue.set_dependencies(self.bot, self.rate_limiter)
        
        # Handler'ları kaydet
        self._register_handlers()
        
        # Message queue'yu başlat
        self.message_queue.start()
        
        logging.info("Telegram bot başlatıldı.")
    
    def _register_handlers(self):
        """Handler'ları kaydet"""
        dispatcher = self.updater.dispatcher
        
        # Komut handler'ları
        dispatcher.add_handler(CommandHandler("start", self.start_command))
        dispatcher.add_handler(CommandHandler("help", self.help_command))
        dispatcher.add_handler(CommandHandler("status", self.status_command))
        dispatcher.add_handler(CommandHandler("config", self.config_command))
        dispatcher.add_handler(CommandHandler("chatlist", self.chat_list_command))
        dispatcher.add_handler(CommandHandler("test", self.test_command))
        
        # Callback query handler
        dispatcher.add_handler(CallbackQueryHandler(self.callback_handler))
        
        # Mesaj handler (konfigürasyon için)
        dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, self.message_handler))
    
    def start_command(self, update: Update, context: CallbackContext):
        """Start komutu"""
        user = update.effective_user
        chat = update.effective_chat
        
        # Kullanıcıyı veritabanına ekle
        config = ChatConfig(
            chat_id=chat.id,
            chat_name=chat.title or f"{user.first_name} {user.last_name or ''}".strip(),
            notification_level="info",
            last_config_update=datetime.now().isoformat(),
            created_at=datetime.now().isoformat()
        )
        self.db_manager.add_chat_config(config)
        
        welcome_text = f"""
🤖 *Hoş geldiniz!*

Bu Telegram bot bildirim sistemi aşağıdaki özellikleri destekler:

📊 *Desteklenen Özellikler:*
• Zengin mesaj formatı (Markdown)
• Grafik ve medya desteği
• Inline klavye
• Konfigürasyon yönetimi
• Rate limiting
• Mesaj kuyruğu yönetimi

📋 *Mevcut Komutlar:*
/help - Bu yardım menüsü
/status - Sistem durumu
/config - Bot konfigürasyonu
/chatlist - Aktif chat listesi
/test - Test mesajı gönder

⚙️ *Başlangıç:*
Bot'unuzu kullanmaya başlamak için /config komutu ile konfigürasyonu ayarlayın.
        """
        
        keyboard = [
            [InlineKeyboardButton("📊 Status", callback_data="status"),
             InlineKeyboardButton("⚙️ Config", callback_data="config")],
            [InlineKeyboardButton("🧪 Test", callback_data="test"),
             InlineKeyboardButton("💬 Chats", callback_data="chatlist")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        update.message.reply_text(
            welcome_text, 
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
    
    def help_command(self, update: Update, context: CallbackContext):
        """Help komutu"""
        help_text = """
🆘 *Yardım Menüsü*

*Komutlar:*

/start - Bot'u başlatır ve karşılar
/help - Bu yardım menüsünü gösterir
/status - Bot durumu ve istatistikler
/config - Bot konfigürasyonu
/chatlist - Aktif chat listesi
/test - Test mesajı gönder

*Özellikler:*

📝 *Mesaj Türleri:*
• Metin mesajları (Markdown formatında)
• Fotoğraf ve resimler
• Belgeler ve PDF'ler
• Otomatik grafik oluşturma

⚡ *Performans:*
• Otomatik retry logic
• Rate limiting koruması
• Mesaj kuyruğu yönetimi
• Hata toleransı

📊 *Medya Desteği:*
• Çizgi grafikleri
• Bar grafikleri
• Pasta grafikleri
• Heatmap'ler
• Özel metin resimleri

*Destek:*
Herhangi bir sorun yaşarsanız lütfen sistem yöneticisi ile iletişime geçin.
        """
        
        update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)
    
    def status_command(self, update: Update, context: CallbackContext):
        """Status komutu"""
        # Rate limiting istatistikleri
        current_hour = datetime.now().strftime("%Y-%m-%d-%H")
        hourly_count = self.db_manager.get_rate_limit_count(current_hour)
        rate_config = self.config.get("rate_limit", {})
        
        # Chat istatistikleri
        configs = self.db_manager.get_chat_configs()
        
        # Queue istatistikleri (basit veritabanı sorgusu)
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM message_queue WHERE status = "pending"')
        pending_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM message_queue WHERE status = "sent" AND sent_at > datetime("now", "-24 hours")')
        sent_24h = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM message_queue WHERE status = "failed"')
        failed_count = cursor.fetchone()[0]
        
        conn.close()
        
        status_text = f"""
📊 *Sistem Durumu*

🤖 *Bot Bilgileri:*
• Token: {'✅ Yapılandırılmış' if self.config.get('bot_token') else '❌ Yapılandırılmamış'}
• API URL: {self.config.get('api_url', 'https://api.telegram.org')}

📈 *İstatistikler (Son 24 Saat):*
• Gönderilen: {sent_24h}
• Bekleyen: {pending_count}
• Başarısız: {failed_count}

⚡ *Rate Limiting:*
• Bu saat: {hourly_count}/{rate_config.get('messages_per_hour', 5000)}
• Limit/saniye: {rate_config.get('messages_per_second', 30)}
• Limit/dakika: {rate_config.get('messages_per_minute', 1000)}

💬 *Aktif Chatler:* {len(configs)}

🔧 *Konfigürasyon:*
• Max queue: {self.config.get('queue_config', {}).get('max_queue_size', 1000)}
• Worker thread: {self.config.get('queue_config', {}).get('worker_threads', 3)}
• Media max size: {self.config.get('media_config', {}).get('max_file_size', 50*1024*1024) // (1024*1024)}MB
        """
        
        update.message.reply_text(status_text, parse_mode=ParseMode.MARKDOWN)
    
    def config_command(self, update: Update, context: CallbackContext):
        """Config komutu"""
        configs = self.db_manager.get_chat_configs()
        
        if not configs:
            update.message.reply_text("❌ Henüz kayıtlı chat bulunmuyor.")
            return
        
        config_text = "⚙️ *Bot Konfigürasyonu*\n\n"
        
        for i, config in enumerate(configs):
            config_text += f"""
*Chat {i+1}:*
• ID: `{config.chat_id}`
• İsim: {config.chat_name}
• Aktif: {'✅' if config.is_active else '❌'}
• Bildirim seviyesi: `{config.notification_level}`
• Son güncelleme: {config.last_config_update[:19] if config.last_config_update else 'Bilinmiyor'}
            """
        
        config_text += """
\n*Komutlar:*
• Chat'i aktif/pasif yapmak için inline klavye kullanın
• Bildirim seviyesi: debug, info, warning, error, critical
        """
        
        keyboard = []
        for config in configs:
            status_btn = f"❌ {config.chat_name}" if config.is_active else f"✅ {config.chat_name}"
            keyboard.append([
                InlineKeyboardButton(status_btn, callback_data=f"toggle_{config.chat_id}"),
                InlineKeyboardButton(f"Seviye: {config.notification_level}", 
                                   callback_data=f"level_{config.chat_id}")
            ])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        update.message.reply_text(config_text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
    
    def chat_list_command(self, update: Update, context: CallbackContext):
        """Chat listesi komutu"""
        configs = self.db_manager.get_chat_configs()
        
        if not configs:
            update.message.reply_text("❌ Henüz kayıtlı chat bulunmuyor.")
            return
        
        chat_text = f"💬 *Aktif Chatler ({len(configs)} adet)*\n\n"
        
        for i, config in enumerate(configs, 1):
            chat_text += f"""
*{i}. {config.chat_name}*
• ID: `{config.chat_id}`
• Seviye: `{config.notification_level}`
• Durum: {'🟢 Aktif' if config.is_active else '🔴 Pasif'}
            """
        
        update.message.reply_text(chat_text, parse_mode=ParseMode.MARKDOWN)
    
    def test_command(self, update: Update, context: CallbackContext):
        """Test komutu"""
        # Test mesajı oluştur
        test_message = MessageQueueItem(
            message_id=f"test_{int(time.time())}_{update.effective_chat.id}",
            chat_id=update.effective_chat.id,
            message_type="text",
            content="🧪 *Test Mesajı*\n\nBu bir test mesajıdır. Bot düzgün çalışıyor!",
            parse_mode=ParseMode.MARKDOWN,
            priority=1
        )
        
        # Kuyruğa ekle
        success = self.message_queue.add_message(test_message)
        
        if success:
            response_text = "✅ Test mesajı kuyruğa eklendi ve gönderilecek!"
            
            # Test grafik de gönder
            try:
                chart_data = {
                    "line_chart": {
                        "x": list(range(10)),
                        "y": [i**2 for i in range(10)],
                        "labels": {
                            "title": "Test Grafik",
                            "xlabel": "X Değeri",
                            "ylabel": "X²"
                        }
                    }
                }
                
                chart_image = self.media_processor.create_chart(chart_data, "line")
                
                chart_message = MessageQueueItem(
                    message_id=f"chart_test_{int(time.time())}_{update.effective_chat.id}",
                    chat_id=update.effective_chat.id,
                    message_type="chart",
                    content=chart_image,
                    metadata={"caption": "📊 Test Grafik - Bot çalışıyor!"}
                )
                
                self.message_queue.add_message(chart_message)
                response_text += "\n📊 Test grafiği de gönderildi!"
                
            except Exception as e:
                logging.warning(f"Test grafiği oluşturulamadı: {e}")
            
            update.message.reply_text(response_text)
        else:
            update.message.reply_text("❌ Test mesajı gönderilemedi!")
    
    def callback_handler(self, update: Update, context: CallbackContext):
        """Inline keyboard callback handler"""
        query = update.callback_query
        query.answer()
        
        data = query.data
        
        if data.startswith("toggle_"):
            # Chat aktif/pasif toggle
            chat_id = int(data.split("_")[1])
            self._toggle_chat_status(chat_id, query)
            
        elif data.startswith("level_"):
            # Bildirim seviyesi değiştir
            chat_id = int(data.split("_")[1])
            self._change_notification_level(chat_id, query)
            
        elif data == "status":
            # Status göster
            self.status_command(update, context)
            
        elif data == "config":
            # Config göster
            self.config_command(update, context)
            
        elif data == "test":
            # Test gönder
            self.test_command(update, context)
            
        elif data == "chatlist":
            # Chat listesi göster
            self.chat_list_command(update, context)
    
    def _toggle_chat_status(self, chat_id: int, query):
        """Chat durumunu değiştir"""
        configs = self.db_manager.get_chat_configs()
        config = next((c for c in configs if c.chat_id == chat_id), None)
        
        if config:
            config.is_active = not config.is_active
            config.last_config_update = datetime.now().isoformat()
            self.db_manager.add_chat_config(config)
            
            status = "aktif" if config.is_active else "pasif"
            query.edit_message_text(f"✅ Chat `{config.chat_name}` şimdi *{status}* olarak ayarlandı.")
        else:
            query.edit_message_text("❌ Chat bulunamadı!")
    
    def _change_notification_level(self, chat_id: int, query):
        """Bildirim seviyesini değiştir"""
        configs = self.db_manager.get_chat_configs()
        config = next((c for c in configs if c.chat_id == chat_id), None)
        
        if config:
            levels = ["debug", "info", "warning", "error", "critical"]
            current_index = levels.index(config.notification_level)
            next_index = (current_index + 1) % len(levels)
            
            config.notification_level = levels[next_index]
            config.last_config_update = datetime.now().isoformat()
            self.db_manager.add_chat_config(config)
            
            query.edit_message_text(f"✅ Bildirim seviyesi: `{config.notification_level}`")
        else:
            query.edit_message_text("❌ Chat bulunamadı!")
    
    def message_handler(self, update: Update, context: CallbackContext):
        """Genel mesaj handler (konfigürasyon için)"""
        update.message.reply_text(
            "💬 Mesajınız alındı! Komutlar için /help kullanabilirsiniz.",
            parse_mode=ParseMode.MARKDOWN
        )
    
    def send_notification(self, chat_id: int, message: str, 
                         message_type: str = "text",
                         parse_mode: str = ParseMode.MARKDOWN,
                         priority: int = 5,
                         metadata: Dict = None,
                         reply_markup: InlineKeyboardMarkup = None):
        """Bildirim gönder"""
        message_id = f"msg_{int(time.time() * 1000)}_{chat_id}"
        
        item = MessageQueueItem(
            message_id=message_id,
            chat_id=chat_id,
            message_type=message_type,
            content=message,
            parse_mode=parse_mode,
            priority=priority,
            metadata=metadata or {},
            reply_markup=reply_markup
        )
        
        return self.message_queue.add_message(item)
    
    def send_chart(self, chat_id: int, chart_data: Dict, 
                   chart_type: str = "line",
                   caption: str = "",
                   priority: int = 5,
                   metadata: Dict = None):
        """Grafik gönder"""
        try:
            chart_image = self.media_processor.create_chart(chart_data, chart_type)
            
            message_id = f"chart_{int(time.time() * 1000)}_{chat_id}"
            
            item = MessageQueueItem(
                message_id=message_id,
                chat_id=chat_id,
                message_type="chart",
                content=chart_image,
                priority=priority,
                metadata={**(metadata or {}), "caption": caption}
            )
            
            return self.message_queue.add_message(item)
            
        except Exception as e:
            logging.error(f"Grafik gönderilemedi: {e}")
            return False
    
    def send_image(self, chat_id: int, image_data: Union[str, bytes],
                   caption: str = "",
                   priority: int = 5,
                   metadata: Dict = None):
        """Resim gönder"""
        message_id = f"img_{int(time.time() * 1000)}_{chat_id}"
        
        item = MessageQueueItem(
            message_id=message_id,
            chat_id=chat_id,
            message_type="photo",
            content=image_data,
            priority=priority,
            metadata={**(metadata or {}), "caption": caption}
        )
        
        return self.message_queue.add_message(item)
    
    def send_document(self, chat_id: int, document_data: Union[str, bytes],
                      caption: str = "",
                      priority: int = 5,
                      metadata: Dict = None):
        """Döküman gönder"""
        message_id = f"doc_{int(time.time() * 1000)}_{chat_id}"
        
        item = MessageQueueItem(
            message_id=message_id,
            chat_id=chat_id,
            message_type="document",
            content=document_data,
            priority=priority,
            metadata={**(metadata or {}), "caption": caption}
        )
        
        return self.message_queue.add_message(item)
    
    def run(self):
        """Bot'u çalıştır"""
        try:
            self.initialize_bot()
            
            # Logging konfigürasyonu
            log_config = self.config.get("logging", {})
            logging.basicConfig(
                level=getattr(logging, log_config.get("level", "INFO")),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_config.get("file", "telegram_bot.log")),
                    logging.StreamHandler()
                ]
            )
            
            logging.info("Telegram Bot başlatılıyor...")
            
            # Bot'u başlat
            self.updater.start_polling(drop_pending_updates=True)
            self.updater.idle()
            
        except InvalidToken:
            logging.error("Geçersiz bot token! Lütfen geçerli bir token girin.")
            return False
            
        except KeyboardInterrupt:
            logging.info("Bot kapatılıyor...")
            self.stop()
            
        except Exception as e:
            logging.error(f"Bot başlatılamadı: {e}")
            return False
        
        return True
    
    def stop(self):
        """Bot'u durdur"""
        if self.message_queue:
            self.message_queue.stop()
        
        if self.updater:
            self.updater.stop()
        
        logging.info("Bot durduruldu.")


def create_sample_config():
    """Örnek konfigürasyon dosyası oluştur"""
    sample_config = {
        "bot_token": "YOUR_BOT_TOKEN_HERE",
        "api_url": "https://api.telegram.org",
        "rate_limit": {
            "messages_per_second": 30,
            "messages_per_minute": 1000,
            "messages_per_hour": 5000
        },
        "retry_config": {
            "max_retries": 3,
            "retry_delay": 1,
            "exponential_backoff": True
        },
        "queue_config": {
            "max_queue_size": 1000,
            "worker_threads": 3,
            "queue_timeout": 300
        },
        "media_config": {
            "max_file_size": 52428800,
            "supported_formats": ["jpg", "jpeg", "png", "gif", "pdf", "doc", "docx"],
            "chart_dpi": 100,
            "chart_dpi_high": 200
        },
        "logging": {
            "level": "INFO",
            "file": "telegram_bot.log",
            "max_size": 10485760,
            "backup_count": 5
        },
        "database": {
            "path": "telegram_bot.db"
        }
    }
    
    with open("bot_config.json", "w", encoding="utf-8") as f:
        json.dump(sample_config, f, indent=4, ensure_ascii=False)
    
    print("✅ Örnek konfigürasyon dosyası oluşturuldu: bot_config.json")
    print("⚠️  Lütfen bot_token değerini BotFather'dan aldığınız token ile değiştirin!")


def main():
    """Ana fonksiyon"""
    print("🤖 Telegram Bot Notification System")
    print("=" * 50)
    
    # Konfigürasyon dosyası kontrolü
    if not os.path.exists("bot_config.json"):
        print("❌ bot_config.json dosyası bulunamadı.")
        create_sample_config()
        return
    
    # Bot token kontrolü
    with open("bot_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    if not config.get("bot_token") or config.get("bot_token") == "YOUR_BOT_TOKEN_HERE":
        print("❌ Bot token ayarlanmamış!")
        print("Lütfen bot_config.json dosyasında bot_token değerini ayarlayın.")
        return
    
    # Bot'u başlat
    bot = TelegramNotifierBot()
    
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\n👋 Bot kapatıldı.")
    except Exception as e:
        print(f"❌ Hata: {e}")
        logging.error(f"Bot çalıştırılamadı: {e}")


if __name__ == "__main__":
    main()
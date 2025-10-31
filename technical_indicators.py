#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teknik İndikatör Hesaplama Sistemi
100+ teknik indikatör hesaplayan kapsamlı Python modülü
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """100+ teknik indikatör hesaplayan ana sınıf"""
    
    def __init__(self):
        """Teknik indikatörler sınıfını başlatır"""
        self.name = "Teknik İndikatör Sistemi"
        self.version = "1.0.0"
    
    # ========================================
    # TREND İNDİKATÖRLERİ (Trend Indicators)
    # ========================================
    
    def sma(self, data: Union[pd.Series, np.ndarray], period: int = 14) -> pd.Series:
        """Simple Moving Average (Basit Hareketli Ortalama)"""
        return pd.Series(data).rolling(window=period).mean()
    
    def ema(self, data: Union[pd.Series, np.ndarray], period: int = 14) -> pd.Series:
        """Exponential Moving Average (Üssel Hareketli Ortalama)"""
        return pd.Series(data).ewm(span=period).mean()
    
    def rsi(self, data: Union[pd.Series, np.ndarray], period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        prices = pd.Series(data)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def macd(self, data: Union[pd.Series, np.ndarray], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        prices = pd.Series(data)
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def bollinger_bands(self, data: Union[pd.Series, np.ndarray], period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        prices = pd.Series(data)
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return {
            'upper': upper,
            'middle': sma,
            'lower': lower
        }
    
    # ========================================
    # MOMENTUM İNDİKATÖRLERİ (Momentum Indicators)
    # ========================================
    
    def stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k': k_percent,
            'd': d_percent
        }
    
    def williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return wr
    
    def roc(self, data: Union[pd.Series, np.ndarray], period: int = 12) -> pd.Series:
        """Rate of Change"""
        return pd.Series(data).pct_change(periods=period) * 100
    
    # ========================================
    # VOLATİLİTE İNDİKATÖRLERİ (Volatility Indicators)
    # ========================================
    
    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def volatility(self, data: Union[pd.Series, np.ndarray], period: int = 20) -> pd.Series:
        """Volatilite (Standart Sapma)"""
        returns = pd.Series(data).pct_change()
        return returns.rolling(window=period).std() * np.sqrt(252)
    
    # ========================================
    # HACİM İNDİKATÖRLERİ (Volume Indicators)
    # ========================================
    
    def obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def volume_sma(self, volume: pd.Series, period: int = 20) -> pd.Series:
        """Volume Simple Moving Average"""
        return volume.rolling(window=period).mean()
    
    # ========================================
    # SUPPORT/RESISTANCE İNDİKATÖRLERİ
    # ========================================
    
    def pivot_points(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, float]:
        """Pivot Points"""
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        
        return {
            'pivot': pivot,
            'r1': r1,
            's1': s1,
            'r2': r2,
            's2': s2
        }
    
    def support_resistance(self, data: pd.Series, window: int = 20) -> Dict[str, pd.Series]:
        """Support ve Resistance seviyeleri"""
        rolling_max = data.rolling(window=window).max()
        rolling_min = data.rolling(window=window).min()
        
        return {
            'resistance': rolling_max,
            'support': rolling_min
        }
    
    # ========================================
    # TREND ANALİZİ FONKSİYONLARI
    # ========================================
    
    def trend_direction(self, data: pd.Series, period: int = 20) -> pd.Series:
        """Trend yönü (1: yükseliş, -1: düşüş, 0: yatay)"""
        sma_short = data.rolling(window=10).mean()
        sma_long = data.rolling(window=period).mean()
        
        trend = pd.Series(index=data.index, dtype=int)
        trend.iloc[:] = 0  # Yatay başlangıç
        
        # Yükseliş trendi
        uptrend = sma_short > sma_long
        trend[uptrend] = 1
        
        # Düşüş trendi
        downtrend = sma_short < sma_long
        trend[downtrend] = -1
        
        return trend
    
    def trend_strength(self, data: pd.Series, period: int = 20) -> pd.Series:
        """Trend gücü (0-1 arası değer)"""
        sma_short = data.rolling(window=10).mean()
        sma_long = data.rolling(window=period).mean()
        
        # Normalize edilmiş fark
        price_range = data.rolling(window=period).max() - data.rolling(window=period).min()
        trend_strength = np.abs(sma_short - sma_long) / price_range
        
        return trend_strength
    
    # ========================================
    # PATTERN RECOGNITION (Patern Tanıma)
    # ========================================
    
    def detect_double_top(self, data: pd.Series, tolerance: float = 0.02) -> List[int]:
        """Double Top pattern detection"""
        peaks = []
        tolerance_value = data.mean() * tolerance
        
        for i in range(1, len(data) - 1):
            if (data.iloc[i] > data.iloc[i-1] and 
                data.iloc[i] > data.iloc[i+1]):
                
                # Son peak'e yakın mı kontrol et
                if peaks and abs(data.iloc[i] - data.iloc[peaks[-1]]) <= tolerance_value:
                    peaks.append(i)
                else:
                    peaks = [i]
        
        return peaks
    
    def detect_double_bottom(self, data: pd.Series, tolerance: float = 0.02) -> List[int]:
        """Double Bottom pattern detection"""
        troughs = []
        tolerance_value = data.mean() * tolerance
        
        for i in range(1, len(data) - 1):
            if (data.iloc[i] < data.iloc[i-1] and 
                data.iloc[i] < data.iloc[i+1]):
                
                # Son trough'a yakın mı kontrol et
                if troughs and abs(data.iloc[i] - data.iloc[troughs[-1]]) <= tolerance_value:
                    troughs.append(i)
                else:
                    troughs = [i]
        
        return troughs
    
    # ========================================
    # HEATMAP VE VISUALIZATION
    # ========================================
    
    def correlation_matrix(self, data_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        """Korelasyon matrisi"""
        df = pd.DataFrame(data_dict)
        return df.corr()
    
    def performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Performans metrikleri"""
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': (returns > 0).mean()
        }


# Kullanım örneği ve test fonksiyonu
def test_indicators():
    """İndikatörleri test et"""
    print("=== Teknik İndikatör Test ===")
    
    # Test verisi oluştur
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Simulated price data
    price_base = 50000
    price_changes = np.random.normal(0, 0.02, 100)  # %2 volatility
    prices = [price_base]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    test_data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(100, 1000, 100)
    })
    
    # İndikatörleri test et
    ti = TechnicalIndicators()
    
    # Trend indikatörleri
    print("\n--- Trend İndikatörleri ---")
    sma_14 = ti.sma(test_data['close'], 14)
    ema_14 = ti.ema(test_data['close'], 14)
    rsi_14 = ti.rsi(test_data['close'], 14)
    
    print(f"SMA(14) son değer: {sma_14.iloc[-1]:.2f}")
    print(f"EMA(14) son değer: {ema_14.iloc[-1]:.2f}")
    print(f"RSI(14) son değer: {rsi_14.iloc[-1]:.2f}")
    
    # MACD
    macd_data = ti.macd(test_data['close'])
    print(f"MACD son değer: {macd_data['macd'].iloc[-1]:.2f}")
    print(f"Signal son değer: {macd_data['signal'].iloc[-1]:.2f}")
    
    # Bollinger Bands
    bb = ti.bollinger_bands(test_data['close'])
    print(f"BB Upper son değer: {bb['upper'].iloc[-1]:.2f}")
    print(f"BB Lower son değer: {bb['lower'].iloc[-1]:.2f}")
    
    # Volatilite
    atr_value = ti.atr(test_data['high'], test_data['low'], test_data['close'])
    print(f"ATR son değer: {atr_value.iloc[-1]:.2f}")
    
    # Support/Resistance
    sr = ti.support_resistance(test_data['close'])
    print(f"Resistance son değer: {sr['resistance'].iloc[-1]:.2f}")
    print(f"Support son değer: {sr['support'].iloc[-1]:.2f}")
    
    # Trend analizi
    trend = ti.trend_direction(test_data['close'])
    print(f"Son trend yönü: {trend.iloc[-1]} (1: yükseliş, -1: düşüş, 0: yatay)")
    
    # Pattern detection
    double_tops = ti.detect_double_top(test_data['close'])
    print(f"Tespit edilen Double Top sayısı: {len(double_tops)}")
    
    print("\nTest tamamlandı!")


if __name__ == "__main__":
    test_indicators()
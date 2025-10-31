#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Akıllı Pozisyon Boyutlandırma Sistemi
=====================================

Bu modül çeşitli risk yönetimi ve pozisyon boyutlandırma stratejilerini içerir:
- Volatilite Hedefleme (Volatility Targeting)
- Kelly Kriteri
- Fixed Fractional
- Risk Parity
- ATR Bazlı Hesaplamalar
- Dinamik Pozisyon Büyüklüğü Belirleme

Yazar: AI Assistant
Tarih: 2025-10-31
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class SizingMethod(Enum):
    """Pozisyon boyutlandırma yöntemleri"""
    VOLATILITY_TARGETING = "volatility_targeting"
    KELLY_CRITERION = "kelly_criterion"
    FIXED_FRACTIONAL = "fixed_fractional"
    RISK_PARITY = "risk_parity"
    ATR_BASED = "atr_based"
    DYNAMIC = "dynamic"


@dataclass
class RiskParameters:
    """Risk parametreleri"""
    max_risk_per_trade: float = 0.02  # Maksimum risk oranı (%2)
    max_portfolio_risk: float = 0.10  # Maksimum portföy riski (%10)
    volatility_target: float = 0.15   # Hedef volatilite (%15)
    kelly_fraction: float = 0.25      # Kelly kriteri kesir (0.25 = %25)
    atr_multiplier: float = 2.0       # ATR çarpanı
    lookback_period: int = 20         # Geriye dönük periyot
    min_position_size: float = 0.01   # Minimum pozisyon boyutu
    max_position_size: float = 1.0    # Maksimum pozisyon boyutu


class ATRCalculator:
    """Average True Range hesaplayıcısı"""
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = 14) -> pd.Series:
        """
        ATR hesapla
        
        Args:
            high: Yüksek fiyat serisi
            low: Düşük fiyat serisi
            close: Kapanış fiyat serisi
            period: ATR periyodu
            
        Returns:
            ATR serisi
        """
        high_low = high - low
        high_close_prev = np.abs(high - close.shift(1))
        low_close_prev = np.abs(low - close.shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, period: int = 20) -> pd.Series:
        """
        Volatilite hesapla
        
        Args:
            returns: Getiri serisi
            period: Volatilite periyodu
            
        Returns:
            Volatilite serisi (annualized)
        """
        volatility = returns.rolling(window=period).std() * np.sqrt(252)
        return volatility


class PositionSizer:
    """Ana pozisyon boyutlandırma sınıfı"""
    
    def __init__(self, risk_params: RiskParameters = None):
        """
        PositionSizer başlatıcısı
        
        Args:
            risk_params: Risk parametreleri
        """
        self.risk_params = risk_params or RiskParameters()
        self.atr_calc = ATRCalculator()
    
    def volatility_targeting(self, expected_return: float, current_volatility: float, 
                           portfolio_value: float) -> float:
        """
        Volatilite hedefleme yöntemi
        
        Args:
            expected_return: Beklenen getiri
            current_volatility: Mevcut volatilite
            portfolio_value: Portföy değeri
            
        Returns:
            Pozisyon boyutu
        """
        if current_volatility <= 0:
            return 0.0
        
        # Volatilite oranı hesapla
        volatility_ratio = self.risk_params.volatility_target / current_volatility
        
        # Pozisyon boyutu = (beklenen getiri / mevcut volatilite) * hedef volatilite
        position_size = (expected_return / current_volatility) * volatility_ratio
        
        # Risk sınırları uygula
        position_size = self._apply_risk_limits(position_size, portfolio_value)
        
        return max(0.0, min(position_size, self.risk_params.max_position_size))
    
    def kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float, 
                       portfolio_value: float) -> float:
        """
        Kelly Kriteri ile pozisyon boyutlandırma
        
        Args:
            win_rate: Kazanma oranı
            avg_win: Ortalama kazanç
            avg_loss: Ortalama kayıp (pozitif değer)
            portfolio_value: Portföy değeri
            
        Returns:
            Pozisyon boyutu
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        # Kelly oranı hesapla
        b = avg_win / avg_loss  # Kazanç/kayıp oranı
        kelly_fraction = win_rate - ((1 - win_rate) / b)
        
        # Kelly kriteri güvenlik kesri uygula
        adjusted_kelly = kelly_fraction * self.risk_params.kelly_fraction
        
        # Pozisyon büyüklüğü = Kelly oranı * portföy değeri
        position_size = adjusted_kelly * portfolio_value
        
        # Risk sınırları uygula
        position_value = self._apply_risk_limits(position_size, portfolio_value)
        
        return max(0.0, min(position_value / portfolio_value, self.risk_params.max_position_size))
    
    def fixed_fractional(self, portfolio_value: float, fixed_fraction: float = None) -> float:
        """
        Sabit kesir yöntemi
        
        Args:
            portfolio_value: Portföy değeri
            fixed_fraction: Sabit kesir (varsayılan: max_risk_per_trade)
            
        Returns:
            Pozisyon boyutu
        """
        fraction = fixed_fraction or self.risk_params.max_risk_per_trade
        position_size = portfolio_value * fraction
        
        return max(0.0, min(position_size, portfolio_value * self.risk_params.max_position_size))
    
    def risk_parity(self, volatilities: Dict[str, float], correlations: Dict[Tuple[str, str], float], 
                   total_risk_budget: float) -> Dict[str, float]:
        """
        Risk parity yöntemi
        
        Args:
            volatilities: Varlık volatiliteleri
            correlations: Korelasyon matrisi
            total_risk_budget: Toplam risk bütçesi
            
        Returns:
            Varlık başına pozisyon boyutları
        """
        assets = list(volatilities.keys())
        n_assets = len(assets)
        
        if n_assets == 0:
            return {}
        
        # Volatilite vektörü
        vol_vector = np.array([volatilities[asset] for asset in assets])
        
        # Korelasyon matrisi
        corr_matrix = np.eye(n_assets)
        for i, asset_i in enumerate(assets):
            for j, asset_j in enumerate(assets):
                if i != j:
                    corr_key = (asset_i, asset_j) if (asset_i, asset_j) in correlations else (asset_j, asset_i)
                    if corr_key in correlations:
                        corr_matrix[i, j] = correlations[corr_key]
        
        # Kovaryans matrisi
        vol_diag = np.diag(vol_vector)
        cov_matrix = vol_diag @ corr_matrix @ vol_diag
        
        # Risk parity ağırlıkları (eşit marjinal risk)
        try:
            inv_cov = np.linalg.inv(cov_matrix)
            ones = np.ones(n_assets)
            weights = inv_cov @ ones
            weights = weights / (ones.T @ weights)
        except np.linalg.LinAlgError:
            # Eğer matris tekil ise, eşit ağırlıklar kullan
            weights = np.ones(n_assets) / n_assets
        
        # Risk bütçesini uygula
        risk_adjusted_weights = weights * total_risk_budget
        
        # Pozisyon boyutlarını sözlük olarak döndür
        position_sizes = {}
        for i, asset in enumerate(assets):
            position_sizes[asset] = max(0.0, risk_adjusted_weights[i])
        
        return position_sizes
    
    def atr_based_sizing(self, entry_price: float, stop_loss: float, 
                        atr_value: float, portfolio_value: float, 
                        atr_multiplier: float = None) -> float:
        """
        ATR bazlı pozisyon boyutlandırma
        
        Args:
            entry_price: Giriş fiyatı
            stop_loss: Stop loss fiyatı
            atr_value: ATR değeri
            portfolio_value: Portföy değeri
            atr_multiplier: ATR çarpanı
            
        Returns:
            Pozisyon boyutu
        """
        if atr_value <= 0:
            return 0.0
        
        multiplier = atr_multiplier or self.risk_params.atr_multiplier
        
        # Risk mesafesi = ATR * çarpan
        risk_distance = atr_value * multiplier
        
        # Pozisyon büyüklüğü = (riske edilecek miktar) / risk mesafesi
        risk_amount = portfolio_value * self.risk_params.max_risk_per_trade
        position_size = risk_amount / risk_distance
        
        # Riske edilen gerçek miktarı hesapla
        actual_risk = position_size * risk_distance
        
        # Risk limitleri kontrolü
        max_risk_amount = portfolio_value * self.risk_params.max_portfolio_risk
        if actual_risk > max_risk_amount:
            position_size = max_risk_amount / risk_distance
        
        return max(0.0, min(position_size, portfolio_value * self.risk_params.max_position_size))
    
    def dynamic_position_sizing(self, market_conditions: Dict[str, float], 
                              signal_strength: float, portfolio_value: float,
                              historical_performance: Dict[str, float] = None) -> float:
        """
        Dinamik pozisyon boyutlandırma
        
        Args:
            market_conditions: Piyasa koşulları (volatilite, trend gücü, vb.)
            signal_strength: Sinyal gücü (0-1)
            portfolio_value: Portföy değeri
            historical_performance: Geçmiş performans metrikleri
            
        Returns:
            Pozisyon boyutu
        """
        # Temel pozisyon boyutu
        base_position = signal_strength * self.risk_params.max_risk_per_trade
        
        # Volatilite ayarlaması
        current_vol = market_conditions.get('volatility', 0.15)
        vol_adjustment = self.risk_params.volatility_target / current_vol if current_vol > 0 else 1.0
        
        # Trend gücü ayarlaması
        trend_strength = market_conditions.get('trend_strength', 0.0)
        trend_adjustment = 1.0 + trend_strength
        
        # Likidite ayarlaması
        liquidity = market_conditions.get('liquidity', 1.0)
        liquidity_adjustment = min(liquidity, 1.0)
        
        # Geçmiş performans ayarlaması
        performance_adjustment = 1.0
        if historical_performance:
            sharpe_ratio = historical_performance.get('sharpe_ratio', 1.0)
            win_rate = historical_performance.get('win_rate', 0.5)
            
            # Performans temelli ayarlama
            if sharpe_ratio > 1.5 and win_rate > 0.6:
                performance_adjustment = 1.2  # Pozitif ayarlama
            elif sharpe_ratio < 0.5 or win_rate < 0.4:
                performance_adjustment = 0.8  # Negatif ayarlama
        
        # Dinamik pozisyon boyutu hesapla
        dynamic_position = (base_position * vol_adjustment * trend_adjustment * 
                           liquidity_adjustment * performance_adjustment)
        
        # Risk sınırları uygula
        position_value = self._apply_risk_limits(dynamic_position * portfolio_value, portfolio_value)
        
        return max(self.risk_params.min_position_size, 
                  min(position_value / portfolio_value, self.risk_params.max_position_size))
    
    def _apply_risk_limits(self, position_value: float, portfolio_value: float) -> float:
        """
        Risk limitlerini uygula
        
        Args:
            position_value: Pozisyon değeri
            portfolio_value: Portföy değeri
            
        Returns:
            Risk limitleri uygulanmış pozisyon değeri
        """
        # Maksimum pozisyon limiti
        max_position = portfolio_value * self.risk_params.max_position_size
        
        # Maksimum risk limiti
        max_risk = portfolio_value * self.risk_params.max_portfolio_risk
        
        # Risk sınırını uygula
        if position_value > max_risk:
            position_value = max_risk
        
        return min(position_value, max_position)
    
    def calculate_position_size(self, method: SizingMethod, **kwargs) -> Union[float, Dict[str, float]]:
        """
        Genel pozisyon boyutlandırma fonksiyonu
        
        Args:
            method: Kullanılacak yöntem
            **kwargs: Yöntem parametreleri
            
        Returns:
            Pozisyon boyutu(ları)
        """
        if method == SizingMethod.VOLATILITY_TARGETING:
            return self.volatility_targeting(
                expected_return=kwargs['expected_return'],
                current_volatility=kwargs['current_volatility'],
                portfolio_value=kwargs['portfolio_value']
            )
        
        elif method == SizingMethod.KELLY_CRITERION:
            return self.kelly_criterion(
                win_rate=kwargs['win_rate'],
                avg_win=kwargs['avg_win'],
                avg_loss=kwargs['avg_loss'],
                portfolio_value=kwargs['portfolio_value']
            )
        
        elif method == SizingMethod.FIXED_FRACTIONAL:
            return self.fixed_fractional(
                portfolio_value=kwargs['portfolio_value'],
                fixed_fraction=kwargs.get('fixed_fraction')
            )
        
        elif method == SizingMethod.RISK_PARITY:
            return self.risk_parity(
                volatilities=kwargs['volatilities'],
                correlations=kwargs['correlations'],
                total_risk_budget=kwargs['total_risk_budget']
            )
        
        elif method == SizingMethod.ATR_BASED:
            return self.atr_based_sizing(
                entry_price=kwargs['entry_price'],
                stop_loss=kwargs['stop_loss'],
                atr_value=kwargs['atr_value'],
                portfolio_value=kwargs['portfolio_value'],
                atr_multiplier=kwargs.get('atr_multiplier')
            )
        
        elif method == SizingMethod.DYNAMIC:
            return self.dynamic_position_sizing(
                market_conditions=kwargs['market_conditions'],
                signal_strength=kwargs['signal_strength'],
                portfolio_value=kwargs['portfolio_value'],
                historical_performance=kwargs.get('historical_performance')
            )
        
        else:
            raise ValueError(f"Desteklenmeyen yöntem: {method}")


class PortfolioRiskManager:
    """Portföy risk yöneticisi"""
    
    def __init__(self, position_sizer: PositionSizer):
        """
        PortfolioRiskManager başlatıcısı
        
        Args:
            position_sizer: PositionSizer instance
        """
        self.position_sizer = position_sizer
        self.positions = {}
        self.total_portfolio_value = 0.0
    
    def add_position(self, symbol: str, size: float, entry_price: float, 
                    stop_loss: float = None, take_profit: float = None):
        """
        Pozisyon ekle
        
        Args:
            symbol: Varlık sembolü
            size: Pozisyon büyüklüğü
            entry_price: Giriş fiyatı
            stop_loss: Stop loss fiyatı
            take_profit: Take profit fiyatı
        """
        self.positions[symbol] = {
            'size': size,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'current_value': size * entry_price
        }
    
    def calculate_portfolio_risk(self) -> Dict[str, float]:
        """
        Portföy riskini hesapla
        
        Returns:
            Risk metrikleri
        """
        total_value = sum(pos['current_value'] for pos in self.positions.values())
        
        if total_value == 0:
            return {'total_risk': 0.0, 'risk_per_position': {}, 'concentration_risk': 0.0}
        
        # Pozisyon riskleri
        position_risks = {}
        total_risk = 0.0
        
        for symbol, position in self.positions.items():
            position_value = position['current_value']
            position_weight = position_value / total_value
            
            # Pozisyon riskini hesapla
            if position['stop_loss']:
                risk_per_unit = abs(position['entry_price'] - position['stop_loss'])
                position_risk = position['size'] * risk_per_unit
            else:
                # Stop loss yoksa volatilite bazlı risk
                position_risk = position_value * 0.05  # Varsayılan %5 volatilite
            
            position_risks[symbol] = {
                'value': position_risk,
                'percentage': position_risk / total_value
            }
            
            total_risk += position_risk
        
        # Konsantrasyon riski
        max_position_weight = max(pos['current_value'] / total_value for pos in self.positions.values())
        
        return {
            'total_risk': total_risk,
            'total_value': total_value,
            'total_risk_percentage': total_risk / total_value,
            'risk_per_position': position_risks,
            'concentration_risk': max_position_weight,
            'diversification_score': 1.0 - (max_position_weight ** 2)
        }
    
    def optimize_portfolio(self) -> Dict[str, Dict[str, float]]:
        """
        Portföy optimizasyonu
        
        Returns:
            Önerilen pozisyon ayarlamaları
        """
        portfolio_risk = self.calculate_portfolio_risk()
        risk_per_position = portfolio_risk['risk_per_position']
        
        adjustments = {}
        
        for symbol, position in self.positions.items():
            current_risk = risk_per_position[symbol]['percentage']
            
            # Eğer pozisyon riski çok yüksekse azalt
            if current_risk > 0.05:  # %5'ten fazla
                adjustment_factor = 0.8
                action = 'reduce'
            # Eğer pozisyon riski çok düşükse artır
            elif current_risk < 0.01:  # %1'den az
                adjustment_factor = 1.1
                action = 'increase'
            else:
                adjustment_factor = 1.0
                action = 'hold'
            
            adjustments[symbol] = {
                'action': action,
                'adjustment_factor': adjustment_factor,
                'current_risk': current_risk,
                'suggested_size': position['size'] * adjustment_factor
            }
        
        return adjustments


# Örnek kullanım ve test fonksiyonları
def example_usage():
    """Örnek kullanım"""
    print("=== Akıllı Pozisyon Boyutlandırma Sistemi Örnekleri ===\n")
    
    # Risk parametreleri
    risk_params = RiskParameters(
        max_risk_per_trade=0.02,
        max_portfolio_risk=0.10,
        volatility_target=0.15,
        kelly_fraction=0.25
    )
    
    # PositionSizer oluştur
    sizer = PositionSizer(risk_params)
    
    # Portföy değeri
    portfolio_value = 100000
    
    print("1. Volatilite Hedefleme Örneği:")
    expected_return = 0.10  # %10 beklenen getiri
    current_volatility = 0.20  # %20 mevcut volatilite
    
    vol_position = sizer.volatility_targeting(expected_return, current_volatility, portfolio_value)
    print(f"   Pozisyon Boyutu: {vol_position:.2%} ({vol_position * portfolio_value:,.0f} TL)")
    print(f"   Beklenen Risk: {current_volatility * vol_position:.2%}")
    print()
    
    print("2. Kelly Kriteri Örneği:")
    win_rate = 0.60  # %60 kazanma oranı
    avg_win = 0.05   # Ortalama %5 kazanç
    avg_loss = 0.03  # Ortalama %3 kayıp
    
    kelly_position = sizer.kelly_criterion(win_rate, avg_win, avg_loss, portfolio_value)
    print(f"   Pozisyon Boyutu: {kelly_position:.2%} ({kelly_position * portfolio_value:,.0f} TL)")
    print(f"   Kelly Oranı: {kelly_position / risk_params.kelly_fraction:.4f}")
    print()
    
    print("3. ATR Bazlı Boyutlandırma Örneği:")
    entry_price = 100
    stop_loss = 95
    atr_value = 2.5
    
    atr_position = sizer.atr_based_sizing(entry_price, stop_loss, atr_value, portfolio_value)
    print(f"   Pozisyon Boyutu: {atr_position:.2%} ({atr_position * portfolio_value:,.0f} TL)")
    print(f"   Risk Mesafesi: {abs(entry_price - stop_loss)}")
    print()
    
    print("4. Dinamik Pozisyon Boyutlandırma Örneği:")
    market_conditions = {
        'volatility': 0.18,
        'trend_strength': 0.3,
        'liquidity': 0.9
    }
    signal_strength = 0.8
    
    historical_performance = {
        'sharpe_ratio': 1.2,
        'win_rate': 0.65
    }
    
    dynamic_position = sizer.dynamic_position_sizing(
        market_conditions, signal_strength, portfolio_value, historical_performance
    )
    print(f"   Pozisyon Boyutu: {dynamic_position:.2%} ({dynamic_position * portfolio_value:,.0f} TL)")
    print()
    
    print("5. Risk Parity Örneği:")
    volatilities = {'BTC': 0.25, 'ETH': 0.30, 'SOL': 0.35}
    correlations = {('BTC', 'ETH'): 0.7, ('BTC', 'SOL'): 0.6, ('ETH', 'SOL'): 0.8}
    total_risk_budget = portfolio_value * 0.10
    
    risk_parity_positions = sizer.risk_parity(volatilities, correlations, total_risk_budget)
    print("   Risk Parity Pozisyonları:")
    for asset, position in risk_parity_positions.items():
        print(f"   {asset}: {position:,.0f} TL ({position/total_risk_budget:.1%})")
    print()
    
    print("6. Portföy Risk Yönetimi Örneği:")
    risk_manager = PortfolioRiskManager(sizer)
    
    # Örnek pozisyonlar ekle
    risk_manager.add_position('BTC', 0.3, 45000, 42000)
    risk_manager.add_position('ETH', 0.2, 3000, 2800)
    risk_manager.add_position('SOL', 0.1, 80, 75)
    
    portfolio_risk = risk_manager.calculate_portfolio_risk()
    print(f"   Toplam Portföy Değeri: {portfolio_risk['total_value']:,.0f} TL")
    print(f"   Toplam Risk: {portfolio_risk['total_risk']:,.0f} TL ({portfolio_risk['total_risk_percentage']:.2%})")
    print(f"   Konsantrasyon Riski: {portfolio_risk['concentration_risk']:.2%}")
    print(f"   Çeşitlendirme Skoru: {portfolio_risk['diversification_score']:.3f}")
    
    # Portföy optimizasyonu
    optimizations = risk_manager.optimize_portfolio()
    print("\n   Önerilen Ayarlamalar:")
    for symbol, optimization in optimizations.items():
        action = optimization['action']
        factor = optimization['adjustment_factor']
        print(f"   {symbol}: {action.upper()} (%{factor:.0f})")


def performance_analysis():
    """Performans analizi"""
    print("\n=== Performans Analizi ===")
    
    # Simüle edilmiş veri
    np.random.seed(42)
    days = 252
    
    # Fiyat serileri
    initial_prices = {'BTC': 45000, 'ETH': 3000, 'SOL': 80}
    returns = {}
    
    for symbol, price in initial_prices.items():
        daily_returns = np.random.normal(0.001, 0.02, days)
        returns[symbol] = pd.Series(daily_returns)
    
    # PositionSizer test
    sizer = PositionSizer()
    portfolio_value = 100000
    
    results = {}
    
    for symbol in initial_prices.keys():
        symbol_returns = returns[symbol]
        
        # Volatilite hedefleme
        current_vol = symbol_returns.tail(20).std() * np.sqrt(252)
        expected_return = symbol_returns.mean() * 252
        
        vol_size = sizer.volatility_targeting(expected_return, current_vol, portfolio_value)
        
        # ATR hesaplama
        # Simüle edilmiş OHLC verisi
        price_series = pd.Series([initial_prices[symbol]] * (days + 1))
        for i in range(1, len(price_series)):
            price_series.iloc[i] = price_series.iloc[i-1] * (1 + returns[symbol].iloc[i-1])
        
        high = price_series * 1.01
        low = price_series * 0.99
        close = price_series
        
        atr = ATRCalculator.calculate_atr(high, low, close)
        current_atr = atr.iloc[-1]
        
        atr_size = sizer.atr_based_sizing(
            initial_prices[symbol], 
            initial_prices[symbol] * 0.97,  # %3 stop loss
            current_atr, 
            portfolio_value
        )
        
        results[symbol] = {
            'volatility_targeting': vol_size,
            'atr_based': atr_size,
            'volatility': current_vol,
            'expected_return': expected_return,
            'atr': current_atr
        }
    
    # Sonuçları yazdır
    for symbol, result in results.items():
        print(f"\n{symbol}:")
        print(f"  Volatilite Hedefleme: {result['volatility_targeting']:.2%}")
        print(f"  ATR Bazlı: {result['atr_based']:.2%}")
        print(f"  Mevcut Volatilite: {result['volatility']:.2%}")
        print(f"  Beklenen Getiri: {result['expected_return']:.2%}")
        print(f"  ATR: {result['atr']:.2f}")


if __name__ == "__main__":
    # Örnekleri çalıştır
    example_usage()
    performance_analysis()
    
    print("\n" + "="*50)
    print("Pozisyon Boyutlandırma Sistemi Başarıyla Yüklendi!")
    print("Sistem şu yöntemleri destekler:")
    print("• Volatilite Hedefleme")
    print("• Kelly Kriteri")
    print("• Fixed Fractional")
    print("• Risk Parity")
    print("• ATR Bazlı Boyutlandırma")
    print("• Dinamik Pozisyon Boyutlandırma")
    print("="*50)
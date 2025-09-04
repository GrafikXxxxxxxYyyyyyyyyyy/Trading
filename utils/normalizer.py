# utils/normalizer.py
import numpy as np



class TradingNormalizer:
    """
    Нормализатор данных для задачи прогнозирования цен.
    Поддерживает различные методы нормализации.
    """
    def __init__(self, method='log_returns'):
        """
        Args:
            method (str): Метод нормализации. Поддерживаемые значения: 'log_returns'.
        """
        self.method = method
        if self.method not in ['log_returns']:
            raise ValueError(f"Unsupported normalization method: {self.method}")
        

    def __call__(self, chunk):
        """
        Применяет нормализацию к чанку данных.
        
        Args:
            chunk (np.ndarray): Входной чанк данных формы [history_len + target_len, 5] (OHLCV).
            
        Returns:
            tuple: (normalized_chunk, stats)
                normalized_chunk (np.ndarray): Нормализованный чанк формы [history_len + target_len, 5].
                stats (dict): Статистики для денормализации, если метод требует.
        """
        if self.method == 'log_returns':
            return self._log_returns_normalize(chunk)
        else:
            # По умолчанию возвращаем исходный чанк без нормализации
            return chunk, {}


    def _log_returns_normalize(self, chunk):
        """
        Нормализует данные с использованием лог-доходностей для цен и z-score для объема.
        
        Args:
            chunk (np.ndarray): Входной чанк данных формы [T, 5] (OHLCV).
            
        Returns:
            tuple: (normalized_chunk, stats)
                normalized_chunk (np.ndarray): Нормализованный чанк формы [T, 5].
                stats (dict): Статистики для денормализации.
        """
        if chunk.shape[1] != 5:
            raise ValueError(f"Expected chunk with 5 columns (OHLCV), got {chunk.shape[1]}")
            
        normalized_chunk = np.zeros_like(chunk, dtype=np.float32)
        stats = {}
        
        # Индексы: 0=Open, 1=High, 2=Low, 3=Close, 4=Volume
        price_indices = [0, 1, 2, 3]  # OHLC
        volume_index = 4              # Volume
        
        # Нормализация цен (O, H, L, C) с использованием лог-доходностей
        for i in price_indices:
            # Вычисляем лог-доходности: ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
            prices = chunk[:, i]
            log_prices = np.log(prices + 1e-8)  # Добавляем маленькое число для избежания log(0)
            
            # Для первой точки используем оригинальное значение (или 0)
            log_returns = np.zeros_like(log_prices)
            if len(log_prices) > 1:
                log_returns[1:] = np.diff(log_prices)  # log(P_t) - log(P_{t-1})
                
            normalized_chunk[:, i] = log_returns
            
            # Сохраняем первую цену для денормализации
            stats[f'first_price_{i}'] = prices[0]
            
        # Нормализация объема по z-score
        volumes = chunk[:, volume_index]
        volume_mean = np.mean(volumes)
        volume_std = np.std(volumes)
        
        # Избегаем деления на ноль
        if volume_std < 1e-8:
            volume_std = 1.0
            
        normalized_volumes = (volumes - volume_mean) / volume_std
        normalized_chunk[:, volume_index] = normalized_volumes
        
        # Сохраняем статистики объема для денормализации
        stats['volume_mean'] = volume_mean
        stats['volume_std'] = volume_std
        
        return normalized_chunk, stats
import numpy as np



class TradingNormalizer:
    """
    Нормализатор данных для задачи прогнозирования цен.
    Поддерживает различные методы нормализации.
    """
    def __init__(self, method='log_returns', history_len=256):
        """
        Args:
            method (str): Метод нормализации. Поддерживаемые значения: 'log_returns'.
            history_len (int): Длина исторических данных в чанке.
        """
        self.method = method
        self.history_len = history_len
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
        Нормализует данные с использованием лог-доходностей для цен,
        а затем приводит лог-доходности к нулевому среднему и единичной дисперсии
        по историческим данным. Объем нормализуется по z-score по всему чанку.
        
        Args:
            chunk (np.ndarray): Входной чанк данных формы [T, 5] (OHLCV).
            
        Returns:
            tuple: (normalized_chunk, stats)
                normalized_chunk (np.ndarray): Нормализованный чанк формы [T, 5].
                stats (dict): Статистики для денормализации.
        """
        if chunk.shape[1] != 5:
            raise ValueError(f"Expected chunk with 5 columns (OHLCV), got {chunk.shape[1]}")
            
        T = chunk.shape[0]
        if T < self.history_len:
             raise ValueError(f"Chunk length {T} is less than history_len {self.history_len}")
            
        normalized_chunk = np.zeros_like(chunk, dtype=np.float32)
        stats = {}
        
        # Индексы: 0=Open, 1=High, 2=Low, 3=Close, 4=Volume
        price_indices = [0, 1, 2, 3]  # OHLC
        volume_index = 4              # Volume
        
        # Нормализация цен (O, H, L, C)
        for i in price_indices:
            # Вычисляем лог-доходности для всего чанка: ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
            prices = chunk[:, i]
            log_prices = np.log(prices + 1e-8)  # Добавляем маленькое число для избежания log(0)
            
            # Для первой точки используем оригинальное значение лог-цены
            log_returns = np.zeros_like(log_prices)
            log_returns[0] = log_prices[0] # Это ln(P_0)
            if len(log_prices) > 1:
                log_returns[1:] = np.diff(log_prices)  # log(P_t) - log(P_{t-1})
                
            # Вычисляем статистики по истории (первые history_len точек)
            history_log_returns = log_returns[:self.history_len] # [history_len]
            mean_history = np.mean(history_log_returns[1:]) # Исключаем первую точку, так как она это ln(P_0)
            std_history = np.std(history_log_returns[1:])   # Исключаем первую точку
            
            # Избегаем деления на ноль
            if std_history < 1e-8:
                std_history = 1.0
                
            # Нормализуем лог-доходности (всё, включая таргет) по статистикам истории
            # Для первой точки оставляем как есть (это ln(P_0)), для остальных нормализуем
            normalized_log_returns = np.copy(log_returns)
            normalized_log_returns[1:] = (log_returns[1:] - mean_history) / std_history
            
            normalized_chunk[:, i] = normalized_log_returns
            
            # Сохраняем статистики для денормализации
            stats[f'first_log_price_{i}'] = log_prices[0] # ln(P_0)
            stats[f'log_return_mean_{i}'] = mean_history
            stats[f'log_return_std_{i}'] = std_history
            
        # Нормализация объема по z-score по всему чанку
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
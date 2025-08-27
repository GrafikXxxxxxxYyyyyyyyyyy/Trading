import torch
import torch.nn.functional as F

class TradingProcessor:
    """
    Обрабатывает исторические торговые данные, добавляя технические индикаторы.
    Преобразует тензор [B, T, 5] (Open, High, Low, Close, Volume) в [B, T, feature_dim].
    """
    def __init__(self, eps=1e-8):
        """
        Args:
            eps (float): Малое значение для избежания деления на ноль.
        """
        self.eps = eps

    def _safe_division(self, numerator, denominator):
        """Безопасное деление с заменой NaN и Inf на 0."""
        # Избегаем деления на ноль, заменяя малые знаменатели
        denominator_safe = torch.where(
            denominator.abs() < self.eps, 
            torch.sign(denominator) * self.eps, 
            denominator
        )
        result = numerator / denominator_safe
        # Заменяем любые оставшиеся NaN или Inf на 0
        result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result

    def _compute_basic_prices(self, open_p, high, low, close):
        """Вычисляет базовые ценовые признаки."""
        features = []
        
        # Самые базовые цены (уже нормализованы)
        features.append(open_p)   # 1
        features.append(high)     # 2
        features.append(low)      # 3
        features.append(close)    # 4
        
        # Типичная цена
        typical_price = (high + low + close) / 3.0 # 5
        features.append(typical_price)
        
        # Взвешенная по близости к закрытию цена
        weighted_close = (high + low + 2 * close) / 4.0 # 6
        features.append(weighted_close)

        return features

    def _compute_price_changes(self, close):
        """Вычисляет абсолютные и относительные изменения цен."""
        features = []
        
        # Абсолютное изменение цены закрытия
        abs_close_change = torch.cat([torch.zeros_like(close[:, :1]), close[:, 1:] - close[:, :-1]], dim=1) # 7
        features.append(abs_close_change)
        
        # Относительное изменение цены закрытия (%)
        rel_close_change = self._safe_division(abs_close_change, torch.roll(close, shifts=1, dims=1)) # 8
        features.append(rel_close_change)

        # Логарифмическое изменение
        log_close_change = torch.log(torch.roll(close, shifts=-1, dims=1) / (close + self.eps) + self.eps) # 9
        # Заполняем последнее значение, так как log(1) = 0
        log_close_change[:, -1] = 0.0
        features.append(log_close_change)

        return features

    def _compute_volatility_indicators(self, high, low, close, window=14):
        """Вычисляет индикаторы волатильности."""
        features = []
        batch_size, seq_len, *_ = close.shape

        # True Range (TR)
        high_low = high - low
        high_close_prev = torch.abs(high - torch.roll(close, shifts=1, dims=1))
        low_close_prev = torch.abs(low - torch.roll(close, shifts=1, dims=1))
        tr = torch.maximum(torch.maximum(high_low, high_close_prev), low_close_prev) # 10
        tr[:, 0] = high_low[:, 0] # Первое значение TR
        features.append(tr)

        # Average True Range (ATR) - простое скользящее среднее
        # Используем свертку с равномерными весами для вычисления SMA
        # Создаем веса для свертки
        weights = torch.ones(window, device=tr.device) / window
        # Применяем свертку с паддингом 'replicate' для сохранения длины
        padded_tr = F.pad(tr.transpose(-1, -2), (window//2, window-1-window//2), mode='replicate') 
        atr = F.conv1d(padded_tr, weights.view(1, 1, -1)) 
        atr = atr.transpose(-1, -2) # 11
        features.append(atr)

        return features

    def _compute_moving_averages(self, close, volumes):
        """Вычисляет различные скользящие средние."""
        features = []
        
        # Экспоненциально взвешенное скользящее среднее (EWMA) для разных периодов
        # Используем приближение EWMA через свертку с экспоненциально убывающими весами
        def compute_ewma_approx(data, span):
            alpha = 2.0 / (span + 1)
            # Создаем веса для свертки
            weights = torch.pow(1 - alpha, torch.arange(span, device=data.device)).flip(0)
            weights = weights / weights.sum() # Нормализация
            # Применяем свертку
            # [B, 1, T] -> [B, 1, T]
            padded_data = F.pad(data.transpose(-1, -2), (span-1, 0), mode='replicate') 
            ewma = F.conv1d(padded_data, weights.view(1, 1, -1)) 
            return ewma.transpose(-1, -2)

        # Закрываем цены для расчета MA
        close_t = close.transpose(-1, -2) # [B, 1, T]
        volumes_t = volumes.transpose(-1, -2) # [B, 1, T]

        # SMA
        def compute_sma(data_t, window):
            weights = torch.ones(window, device=data_t.device) / window
            pad_left = window // 2
            pad_right = window - 1 - pad_left
            padded_data = F.pad(data_t, (pad_left, pad_right), mode='replicate')
            return F.conv1d(padded_data, weights.view(1, 1, -1)).transpose(-1, -2)

        sma_5 = compute_sma(close_t, 5) # 12
        sma_15 = compute_sma(close_t, 15) # 13
        sma_30 = compute_sma(close_t, 30) # 14
        features.extend([sma_5, sma_15, sma_30])

        # EWMA (приблизительные)
        ewma_5 = compute_ewma_approx(close, 5) # 15
        ewma_15 = compute_ewma_approx(close, 15) # 16
        ewma_30 = compute_ewma_approx(close, 30) # 17
        features.extend([ewma_5, ewma_15, ewma_30])

        # VWAP (Volume Weighted Average Price) - упрощенный расчет на основе доступных данных
        # Используем типичную цену как цену для VWAP
        typical_price = (close + close + close) / 3.0 # Упрощение, можно использовать (H+L+C)/3 из других фич
        # vwap_numerator = F.conv1d((typical_price * volumes).transpose(-1, -2), torch.ones(20, device=close.device).view(1, 1, -1), padding=9) 
        # vwap_denominator = F.conv1d(volumes_t, torch.ones(20, device=close.device).view(1, 1, -1), padding=9)
        # Используем правильный паддинг
        tp_volumes_t = (typical_price * volumes).transpose(-1, -2)
        weights_20 = torch.ones(20, device=close.device) / 20.0
        padded_tp_volumes = F.pad(tp_volumes_t, (9, 10), mode='replicate') # padding=9 для окна 20
        padded_volumes_for_vwap = F.pad(volumes_t, (9, 10), mode='replicate')
        vwap_numerator = F.conv1d(padded_tp_volumes, weights_20.view(1, 1, -1))
        vwap_denominator = F.conv1d(padded_volumes_for_vwap, weights_20.view(1, 1, -1))
        vwap = self._safe_division(vwap_numerator, vwap_denominator).transpose(-1, -2) # 18
        features.append(vwap)

        return features

    def _compute_momentum_indicators(self, close, window=14):
        """Вычисляет индикаторы импульса."""
        features = []
        
        # Rate of Change (ROC)
        roc = self._safe_division(close - torch.roll(close, shifts=window, dims=1), torch.roll(close, shifts=window, dims=1)) # 19
        roc[:, :window] = 0.0 # Заполняем начальные значения
        features.append(roc)

        # William's %R
        # Используем свертку для нахождения минимума и максимума
        def rolling_min_max(data_t, window):
            # Max pooling
            pad_left = window // 2
            pad_right = window - 1 - pad_left
            padded_data_max = F.pad(data_t, (pad_left, pad_right), mode='replicate')
            max_vals = F.max_pool1d(padded_data_max, kernel_size=window, stride=1)
            
            # Min pooling (через max от отрицательных значений)
            padded_data_min = F.pad(-data_t, (pad_left, pad_right), mode='replicate')
            min_vals = -F.max_pool1d(padded_data_min, kernel_size=window, stride=1)
            
            return min_vals.transpose(-1, -2), max_vals.transpose(-1, -2)

        lowest_low, highest_high = rolling_min_max(-close.transpose(-1, -2), window)
        williams_r_numerator = highest_high - close
        williams_r_denominator = highest_high - lowest_low
        williams_r = -100 * self._safe_division(williams_r_numerator, williams_r_denominator) # 20
        features.append(williams_r)

        return features

    def _compute_oscillators(self, close, high, low, volumes, window=14):
        """Вычисляет осцилляторы."""
        features = []
        batch_size, seq_len, *_ = close.shape

        # RSI
        delta = close - torch.roll(close, shifts=1, dims=1)
        delta[:, 0] = 0
        gain = delta.clamp(min=0)
        loss = -delta.clamp(max=0)

        # Используем EMA для сглаживания
        def rma(series, n):
            alpha = 1.0 / n
            weights = torch.pow(1 - alpha, torch.arange(n, device=series.device)).flip(0)
            weights = weights / weights.sum()
            padded_series = F.pad(series.transpose(-1, -2), (n-1, 0), mode='replicate')
            return F.conv1d(padded_series, weights.view(1, 1, -1)).transpose(-1, -2)

        avg_gain = rma(gain, window) # 21
        avg_loss = rma(loss, window) # 22
        rs = self._safe_division(avg_gain, avg_loss)
        rsi = 100 - (100 / (1 + rs)) # 23
        features.extend([avg_gain, avg_loss, rsi])

        # Stochastic Oscillator (%K и %D)
        def rolling_min_max_stoch(data_t, window):
            pad_left = window // 2
            pad_right = window - 1 - pad_left
            padded_data_max = F.pad(data_t, (pad_left, pad_right), mode='replicate')
            max_vals = F.max_pool1d(padded_data_max, kernel_size=window, stride=1)
            
            padded_data_min = F.pad(-data_t, (pad_left, pad_right), mode='replicate')
            min_vals = -F.max_pool1d(padded_data_min, kernel_size=window, stride=1)
            
            return min_vals.transpose(-1, -2), max_vals.transpose(-1, -2)

        low_min, high_max = rolling_min_max_stoch(-low.transpose(-1, -2), window)
        stoch_k_numerator = close - low_min
        stoch_k_denominator = high_max - low_min
        stoch_k = 100 * self._safe_division(stoch_k_numerator, stoch_k_denominator) # 24
        # %D - это 3-периодное SMA от %K
        # stoch_d = F.avg_pool1d(stoch_k.transpose(-1, -2), kernel_size=3, stride=1, padding=1, count_include_pad=True).transpose(-1, -2) # 25
        # Используем свертку вместо avg_pool1d для %D
        weights_3 = torch.ones(3, device=stoch_k.device) / 3.0
        padded_stoch_k = F.pad(stoch_k.transpose(-1, -2), (1, 1), mode='replicate') # padding=1 для окна 3
        stoch_d = F.conv1d(padded_stoch_k, weights_3.view(1, 1, -1)).transpose(-1, -2) # 25
        features.extend([stoch_k, stoch_d])

        # Chaikin Oscillator (упрощенный)
        # Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
        # Money Flow Volume = Money Flow Multiplier * Volume
        # ADL = cumsum(Money Flow Volume)
        # Chaikin Osc = EMA_3(ADL) - EMA_10(ADL)
        # Для простоты возьмем разницу между двумя SMA объемов
        # vol_sma_short = F.avg_pool1d(volumes.transpose(-1, -2), kernel_size=3, stride=1, padding=1, count_include_pad=True).transpose(-1, -2)
        # vol_sma_long = F.avg_pool1d(volumes.transpose(-1, -2), kernel_size=10, stride=1, padding=5, count_include_pad=True).transpose(-1, -2)
        # chaikin_osc = vol_sma_short - vol_sma_long # 26
        # Используем свертки вместо avg_pool1d для Chaikin Osc
        def compute_sma_conv(data_t, window):
            weights = torch.ones(window, device=data_t.device) / window
            pad_left = window // 2
            pad_right = window - 1 - pad_left
            padded_data = F.pad(data_t, (pad_left, pad_right), mode='replicate')
            return F.conv1d(padded_data, weights.view(1, 1, -1))
        
        vol_sma_short_t = compute_sma_conv(volumes.transpose(-1, -2), 3) # [B, 1, T]
        vol_sma_long_t = compute_sma_conv(volumes.transpose(-1, -2), 10) # [B, 1, T]
        chaikin_osc_t = vol_sma_short_t - vol_sma_long_t # [B, 1, T]
        chaikin_osc = chaikin_osc_t.transpose(-1, -2) # [B, T, 1] # 26
        features.append(chaikin_osc)

        return features

    def _compute_volume_indicators(self, volumes, close):
        """Вычисляет индикаторы, основанные на объеме."""
        features = []
        
        # Отношение объема к среднему объему
        # avg_volume_20 = F.avg_pool1d(volumes.transpose(-1, -2), kernel_size=20, stride=1, padding=10, count_include_pad=True).transpose(-1, -2) # 27
        # Используем свертку вместо avg_pool1d
        weights_20 = torch.ones(20, device=volumes.device) / 20.0
        padded_volumes = F.pad(volumes.transpose(-1, -2), (10, 9), mode='replicate') # padding=10 для окна 20 (чтобы сохранить длину)
        avg_volume_20_t = F.conv1d(padded_volumes, weights_20.view(1, 1, -1)) # [B, 1, T]
        avg_volume_20 = avg_volume_20_t.transpose(-1, -2) # [B, T, 1] # 27
        
        volume_ratio = self._safe_division(volumes, avg_volume_20) # 28
        features.extend([avg_volume_20, volume_ratio])

        # Volume Price Trend (VPT) - упрощенная версия
        # VPT = VPT_prev + (Close_change / Close_prev) * Volume
        close_change = close - torch.roll(close, shifts=1, dims=1)
        close_change_pct = self._safe_division(close_change, torch.roll(close, shifts=1, dims=1))
        vpt_component = close_change_pct * volumes
        vpt = torch.cumsum(vpt_component, dim=1) # 29
        features.append(vpt)

        return features

    def __call__(self, history_prices):
        """
        Предобрабатывает историю цен, добавляя к ним осцилляторы в качестве новых фичей.
        
        Args:
            history_prices: torch.Tensor of shape [batch_size, seq_len, 5] 
                            (Open, High, Low, Close, Volume)
            
        Returns:
            torch.Tensor of shape [batch_size, seq_len, num_features]
        """
        if not isinstance(history_prices, torch.Tensor):
            raise TypeError("Input history_prices must be a torch.Tensor")

        # Проверка на NaN во входных данных
        if torch.isnan(history_prices).any():
            # Заменяем NaN на 0 во входных данных
            history_prices = torch.nan_to_num(history_prices, nan=0.0)

        # Извлекаем отдельные компоненты
        open_p = history_prices[..., 0:1]   # [B, T, 1]
        high = history_prices[..., 1:2]     # [B, T, 1]
        low = history_prices[..., 2:3]      # [B, T, 1]
        close = history_prices[..., 3:4]    # [B, T, 1]
        volumes = history_prices[..., 4:5]  # [B, T, 1]

        all_features = []

        # 1. Базовые цены
        all_features.extend(self._compute_basic_prices(open_p, high, low, close))

        # 2. Изменения цен
        all_features.extend(self._compute_price_changes(close))

        # 3. Волатильность
        all_features.extend(self._compute_volatility_indicators(high, low, close))

        # 4. Скользящие средние
        all_features.extend(self._compute_moving_averages(close, volumes))

        # 5. Импульс
        all_features.extend(self._compute_momentum_indicators(close))

        # 6. Осцилляторы
        all_features.extend(self._compute_oscillators(close, high, low, volumes))

        # 7. Индикаторы объема
        all_features.extend(self._compute_volume_indicators(volumes, close))

        # Объединяем все признаки
        # Проверяем, что все тензоры имеют одинаковую длину по временной оси
        seq_len = close.shape[1]
        aligned_features = []
        for feat in all_features:
            if feat.shape[1] != seq_len:
                # Если длина не совпадает (из-за паддинга в свертках/пулингах), обрезаем или дополняем
                if feat.shape[1] > seq_len:
                    # Обрезаем
                    feat = feat[:, :seq_len, :]
                else:
                    # Дополняем NaN и затем 0
                    pad_len = seq_len - feat.shape[1]
                    padding = torch.full((feat.shape[0], pad_len, feat.shape[2]), float('nan'), device=feat.device)
                    feat = torch.cat([padding, feat], dim=1)
            # После всех операций убедимся, что нет NaN
            feat = torch.nan_to_num(feat, nan=0.0)
            aligned_features.append(feat)

        # Конкатенируем по последней размерности
        processed_data = torch.cat(aligned_features, dim=-1) # [B, T, num_features]
        
        return processed_data
    
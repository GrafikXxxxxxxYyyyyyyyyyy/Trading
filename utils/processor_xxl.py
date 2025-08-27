# utils/processor_xxl.py
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

        # Цена медианы
        median_price = (high + low) / 2.0 # 7
        features.append(median_price)

        # Body (тело свечи)
        body = torch.abs(close - open_p) # 8
        features.append(body)

        # Upper Shadow (верхняя тень)
        upper_shadow = high - torch.maximum(open_p, close) # 9
        features.append(upper_shadow)

        # Lower Shadow (нижняя тень)
        lower_shadow = torch.minimum(open_p, close) - low # 10
        features.append(lower_shadow)

        # Цена открытия предыдущего дня
        prev_open = torch.roll(open_p, shifts=1, dims=1) # 11
        prev_open[:, 0] = open_p[:, 0]
        features.append(prev_open)

        # Цена закрытия предыдущего дня
        prev_close = torch.roll(close, shifts=1, dims=1) # 12
        prev_close[:, 0] = close[:, 0]
        features.append(prev_close)

        # Цена открытия 2 дня назад
        prev_open_2 = torch.roll(open_p, shifts=2, dims=1) # 13
        prev_open_2[:, :2] = open_p[:, :2]
        features.append(prev_open_2)

        # Цена закрытия 2 дня назад
        prev_close_2 = torch.roll(close, shifts=2, dims=1) # 14
        prev_close_2[:, :2] = close[:, :2]
        features.append(prev_close_2)

        # Цена открытия 3 дня назад
        prev_open_3 = torch.roll(open_p, shifts=3, dims=1) # 15
        prev_open_3[:, :3] = open_p[:, :3]
        features.append(prev_open_3)

        # Цена закрытия 3 дня назад
        prev_close_3 = torch.roll(close, shifts=3, dims=1) # 16
        prev_close_3[:, :3] = close[:, :3]
        features.append(prev_close_3)

        return features

    def _compute_price_changes(self, close):
        """Вычисляет абсолютные и относительные изменения цен."""
        features = []
        
        # Абсолютное изменение цены закрытия
        abs_close_change = torch.cat([torch.zeros_like(close[:, :1]), close[:, 1:] - close[:, :-1]], dim=1) # 17
        features.append(abs_close_change)
        
        # Относительное изменение цены закрытия (%)
        rel_close_change = self._safe_division(abs_close_change, torch.roll(close, shifts=1, dims=1)) # 18
        features.append(rel_close_change)

        # Логарифмическое изменение
        log_close_change = torch.log(torch.roll(close, shifts=-1, dims=1) / (close + self.eps) + self.eps) # 19
        # Заполняем последнее значение, так как log(1) = 0
        log_close_change[:, -1] = 0.0
        features.append(log_close_change)

        # Процентное изменение за 2 дня
        close_change_2 = self._safe_division(close - torch.roll(close, shifts=2, dims=1), torch.roll(close, shifts=2, dims=1)) # 20
        close_change_2[:, :2] = 0.0
        features.append(close_change_2)

        # Процентное изменение за 3 дня
        close_change_3 = self._safe_division(close - torch.roll(close, shifts=3, dims=1), torch.roll(close, shifts=3, dims=1)) # 21
        close_change_3[:, :3] = 0.0
        features.append(close_change_3)

        # Процентное изменение за 5 дней
        close_change_5 = self._safe_division(close - torch.roll(close, shifts=5, dims=1), torch.roll(close, shifts=5, dims=1)) # 22
        close_change_5[:, :5] = 0.0
        features.append(close_change_5)

        # Процентное изменение за 10 дней
        close_change_10 = self._safe_division(close - torch.roll(close, shifts=10, dims=1), torch.roll(close, shifts=10, dims=1)) # 23
        close_change_10[:, :10] = 0.0
        features.append(close_change_10)

        return features

    def _compute_volatility_indicators(self, high, low, close, window=14):
        """Вычисляет индикаторы волатильности."""
        features = []
        batch_size, seq_len, *_ = close.shape

        # True Range (TR)
        high_low = high - low
        high_close_prev = torch.abs(high - torch.roll(close, shifts=1, dims=1))
        low_close_prev = torch.abs(low - torch.roll(close, shifts=1, dims=1))
        tr = torch.maximum(torch.maximum(high_low, high_close_prev), low_close_prev) # 24
        tr[:, 0] = high_low[:, 0] # Первое значение TR
        features.append(tr)

        # Average True Range (ATR) - простое скользящее среднее
        # Используем свертку с равномерными весами для вычисления SMA
        # Создаем веса для свертки
        weights = torch.ones(window, device=tr.device) / window
        # Применяем свертку с паддингом 'replicate' для сохранения длины
        padded_tr = F.pad(tr.transpose(-1, -2), (window//2, window-1-window//2), mode='replicate') 
        atr = F.conv1d(padded_tr, weights.view(1, 1, -1)) 
        atr = atr.transpose(-1, -2) # 25
        features.append(atr)

        # Volatility (стандартное отклонение цен закрытия)
        def compute_std(data_t, window):
            # Вычисляем скользящее среднее
            weights_mean = torch.ones(window, device=data_t.device) / window
            pad_left_mean = window // 2
            pad_right_mean = window - 1 - pad_left_mean
            padded_data_mean = F.pad(data_t, (pad_left_mean, pad_right_mean), mode='replicate')
            mean_vals = F.conv1d(padded_data_mean, weights_mean.view(1, 1, -1))
            
            # Вычисляем (x - mean)^2
            diff_sq = (data_t - mean_vals) ** 2
            
            # Вычисляем скользящее среднее (x - mean)^2
            padded_diff_sq = F.pad(diff_sq, (pad_left_mean, pad_right_mean), mode='replicate')
            var_vals = F.conv1d(padded_diff_sq, weights_mean.view(1, 1, -1))
            
            # Стандартное отклонение
            std_vals = torch.sqrt(var_vals + self.eps)
            return std_vals.transpose(-1, -2)
        
        volatility = compute_std(close.transpose(-1, -2), window) # 26
        features.append(volatility)

        # Bollinger Bands
        def compute_bollinger_bands(data_t, window, num_std=2):
            # Вычисляем скользящее среднее
            weights_mean = torch.ones(window, device=data_t.device) / window
            pad_left_mean = window // 2
            pad_right_mean = window - 1 - pad_left_mean
            padded_data_mean = F.pad(data_t, (pad_left_mean, pad_right_mean), mode='replicate')
            mean_vals = F.conv1d(padded_data_mean, weights_mean.view(1, 1, -1))
            
            # Вычисляем (x - mean)^2
            diff_sq = (data_t - mean_vals) ** 2
            
            # Вычисляем скользящее среднее (x - mean)^2
            padded_diff_sq = F.pad(diff_sq, (pad_left_mean, pad_right_mean), mode='replicate')
            var_vals = F.conv1d(padded_diff_sq, weights_mean.view(1, 1, -1))
            
            # Стандартное отклонение
            std_vals = torch.sqrt(var_vals + self.eps)
            
            upper_band = mean_vals + num_std * std_vals
            lower_band = mean_vals - num_std * std_vals
            
            return mean_vals.transpose(-1, -2), upper_band.transpose(-1, -2), lower_band.transpose(-1, -2)
        
        bb_mean, bb_upper, bb_lower = compute_bollinger_bands(close.transpose(-1, -2), window) # 27, 28, 29
        bb_width = bb_upper - bb_lower # 30
        bb_position = self._safe_division(close - bb_lower, bb_width) # 31
        features.extend([bb_mean, bb_upper, bb_lower, bb_width, bb_position])

        # Keltner Channels
        def compute_keltner_channels(close_t, high_t, low_t, window, atr_t):
            # Вычисляем EMA цены закрытия
            alpha = 2.0 / (window + 1)
            weights = torch.pow(1 - alpha, torch.arange(window, device=close_t.device)).flip(0)
            weights = weights / weights.sum()
            padded_close = F.pad(close_t, (window-1, 0), mode='replicate')
            ema_close = F.conv1d(padded_close, weights.view(1, 1, -1))
            
            # Keltner Channels
            keltner_upper = ema_close + 2 * atr_t.transpose(-1, -2)
            keltner_lower = ema_close - 2 * atr_t.transpose(-1, -2)
            
            return ema_close.transpose(-1, -2), keltner_upper.transpose(-1, -2), keltner_lower.transpose(-1, -2)
        
        keltner_mean, keltner_upper, keltner_lower = compute_keltner_channels(
            close.transpose(-1, -2), high.transpose(-1, -2), low.transpose(-1, -2), window, atr
        ) # 32, 33, 34
        features.extend([keltner_mean, keltner_upper, keltner_lower])

        # Donchian Channels
        def compute_donchian_channels(high_t, low_t, window):
            pad_left = window // 2
            pad_right = window - 1 - pad_left
            padded_high = F.pad(high_t, (pad_left, pad_right), mode='replicate')
            padded_low = F.pad(low_t, (pad_left, pad_right), mode='replicate')
            
            # Max pooling для Highest High
            hh_vals = F.max_pool1d(padded_high, kernel_size=window, stride=1)
            
            # Min pooling для Lowest Low (через max от отрицательных значений)
            ll_vals = -F.max_pool1d(-padded_low, kernel_size=window, stride=1)
            
            return hh_vals.transpose(-1, -2), ll_vals.transpose(-1, -2)
        
        donchian_upper, donchian_lower = compute_donchian_channels(high.transpose(-1, -2), low.transpose(-1, -2), window) # 35, 36
        donchian_middle = (donchian_upper + donchian_lower) / 2.0 # 37
        features.extend([donchian_upper, donchian_lower, donchian_middle])

        # Ulcer Index
        def compute_ulcer_index(close_t, window=14):
            # Вычисляем максимальную цену за период
            pad_left = window // 2
            pad_right = window - 1 - pad_left
            padded_close = F.pad(close_t, (pad_left, pad_right), mode='replicate')
            
            # Max pooling для Highest Close
            max_close_vals = F.max_pool1d(padded_close, kernel_size=window, stride=1)
            max_close = max_close_vals # Уже в формате [B, 1, T]
            
            # Процентное падение от максимума
            # close_t уже в формате [B, 1, T]
            pct_drawdown = self._safe_division(close_t - max_close, max_close) * 100
            
            # Квадрат падения
            drawdown_squared = pct_drawdown ** 2
            
            # Скользящее среднее квадратов падений
            weights_mean = torch.ones(window, device=drawdown_squared.device) / window
            pad_left_mean = window // 2
            pad_right_mean = window - 1 - pad_left_mean
            # Убедимся, что drawdown_squared в правильном формате [B, 1, T]
            if drawdown_squared.dim() == 3 and drawdown_squared.shape[1] != 1:
                # Если это [B, T, 1], транспонируем
                drawdown_squared_input = drawdown_squared.transpose(-1, -2)
            elif drawdown_squared.dim() == 2:
                # Если это [B, T], добавим размерность канала
                drawdown_squared_input = drawdown_squared.unsqueeze(1)
            else:
                drawdown_squared_input = drawdown_squared
                
            padded_drawdown_squared = F.pad(drawdown_squared_input, (pad_left_mean, pad_right_mean), mode='replicate')
            mean_drawdown_squared = F.conv1d(padded_drawdown_squared, weights_mean.view(1, 1, -1))
            
            # Квадратный корень из среднего
            ulcer_index = torch.sqrt(mean_drawdown_squared + self.eps) # 38
            return ulcer_index.transpose(-1, -2)
        
        ulcer_index = compute_ulcer_index(close.transpose(-1, -2)) # 38
        features.append(ulcer_index)

        return features

    def _compute_moving_averages(self, close, volumes, open_p, high, low):
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
        open_t = open_p.transpose(-1, -2) # [B, 1, T]
        high_t = high.transpose(-1, -2) # [B, 1, T]
        low_t = low.transpose(-1, -2) # [B, 1, T]

        # SMA
        def compute_sma(data_t, window):
            weights = torch.ones(window, device=data_t.device) / window
            pad_left = window // 2
            pad_right = window - 1 - pad_left
            padded_data = F.pad(data_t, (pad_left, pad_right), mode='replicate')
            return F.conv1d(padded_data, weights.view(1, 1, -1)).transpose(-1, -2)

        sma_3 = compute_sma(close_t, 3) # 39
        sma_5 = compute_sma(close_t, 5) # 40
        sma_7 = compute_sma(close_t, 7) # 41
        sma_10 = compute_sma(close_t, 10) # 42
        sma_15 = compute_sma(close_t, 15) # 43
        sma_20 = compute_sma(close_t, 20) # 44
        sma_30 = compute_sma(close_t, 30) # 45
        sma_50 = compute_sma(close_t, 50) # 46
        features.extend([sma_3, sma_5, sma_7, sma_10, sma_15, sma_20, sma_30, sma_50])

        # EWMA (приблизительные)
        ewma_3 = compute_ewma_approx(close, 3) # 47
        ewma_5 = compute_ewma_approx(close, 5) # 48
        ewma_7 = compute_ewma_approx(close, 7) # 49
        ewma_10 = compute_ewma_approx(close, 10) # 50
        ewma_15 = compute_ewma_approx(close, 15) # 51
        ewma_20 = compute_ewma_approx(close, 20) # 52
        ewma_30 = compute_ewma_approx(close, 30) # 53
        ewma_50 = compute_ewma_approx(close, 50) # 54
        features.extend([ewma_3, ewma_5, ewma_7, ewma_10, ewma_15, ewma_20, ewma_30, ewma_50])

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
        vwap = self._safe_division(vwap_numerator, vwap_denominator).transpose(-1, -2) # 55
        features.append(vwap)

        # MACD
        ema_12 = compute_ewma_approx(close, 12)
        ema_26 = compute_ewma_approx(close, 26)
        macd_line = ema_12 - ema_26 # 56
        signal_line = compute_ewma_approx(macd_line, 9) # 57
        macd_histogram = macd_line - signal_line # 58
        features.extend([macd_line, signal_line, macd_histogram])

        # Price Channels (Highest High и Lowest Low)
        def compute_price_channels(high_t, low_t, window):
            pad_left = window // 2
            pad_right = window - 1 - pad_left
            padded_high = F.pad(high_t, (pad_left, pad_right), mode='replicate')
            padded_low = F.pad(low_t, (pad_left, pad_right), mode='replicate')
            
            # Max pooling для Highest High
            hh_vals = F.max_pool1d(padded_high, kernel_size=window, stride=1)
            
            # Min pooling для Lowest Low (через max от отрицательных значений)
            ll_vals = -F.max_pool1d(-padded_low, kernel_size=window, stride=1)
            
            return hh_vals.transpose(-1, -2), ll_vals.transpose(-1, -2)
        
        hh_20, ll_20 = compute_price_channels(high_t, low_t, 20) # 59, 60
        hh_50, ll_50 = compute_price_channels(high_t, low_t, 50) # 61, 62
        features.extend([hh_20, ll_20, hh_50, ll_50])

        return features

    def _compute_momentum_indicators(self, close, window=14):
        """Вычисляет индикаторы импульса."""
        features = []
        
        # Rate of Change (ROC)
        roc = self._safe_division(close - torch.roll(close, shifts=window, dims=1), torch.roll(close, shifts=window, dims=1)) # 63
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
        williams_r = -100 * self._safe_division(williams_r_numerator, williams_r_denominator) # 64
        features.append(williams_r)

        # Momentum
        momentum = close - torch.roll(close, shifts=window, dims=1) # 65
        momentum[:, :window] = 0.0
        features.append(momentum)

        # Price Rate of Change (PROC) - нормализованная версия Momentum
        proc = self._safe_division(momentum, torch.roll(close, shifts=window, dims=1)) # 66
        features.append(proc)

        # Rate of Change (ROC) для 3, 5 и 20 дней
        roc_3 = self._safe_division(close - torch.roll(close, shifts=3, dims=1), torch.roll(close, shifts=3, dims=1)) # 67
        roc_3[:, :3] = 0.0
        features.append(roc_3)

        roc_5 = self._safe_division(close - torch.roll(close, shifts=5, dims=1), torch.roll(close, shifts=5, dims=1)) # 68
        roc_5[:, :5] = 0.0
        features.append(roc_5)

        roc_20 = self._safe_division(close - torch.roll(close, shifts=20, dims=1), torch.roll(close, shifts=20, dims=1)) # 69
        roc_20[:, :20] = 0.0
        features.append(roc_20)

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

        avg_gain = rma(gain, window) # 71
        avg_loss = rma(loss, window) # 72
        rs = self._safe_division(avg_gain, avg_loss)
        rsi = 100 - (100 / (1 + rs)) # 73
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
        stoch_k = 100 * self._safe_division(stoch_k_numerator, stoch_k_denominator) # 74
        # %D - это 3-периодное SMA от %K
        # stoch_d = F.avg_pool1d(stoch_k.transpose(-1, -2), kernel_size=3, stride=1, padding=1, count_include_pad=True).transpose(-1, -2) # 75
        # Используем свертку вместо avg_pool1d для %D
        weights_3 = torch.ones(3, device=stoch_k.device) / 3.0
        padded_stoch_k = F.pad(stoch_k.transpose(-1, -2), (1, 1), mode='replicate') # padding=1 для окна 3
        stoch_d = F.conv1d(padded_stoch_k, weights_3.view(1, 1, -1)).transpose(-1, -2) # 75
        features.extend([stoch_k, stoch_d])

        # Chaikin Oscillator (упрощенный)
        # Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
        # Money Flow Volume = Money Flow Multiplier * Volume
        # ADL = cumsum(Money Flow Volume)
        # Chaikin Osc = EMA_3(ADL) - EMA_10(ADL)
        # Для простоты возьмем разницу между двумя SMA объемов
        # vol_sma_short = F.avg_pool1d(volumes.transpose(-1, -2), kernel_size=3, stride=1, padding=1, count_include_pad=True).transpose(-1, -2)
        # vol_sma_long = F.avg_pool1d(volumes.transpose(-1, -2), kernel_size=10, stride=1, padding=5, count_include_pad=True).transpose(-1, -2)
        # chaikin_osc = vol_sma_short - vol_sma_long # 76
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
        chaikin_osc = chaikin_osc_t.transpose(-1, -2) # [B, T, 1] # 76
        features.append(chaikin_osc)

        # CCI (Commodity Channel Index)
        def compute_cci(high_t, low_t, close_t, window):
            # Typical Price
            tp = (high_t + low_t + close_t) / 3.0
            
            # SMA of TP
            weights = torch.ones(window, device=tp.device) / window
            pad_left = window // 2
            pad_right = window - 1 - pad_left
            padded_tp = F.pad(tp, (pad_left, pad_right), mode='replicate')
            sma_tp = F.conv1d(padded_tp, weights.view(1, 1, -1))
            
            # Mean Deviation
            diff_abs = torch.abs(tp - sma_tp)
            padded_diff_abs = F.pad(diff_abs, (pad_left, pad_right), mode='replicate')
            mean_dev = F.conv1d(padded_diff_abs, weights.view(1, 1, -1))
            
            # CCI
            cci = self._safe_division(tp - sma_tp, 0.015 * mean_dev) # 77
            return cci.transpose(-1, -2)
        
        cci = compute_cci(high.transpose(-1, -2), low.transpose(-1, -2), close.transpose(-1, -2), window)
        features.append(cci)

        # Ultimate Oscillator
        def compute_ultimate_oscillator(close, high, low, short_window=7, medium_window=14, long_window=28):
            buying_pressure = close - torch.minimum(low, torch.roll(close, shifts=1, dims=1))
            buying_pressure[:, 0] = 0
            
            true_range = torch.maximum(
                torch.maximum(high - low, torch.abs(high - torch.roll(close, shifts=1, dims=1))),
                torch.abs(low - torch.roll(close, shifts=1, dims=1))
            )
            true_range[:, 0] = high[:, 0] - low[:, 0]
            
            def avg(bp, tr, n):
                # Простое скользящее среднее
                weights = torch.ones(n, device=bp.device) / n
                pad_left = n // 2
                pad_right = n - 1 - pad_left
                padded_bp = F.pad(bp, (pad_left, pad_right), mode='replicate')
                padded_tr = F.pad(tr, (pad_left, pad_right), mode='replicate')
                bp_avg = F.conv1d(padded_bp, weights.view(1, 1, -1))
                tr_avg = F.conv1d(padded_tr, weights.view(1, 1, -1))
                return bp_avg.transpose(-1, -2), tr_avg.transpose(-1, -2)
            
            bp_short, tr_short = avg(buying_pressure.transpose(-1, -2), true_range.transpose(-1, -2), short_window)
            bp_medium, tr_medium = avg(buying_pressure.transpose(-1, -2), true_range.transpose(-1, -2), medium_window)
            bp_long, tr_long = avg(buying_pressure.transpose(-1, -2), true_range.transpose(-1, -2), long_window)
            
            avg7 = self._safe_division(bp_short, tr_short)
            avg14 = self._safe_division(bp_medium, tr_medium)
            avg28 = self._safe_division(bp_long, tr_long)
            
            ult_osc = 100 * (4*avg7 + 2*avg14 + avg28) / 7
            return ult_osc
        
        ultimate_osc = compute_ultimate_oscillator(close, high, low) # 78
        features.append(ultimate_osc)

        # Triple Smoothed Oscillator (TRIX)
        # ИСПРАВЛЕНИЕ: Передаем self._compute_ewma_approx как аргумент
        def compute_trix(close_t, window, ewma_func):
            # TRIX - это тройная экспоненциально сглаженная скорость изменения
            ema1 = ewma_func(close_t, window)
            ema2 = ewma_func(ema1, window)
            ema3 = ewma_func(ema2, window)
            
            # Процентное изменение EMA3
            trix = self._safe_division(ema3 - torch.roll(ema3, shifts=1, dims=1), torch.roll(ema3, shifts=1, dims=1)) * 100
            trix[:, 0] = 0
            return trix
        
        # ИСПРАВЛЕНИЕ: Передаем self._compute_ewma_approx
        trix = compute_trix(close, 15, self._compute_ewma_approx) # 79
        features.append(trix)

        # Mass Index
        # ИСПРАВЛЕНИЕ: Передаем self._compute_ewma_approx как аргумент
        def compute_mass_index(high_t, low_t, ewma_func, window=25):
            # Вычисляем диапазон
            range_t = high_t - low_t
            
            # Вычисляем EMA диапазона
            ema_range = ewma_func(range_t, 9)
            
            # Вычисляем двойную EMA диапазона
            ema_ema_range = ewma_func(ema_range, 9)
            
            # Вычисляем отношение
            ratio = self._safe_division(ema_range, ema_ema_range)
            
            # Суммируем за последние 25 периодов
            weights = torch.ones(window, device=ratio.device) / window
            pad_left = window // 2
            pad_right = window - 1 - pad_left
            padded_ratio = F.pad(ratio.transpose(-1, -2), (pad_left, pad_right), mode='replicate')
            mass_index = F.conv1d(padded_ratio, weights.view(1, 1, -1)).transpose(-1, -2) * window # 80
            
            return mass_index
        
        # ИСПРАВЛЕНИЕ: Передаем self._compute_ewma_approx
        mass_index = compute_mass_index(high, low, self._compute_ewma_approx) # 80
        features.append(mass_index)

        # Aroon Oscillator
        def compute_aroon_oscillator(high_t, low_t, window=25):
            pad_left = window // 2
            pad_right = window - 1 - pad_left
            
            # Padding для high и low
            padded_high = F.pad(high_t, (pad_left, pad_right), mode='replicate')
            padded_low = F.pad(low_t, (pad_left, pad_right), mode='replicate')
            
            # Max pooling для Highest High и argmax индекса
            max_pool_result = F.max_pool1d_with_indices(padded_high, kernel_size=window, stride=1)
            max_vals = max_pool_result[0]
            max_indices = max_pool_result[1]
            
            # Min pooling для Lowest Low и argmin индекса (через max от отрицательных значений)
            neg_padded_low = -padded_low
            min_pool_result = F.max_pool1d_with_indices(neg_padded_low, kernel_size=window, stride=1)
            min_vals = -min_pool_result[0]
            min_indices = min_pool_result[1]
            
            # Преобразуем индексы в позиции внутри окна
            aroon_up = (window - 1 - (max_indices % window)).float() / (window - 1) * 100 # 81
            aroon_down = (window - 1 - (min_indices % window)).float() / (window - 1) * 100 # 82
            
            aroon_osc = aroon_up.transpose(-1, -2) - aroon_down.transpose(-1, -2) # 83
            return aroon_osc
        
        aroon_osc = compute_aroon_oscillator(high.transpose(-1, -2), low.transpose(-1, -2)) # 81, 82, 83
        features.append(aroon_osc)

        return features

    # ИСПРАВЛЕНИЕ 1: Добавлен параметр window=14
    def _compute_volume_indicators(self, volumes, close, window=14):
        """Вычисляет индикаторы, основанные на объеме."""
        features = []
        
        # Отношение объема к среднему объему
        # avg_volume_20 = F.avg_pool1d(volumes.transpose(-1, -2), kernel_size=20, stride=1, padding=10, count_include_pad=True).transpose(-1, -2) # 84
        # Используем свертку вместо avg_pool1d
        weights_20 = torch.ones(20, device=volumes.device) / 20.0
        padded_volumes = F.pad(volumes.transpose(-1, -2), (10, 9), mode='replicate') # padding=10 для окна 20 (чтобы сохранить длину)
        avg_volume_20_t = F.conv1d(padded_volumes, weights_20.view(1, 1, -1)) # [B, 1, T]
        avg_volume_20 = avg_volume_20_t.transpose(-1, -2) # [B, T, 1] # 84
        
        volume_ratio = self._safe_division(volumes, avg_volume_20) # 85
        features.extend([avg_volume_20, volume_ratio])

        # Volume Price Trend (VPT) - упрощенная версия
        # VPT = VPT_prev + (Close_change / Close_prev) * Volume
        close_change = close - torch.roll(close, shifts=1, dims=1)
        close_change_pct = self._safe_division(close_change, torch.roll(close, shifts=1, dims=1))
        vpt_component = close_change_pct * volumes
        vpt = torch.cumsum(vpt_component, dim=1) # 86
        features.append(vpt)

        # On-Balance Volume (OBV)
        sign_changes = torch.sign(close_change)
        sign_changes[:, 0] = 0  # Нет предыдущего значения для первого элемента
        obv_flow = sign_changes * volumes
        obv = torch.cumsum(obv_flow, dim=1) # 87
        features.append(obv)

        # Volume Rate of Change (VROC)
        # ИСПРАВЛЕНИЕ 2: Теперь переменная window определена
        vroc = self._safe_division(volumes - torch.roll(volumes, shifts=window, dims=1), torch.roll(volumes, shifts=window, dims=1)) # 88
        vroc[:, :window] = 0.0
        features.append(vroc)

        # Volume Oscillator (разница между двумя SMA объема)
        def compute_volume_oscillator(volumes_t, short_window=5, long_window=10):
            # Вычисляем короткое SMA объема
            weights_short = torch.ones(short_window, device=volumes_t.device) / short_window
            pad_left_short = short_window // 2
            pad_right_short = short_window - 1 - pad_left_short
            padded_volumes_short = F.pad(volumes_t, (pad_left_short, pad_right_short), mode='replicate')
            short_sma = F.conv1d(padded_volumes_short, weights_short.view(1, 1, -1))
            
            # Вычисляем длинное SMA объема
            weights_long = torch.ones(long_window, device=volumes_t.device) / long_window
            pad_left_long = long_window // 2
            pad_right_long = long_window - 1 - pad_left_long
            padded_volumes_long = F.pad(volumes_t, (pad_left_long, pad_right_long), mode='replicate')
            long_sma = F.conv1d(padded_volumes_long, weights_long.view(1, 1, -1))
            
            # Volume Oscillator как разница в процентах
            vol_osc = self._safe_division(short_sma - long_sma, long_sma) * 100 # 90
            return vol_osc.transpose(-1, -2)
        
        vol_osc = compute_volume_oscillator(volumes.transpose(-1, -2)) # 90
        features.append(vol_osc)

        return features

    def _compute_advanced_indicators(self, open_p, high, low, close, volumes):
        """Вычисляет дополнительные продвинутые индикаторы."""
        features = []
        
        # Accumulation/Distribution Line (ADL)
        clv = self._safe_division((close - low) - (high - close), high - low) # 91
        adl_flow = clv * volumes
        adl = torch.cumsum(adl_flow, dim=1) # 92
        features.append(adl)

        # Money Flow Index (MFI) - упрощенная версия
        typical_price = (high + low + close) / 3.0
        money_flow = typical_price * volumes
        delta_tp = typical_price - torch.roll(typical_price, shifts=1, dims=1)
        delta_tp[:, 0] = 0
        
        positive_flow = torch.where(delta_tp > 0, money_flow, torch.zeros_like(money_flow))
        negative_flow = torch.where(delta_tp < 0, money_flow, torch.zeros_like(money_flow))
        
        # Суммируем за последние 14 дней
        def sum_window(data_t, window):
            weights = torch.ones(window, device=data_t.device)
            pad_left = window // 2
            pad_right = window - 1 - pad_left
            padded_data = F.pad(data_t, (pad_left, pad_right), mode='replicate')
            return F.conv1d(padded_data, weights.view(1, 1, -1)).transpose(-1, -2)
        
        positive_flow_sum = sum_window(positive_flow.transpose(-1, -2), 14) # 93
        negative_flow_sum = sum_window(negative_flow.transpose(-1, -2), 14) # 94
        
        money_ratio = self._safe_division(positive_flow_sum, negative_flow_sum)
        mfi = 100 - (100 / (1 + money_ratio)) # 95
        features.extend([positive_flow_sum, negative_flow_sum, mfi])

        # Price Volume Trend (PVT)
        close_change_pct = self._safe_division(close - torch.roll(close, shifts=1, dims=1), torch.roll(close, shifts=1, dims=1))
        close_change_pct[:, 0] = 0
        pvt_flow = close_change_pct * volumes
        pvt = torch.cumsum(pvt_flow, dim=1) # 96
        features.append(pvt)

        # Force Index
        force_index = (close - torch.roll(close, shifts=1, dims=1)) * volumes # 97
        force_index[:, 0] = 0
        # Сглаживаем с помощью EMA
        force_index_ema = self._compute_ewma_approx(force_index, 13) # 98
        features.extend([force_index, force_index_ema])

        # Elder's Force Index (EFI) - это то же самое, что и Force Index
        # Elder's Ray (Bull Power и Bear Power)
        def compute_ema(data, span):
            alpha = 2.0 / (span + 1)
            weights = torch.pow(1 - alpha, torch.arange(span, device=data.device)).flip(0)
            weights = weights / weights.sum()
            padded_data = F.pad(data.transpose(-1, -2), (span-1, 0), mode='replicate') 
            ema = F.conv1d(padded_data, weights.view(1, 1, -1)) 
            return ema.transpose(-1, -2)
        
        ema_13 = compute_ema(close, 13) # 99
        bull_power = high - ema_13 # 100
        bear_power = low - ema_13 # 101
        features.extend([ema_13, bull_power, bear_power])

        # Detrended Price Oscillator (DPO)
        # ИСПРАВЛЕНИЕ: Передаем self._compute_sma как аргумент
        def compute_dpo(close_t, window, sma_func):
            # Вычисляем SMA
            sma = sma_func(close_t.transpose(-1, -2), window).transpose(-1, -2)
            
            # Сдвигаем SMA на период/2 + 1
            shift = window // 2 + 1
            shifted_sma = torch.roll(sma, shifts=-shift, dims=1)
            
            # DPO = Цена - Сдвинутая SMA
            dpo = close_t - shifted_sma # 102
            return dpo
        
        # ИСПРАВЛЕНИЕ: Передаем self._compute_sma
        dpo = compute_dpo(close, 20, self._compute_sma) # 102
        features.append(dpo)

        # Vortex Indicator
        def compute_vortex(high_t, low_t, close_t, window=14):
            # True Range
            tr = torch.maximum(
                torch.maximum(high_t - low_t, torch.abs(high_t - torch.roll(close_t, shifts=1, dims=1))),
                torch.abs(low_t - torch.roll(close_t, shifts=1, dims=1))
            )
            tr[:, 0] = high_t[:, 0] - low_t[:, 0]
            
            # VM+ и VM-
            vm_plus = torch.abs(high_t - torch.roll(low_t, shifts=1, dims=1))
            vm_minus = torch.abs(low_t - torch.roll(high_t, shifts=1, dims=1))
            
            # Суммируем за период
            weights = torch.ones(window, device=tr.device) / window
            pad_left = window // 2
            pad_right = window - 1 - pad_left
            
            padded_tr = F.pad(tr.transpose(-1, -2), (pad_left, pad_right), mode='replicate')
            padded_vm_plus = F.pad(vm_plus.transpose(-1, -2), (pad_left, pad_right), mode='replicate')
            padded_vm_minus = F.pad(vm_minus.transpose(-1, -2), (pad_left, pad_right), mode='replicate')
            
            tr_sum = F.conv1d(padded_tr, weights.view(1, 1, -1)).transpose(-1, -2) * window
            vm_plus_sum = F.conv1d(padded_vm_plus, weights.view(1, 1, -1)).transpose(-1, -2) * window
            vm_minus_sum = F.conv1d(padded_vm_minus, weights.view(1, 1, -1)).transpose(-1, -2) * window
            
            # Vortex Indicator
            vi_plus = self._safe_division(vm_plus_sum, tr_sum) # 103
            vi_minus = self._safe_division(vm_minus_sum, tr_sum) # 104
            
            return vi_plus, vi_minus
        
        vi_plus, vi_minus = compute_vortex(high, low, close) # 103, 104
        features.extend([vi_plus, vi_minus])

        # Coppock Curve
        def compute_coppock_curve(close_t, short_window=11, long_window=14):
            # Вычисляем ROC для короткого и длинного периодов
            roc_short = self._safe_division(close_t - torch.roll(close_t, shifts=short_window, dims=1), torch.roll(close_t, shifts=short_window, dims=1))
            roc_short[:, :short_window] = 0.0
            
            roc_long = self._safe_division(close_t - torch.roll(close_t, shifts=long_window, dims=1), torch.roll(close_t, shifts=long_window, dims=1))
            roc_long[:, :long_window] = 0.0
            
            # Сумма ROC
            roc_sum = roc_short + roc_long
            
            # WMA (Weighted Moving Average) суммы ROC
            # Для упрощения используем EMA вместо WMA
            coppock = self._compute_ewma_approx(roc_sum, 10) # 105
            return coppock
        
        coppock = compute_coppock_curve(close) # 105
        features.append(coppock)

        # Chande Momentum Oscillator (CMO)
        def compute_cmo(close_t, window=14):
            delta = close_t - torch.roll(close_t, shifts=1, dims=1)
            delta[:, 0] = 0
            
            # Суммируем положительные и отрицательные изменения
            up = delta.clamp(min=0)
            down = -delta.clamp(max=0)
            
            # Вычисляем скользящие суммы
            weights = torch.ones(window, device=up.device) / window
            pad_left = window // 2
            pad_right = window - 1 - pad_left
            
            padded_up = F.pad(up.transpose(-1, -2), (pad_left, pad_right), mode='replicate')
            padded_down = F.pad(down.transpose(-1, -2), (pad_left, pad_right), mode='replicate')
            
            sum_up = F.conv1d(padded_up, weights.view(1, 1, -1)).transpose(-1, -2) * window
            sum_down = F.conv1d(padded_down, weights.view(1, 1, -1)).transpose(-1, -2) * window
            
            # CMO
            cmo = self._safe_division(sum_up - sum_down, sum_up + sum_down) * 100 # 106
            return cmo
        
        cmo = compute_cmo(close) # 106
        features.append(cmo)

        return features

    def _compute_candlestick_patterns(self, open_p, high, low, close):
        """Вычисляет признаки свечных паттернов."""
        features = []
        
        # Размеры свечи
        body = torch.abs(close - open_p)
        upper_shadow = high - torch.maximum(open_p, close)
        lower_shadow = torch.minimum(open_p, close) - low
        total_range = high - low
        
        # Нормализация размеров (чтобы избежать деления на 0)
        small_body = body < (total_range * 0.3)
        large_body = body > (total_range * 0.7)
        # Исправление: используем torch.roll с правильной обработкой начальных значений
        prev_total_range = torch.roll(total_range, shifts=1, dims=1)
        prev_total_range[:, 0] = total_range[:, 0] # Заполняем начальное значение
        
        small_range = total_range < prev_total_range * 0.5
        large_range = total_range > prev_total_range * 1.5
        
        # Doji (дожи) - свеча с очень маленьким телом
        doji = small_body & (total_range > self.eps) # 107
        features.append(doji.float())
        
        # Marubozu (марубозу) - свеча без теней или с очень маленькими тенями
        marubozu = large_body & (upper_shadow + lower_shadow < total_range * 0.1) # 108
        features.append(marubozu.float())
        
        # Hammer (молот) - длинная нижняя тень, маленькое тело в верхней части
        hammer = (lower_shadow > body * 2) & (upper_shadow < body * 0.5) & (close > open_p) # 109
        features.append(hammer.float())
        
        # Hanging Man (повешенный) - длинная нижняя тень, маленькое тело в верхней части, но в downtrend
        hanging_man = (lower_shadow > body * 2) & (upper_shadow < body * 0.5) & (close < open_p) # 110
        features.append(hanging_man.float())
        
        # Inverted Hammer (перевернутый молот)
        inverted_hammer = (upper_shadow > body * 2) & (lower_shadow < body * 0.5) & (close > open_p) # 111
        features.append(inverted_hammer.float())
        
        # Shooting Star (падающая звезда)
        shooting_star = (upper_shadow > body * 2) & (lower_shadow < body * 0.5) & (close < open_p) # 112
        features.append(shooting_star.float())
        
        # Bullish Engulfing (бычье поглощение)
        prev_open = torch.roll(open_p, shifts=1, dims=1)
        prev_close = torch.roll(close, shifts=1, dims=1)
        # Исправление: заполняем начальные значения
        prev_open[:, 0] = open_p[:, 0]
        prev_close[:, 0] = close[:, 0]
        bullish_engulfing = (close > open_p) & (prev_close < prev_open) & (close > prev_open) & (open_p < prev_close) # 113
        features.append(bullish_engulfing.float())
        
        # Bearish Engulfing (медвежье поглощение)
        # Исправление: заполняем начальные значения
        prev_open_bear = torch.roll(open_p, shifts=1, dims=1)
        prev_close_bear = torch.roll(close, shifts=1, dims=1)
        prev_open_bear[:, 0] = open_p[:, 0]
        prev_close_bear[:, 0] = close[:, 0]
        bearish_engulfing = (close < open_p) & (prev_close_bear > prev_open_bear) & (close < prev_open_bear) & (open_p > prev_close_bear) # 114
        features.append(bearish_engulfing.float())
        
        # Tweezer Tops (вершины-пинцеты)
        prev_high = torch.roll(high, shifts=1, dims=1)
        # Исправление: заполняем начальное значение
        prev_high[:, 0] = high[:, 0]
        tweezer_tops = (torch.abs(high - prev_high) < self.eps) & (close < open_p) & (torch.roll(close, shifts=1, dims=1) > torch.roll(open_p, shifts=1, dims=1)) # 115
        # Исправление: заполняем начальное значение
        tweezer_tops[:, 0] = 0
        features.append(tweezer_tops.float())
        
        # Tweezer Bottoms (днища-пинцеты)
        prev_low = torch.roll(low, shifts=1, dims=1)
        # Исправление: заполняем начальное значение
        prev_low[:, 0] = low[:, 0]
        tweezer_bottoms = (torch.abs(low - prev_low) < self.eps) & (close > open_p) & (torch.roll(close, shifts=1, dims=1) < torch.roll(open_p, shifts=1, dims=1)) # 116
        # Исправление: заполняем начальное значение
        tweezer_bottoms[:, 0] = 0
        features.append(tweezer_bottoms.float())
        
        # Three White Soldiers (три белых солдата)
        prev_prev_close = torch.roll(close, shifts=2, dims=1)
        # Исправление: заполняем начальные значения
        prev_prev_close[:, :2] = close[:, :2]
        three_white_soldiers = (
            (close > open_p) & 
            (torch.roll(close, shifts=1, dims=1) > torch.roll(open_p, shifts=1, dims=1)) & 
            (prev_prev_close > torch.roll(open_p, shifts=2, dims=1)) &
            (close > torch.roll(close, shifts=1, dims=1)) & 
            (torch.roll(close, shifts=1, dims=1) > prev_prev_close)
        ) # 117
        # Исправление: заполняем начальные значения
        three_white_soldiers[:, :2] = 0
        features.append(three_white_soldiers.float())
        
        # Three Black Crows (три черных ворона)
        prev_prev_close_black = torch.roll(close, shifts=2, dims=1)
        # Исправление: заполняем начальные значения
        prev_prev_close_black[:, :2] = close[:, :2]
        three_black_crows = (
            (close < open_p) & 
            (torch.roll(close, shifts=1, dims=1) < torch.roll(open_p, shifts=1, dims=1)) & 
            (prev_prev_close_black < torch.roll(open_p, shifts=2, dims=1)) &
            (close < torch.roll(close, shifts=1, dims=1)) & 
            (torch.roll(close, shifts=1, dims=1) < prev_prev_close_black)
        ) # 118
        # Исправление: заполняем начальные значения
        three_black_crows[:, :2] = 0
        features.append(three_black_crows.float())
        
        # Morning Star (утренняя звезда)
        prev_prev_close_morning = torch.roll(close, shifts=2, dims=1)
        prev_open_morning = torch.roll(open_p, shifts=1, dims=1)
        prev_high_morning = torch.roll(high, shifts=1, dims=1)
        prev_low_morning = torch.roll(low, shifts=1, dims=1)
        # Исправление: заполняем начальные значения
        prev_prev_close_morning[:, :2] = close[:, :2]
        prev_open_morning[:, :2] = open_p[:, :2]
        prev_high_morning[:, :2] = high[:, :2]
        prev_low_morning[:, :2] = low[:, :2]
        morning_star = (
            (prev_prev_close_morning < torch.roll(open_p, shifts=2, dims=1)) & # Первый - медвежья свеча
            (torch.abs(torch.roll(close, shifts=1, dims=1) - prev_open_morning) < (prev_high_morning - prev_low_morning) * 0.3) & # Второй - дожи
            (close > open_p) & # Третий - бычья свеча
            (close > (prev_prev_close_morning + torch.roll(open_p, shifts=2, dims=1)) / 2) # Закрытие третьей свечи выше середины первой
        ) # 119
        # Исправление: заполняем начальные значения
        morning_star[:, :2] = 0
        features.append(morning_star.float())
        
        # Evening Star (вечерняя звезда)
        prev_prev_close_evening = torch.roll(close, shifts=2, dims=1)
        prev_open_evening = torch.roll(open_p, shifts=1, dims=1)
        prev_high_evening = torch.roll(high, shifts=1, dims=1)
        prev_low_evening = torch.roll(low, shifts=1, dims=1)
        # Исправление: заполняем начальные значения
        prev_prev_close_evening[:, :2] = close[:, :2]
        prev_open_evening[:, :2] = open_p[:, :2]
        prev_high_evening[:, :2] = high[:, :2]
        prev_low_evening[:, :2] = low[:, :2]
        evening_star = (
            (prev_prev_close_evening > torch.roll(open_p, shifts=2, dims=1)) & # Первый - бычья свеча
            (torch.abs(torch.roll(close, shifts=1, dims=1) - prev_open_evening) < (prev_high_evening - prev_low_evening) * 0.3) & # Второй - дожи
            (close < open_p) & # Третий - медвежья свеча
            (close < (prev_prev_close_evening + torch.roll(open_p, shifts=2, dims=1)) / 2) # Закрытие третьей свечи ниже середины первой
        ) # 120
        # Исправление: заполняем начальные значения
        evening_star[:, :2] = 0
        features.append(evening_star.float())
        
        # Piercing Line (линия пробоя)
        prev_close_piercing = torch.roll(close, shifts=1, dims=1)
        prev_open_piercing = torch.roll(open_p, shifts=1, dims=1)
        prev_low_piercing = torch.roll(low, shifts=1, dims=1)
        # Исправление: заполняем начальное значение
        prev_close_piercing[:, 0] = close[:, 0]
        prev_open_piercing[:, 0] = open_p[:, 0]
        prev_low_piercing[:, 0] = low[:, 0]
        piercing_line = (
            (prev_close_piercing < prev_open_piercing) & # Первая - медвежья свеча
            (close > open_p) & # Вторая - бычья свеча
            (open_p < prev_low_piercing) & # Открытие второй ниже минимума первой
            (close > (prev_close_piercing + prev_open_piercing) / 2) & # Закрытие второй выше середины первой
            (close < prev_open_piercing) # Но закрытие второй ниже открытия первой
        ) # 121
        # Исправление: заполняем начальное значение
        piercing_line[:, 0] = 0
        features.append(piercing_line.float())
        
        # Dark Cloud Cover (темное облако)
        prev_close_dark = torch.roll(close, shifts=1, dims=1)
        prev_open_dark = torch.roll(open_p, shifts=1, dims=1)
        prev_high_dark = torch.roll(high, shifts=1, dims=1)
        # Исправление: заполняем начальное значение
        prev_close_dark[:, 0] = close[:, 0]
        prev_open_dark[:, 0] = open_p[:, 0]
        prev_high_dark[:, 0] = high[:, 0]
        dark_cloud_cover = (
            (prev_close_dark > prev_open_dark) & # Первая - бычья свеча
            (close < open_p) & # Вторая - медвежья свеча
            (open_p > prev_high_dark) & # Открытие второй выше максимума первой
            (close < (prev_close_dark + prev_open_dark) / 2) & # Закрытие второй ниже середины первой
            (close > prev_open_dark) # Но закрытие второй выше открытия первой
        ) # 122
        # Исправление: заполняем начальное значение
        dark_cloud_cover[:, 0] = 0
        features.append(dark_cloud_cover.float())
        
        # Inside Bar (внутренняя свеча)
        prev_high_inside = torch.roll(high, shifts=1, dims=1)
        prev_low_inside = torch.roll(low, shifts=1, dims=1)
        prev_high_inside[:, 0] = high[:, 0]
        prev_low_inside[:, 0] = low[:, 0]
        inside_bar = (high < prev_high_inside) & (low > prev_low_inside) # 123
        inside_bar[:, 0] = 0
        features.append(inside_bar.float())
        
        # Outside Bar (внешняя свеча)
        prev_high_outside = torch.roll(high, shifts=1, dims=1)
        prev_low_outside = torch.roll(low, shifts=1, dims=1)
        prev_high_outside[:, 0] = high[:, 0]
        prev_low_outside[:, 0] = low[:, 0]
        outside_bar = (high > prev_high_outside) & (low < prev_low_outside) # 124
        outside_bar[:, 0] = 0
        features.append(outside_bar.float())
        
        # Harami Pattern (харами)
        prev_open_harami = torch.roll(open_p, shifts=1, dims=1)
        prev_close_harami = torch.roll(close, shifts=1, dims=1)
        prev_open_harami[:, 0] = open_p[:, 0]
        prev_close_harami[:, 0] = close[:, 0]
        harami = (
            (torch.maximum(open_p, close) < torch.maximum(prev_open_harami, prev_close_harami)) &
            (torch.minimum(open_p, close) > torch.minimum(prev_open_harami, prev_close_harami))
        ) # 125
        harami[:, 0] = 0
        features.append(harami.float())
        
        # Gap Up (разрыв вверх)
        prev_close_gap = torch.roll(close, shifts=1, dims=1)
        prev_close_gap[:, 0] = close[:, 0]
        gap_up = (torch.minimum(open_p, close) > prev_close_gap) # 126
        gap_up[:, 0] = 0
        features.append(gap_up.float())
        
        # Gap Down (разрыв вниз)
        prev_close_gap_down = torch.roll(close, shifts=1, dims=1)
        prev_close_gap_down[:, 0] = close[:, 0]
        gap_down = (torch.maximum(open_p, close) < prev_close_gap_down) # 127
        gap_down[:, 0] = 0
        features.append(gap_down.float())
        
        return features

    def _compute_ewma_approx(self, data, span):
        """Вспомогательная функция для вычисления приближенного EWMA."""
        alpha = 2.0 / (span + 1)
        weights = torch.pow(1 - alpha, torch.arange(span, device=data.device)).flip(0)
        weights = weights / weights.sum()
        padded_data = F.pad(data.transpose(-1, -2), (span-1, 0), mode='replicate') 
        ewma = F.conv1d(padded_data, weights.view(1, 1, -1)) 
        return ewma.transpose(-1, -2)
    
    def _compute_sma(self, data_t, window):
        """Вспомогательная функция для вычисления SMA."""
        weights = torch.ones(window, device=data_t.device) / window
        pad_left = window // 2
        pad_right = window - 1 - pad_left
        padded_data = F.pad(data_t, (pad_left, pad_right), mode='replicate')
        return F.conv1d(padded_data, weights.view(1, 1, -1))

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
        all_features.extend(self._compute_moving_averages(close, volumes, open_p, high, low))

        # 5. Импульс
        all_features.extend(self._compute_momentum_indicators(close))

        # 6. Осцилляторы
        all_features.extend(self._compute_oscillators(close, high, low, volumes))

        # ИСПРАВЛЕНИЕ 3: Передаем window=14 в вызов _compute_volume_indicators
        # 7. Индикаторы объема
        all_features.extend(self._compute_volume_indicators(volumes, close, window=14))

        # 8. Продвинутые индикаторы
        all_features.extend(self._compute_advanced_indicators(open_p, high, low, close, volumes))

        # 9. Свечные паттерны
        all_features.extend(self._compute_candlestick_patterns(open_p, high, low, close))

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
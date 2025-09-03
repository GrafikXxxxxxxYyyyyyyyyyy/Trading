# src/lstm_ffn/lstm_ffn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TradingLSTMFFN(nn.Module):
    """
    Гибридная архитектура LSTM + FFN для прогнозирования цен на фондовом рынке.
    
    Архитектура:
    1. Bidirectional LSTM для захвата временных зависимостей
    2. Self-Attention для фокусировки на важных временных шагах
    3. Глубокая FFN для нелинейной обработки признаков
    4. Выходной слой для прогнозирования 32 цен закрытия
    
    Принимает: [B, 256, 256] - обогащённые признаки от SAE
    Возвращает: [B, 32, 1] - прогноз 32 цен закрытия
    """
    def __init__(
        self,
        feature_size=256,
        history_len=256,
        target_len=32,
        lstm_hidden_size=512,
        lstm_num_layers=2,
        lstm_dropout=0.2,
        ffn_hidden_dims=[1024, 512, 256],
        ffn_dropout=0.2,
        use_layer_norm=True,
        use_bidirectional=True,
        num_attention_heads=8
    ):
        """
        Args:
            feature_size (int): Размер признаков на каждом временном шаге (256).
            history_len (int): Длина исторической последовательности (256).
            target_len (int): Длина прогнозируемой последовательности (32).
            lstm_hidden_size (int): Размер скрытого состояния LSTM.
            lstm_num_layers (int): Количество слоев LSTM.
            lstm_dropout (float): Вероятность Dropout в LSTM.
            ffn_hidden_dims (list): Размеры скрытых слоёв FFN.
            ffn_dropout (float): Вероятность Dropout в FFN.
            use_layer_norm (bool): Использовать ли LayerNorm.
            use_bidirectional (bool): Использовать ли_bidirectional LSTM.
            num_attention_heads (int): Количество голов внимания.
        """
        super(TradingLSTMFFN, self).__init__()
        self.feature_size = feature_size
        self.history_len = history_len
        self.target_len = target_len
        self.use_layer_norm = use_layer_norm
        self.use_bidirectional = use_bidirectional
        self.num_directions = 2 if use_bidirectional else 1
        
        # Вычисляем фактический размер выхода LSTM
        self.lstm_output_size = lstm_hidden_size * self.num_directions
        
        # Входная нормализация
        if use_layer_norm:
            self.input_norm = nn.LayerNorm(feature_size)
            
        # Bidirectional LSTM для захвата временных зависимостей
        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=use_bidirectional
        )
        
        # Self-Attention для фокусировки на важных временных шагах
        # ВАЖНО: embed_dim должен соответствовать размеру выхода LSTM
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.lstm_output_size,  # Используем правильный размер
            num_heads=num_attention_heads,
            dropout=lstm_dropout,
            batch_first=True
        )
        
        # Глубокая FFN часть
        layers = []
        input_dim = self.lstm_output_size * history_len  # Сглаживаем всю историю
        
        for i, hidden_dim in enumerate(ffn_hidden_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(ffn_dropout))
            input_dim = hidden_dim
            
        self.ffn = nn.Sequential(*layers)
        
        # Выходной слой для прогнозирования target_len значений
        self.output_projection = nn.Linear(ffn_hidden_dims[-1], target_len)
        
        # Инициализация весов
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов для лучшей сходимости."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Инициализация весов линейного слоя
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.xavier_uniform_(m.weight)
                # Инициализация смещения, если оно существует
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                # Инициализация весов LSTM
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        # Установка смещений забывательной части в 1 для лучшей инициализации LSTM
                        param.data.fill_(0)
                        # Смещение забывательной части (bias_hh[bias_size:2*bias_size])
                        n = param.size(0)
                        start, end = n//4, n//2
                        param.data[start:end].fill_(1.)

    def forward(self, src, tgt=None):
        """
        Прямой проход модели.
        
        Args:
            src (torch.Tensor): Исторические данные [B, history_len, feature_size].
                                Должны быть уже обработаны TradingFeatureExtractor.
            tgt (torch.Tensor, optional): Целевые значения [B, target_len, 1].
            
        Returns:
            torch.Tensor: Прогнозы [B, target_len, 1].
        """
        batch_size = src.size(0)
        
        # 1. Нормализация входа (если включено)
        if self.use_layer_norm:
            x = self.input_norm(src)  # [B, 256, 256]
        else:
            x = src
            
        # 2. Пропускаем через LSTM
        lstm_out, (hidden, cell) = self.lstm(x)  # [B, 256, lstm_output_size]
        
        # 3. Применяем self-attention для фокусировки на важных частях
        # ВАЖНО: embed_dim в MultiheadAttention должен совпадать с последней размерностью lstm_out
        attn_out, _ = self.self_attention(lstm_out, lstm_out, lstm_out)  # [B, 256, lstm_output_size]
        
        # 4. Сглаживаем временные шаги для FFN
        # ИСПРАВЛЕНО: используем reshape вместо view для корректной работы с не-непрерывными тензорами
        flattened = attn_out.reshape(batch_size, -1)  # [B, 256*lstm_output_size]
        
        # 5. Пропускаем через глубокую FFN
        features = self.ffn(flattened)  # [B, ffn_hidden_dims[-1]]
        
        # 6. Прогнозируем target_len значений
        output = self.output_projection(features)  # [B, 32]
        
        # 7. Добавляем размерность для совместимости с датасетом
        return output.unsqueeze(-1)  # [B, 32, 1]




# # Пример использования
# if __name__ == "__main__":
#     # Параметры модели
#     B, T_hist, feature_size = 4, 256, 256
#     T_pred = 32
#     output_size = 1
    
#     # Создание модели
#     model = TradingLSTMFFN(
#         feature_size=feature_size,
#         history_len=T_hist,
#         target_len=T_pred,
#         lstm_hidden_size=512,
#         lstm_num_layers=2,
#         lstm_dropout=0.2,
#         ffn_hidden_dims=[1024, 512, 256],  # Глубокая архитектура
#         ffn_dropout=0.2,
#         use_layer_norm=True,
#         use_bidirectional=True,
#         num_attention_heads=8
#     )
    
#     # Примерные входные данные (после SAE)
#     src = torch.randn(B, T_hist, feature_size)  # История после TradingFeatureExtractor
#     tgt = torch.randn(B, T_pred, output_size)   # Целевые значения (опционально)
    
#     # Прогон модели
#     model.eval()
#     with torch.no_grad():
#         output = model(src, tgt)
#         print(f"Выход модели: {output.shape}")  # [B, 32, 1]
        
#     # Подсчет параметров
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"Общее количество параметров: {total_params:,}")
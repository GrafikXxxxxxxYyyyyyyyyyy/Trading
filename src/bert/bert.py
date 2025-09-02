# models/trading_bert.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Позиционное кодирование для добавления информации о позиции в последовательности.
    Сlightly адаптировано для временных рядов.
    """
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        """
        Args:
            d_model (int): Размерность модели (признаков).
            max_len (int): Максимальная длина последовательности.
            dropout (float): Вероятность Dropout.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Создаем матрицу позиционных кодировок: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # [max_len, 1]
        # Используем разные индексы для sin и cos
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        
        # Добавляем размерность для батча: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        # Регистрируем как буфер, чтобы он сохранялся при сериализации модели
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (Tensor): Входной тензор [B, seq_len, d_model]
        Returns:
            Tensor: Выходной тензор [B, seq_len, d_model]
        """
        # x: [B, seq_len, d_model]
        # self.pe: [1, max_len, d_model]
        # seq_len <= max_len
        x = x + self.pe[:, :x.size(1)] # Broadcasting по батчу и длине
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Многоголовое внимание (Multi-Head Attention).
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Args:
            d_model (int): Размерность модели.
            num_heads (int): Количество голов внимания.
            dropout (float): Вероятность Dropout.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # Размерность ключа/запроса/значения на одну голову

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query (Tensor): Запросы [B, len_q, d_model]
            key (Tensor): Ключи [B, len_k, d_model]
            value (Tensor): Значения [B, len_v, d_model]
            mask (Tensor, optional): Маска [B, 1, len_q, len_k] или [B, num_heads, len_q, len_k]
        Returns:
            Tensor: Выход [B, len_q, d_model]
            Tensor: Веса внимания [B, num_heads, len_q, len_k] (для визуализации, не используется здесь)
        """
        batch_size = query.size(0)

        # Линейные преобразования и разделение на головы
        # [B, len, d_model] -> [B, len, num_heads, d_k] -> [B, num_heads, len, d_k]
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Матрица оценок внимания
        # [B, num_heads, len_q, d_k] x [B, num_heads, d_k, len_k] -> [B, num_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Применение маски (если есть)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Вычисление весов внимания
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Применение весов к значениям
        # [B, num_heads, len_q, len_k] x [B, num_heads, len_v, d_k] -> [B, num_heads, len_q, d_k]
        # len_k == len_v
        attn_output = torch.matmul(attn_weights, V)

        # Конкатенация голов и линейное преобразование
        # [B, num_heads, len_q, d_k] -> [B, len_q, num_heads, d_k] -> [B, len_q, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)

        return output #, attn_weights # Веса можно вернуть для анализа


class PositionwiseFeedForward(nn.Module):
    """
    Позиционно-независимый полносвязный слой (Feed-Forward Network).
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model (int): Размерность модели.
            d_ff (int): Размерность внутреннего слоя.
            dropout (float): Вероятность Dropout.
        """
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU() # Используем GELU вместо ReLU, как в BERT

    def forward(self, x):
        """
        Args:
            x (Tensor): Входной тензор [B, seq_len, d_model]
        Returns:
            Tensor: Выходной тензор [B, seq_len, d_model]
        """
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EncoderBlock(nn.Module):
    """
    Блок энкодера BERT-like: Multi-Head Attention + Feed Forward.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model (int): Размерность модели.
            num_heads (int): Количество голов внимания.
            d_ff (int): Размерность внутреннего слоя FFN.
            dropout (float): Вероятность Dropout.
        """
        super(EncoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        """
        Args:
            src (Tensor): Входные данные [B, seq_len, d_model]
            src_mask (Tensor, optional): Маска для внимания [B, 1, 1, seq_len] или подобная
        Returns:
            Tensor: Выходные данные [B, seq_len, d_model]
        """
        # Self-attention
        attn_out = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout1(attn_out)) # Residual connection + LayerNorm
        
        # Feed forward
        ffn_out = self.ffn(src)
        out = self.norm2(src + self.dropout2(ffn_out)) # Residual connection + LayerNorm
        
        return out


class TradingBERT(nn.Module):
    """
    Мощная BERT-like модель (Encoder-only Transformer) для прогнозирования цен на фондовом рынке.
    """
    def __init__(
        self,
        feature_size=128,           # Размер признаков от TradingProcessor
        d_model=512,                # Размерность внутреннего представления
        num_layers=8,               # Количество слоев энкодера
        num_heads=8,                # Количество голов внимания
        d_ff=2048,                  # Размерность внутреннего слоя FFN
        target_len=32,              # Длина прогноза
        dropout=0.1,                # Dropout
        use_layer_norm=True,        # Использовать LayerNorm на входе
        max_seq_len=1000            # Максимальная длина последовательности для поз. кодирования
    ):
        """
        Args:
            feature_size (int): Размер входных признаков (128).
            d_model (int): Размерность внутреннего представления модели.
            num_layers (int): Количество слоев энкодера.
            num_heads (int): Количество голов внимания.
            d_ff (int): Размерность внутреннего слоя FFN.
            target_len (int): Длина прогнозируемой последовательности (32).
            dropout (float): Вероятность Dropout.
            use_layer_norm (bool): Применять ли LayerNorm к входу.
            max_seq_len (int): Максимальная длина последовательности для поз. кодирования.
        """
        super(TradingBERT, self).__init__()
        self.feature_size = feature_size
        self.d_model = d_model
        self.target_len = target_len
        self.use_layer_norm = use_layer_norm
        
        # Входное линейное преобразование (Embedding)
        self.input_projection = nn.Linear(feature_size, d_model)
        
        # Позиционное кодирование
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # LayerNorm для нормализации входа (опционально)
        if use_layer_norm:
            self.input_layer_norm = nn.LayerNorm(d_model) # Применяем к d_model после проекции
            
        # Стек Encoder блоков
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        # Выходной слой (Prediction Head) для прогнозирования
        # Прогнозируем только цену закрытия, поэтому output_size=1
        # Используем линейный слой, который будет применяться к выходам последних target_len позиций
        # Вариант 1: Простая голова
        # self.output_projection = nn.Linear(d_model, 1)
        
        # Вариант 2: Более сложная голова, как в BERT
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1) # Выход: прогноз цены закрытия
        )
        
        # Инициализация весов
        self._init_weights()
        
    def _init_weights(self):
        """Инициализация весов."""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                # Для матриц используем Xavier_uniform или нормальное распределение как в BERT
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        # Инициализация весов LayerNorm
        for name, param in self.named_parameters():
             if 'LayerNorm' in name and 'weight' in name:
                 nn.init.constant_(param, 1.0)
             elif 'LayerNorm' in name and 'bias' in name:
                 nn.init.constant_(param, 0.0)

    def forward(self, src, tgt=None):
        """
        Прямой проход модели.
        
        Args:
            src (torch.Tensor): Исторические данные [B, T_hist=256, F_hist=128].
                                Должны быть уже обработаны TradingProcessor.
            tgt (torch.Tensor, optional): Целевые значения [B, T_pred=32, 1].
                                          Используется для совместимости с интерфейсом, 
                                          но не влияет на вычисления в этой архитектуре.
            
        Returns:
            torch.Tensor: Прогнозы [B, T_pred=32, 1].
        """
        # src: [B, 256, 128]
        batch_size, seq_len, _ = src.size()
        
        # 1. Линейная проекция входа в d_model
        x = self.input_projection(src) # [B, 256, d_model]
        
        # 2. Добавляем позиционное кодирование
        x = self.positional_encoding(x) # [B, 256, d_model]
        
        # 3. Применяем Layer Normalization (если включено)
        if self.use_layer_norm:
            x = self.input_layer_norm(x) # [B, 256, d_model]
            
        # 4. Пропускаем через стек Encoder блоков
        # x: [B, 256, d_model]
        for layer in self.encoder_layers:
            x = layer(x) # [B, 256, d_model]
        # После всех слоев: encoder_output: [B, 256, d_model]
        
        # 5. Извлекаем выходы последних target_len позиций для прогнозирования
        # Предполагаем, что последние target_len позиций используются для прогноза
        # Это простой и эффективный способ без использования специальных токенов
        prediction_features = x[:, -self.target_len:, :] # [B, 32, d_model]
        
        # 6. Применяем Prediction Head к этим признакам
        # prediction_features: [B, 32, d_model]
        # output: [B, 32, 1]
        output = self.prediction_head(prediction_features)
        
        return output



# # --- Пример использования ---
# if __name__ == "__main__":
#     # Параметры модели
#     B, T_hist, feature_size = 4, 256, 128
#     T_pred = 32
#     output_size = 1
    
#     # Создание модели
#     model = TradingBERT(
#         feature_size=feature_size,
#         d_model=512,
#         num_layers=6,
#         num_heads=8,
#         d_ff=2048,
#         target_len=T_pred,
#         dropout=0.1,
#         use_layer_norm=True,
#         max_seq_len=1000
#     )
    
#     # Примерные входные данные
#     src = torch.randn(B, T_hist, feature_size)  # История после TradingProcessor
#     tgt = torch.randn(B, T_pred, output_size)   # Целевые значения (опционально)
    
#     # Прогон модели
#     model.eval()
#     with torch.no_grad():
#         output = model(src, tgt)
#         print(f"Выход модели: {output.shape}") # [B, 32, 1]

#     # Проверка количества параметров
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Общее количество параметров: {total_params}")
#     print(f"Количество обучаемых параметров: {trainable_params}")

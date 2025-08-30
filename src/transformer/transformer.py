import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class PositionalEncoding(nn.Module):
    """
    Позиционное кодирование для добавления информации о позиции в последовательности.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
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
        x = x + self.pe[:, :x.size(1)] # Broadcasting по батчу
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
            Tensor: Веса внимания [B, num_heads, len_q, len_k]
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

        return output, attn_weights



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
        self.relu = nn.ReLU()


    def forward(self, x):
        """
        Args:
            x (Tensor): Входной тензор [B, seq_len, d_model]
        Returns:
            Tensor: Выходной тензор [B, seq_len, d_model]
        """
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



class EncoderBlock(nn.Module):
    """
    Блок энкодера: Multi-Head Attention + Feed Forward.
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
        attn_out, _ = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout1(attn_out))
        
        # Feed forward
        ffn_out = self.ffn(src)
        out = self.norm2(src + self.dropout2(ffn_out))
        
        return out



class DecoderBlock(nn.Module):
    """
    Блок декодера: Masked Multi-Head Attention + Multi-Head Attention + Feed Forward.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model (int): Размерность модели.
            num_heads (int): Количество голов внимания.
            d_ff (int): Размерность внутреннего слоя FFN.
            dropout (float): Вероятность Dropout.
        """
        super(DecoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)


    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Args:
            tgt (Tensor): Вход декодера (таргеты) [B, tgt_len, d_model]
            memory (Tensor): Выход энкодера [B, src_len, d_model]
            tgt_mask (Tensor, optional): Маска для self-attention в декодере [B, 1, tgt_len, tgt_len]
            memory_mask (Tensor, optional): Маска для cross-attention [B, 1, 1, src_len]
        Returns:
            Tensor: Выход декодера [B, tgt_len, d_model]
        """
        # Masked self-attention (causal)
        self_attn_out, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(self_attn_out))
        
        # Cross-attention
        cross_attn_out, _ = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout2(cross_attn_out))
        
        # Feed forward
        ffn_out = self.ffn(tgt)
        out = self.norm3(tgt + self.dropout3(ffn_out))
        
        return out



class TransformerEncoder(nn.Module):
    """
    Полный энкодер трансформера.
    """
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            num_layers (int): Количество блоков энкодера.
            d_model (int): Размерность модели.
            num_heads (int): Количество голов внимания.
            d_ff (int): Размерность внутреннего слоя FFN.
            dropout (float): Вероятность Dropout.
        """
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)


    def forward(self, src, src_mask=None):
        """
        Args:
            src (Tensor): Входные данные [B, src_len, d_model]
            src_mask (Tensor, optional): Маска для внимания [B, 1, 1, src_len]
        Returns:
            Tensor: Выходные данные [B, src_len, d_model]
        """
        output = src
        for layer in self.layers:
            output = layer(output, src_mask)
        return self.norm(output)



class TransformerDecoder(nn.Module):
    """
    Полный декодер трансформера.
    """
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            num_layers (int): Количество блоков декодера.
            d_model (int): Размерность модели.
            num_heads (int): Количество голов внимания.
            d_ff (int): Размерность внутреннего слоя FFN.
            dropout (float): Вероятность Dropout.
        """
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)


    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Args:
            tgt (Tensor): Вход декодера (таргеты) [B, tgt_len, d_model]
            memory (Tensor): Выход энкодера [B, src_len, d_model]
            tgt_mask (Tensor, optional): Маска для self-attention в декодере [B, 1, tgt_len, tgt_len]
            memory_mask (Tensor, optional): Маска для cross-attention [B, 1, 1, src_len]
        Returns:
            Tensor: Выход декодера [B, tgt_len, d_model]
        """
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask, memory_mask)
        return self.norm(output)



class TradingTransformer(nn.Module):
    """
    Полная модель Transformer для прогнозирования цен на фондовом рынке.
    """
    def __init__(
        self,
        feature_size=128,           # Размер признаков от TradingProcessor
        d_model=512,                # Размерность внутреннего представления
        num_encoder_layers=6,       # Количество слоев в энкодере
        num_decoder_layers=6,       # Количество слоев в декодере
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
            num_encoder_layers (int): Количество слоев в энкодере.
            num_decoder_layers (int): Количество слоев в декодере.
            num_heads (int): Количество голов внимания.
            d_ff (int): Размерность внутреннего слоя FFN.
            target_len (int): Длина прогнозируемой последовательности (32).
            dropout (float): Вероятность Dropout.
            use_layer_norm (bool): Применять ли LayerNorm к входу.
            max_seq_len (int): Максимальная длина последовательности для поз. кодирования.
        """
        super(TradingTransformer, self).__init__()
        self.feature_size = feature_size
        self.d_model = d_model
        self.target_len = target_len
        self.use_layer_norm = use_layer_norm
        
        # Входные эмбеддинги и позиционное кодирование
        # Для истории
        self.src_embedding = nn.Linear(feature_size, d_model)
        # Для таргетов (предполагаем, что таргет это цена закрытия, 1 признак)
        self.tgt_embedding = nn.Linear(1, d_model)
        
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # LayerNorm для нормализации входа (опционально)
        if use_layer_norm:
            self.src_layer_norm = nn.LayerNorm(feature_size)
            self.tgt_layer_norm = nn.LayerNorm(1) # Таргет это 1 признак
            
        # Энкодер и Декодер
        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )
        
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # Выходной слой для прогнозирования
        # Проектируем из d_model в размерность таргета (1)
        self.output_projection = nn.Linear(d_model, 1)
        
        # Инициализация весов
        self._init_weights()
        

    def _init_weights(self):
        """Инициализация весов."""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)


    def generate_square_subsequent_mask(self, sz):
        """
        Генерирует квадратную маску для предотвращения заглядывания в будущее.
        Args:
            sz (int): Размер квадратной матрицы.
        Returns:
            Tensor: Маска [sz, sz] с 0 для запрещенных позиций и -inf для разрешенных.
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


    def _prepare_encoder_input(self, src):
        """Подготовка входа для энкодера."""
        batch_size, src_seq_len, _ = src.size()
        
        # Применяем Layer Normalization (если включено)
        if self.use_layer_norm:
            src_norm = self.src_layer_norm(src) # [B, src_seq_len, 128]
        else:
            src_norm = src
            
        # Линейное преобразование в d_model
        src_embedded = self.src_embedding(src_norm) # [B, src_seq_len, d_model]
        
        # Добавляем позиционное кодирование
        src_embedded = self.positional_encoding(src_embedded) # [B, src_seq_len, d_model]
        
        return src_embedded


    def _prepare_decoder_input(self, tgt):
        """Подготовка входа для декодера."""
        batch_size, tgt_seq_len, _ = tgt.size()
        
        # Применяем Layer Normalization к таргету (если включено)
        if self.use_layer_norm:
            tgt_norm = self.tgt_layer_norm(tgt) # [B, tgt_seq_len, 1]
        else:
            tgt_norm = tgt
            
        # Линейное преобразование в d_model
        tgt_embedded = self.tgt_embedding(tgt_norm) # [B, tgt_seq_len, d_model]
        
        # Добавляем позиционное кодирование
        tgt_embedded = self.positional_encoding(tgt_embedded) # [B, tgt_seq_len, d_model]
        
        return tgt_embedded


    def forward(self, src, tgt=None):
        """
        Прямой проход модели.
        
        Args:
            src (torch.Tensor): Исторические данные [B, T_hist=256, F_hist=128].
                                Должны быть уже обработаны TradingProcessor.
            tgt (torch.Tensor, optional): Целевые значения [B, T_pred=32, 1].
            
        Returns:
            torch.Tensor: Прогнозы [B, T_pred=32, 1].
        """
        # src: [B, 256, 128]
        # tgt: [B, 32, 1] (если предоставлено)
        
        # --- Обработка входа (энкодер) ---
        src_embedded = self._prepare_encoder_input(src)
        
        # Пропускаем через энкодер
        # memory: [B, 256, d_model]
        memory = self.encoder(src_embedded)
        
        # --- Обработка таргета (декодер) ---
        if self.training or tgt is not None:
            # Во время обучения используем предоставленные таргеты (teacher forcing)
            if tgt is None:
                raise ValueError("tgt must be provided during training")
                
            tgt_input = tgt # [B, 32, 1]
            tgt_embedded = self._prepare_decoder_input(tgt_input)
            
            # Генерируем маску для предотвращения заглядывания в будущее
            tgt_seq_len = tgt_embedded.size(1)
            tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(tgt_embedded.device)
            # tgt_mask: [tgt_seq_len, tgt_seq_len] -> [1, 1, tgt_seq_len, tgt_seq_len]
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)
            
            # Пропускаем через декодер
            # decoder_output: [B, 32, d_model]
            decoder_output = self.decoder(
                tgt=tgt_embedded,
                memory=memory,
                tgt_mask=tgt_mask
            )
            
            # Применяем выходную проекцию
            # output: [B, 32, d_model] -> [B, 32, 1]
            output = self.output_projection(decoder_output)
            
        else:
            # Во время инференса генерируем таргеты авторегрессивно
            output = self.generate(src, memory)
            
        return output

    def generate(self, src, memory=None):
        """
        Авторегрессивная генерация прогноза.
        
        Args:
            src (torch.Tensor): Исторические данные [B, T_hist, F_hist].
            memory (torch.Tensor, optional): Предвычисленный выход энкодера [B, T_hist, d_model].
            
        Returns:
            torch.Tensor: Прогнозы [B, T_pred, 1].
        """
        batch_size = src.size(0)
        device = src.device
        
        if memory is None:
            # Вычисляем memory, если оно не предоставлено
            src_embedded = self._prepare_encoder_input(src)
            memory = self.encoder(src_embedded)
        
        # Начинаем с токена начала последовательности или последнего известного значения
        # Для простоты начнем с нулей
        # tgt: [B, 1, 1] - начинаем с одного токена
        tgt = torch.zeros(batch_size, 1, 1, device=device)
        
        # Список для хранения прогнозов
        predictions = []
        
        for _ in range(self.target_len):
            # Подготавливаем текущий tgt для декодера
            tgt_embedded = self._prepare_decoder_input(tgt) # [B, текущая_длина, d_model]
            
            # Генерируем маску для текущей длины
            current_tgt_len = tgt_embedded.size(1)
            tgt_mask = self.generate_square_subsequent_mask(current_tgt_len).to(device)
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0) # [1, 1, текущая_длина, текущая_длина]
            
            # Пропускаем через декодер
            # decoder_output: [B, текущая_длина, d_model]
            decoder_output = self.decoder(
                tgt=tgt_embedded,
                memory=memory,
                tgt_mask=tgt_mask
            )
            
            # Берем выход последнего токена (наш прогноз)
            # last_output: [B, d_model]
            last_output = decoder_output[:, -1, :]
            
            # Применяем выходную проекцию
            # next_prediction: [B, 1]
            next_prediction = self.output_projection(last_output)
            predictions.append(next_prediction.unsqueeze(1)) # [B, 1, 1]
            
            # Обновляем tgt для следующей итерации
            # Добавляем прогноз к tgt
            # tgt: [B, текущая_длина, 1]
            # next_prediction.unsqueeze(1): [B, 1, 1]
            tgt = torch.cat([tgt, next_prediction.unsqueeze(1)], dim=1) # [B, текущая_длина+1, 1]
            
        # Конкатенируем все прогнозы
        # Каждый элемент predictions: [B, 1, 1]
        # final_output: [B, T_pred, 1]
        final_output = torch.cat(predictions, dim=1)
        
        return final_output



# # --- Пример использования ---
# if __name__ == "__main__":
#     # Параметры модели
#     B, T_hist, feature_size = 4, 256, 128 # Уменьшено B для теста
#     T_pred = 32
#     output_size = 1
    
#     # Создание модели
#     model = TradingTransformer(
#         feature_size=feature_size,
#         d_model=512, 
#         num_encoder_layers=6,
#         num_decoder_layers=6,
#         num_heads=8, 
#         d_ff=2048, 
#         target_len=T_pred,
#         dropout=0.1,
#         use_layer_norm=True,
#         max_seq_len=1000
#     )
    
#     # Примерные входные данные
#     src = torch.randn(B, T_hist, feature_size)  # История после TradingProcessor
#     tgt = torch.randn(B, T_pred, output_size)   # Целевые значения
    
#     # Обучение (с учителем)
#     model.train()
#     output_train = model(src, tgt)
#     print(f"Выход при обучении (с учителем): {output_train.shape}") # [B, 32, 1]
    
#     # Инференс
#     model.eval()
#     with torch.no_grad():
#         output_infer = model(src)
#         print(f"Выход при инференсе (авторегрессивно): {output_infer.shape}") # [B, 32, 1]

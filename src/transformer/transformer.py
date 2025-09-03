# src/transformer/transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Позиционные эмбеддинги для захвата временной информации.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Многоголовое внимание с маскированием для декодера.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        output = torch.matmul(attn_probs, V)
        return output
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Применяем линейные преобразования и разделяем на головы
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Применяем внимание
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Объединяем головы
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.num_heads * self.d_k)
        
        # Финальное линейное преобразование
        output = self.W_o(attn_output)
        return output


class PositionwiseFeedForward(nn.Module):
    """
    Позиционно-независимая полносвязная сеть.
    """
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class EncoderLayer(nn.Module):
    """
    Слой энкодера трансформера.
    """
    def __init__(self, d_model, num_heads, ff_hidden_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, ff_hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, src_mask=None):
        # Самовнимание
        attn_output = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Полносвязная сеть
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """
    Слой декодера трансформера.
    """
    def __init__(self, d_model, num_heads, ff_hidden_dim, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, ff_hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Самовнимание с маской
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Перекрёстное внимание
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Полносвязная сеть
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class Encoder(nn.Module):
    """
    Энкодер трансформера.
    """
    def __init__(self, num_layers, d_model, num_heads, ff_hidden_dim, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """
    Декодер трансформера.
    """
    def __init__(self, num_layers, d_model, num_heads, ff_hidden_dim, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)


class TradingTransformer(nn.Module):
    """
    Мощная архитектура Transformer для прогнозирования цен на фондовом рынке.
    """
    def __init__(
        self,
        feature_size=256,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        ff_hidden_dim=2048,
        target_len=32,
        dropout=0.1,
        use_positional_encoding=True
    ):
        super(TradingTransformer, self).__init__()
        self.feature_size = feature_size
        self.d_model = d_model
        self.target_len = target_len
        self.use_positional_encoding = use_positional_encoding
        
        # Входная проекция
        self.input_projection = nn.Linear(feature_size, d_model)
        
        # Позиционное кодирование
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Энкодер и декодер
        self.encoder = Encoder(
            num_encoder_layers, d_model, num_heads, ff_hidden_dim, dropout
        )
        self.decoder = Decoder(
            num_decoder_layers, d_model, num_heads, ff_hidden_dim, dropout
        )
        
        # Выходной слой
        self.output_projection = nn.Linear(d_model, 1)
        
        # Маска для декодера (предотвращает заглядывание в будущее)
        self.register_buffer(
            'tgt_mask', 
            torch.tril(torch.ones(target_len, target_len)).bool()
        )
        
        # Инициализация весов
        self._init_weights()
        
    def _init_weights(self):
        """Инициализация весов."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def generate_square_subsequent_mask(self, sz):
        """Генерирует маску для предотвращения заглядывания в будущее."""
        mask = torch.tril(torch.ones(sz, sz)).bool()
        return mask
        
    def forward(self, src, tgt=None):
        """
        Прямой проход модели.
        
        Args:
            src (torch.Tensor): Исторические данные [B, T_hist, F_hist=256].
            tgt (torch.Tensor, optional): Целевые значения [B, T_pred, 1].
            
        Returns:
            torch.Tensor: Прогнозы [B, T_pred, 1].
        """
        batch_size = src.size(0)
        
        # Входная проекция
        src_emb = self.input_projection(src)  # [B, T_hist, d_model]
        
        # Позиционное кодирование
        if self.use_positional_encoding:
            src_emb = self.pos_encoding(src_emb)
            
        # Маска для энкодера (обычно None для задач прогнозирования)
        src_mask = None
        
        # Проход через энкодер
        enc_output = self.encoder(src_emb, src_mask)  # [B, T_hist, d_model]
        
        # Для обучения используем teacher forcing
        if self.training and tgt is not None:
            # Подготавливаем декодер вход (сдвигаем tgt вправо и добавляем начальный токен)
            # tgt: [B, T_pred, 1]
            tgt_input = torch.zeros(batch_size, self.target_len, self.d_model, device=src.device)
            # Простая реализация: используем нули как начальное состояние
            # В более сложной версии можно использовать предыдущие прогнозы
            
            # Позиционное кодирование для декодера
            if self.use_positional_encoding:
                tgt_input = self.pos_encoding(tgt_input)
                
            # Маска для декодера
            tgt_mask = self.generate_square_subsequent_mask(self.target_len).to(src.device)
            
            # Проход через декодер
            dec_output = self.decoder(tgt_input, enc_output, src_mask, tgt_mask)
            
            # Выходная проекция
            output = self.output_projection(dec_output)  # [B, T_pred, 1]
            
        else:
            # Инференс: генерируем по одному шагу
            outputs = []
            # Начальное состояние
            dec_input = torch.zeros(batch_size, 1, self.d_model, device=src.device)
            
            for i in range(self.target_len):
                # Позиционное кодирование
                if self.use_positional_encoding:
                    dec_input = self.pos_encoding(dec_input)
                    
                # Маска для декодера
                tgt_mask = self.generate_square_subsequent_mask(i + 1).to(src.device)
                
                # Проход через декодер
                dec_output = self.decoder(
                    dec_input, enc_output, src_mask, tgt_mask
                )
                
                # Получаем прогноз для текущего шага
                step_output = self.output_projection(dec_output[:, -1:, :])  # [B, 1, 1]
                outputs.append(step_output)
                
                # Подготавливаем вход для следующего шага
                # В реальной реализации можно использовать более сложную стратегию
                next_input = torch.zeros(batch_size, 1, self.d_model, device=src.device)
                dec_input = torch.cat([dec_input, next_input], dim=1)
                
            output = torch.cat(outputs, dim=1)  # [B, T_pred, 1]
            
        return output


# Пример использования
if __name__ == "__main__":
    # Параметры модели
    B, T_hist, feature_size = 4, 256, 256
    T_pred = 32
    output_size = 1
    
    # Создание модели
    model = TradingTransformer(
        feature_size=feature_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        ff_hidden_dim=2048,
        target_len=T_pred,
        dropout=0.1,
        use_positional_encoding=True
    )
    
    # Примерные входные данные
    src = torch.randn(B, T_hist, feature_size)  # История после TradingProcessor
    tgt = torch.randn(B, T_pred, output_size)   # Целевые значения (опционально)
    
    # Прогон модели
    model.eval()
    with torch.no_grad():
        output = model(src, tgt)
        print(f"Выход модели: {output.shape}")  # [B, 32, 1]
# src/ffn/ffn.py
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


class Attention(nn.Module):
    """
    Механизм внимания для фокусировки на важных временных шагах.
    """
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super(Attention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, hidden_size]
        Returns:
            Tensor, shape [batch_size, seq_len, hidden_size]
        """
        attn_output, _ = self.multihead_attn(x, x, x)
        # Остаточное соединение и нормализация
        x = self.norm(x + self.dropout(attn_output))
        return x


class TradingFFN(nn.Module):
    """
    Мощная Feed-Forward Network для прогнозирования цен на фондовом рынке.
    """
    def __init__(
        self,
        feature_size=256,
        hidden_sizes=[512, 1024, 512, 256],
        output_size=32,
        seq_len=256,
        num_attention_layers=3,
        num_heads=8,
        dropout=0.2,
        use_layer_norm=True
    ):
        super(TradingFFN, self).__init__()
        self.feature_size = feature_size
        self.seq_len = seq_len
        self.output_size = output_size
        self.use_layer_norm = use_layer_norm

        # Позиционное кодирование
        self.pos_encoding = PositionalEncoding(feature_size, max_len=seq_len, dropout=dropout)
        
        # Стек слоев внимания
        self.attention_layers = nn.ModuleList([
            Attention(feature_size, num_heads, dropout)
            for _ in range(num_attention_layers)
        ])
        
        # LayerNorm после внимания (опционально)
        if use_layer_norm:
            self.attn_layer_norm = nn.LayerNorm(feature_size)
        
        # Стек полносвязных слоев
        ff_layers = []
        input_dim = feature_size * seq_len
        
        for hidden_size in hidden_sizes:
            ff_layers.extend([
                nn.Linear(input_dim, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_size
            
        self.feed_forward = nn.Sequential(*ff_layers)
        
        # Выходной слой
        self.output_layer = nn.Linear(input_dim, output_size)
        
        # Инициализация весов
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
        
        # Применяем позиционное кодирование
        x = self.pos_encoding(src)  # [B, 256, 256]
        
        # Пропускаем через слои внимания
        for attn_layer in self.attention_layers:
            x = attn_layer(x)  # [B, 256, 256]
            
        # Применяем LayerNorm после внимания (опционально)
        if self.use_layer_norm:
            x = self.attn_layer_norm(x)
        
        # Сглаживаем для полносвязных слоев
        x = x.view(batch_size, -1)  # [B, 256*256]
        
        # Пропускаем через полносвязные слои
        features = self.feed_forward(x)  # [B, hidden_sizes[-1]]
        
        # Выходной слой
        output = self.output_layer(features)  # [B, 32]
        
        return output.unsqueeze(-1)  # [B, 32, 1]



# Пример использования
if __name__ == "__main__":
    # Параметры модели
    B, T_hist, feature_size = 4, 256, 256
    T_pred = 32
    output_size = 1
    
    # Создание модели
    model = TradingFFN(
        feature_size=feature_size,
        hidden_sizes=[512, 1024, 512, 256],
        output_size=T_pred,
        seq_len=T_hist,
        num_attention_layers=3,
        num_heads=8,
        dropout=0.2,
        use_layer_norm=True
    )
    
    # Примерные входные данные
    src = torch.randn(B, T_hist, feature_size)  # История после TradingProcessor
    tgt = torch.randn(B, T_pred, output_size)   # Целевые значения (опционально)
    
    # Прогон модели
    model.eval()
    with torch.no_grad():
        output = model(src, tgt)
        print(f"Выход модели: {output.shape}")  # [B, 32, 1]
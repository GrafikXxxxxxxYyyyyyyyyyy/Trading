# src/cnn/cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Остаточный блок для более глубоких CNN.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Слой проекции для остаточного соединения при несовпадении размерностей
        self.projection = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        residual = x
        if self.projection is not None:
            residual = self.projection(residual)
            
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class DilatedConvBlock(nn.Module):
    """
    Блок дилатированных сверток для захвата долгосрочных зависимостей.
    """
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2, 4, 8], dropout=0.1):
        super(DilatedConvBlock, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
            for dilation in dilation_rates
        ])
        self.bn = nn.BatchNorm1d(out_channels * len(dilation_rates))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        conv_outputs = [conv(x) for conv in self.convs]
        out = torch.cat(conv_outputs, dim=1)
        out = F.relu(self.bn(out))
        out = self.dropout(out)
        return out


class AttentionLayer(nn.Module):
    """
    Простой механизм внимания для фокусировки на важных временных шагах.
    """
    def __init__(self, channels):
        super(AttentionLayer, self).__init__()
        self.channels = channels
        self.attention = nn.Sequential(
            nn.Linear(channels, channels // 8),
            nn.ReLU(),
            nn.Linear(channels // 8, 1)
        )
        
    def forward(self, x):
        # x: [B, C, T]
        x_permuted = x.permute(0, 2, 1)  # [B, T, C]
        attention_weights = F.softmax(self.attention(x_permuted), dim=1)  # [B, T, 1]
        context = torch.sum(x * attention_weights.permute(0, 2, 1), dim=2)  # [B, C]
        return context, attention_weights


class TradingCNN(nn.Module):
    """
    Мощная CNN архитектура для прогнозирования цен на фондовом рынке.
    """
    def __init__(
        self,
        feature_size=256,
        cnn_channels=[64, 128, 256, 512],
        kernel_sizes=[3, 3, 3, 3],
        dilations=[1, 2, 4, 8],
        output_size=32,
        dropout=0.2,
        use_residual=True,
        use_dilated=True,
        use_attention=True
    ):
        super(TradingCNN, self).__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.use_residual = use_residual
        self.use_dilated = use_dilated
        self.use_attention = use_attention
        
        layers = []
        in_channels = feature_size
        
        # Стек сверточных слоев
        for i, (out_channels, kernel_size) in enumerate(zip(cnn_channels, kernel_sizes)):
            if use_residual:
                layers.append(ResidualBlock(in_channels, out_channels, kernel_size, dropout=dropout))
            else:
                layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
                layers.append(nn.BatchNorm1d(out_channels))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            in_channels = out_channels
            
        self.conv_layers = nn.Sequential(*layers)
        
        # Дилатированные свертки (опционально)
        if use_dilated:
            self.dilated_block = DilatedConvBlock(
                cnn_channels[-1], 
                cnn_channels[-1] // 4, 
                dilations, 
                dropout
            )
            attention_channels = (cnn_channels[-1] // 4) * len(dilations)
        else:
            attention_channels = cnn_channels[-1]
            
        # Механизм внимания (опционально)
        if use_attention:
            self.attention = AttentionLayer(attention_channels)
            fc_input_size = attention_channels
        else:
            fc_input_size = attention_channels * 256  # 256 временных шагов
            
        # Полносвязные слои для прогнозирования
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_size)
        )
        
        # Инициализация весов
        self._init_weights()
        
    def _init_weights(self):
        """Инициализация весов."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
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
        # src: [B, 256, 256]
        batch_size = src.size(0)
        
        # Переставляем для Conv1d: [B, F_hist, T_hist]
        x = src.transpose(1, 2).contiguous()  # [B, 256, 256]
        
        # Пропускаем через сверточные слои
        x = self.conv_layers(x)  # [B, cnn_channels[-1], 256]
        
        # Пропускаем через дилатированные свертки (если используются)
        if self.use_dilated:
            x = self.dilated_block(x)  # [B, dilated_channels, 256]
            
        # Применяем внимание или сглаживаем
        if self.use_attention:
            context, attention_weights = self.attention(x)  # [B, dilated_channels]
            x = context
        else:
            x = x.view(batch_size, -1)  # [B, dilated_channels * 256]
            
        # Пропускаем через полносвязные слои
        output = self.fc_layers(x)  # [B, 32]
        
        return output.unsqueeze(-1)  # [B, 32, 1]


# Пример использования
if __name__ == "__main__":
    # Параметры модели
    B, T_hist, feature_size = 4, 256, 256
    T_pred = 32
    output_size = 1
    
    # Создание модели
    model = TradingCNN(
        feature_size=feature_size,
        cnn_channels=[64, 128, 256, 512],
        kernel_sizes=[3, 3, 3, 3],
        dilations=[1, 2, 4, 8],
        output_size=T_pred,
        dropout=0.2,
        use_residual=True,
        use_dilated=True,
        use_attention=True
    )
    
    # Примерные входные данные
    src = torch.randn(B, T_hist, feature_size)  # История после TradingProcessor
    tgt = torch.randn(B, T_pred, output_size)   # Целевые значения (опционально)
    
    # Прогон модели
    model.eval()
    with torch.no_grad():
        output = model(src, tgt)
        print(f"Выход модели: {output.shape}")  # [B, 32, 1]
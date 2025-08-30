import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckBlock(nn.Module):
    """
    Bottleneck блок ResNet, как в ResNet-50/101/152.
    Состоит из 1x1 -> 3x3 -> 1x1 сверток.
    """
    expansion = 4  # Коэффициент увеличения размера выхода по каналам

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout=0.1):
        """
        Args:
            in_channels (int): Количество входных каналов.
            out_channels (int): Количество выходных каналов (до expansion).
            stride (int): Шаг свертки (используется в 3x3 свертке для downsampling).
            downsample (nn.Module, optional): Слой для проекции shortcut, если необходимо.
            dropout (float): Вероятность Dropout.
        """
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Третья свертка увеличивает количество каналов в expansion раз
        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (Tensor): Входной тензор [B, C_in, T]
        Returns:
            Tensor: Выходной тензор [B, C_out_expanded, T]
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        # Применяем downsampling к shortcut, если необходимо
        if self.downsample is not None:
            identity = self.downsample(x)

        # Остаточное соединение
        out += identity
        out = self.relu(out)
        out = self.dropout(out)

        return out


class ResNetCNN(nn.Module):
    """
    ResNet-подобная CNN для обработки временных рядов.
    Адаптирована для 1D данных (временные ряды).
    """
    def __init__(self, block, layers, input_channels=128, num_classes=32, dropout=0.2):
        """
        Args:
            block (nn.Module): Тип блока (BottleneckBlock).
            layers (list): Список, определяющий количество блоков в каждом слое.
                          Например, [3, 4, 23, 3] для ResNet-101.
            input_channels (int): Количество входных признаков (128 от TradingProcessor).
            num_classes (int): Размерность выхода (32 точки прогноза).
            dropout (float): Вероятность Dropout в голове.
        """
        super(ResNetCNN, self).__init__()
        self.in_channels = 64  # Начальное количество каналов
        self.dropout = dropout

        # Начальный слой для преобразования входа в последовательность признаков
        # Предполагаем, что вход [B, T, F] -> [B, F, T] для Conv1d
        # Но мы хотим обрабатывать временные зависимости, поэтому свертка по времени
        # Для этого вход будет [B, F_in, T], где F_in=128, T=256
        # Начальный conv1d преобразует F_in -> 64 признаков
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Создаем слои ResNet
        # Для 1D данные, "layer1" не делает downsampling (stride=1)
        self.layer1 = self._make_layer(block, 64, layers[0], dropout=dropout)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dropout=dropout)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dropout=dropout)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dropout=dropout)

        self.avgpool = nn.AdaptiveAvgPool1d(1) # Глобальный average pooling
        
        # Голова для прогнозирования
        # После layer4 у нас будет 512 * expansion = 512 * 4 = 2048 каналов
        self.fc_head = nn.Sequential(
            nn.Linear(512 * block.expansion, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes) # Выход [B, 32]
        )
        
        # Инициализация весов
        self._init_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1, dropout=0.1):
        """
        Создает слой ResNet, состоящий из нескольких bottleneck блоков.
        """
        downsample = None
        # Если stride != 1 или количество каналов изменилось, нужна проекция shortcut
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        # Первый блок может делать downsampling
        layers.append(block(self.in_channels, out_channels, stride, downsample, dropout))
        self.in_channels = out_channels * block.expansion
        # Остальные блоки без downsampling
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, dropout=dropout))

        return nn.Sequential(*layers)

    def _init_weights(self):
        """Инициализация весов."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Прямой проход.
        Args:
            x (Tensor): Входной тензор [B, F_in=128, T=256]
        Returns:
            Tensor: Выходной тензор [B, num_classes=32]
        """
        # x: [B, 128, 256]
        x = self.conv1(x)       # [B, 64, 128]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # [B, 64, 64]

        x = self.layer1(x)      # [B, 256, 64]  (64 = 64 * 4)
        x = self.layer2(x)      # [B, 512, 32]  (512 = 128 * 4)
        x = self.layer3(x)      # [B, 1024, 16] (1024 = 256 * 4)
        x = self.layer4(x)      # [B, 2048, 8]  (2048 = 512 * 4)

        # Глобальный average pooling: [B, 2048, 8] -> [B, 2048, 1] -> [B, 2048]
        x = self.avgpool(x).view(x.size(0), -1)
        
        # Голова прогнозирования: [B, 2048] -> [B, 32]
        x = self.fc_head(x)
        
        # Добавляем размерность для совместимости с другими моделями [B, 32] -> [B, 32, 1]
        x = x.unsqueeze(-1)
        
        return x


class TradingCNN(nn.Module):
    """
    Мощная CNN модель на основе ResNet-101 для прогнозирования цен на фондовом рынке.
    """
    def __init__(
        self,
        feature_size=128,           # Размер признаков от TradingProcessor
        target_len=32,              # Длина прогноза
        dropout=0.2,                # Dropout
        use_layer_norm=True,        # Использовать LayerNorm на входе
        resnet_layers=[3, 4, 23, 3] # Конфигурация слоев ResNet (ResNet-101)
    ):
        """
        Args:
            feature_size (int): Размер входных признаков (128).
            target_len (int): Длина прогнозируемой последовательности (32).
            dropout (float): Вероятность Dropout.
            use_layer_norm (bool): Применять ли LayerNorm к входу.
            resnet_layers (list): Конфигурация слоев ResNet.
        """
        super(TradingCNN, self).__init__()
        self.feature_size = feature_size
        self.target_len = target_len
        self.use_layer_norm = use_layer_norm
        
        # LayerNorm для нормализации входа
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(feature_size)
            
        # Основная ResNet-подобная CNN
        self.resnet_cnn = ResNetCNN(
            block=BottleneckBlock,
            layers=resnet_layers, # [3, 4, 23, 3] для ResNet-101
            input_channels=feature_size,
            num_classes=target_len,
            dropout=dropout
        )

    def forward(self, src, tgt=None):
        """
        Прямой проход модели.
        
        Args:
            src (torch.Tensor): Исторические данные [B, T_hist=256, F_hist=128].
                                Должны быть уже обработаны TradingProcessor.
            tgt (torch.Tensor, optional): Целевые значения [B, T_pred, 1].
            
        Returns:
            torch.Tensor: Прогнозы [B, T_pred=32, 1].
        """
        # src: [B, 256, 128]
        batch_size, seq_len, _ = src.size()
        
        # Применяем Layer Normalization (если включено)
        if self.use_layer_norm:
            src = self.layer_norm(src) # [B, 256, 128]
            
        # Переставляем размерности для CNN: [B, F_hist, T_hist]
        x = src.transpose(1, 2).contiguous() # [B, 128, 256]
        
        # Пропускаем через ResNet CNN
        # y: [B, 32, 1]
        y = self.resnet_cnn(x)
        
        return y
    


# # --- Пример использования ---
# if __name__ == "__main__":
#     # Параметры модели
#     B, T_hist, feature_size = 4, 256, 128
#     T_pred = 32
#     output_size = 1
    
#     # Создание модели (ResNet-101)
#     model = TradingCNN(
#         feature_size=feature_size,
#         target_len=T_pred,
#         dropout=0.2,
#         use_layer_norm=True,
#         resnet_layers=[3, 4, 23, 3] # ResNet-101
#     ).to('mps')
    
#     # Примерные входные данные
#     src = torch.randn(B, T_hist, feature_size).to('mps')  # История после TradingProcessor
#     tgt = torch.randn(B, T_pred, output_size).to('mps')   # Целевые значения (опционально)
    
#     # Прогон модели
#     model.eval()
#     with torch.no_grad():
#         output = model(src, tgt)
#         print(f"Выход модели: {output.shape}") # [B, 32, 1]
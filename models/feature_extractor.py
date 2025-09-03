# models/feature_extractor.py
import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvEncoder(nn.Module):
    """
    Сверточный энкодер для преобразования [B, 256, 5] в [B, 256, 256].
    """
    def __init__(self, input_channels=5, output_channels=256, hidden_dims=[32, 64]):
        """
        Args:
            input_channels (int): Количество входных каналов (5 для OHLCV).
            output_channels (int): Количество выходных каналов (256).
            hidden_dims (list): Список количества каналов для промежуточных слоев.
        """
        super(ConvEncoder, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_dims = hidden_dims

        layers = []
        in_channels = input_channels
        
        # Стек сверточных слоев для увеличения количества каналов
        # Conv1d: (in_channels, out_channels, kernel_size, padding)
        # padding='same' эквивалентно padding = kernel_size // 2 для нечетных kernel_size
        for h_dim in hidden_dims:
            layers.append(
                nn.Conv1d(in_channels, h_dim, kernel_size=3, padding=1)
            )
            layers.append(nn.BatchNorm1d(h_dim)) # Нормализация для стабильности
            layers.append(nn.ReLU())
            in_channels = h_dim
            
        # Финальный слой для получения 256 каналов
        layers.append(
            nn.Conv1d(in_channels, output_channels, kernel_size=3, padding=1)
        )
        layers.append(nn.BatchNorm1d(output_channels))
        # Не добавляем активацию на выходе энкодера, 
        # так как это будут признаки для декодера
        
        self.layers = nn.Sequential(*layers)
        
        self._init_weights()


    def _init_weights(self):
        """Инициализация весов."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        """
        Args:
            x (Tensor): Входной тензор [B, 256, 5].
        Returns:
            Tensor: Признаки [B, 256, 256].
        """
        # x: [B, 256, 5] -> Переставляем для Conv1d: [B, 5, 256]
        x = x.transpose(1, 2)
        # Проходим через сверточные слои
        features = self.layers(x) # [B, 256, 256]
        # Возвращаем к формату [B, 256, 256]
        features = features.transpose(1, 2)

        return features



class ConvDecoder(nn.Module):
    """
    Сверточный декодер для преобразования [B, 256, 256] в [B, 256, 5].
    """
    def __init__(self, input_channels=256, output_channels=5, hidden_dims=[64, 32]):
        """
        Args:
            input_channels (int): Количество входных каналов (256).
            output_channels (int): Количество выходных каналов (5 для OHLCV).
            hidden_dims (list): Список количества каналов для промежуточных слоев.
        """
        super(ConvDecoder, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_dims = hidden_dims

        layers = []
        in_channels = input_channels
        
        # Стек сверточных слоев для уменьшения количества каналов
        for h_dim in hidden_dims:
            layers.append(
                nn.Conv1d(in_channels, h_dim, kernel_size=3, padding=1)
            )
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            in_channels = h_dim
            
        # Финальный слой для получения 5 каналов
        # Используем Tanh или другую ограниченную активацию для выхода,
        # предполагая, что данные нормализованы или ограничены
        layers.append(
            nn.Conv1d(in_channels, output_channels, kernel_size=3, padding=1)
        )
        
        self.layers = nn.Sequential(*layers)
        
        self._init_weights()


    def _init_weights(self):
        """Инициализация весов."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, z):
        """
        Args:
            z (Tensor): Признаки [B, 256, 256].
        Returns:
            Tensor: Реконструированный тензор [B, 256, 5].
        """
        # z: [B, 256, 128] -> Переставляем для Conv1d: [B, 128, 256]
        z = z.transpose(1, 2)
        # Проходим через сверточные слои
        reconstructed = self.layers(z) # [B, 5, 256]
        # Возвращаем к формату [B, 256, 5]
        reconstructed = reconstructed.transpose(1, 2)

        return reconstructed



class TradingFeatureExtractor(nn.Module):
    """
    Автоэнкодер для извлечения признаков из временных рядов OHLCV.
    """
    def __init__(self, input_size=5, feature_size=256):
        """
        Args:
            input_size (int): Размерность входных данных (5 для OHLCV).
            feature_size (int): Размерность извлекаемых признаков (256).
        """
        super(TradingFeatureExtractor, self).__init__()
        self.input_size = input_size
        self.feature_size = feature_size
        
        # Инициализируем энкодер и декодер
        self.encoder = ConvEncoder(
            input_channels=input_size, 
            output_channels=feature_size,
            hidden_dims=[32, 64, 128, 128] 
        )
        self.decoder = ConvDecoder(
            input_channels=feature_size, 
            output_channels=input_size,
            hidden_dims=[128, 128, 64, 32] 
        )


    def forward(self, x):
        """
        Прямой проход автоэнкодера.
        
        Args:
            x (Tensor): Входные данные [B, 256, 5].
            
        Returns:
            reconstructed_x (Tensor): Реконструированные данные [B, 256, 5].
            features (Tensor): Извлеченные признаки [B, 256, 256].
        """
        # Извлекаем признаки
        features = self.encoder(x) # [B, 256, 256]
        # Реконструируем вход
        reconstructed_x = self.decoder(features) # [B, 256, 5]

        return reconstructed_x


    def extract_features(self, x):
        """
        Извлекает признаки из входных данных.
        Используется после обучения автоэнкодера.
        
        Args:
            x (Tensor): Входные данные [B, 256, 5].
            
        Returns:
            features (Tensor): Извлеченные признаки [B, 256, 256].
        """
        with torch.no_grad(): # Отключаем градиенты для инференса
            features = self.encoder(x) # [B, 256, 256]

        return features
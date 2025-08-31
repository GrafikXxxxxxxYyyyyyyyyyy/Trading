import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """
    Отсекает лишние элементы из последовательности после свертки с паддингом.
    Обеспечивает причинность (causality) свертки.
    """
    def __init__(self, chomp_size):
        """
        Args:
            chomp_size (int): Количество элементов для отсечения с правого края.
        """
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size


    def forward(self, x):
        """
        Args:
            x (Tensor): Входной тензор [B, C, T]
        Returns:
            Tensor: Выходной тензор [B, C, T - chomp_size]
        """
        return x[:, :, :-self.chomp_size].contiguous()



class TemporalBlock(nn.Module):
    """
    Базовый блок TCN: две причинные свертки с Dilated Conv + ReLU + Dropout.
    Использует остаточные соединения.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        Args:
            n_inputs (int): Количество входных каналов.
            n_outputs (int): Количество выходных каналов.
            kernel_size (int): Размер ядра свертки.
            stride (int): Шаг свертки.
            dilation (int): Коэффициент расширения (dilation).
            padding (int): Размер паддинга.
            dropout (float): Вероятность Dropout.
        """
        super(TemporalBlock, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Первый сверточный слой
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)  # Убираем паддинг, чтобы сохранить причинность
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Второй сверточный слой
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Последовательность для первого блока
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # Слой проекции для остаточного соединения, если размерности не совпадают
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
        self.init_weights()


    def init_weights(self):
        """Инициализация весов сверточных слоев."""
        # Инициализация для conv1
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        if self.conv1.bias is not None:
            nn.init.constant_(self.conv1.bias, 0)
            
        # Инициализация для conv2
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        if self.conv2.bias is not None:
            nn.init.constant_(self.conv2.bias, 0)
            
        # Инициализация слоя проекции (если он есть)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, mode='fan_out', nonlinearity='relu')
            if self.downsample.bias is not None:
                nn.init.constant_(self.downsample.bias, 0)


    def forward(self, x):
        """
        Прямой проход через блок.
        Args:
            x (Tensor): Входной тензор [B, n_inputs, T]
        Returns:
            Tensor: Выходной тензор [B, n_outputs, T]
        """
        out = self.net(x)
        # Остаточное соединение
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)



class TemporalConvNet(nn.Module):
    """
    Полная TCN архитектура, состоящая из стека TemporalBlock'ов.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        Args:
            num_inputs (int): Количество входных признаков.
            num_channels (list): Список количества каналов для каждого блока.
                             Например, [256, 256, 256] означает 3 блока по 256 каналов.
            kernel_size (int): Размер ядра свертки.
            dropout (float): Вероятность Dropout.
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            # Вычисляем параметры для текущего блока
            dilation_size = 2 ** i  # Экспоненциально увеличиваем dilation
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Добавляем блок
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, 
                                   dilation=dilation_size, padding=(kernel_size-1) * dilation_size, 
                                   dropout=dropout)]

        self.network = nn.Sequential(*layers)
        

    def forward(self, x):
        """
        Прямой проход через всю сеть.
        Args:
            x (Tensor): Входной тензор [B, num_inputs, T]
        Returns:
            Tensor: Выходной тензор [B, num_channels[-1], T]
        """
        return self.network(x)



class TradingTCN(nn.Module):
    """
    Полная модель TCN для прогнозирования цен на фондовом рынке.
    """
    def __init__(
        self,
        feature_size=128,           # Размер признаков от TradingProcessor
        num_channels=[64, 64, 64, 128, 128, 128, 256, 256, 512, 512], # Архитектура TCN
        kernel_size=3,              # Размер ядра свертки
        dropout=0.2,                # Dropout
        target_len=32,              # Длина прогноза
        use_layer_norm=True,        # Использовать LayerNorm
        use_weight_norm=True        # Использовать WeightNorm
    ):
        """
        Args:
            feature_size (int): Размер входных признаков (128).
            num_channels (list): Список количества каналов для каждого TCN блока.
            kernel_size (int): Размер ядра свертки.
            dropout (float): Вероятность Dropout.
            target_len (int): Длина прогнозируемой последовательности (32).
            use_layer_norm (bool): Применять ли LayerNorm к входу.
            use_weight_norm (bool): Применять ли WeightNorm к сверточным слоям.
        """
        super(TradingTCN, self).__init__()
        self.feature_size = feature_size
        self.target_len = target_len
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        
        # LayerNorm для нормализации входа
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(feature_size)
        
        # TCN сеть
        self.tcn = TemporalConvNet(feature_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        
        # Выходной слой для прогнозирования
        # Вход: [B, num_channels[-1], T_hist]
        # Выход: [B, 1, T_hist] (прогнозируем только цену закрытия)
        self.output_projection = nn.Conv1d(num_channels[-1], 1, 1) # 1x1 conv для проекции каналов
        
        # Дополнительный слой для генерации последовательности прогноза
        # Мы можем использовать последние target_len точек из выхода TCN
        # или применить дополнительную обработку
        
        self.init_weights()
        
        # Применяем WeightNorm, если требуется
        if use_weight_norm:
            self.apply_weight_norm()
            

    def apply_weight_norm(self):
        """Применяет WeightNorm к сверточным слоям."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                module = nn.utils.weight_norm(module)


    def init_weights(self):
        """Инициализация весов выходного слоя."""
        nn.init.kaiming_normal_(self.output_projection.weight, mode='fan_out', nonlinearity='relu')
        if self.output_projection.bias is not None:
            nn.init.constant_(self.output_projection.bias, 0)


    def forward(self, src, tgt=None):
        """
        Прямой проход модели.
        
        Args:
            src (torch.Tensor): Исторические данные [B, T_hist, F_hist=128].
                                Должны быть уже обработаны TradingProcessor.
            tgt (torch.Tensor, optional): Целевые значения [B, T_pred, 1].
            
        Returns:
            torch.Tensor: Прогнозы [B, T_pred, 1].
        """
        # src: [B, T_hist, 128]
        batch_size, seq_len, _ = src.size()
        
        # Применяем Layer Normalization (если включено)
        if self.use_layer_norm:
            src = self.layer_norm(src) # [B, T_hist, 128]
            
        # Переставляем размерности для TCN: [B, F_hist, T_hist]
        x = src.transpose(1, 2).contiguous() # [B, 128, T_hist]
        
        # Пропускаем через TCN
        # y: [B, num_channels[-1], T_hist]
        y = self.tcn(x)
        
        # Применяем выходную проекцию для получения прогноза цены закрытия
        # output: [B, 1, T_hist]
        output = self.output_projection(y)
        
        # Берем последние target_len точек для прогноза
        # output: [B, 1, T_hist] -> [B, 1, target_len] -> [B, target_len, 1]
        prediction = output[:, :, -self.target_len:].transpose(1, 2)
        
        return prediction



# # --- Пример использования ---
# if __name__ == "__main__":
#     # Параметры модели
#     B, T_hist, feature_size = 4, 256, 128
#     T_pred = 32
#     output_size = 1
    
#     # Создание модели
#     model = TradingTCN(
#         feature_size=feature_size,
#         num_channels=[64, 64, 64, 128, 128, 128, 256, 256, 256, 512], # Глубокая архитектура
#         kernel_size=3,
#         dropout=0.2,
#         target_len=T_pred,
#         use_layer_norm=True,
#         use_weight_norm=True
#     )
    
#     # Примерные входные данные
#     src = torch.randn(B, T_hist, feature_size)  # История после TradingProcessor
#     tgt = torch.randn(B, T_pred, output_size)   # Целевые значения (опционально)
    
#     # Прогон модели
#     model.eval()
#     with torch.no_grad():
#         output = model(src, tgt)
#         print(f"Выход модели: {output.shape}") # [B, 32, 1]
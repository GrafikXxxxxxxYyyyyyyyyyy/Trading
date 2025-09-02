import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LSTMEncoder(nn.Module):
    """
    LSTM Энкодер для обработки исторических данных.
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0, bidirectional=False):
        """
        Args:
            input_size (int): Размер входных признаков (128 от TradingProcessor).
            hidden_size (int): Размер скрытого состояния LSTM.
            num_layers (int): Количество слоев LSTM.
            dropout (float): Вероятность Dropout между слоями.
            bidirectional (bool): Использовать_bidirectional LSTM.
        """
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        # Если_bidirectional, добавляем слой для проекции выходов в нужный размер
        if bidirectional:
            self.output_projection = nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.output_projection = None
            
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов."""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                # Для весов LSTM используем ортогональную инициализацию
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        """
        Прогоняет вход через LSTM.
        
        Args:
            x (torch.Tensor): Входные данные [B, T_hist, input_size].
            
        Returns:
            outputs (torch.Tensor): Все скрытые состояния [B, T_hist, hidden_size * num_directions].
            (h_n, c_n) (tuple): Финальные скрытое и ячейковое состояния 
                               [num_layers * num_directions, B, hidden_size].
        """
        # outputs: [B, T_hist, hidden_size * num_directions]
        # h_n, c_n: [num_layers * num_directions, B, hidden_size]
        outputs, (h_n, c_n) = self.lstm(x)
        
        # Если_bidirectional, проецируем выходы
        if self.bidirectional:
            outputs = self.output_projection(outputs)
            
        return outputs, (h_n, c_n)


class FFNHead(nn.Module):
    """
    Голова прогнозирования на основе Feed-Forward Network.
    Принимает выходы LSTM и генерирует прогноз.
    """
    def __init__(self, encoder_output_size, target_len, output_size_per_step=1, 
                 hidden_dims=[512, 256], dropout=0.2):
        """
        Args:
            encoder_output_size (int): Размер выхода энкодера (размер признаков для каждого временного шага).
            target_len (int): Длина прогнозируемой последовательности (32).
            output_size_per_step (int): Размер прогноза на один шаг (обычно 1 для цены закрытия).
            hidden_dims (list): Список размеров скрытых слоев FFN.
            dropout (float): Вероятность Dropout.
        """
        super(FFNHead, self).__init__()
        self.encoder_output_size = encoder_output_size
        self.target_len = target_len
        self.output_size_per_step = output_size_per_step
        
        # Вход в FFN: сначала сглаживаем выходы LSTM
        # Мы будем использовать все выходы LSTM [B, T_hist, H] -> [B, T_hist * H]
        # Это позволяет FFN учитывать всю историю
        input_dim = encoder_output_size * 256 # Предполагаем T_hist = 256
        
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
            
        # Выходной слой
        # Выход: [B, target_len * output_size_per_step]
        layers.append(nn.Linear(prev_dim, target_len * output_size_per_step))
        
        self.ffn = nn.Sequential(*layers)
        
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов."""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, encoder_outputs, encoder_hidden_states):
        """
        Генерирует прогноз на основе выходов энкодера.
        
        Args:
            encoder_outputs (torch.Tensor): Выходы LSTM для каждого временного шага 
                                          [B, T_hist, encoder_output_size].
            encoder_hidden_states (tuple): (h_n, c_n) - финальные скрытые состояния энкодера.
            
        Returns:
            torch.Tensor: Прогноз [B, target_len, output_size_per_step].
        """
        batch_size = encoder_outputs.size(0)
        
        # 1. Способ 1: Использовать все выходы LSTM
        # Преобразуем [B, T_hist, H] -> [B, T_hist * H]
        flattened_outputs = encoder_outputs.view(batch_size, -1)
        
        # 2. Пропускаем через FFN
        # Выход FFN: [B, target_len * output_size_per_step]
        predictions_flat = self.ffn(flattened_outputs)
        
        # 3. Преобразуем в нужную форму
        # [B, target_len * output_size_per_step] -> [B, target_len, output_size_per_step]
        predictions = predictions_flat.view(batch_size, self.target_len, self.output_size_per_step)
        
        return predictions


class TradingLSTM_FFN(nn.Module):
    """
    Мощная модель LSTM Encoder + FFN Head для прогнозирования цен на фондовом рынке.
    """
    def __init__(
        self,
        feature_size=128,           # Размер признаков от TradingProcessor
        encoder_hidden_size=256,    # Размер скрытого состояния LSTM энкодера
        encoder_num_layers=2,       # Количество слоев LSTM энкодера
        encoder_bidirectional=True, # Использовать_bidirectional LSTM в энкодере
        ffn_hidden_dims=[512, 256], # Размеры скрытых слоев FFN головы
        target_len=32,              # Длина прогноза
        output_size_per_step=1,     # Размер прогноза на один шаг
        dropout=0.2,                # Dropout
        use_layer_norm=True         # Использовать LayerNorm на входе
    ):
        """
        Args:
            feature_size (int): Размер входных признаков (128).
            encoder_hidden_size (int): Размер скрытого состояния LSTM.
            encoder_num_layers (int): Количество слоев LSTM.
            encoder_bidirectional (bool): Использовать_bidirectional LSTM.
            ffn_hidden_dims (list): Размеры скрытых слоев FFN.
            target_len (int): Длина прогнозируемой последовательности (32).
            output_size_per_step (int): Размер прогноза на один шаг.
            dropout (float): Вероятность Dropout.
            use_layer_norm (bool): Применять ли LayerNorm к входу.
        """
        super(TradingLSTM_FFN, self).__init__()
        self.feature_size = feature_size
        self.target_len = target_len
        self.use_layer_norm = use_layer_norm
        
        # LayerNorm для нормализации входа
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(feature_size)
            
        # LSTM Энкодер
        self.encoder = LSTMEncoder(
            input_size=feature_size,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_num_layers,
            dropout=dropout,
            bidirectional=encoder_bidirectional
        )
        
        # Определяем размер выхода энкодера
        # Если_bidirectional, выход LSTM проецируется до encoder_hidden_size
        encoder_output_size = encoder_hidden_size 
        
        # FFN Голова для прогнозирования
        self.ffn_head = FFNHead(
            encoder_output_size=encoder_output_size,
            target_len=target_len,
            output_size_per_step=output_size_per_step,
            hidden_dims=ffn_hidden_dims,
            dropout=dropout
        )
        
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов."""
        # Веса энкодера и FFN головы инициализируются в их конструкторах
        pass

    def forward(self, src, tgt=None):
        """
        Прямой проход модели.
        
        Args:
            src (torch.Tensor): Исторические данные после обработки 
                              [B, T_hist=256, F_hist=128].
            tgt (torch.Tensor, optional): Целевые значения [B, T_pred, 1]. 
                                        Не используется в этой архитектуре.
            
        Returns:
            torch.Tensor: Прогнозы [B, T_pred, 1].
        """
        # src: [B, 256, 128]
        batch_size, seq_len, _ = src.size()
        
        # Применяем Layer Normalization (если включено)
        if self.use_layer_norm:
            src = self.layer_norm(src) # [B, 256, 128]
        
        # 1. Энкодер
        # encoder_outputs: [B, 256, H_enc]
        # encoder_hidden_states: ([num_layers*dirs, B, H_enc], [num_layers*dirs, B, H_enc])
        encoder_outputs, encoder_hidden_states = self.encoder(src)
        
        # 2. FFN Голова
        # predictions: [B, 32, 1]
        predictions = self.ffn_head(encoder_outputs, encoder_hidden_states)
        
        return predictions
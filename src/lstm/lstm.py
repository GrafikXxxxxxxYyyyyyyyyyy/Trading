import torch
import torch.nn as nn
import torch.nn.functional as F



class BahdanauAttention(nn.Module):
    """
    Механизм внимания Бахданау (Additive Attention).
    Позволяет декодеру фокусироваться на разных частях выхода энкодера.
    """
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, attention_dim):
        """
        Args:
            encoder_hidden_dim (int): Размер скрытого состояния энкодера (H_enc).
            decoder_hidden_dim (int): Размер скрытого состояния декодера (H_dec).
            attention_dim (int): Размер внутреннего представления внимания.
        """
        super(BahdanauAttention, self).__init__()
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.attention_dim = attention_dim

        # Линейные слои для преобразования входов перед вычислением внимания
        self.W_enc = nn.Linear(encoder_hidden_dim, attention_dim, bias=False)
        self.W_dec = nn.Linear(decoder_hidden_dim, attention_dim, bias=False)
        self.V = nn.Linear(attention_dim, 1, bias=False)

        # Инициализация весов для лучшей сходимости
        self._init_weights()


    def _init_weights(self):
        """Инициализация весов для лучшей сходимости."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)


    def forward(self, encoder_outputs, decoder_hidden):
        """
        Вычисляет веса внимания и контекстный вектор.
        
        Args:
            encoder_outputs (torch.Tensor): Выходы энкодера [B, T_hist, H_enc].
            decoder_hidden (torch.Tensor): Скрытое состояние декодера [B, H_dec].
            
        Returns:
            context_vector (torch.Tensor): Контекстный вектор [B, H_enc].
            attention_weights (torch.Tensor): Веса внимания [B, T_hist].
        """
        batch_size, seq_len, _ = encoder_outputs.size()
        
        # decoder_hidden: [B, H_dec] -> [B, 1, H_dec] для broadcast
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1)
        
        # Вычисляем оценку внимания e_tj
        energy = torch.tanh(
            self.W_enc(encoder_outputs) + self.W_dec(decoder_hidden_expanded)
        )
        attention_scores = self.V(energy).squeeze(2) # [B, T_hist]

        # Применяем softmax для получения весов внимания
        attention_weights = F.softmax(attention_scores, dim=1) # [B, T_hist]

        # Вычисляем контекстный вектор как взвешенную сумму выходов энкодера
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context_vector, attention_weights



class LSTMEncoder(nn.Module):
    """
    LSTM Энкодер для обработки исторических данных.
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0, bidirectional=False):
        """
        Args:
            input_size (int): Размер входных признаков (128).
            hidden_size (int): Размер скрытого состояния LSTM.
            num_layers (int): Количество слоев LSTM.
            dropout (float): Вероятность Dropout между слоями.
            bidirectional (bool): Использовать_bidirectional LSTM.
        """
        super(LSTMEncoder, self).__init__()
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
            (h_n, c_n) (tuple): Финальные скрытое и ячейковое состояния [num_layers * num_directions, B, hidden_size].
        """
        outputs, (h_n, c_n) = self.lstm(x)
        
        # Если_bidirectional, проецируем выходы
        if self.bidirectional:
            outputs = self.output_projection(outputs)
            
        return outputs, (h_n, c_n)



class LSTMDecoder(nn.Module):
    """
    LSTM Декодер с механизмом внимания для генерации прогноза.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, attention, dropout=0.0):
        """
        Args:
            input_size (int): Размер входных признаков для декодера (обычно 1 - цена закрытия предыдущего шага).
            hidden_size (int): Размер скрытого состояния LSTM декодера.
            num_layers (int): Количество слоев LSTM декодера.
            output_size (int): Размер выхода на каждом шаге (обычно 1).
            attention (BahdanauAttention): Экземпляр механизма внимания.
            dropout (float): Вероятность Dropout.
        """
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.attention = attention
        
        # Вход декодера состоит из:
        # 1. Предыдущий прогноз/таргет (input_size)
        # 2. Контекстный вектор от внимания (attention.encoder_hidden_dim)
        self.lstm = nn.LSTM(
            input_size=input_size + attention.encoder_hidden_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Полносвязный слой для преобразования выхода LSTM декодера в размер прогноза
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + attention.encoder_hidden_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
        
        self._init_weights()


    def _init_weights(self):
        """Инициализация весов."""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)


    def forward(self, target_len, encoder_outputs, encoder_hidden, encoder_cell, teacher_forcing_ratio=0.5, targets=None, initial_input=None):
        """
        Генерирует последовательность прогнозов.
        
        Args:
            target_len (int): Длина последовательности для генерации (32).
            encoder_outputs (torch.Tensor): Выходы энкодера [B, T_hist, H_enc].
            encoder_hidden (torch.Tensor): Финальное скрытое состояние энкодера [num_layers_enc * num_directions, B, H_enc].
            encoder_cell (torch.Tensor): Финальное ячейковое состояние энкодера [num_layers_enc * num_directions, B, H_enc].
            teacher_forcing_ratio (float): Вероятность использования teacher forcing.
            targets (torch.Tensor, optional): Реальные таргеты [B, T_pred, 1] для teacher forcing.
            initial_input (torch.Tensor, optional): Начальное значение для декодера [B, 1, input_size].
            
        Returns:
            outputs (torch.Tensor): Сгенерированные прогнозы [B, T_pred, output_size].
            attention_weights_list (list): Список весов внимания для каждого шага [T_pred, B, T_hist].
        """
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        
        # Инициализируем выходы декодера
        outputs = torch.zeros(batch_size, target_len, self.output_size, device=device)
        
        # Инициализируем список для хранения весов внимания
        attention_weights_list = []
        
        # Инициализируем вход декодера
        if initial_input is not None:
            decoder_input = initial_input # [B, 1, input_size]
        else:
            # Используем ноль как начальное значение
            decoder_input = torch.zeros(batch_size, 1, 1, device=device) # [B, 1, 1]

        # Инициализируем скрытое и ячейковое состояние декодера
        # Берем последние num_layers слоев из encoder_hidden/encoder_cell
        decoder_hidden = encoder_hidden[-self.num_layers:] if encoder_hidden.size(0) >= self.num_layers else encoder_hidden
        decoder_cell = encoder_cell[-self.num_layers:] if encoder_cell.size(0) >= self.num_layers else encoder_cell
        
        for t in range(target_len):
            # Вычисляем контекстный вектор с помощью внимания
            context_vector, attention_weights = self.attention(encoder_outputs, decoder_hidden[-1])
            attention_weights_list.append(attention_weights)
            
            # Подготовка входа для LSTM декодера
            lstm_input = torch.cat((decoder_input, context_vector.unsqueeze(1)), dim=2)
            
            # Прогоняем через LSTM
            lstm_output, (decoder_hidden, decoder_cell) = self.lstm(lstm_input, (decoder_hidden, decoder_cell))
            
            # Подготовка входа для полносвязного слоя
            fc_input = torch.cat((lstm_output.squeeze(1), context_vector), dim=1)
            
            # Прогноз на текущем шаге
            output = self.fc(fc_input)
            outputs[:, t:t+1] = output.unsqueeze(1)
            
            # Подготовка decoder_input для следующего шага
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            if use_teacher_forcing and targets is not None and t < targets.size(1):
                decoder_input = targets[:, t:t+1]
            else:
                decoder_input = output.unsqueeze(1)

        return outputs, attention_weights_list



class TradingLSTM(nn.Module):
    """
    Полная LSTM Encoder-Decoder модель с вниманием для прогнозирования цен.
    """
    def __init__(
        self,
        feature_size=128,
        encoder_hidden_size=256,
        encoder_num_layers=2,
        decoder_hidden_size=256,
        decoder_num_layers=2,
        attention_dim=128,
        target_len=32,
        dropout=0.2,
        teacher_forcing_ratio=0.5,
        bidirectional_encoder=True,
        use_layer_norm=True
    ):
        """
        Args:
            feature_size (int): Размер входных признаков после TradingProcessor (128).
            encoder_hidden_size (int): Размер скрытого состояния энкодера.
            encoder_num_layers (int): Количество слоев энкодера.
            decoder_hidden_size (int): Размер скрытого состояния декодера.
            decoder_num_layers (int): Количество слоев декодера.
            attention_dim (int): Размер внутреннего представления внимания.
            target_len (int): Длина прогнозируемой последовательности (32).
            dropout (float): Вероятность Dropout.
            teacher_forcing_ratio (float): Вероятность использования teacher forcing при обучении.
            bidirectional_encoder (bool): Использовать_bidirectional LSTM в энкодере.
            use_layer_norm (bool): Использовать Layer Normalization.
        """
        super(TradingLSTM, self).__init__()
        self.target_len = target_len
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.bidirectional_encoder = bidirectional_encoder
        
        # Создаем механизм внимания
        self.attention = BahdanauAttention(
            encoder_hidden_dim=encoder_hidden_size,
            decoder_hidden_dim=decoder_hidden_size,
            attention_dim=attention_dim
        )
        
        # Создаем энкодер
        self.encoder = LSTMEncoder(
            input_size=feature_size,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_num_layers,
            dropout=dropout,
            bidirectional=bidirectional_encoder
        )
        
        # Создаем декодер
        self.decoder = LSTMDecoder(
            input_size=1, # Прогнозируемая цена закрытия
            hidden_size=decoder_hidden_size,
            num_layers=decoder_num_layers,
            output_size=1, # Одна цена закрытия на выходе
            attention=self.attention,
            dropout=dropout
        )
        
        # Дополнительный слой нормализации (опционально)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(feature_size)
            
        self._init_weights()


    def _init_weights(self):
        """Инициализация весов."""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)


    def forward(self, src, tgt=None):
        """
        Прямой проход модели.
        
        Args:
            src (torch.Tensor): Исторические данные после обработки [B, T_hist, F_hist=128].
            tgt (torch.Tensor, optional): Целевые значения (таргеты) [B, T_pred, 1]. Используется для teacher forcing.
            
        Returns:
            output (torch.Tensor): Прогнозы [B, T_pred, 1].
        """
        batch_size = src.size(0)
        
        # Применяем Layer Normalization (если включено)
        if self.use_layer_norm:
            src = self.layer_norm(src)
        
        # 1. Энкодер
        encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder(src)
        
        # 2. Подготовка начального значения для декодера
        # Можно использовать последнее значение цены закрытия из исходных данных
        # Предполагаем, что цена закрытия - это 4-й столбец в исходных данных (до обработки)
        # Но так как у нас уже обработанные данные, можно использовать среднее или другое значение
        # Для простоты возьмем среднее значение последних N точек
        # initial_input = src[:, -1:, 3:4] # Если бы у нас были исходные данные
        # Или просто ноль
        initial_input = None # Пусть декодер сам решит
        
        # 3. Декодер
        decoder_outputs, _ = self.decoder(
            target_len=self.target_len,
            encoder_outputs=encoder_outputs,
            encoder_hidden=encoder_hidden,
            encoder_cell=encoder_cell,
            teacher_forcing_ratio=self.teacher_forcing_ratio if self.training else 0.0,
            targets=tgt,
            initial_input=initial_input
        )
        
        return decoder_outputs



# # --- Пример использования ---
# if __name__ == "__main__":
#     # Параметры модели
#     B, T_hist, feature_size = 4, 256, 128
#     T_pred = 32
#     output_size = 1
    
#     # Создание модели
#     model = TradingLSTM(
#         feature_size=feature_size,
#         encoder_hidden_size=512,
#         encoder_num_layers=3,
#         decoder_hidden_size=512,
#         decoder_num_layers=3,
#         attention_dim=128,
#         target_len=T_pred,
#         dropout=0.2,
#         teacher_forcing_ratio=0.5,
#         bidirectional_encoder=True, # Новое улучшение
#         use_layer_norm=True # Новое улучшение
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
#         print(f"Выход при инференсе: {output_infer.shape}") # [B, 32, 1]

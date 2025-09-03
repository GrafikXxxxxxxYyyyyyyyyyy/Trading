# src/tft/tft.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit для контроля потока информации.
    """
    def __init__(self, input_size, hidden_size=None, dropout=0.0):
        super(GatedLinearUnit, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size or input_size
        self.dropout = dropout
        
        self.fc = nn.Linear(input_size, self.hidden_size * 2)
        self.dropout_layer = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        # x: [B, T, input_size] или [B, input_size]
        x = self.dropout_layer(x)
        x = self.fc(x)
        
        if x.dim() == 3:  # [B, T, hidden_size * 2]
            gate, value = x.chunk(2, dim=-1)
        else:  # [B, hidden_size * 2]
            gate, value = x.chunk(2, dim=-1)
            
        gate = torch.sigmoid(gate)
        value = self.activation(value)
        
        return gate * value


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network с возможностью контекста.
    """
    def __init__(self, input_size, hidden_size, output_size, dropout=0.0, context_size=None):
        super(GatedResidualNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_size = context_size
        self.dropout = dropout
        
        # Элиминация размерности
        if input_size != output_size:
            self.skip_layer = nn.Linear(input_size, output_size)
        else:
            self.skip_layer = None
            
        # Основная сеть
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Контекст (если предоставлен)
        if context_size is not None:
            self.context_layer = nn.Linear(context_size, hidden_size, bias=False)
            
        # GLU для выхода
        self.glu = GatedLinearUnit(output_size, output_size, dropout=dropout)
        
    def forward(self, x, context=None):
        # x: [B, T, input_size] или [B, input_size]
        # context: [B, context_size] (опционально)
        
        # Skip connection
        if self.skip_layer is not None:
            skip = self.skip_layer(x)
        else:
            skip = x
            
        # Основная обработка
        hidden = self.fc1(x)
        if context is not None:
            hidden = hidden + self.context_layer(context).unsqueeze(1) if hidden.dim() == 3 else \
                     hidden + self.context_layer(context)
        hidden = self.activation(hidden)
        hidden = self.dropout_layer(hidden)
        hidden = self.fc2(hidden)
        
        # GLU
        gated_hidden = self.glu(hidden)
        
        # Остаточное соединение
        output = skip + gated_hidden
        
        return output


class VariableSelectionNetwork(nn.Module):
    """
    Сеть выбора переменных для определения релевантности признаков.
    """
    def __init__(self, input_size, num_variables, hidden_size, dropout=0.0, context_size=None):
        super(VariableSelectionNetwork, self).__init__()
        self.input_size = input_size
        self.num_variables = num_variables
        self.hidden_size = hidden_size
        self.context_size = context_size
        
        # GRN для каждого признака
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout, context_size)
            for _ in range(num_variables)
        ])
        
        # GRN для весов переменных
        grn_input_size = num_variables * hidden_size
        self.weight_grn = GatedResidualNetwork(
            grn_input_size, hidden_size, num_variables, dropout, context_size
        )
        
    def forward(self, variables, context=None):
        # variables: список из num_variables тензоров [B, T, input_size] или [B, input_size]
        # context: [B, context_size] (опционально)
        
        # Обрабатываем каждую переменную
        processed_variables = []
        for i, variable in enumerate(variables):
            processed_var = self.variable_grns[i](variable, context)
            processed_variables.append(processed_var)
            
        # Конкатенируем все переменные
        if processed_variables[0].dim() == 3:
            # [B, T, num_variables * hidden_size]
            flat_vars = torch.cat(processed_variables, dim=-1)
        else:
            # [B, num_variables * hidden_size]
            flat_vars = torch.cat(processed_variables, dim=-1)
            
        # Вычисляем веса переменных
        weights = self.weight_grn(flat_vars, context)  # [B, T, num_variables] или [B, num_variables]
        weights = torch.softmax(weights, dim=-1)
        
        # Применяем веса
        if processed_variables[0].dim() == 3:
            # [B, T, num_variables, hidden_size]
            stacked_vars = torch.stack(processed_variables, dim=-2)
            # [B, T, num_variables, 1]
            weights = weights.unsqueeze(-1)
        else:
            # [B, num_variables, hidden_size]
            stacked_vars = torch.stack(processed_variables, dim=-2)
            # [B, num_variables, 1]
            weights = weights.unsqueeze(-1)
            
        # Взвешенная сумма
        selected_vars = torch.sum(stacked_vars * weights, dim=-2)
        
        return selected_vars, weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Интерпретируемый механизм многоголового внимания.
    """
    def __init__(self, input_size, hidden_size, num_heads=8, dropout=0.0):
        super(InterpretableMultiHeadAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Убедимся, что hidden_size делится на num_heads
        assert hidden_size % num_heads == 0, "hidden_size должен делиться на num_heads"
        self.head_size = hidden_size // num_heads
        
        # Проекции
        self.query_projection = nn.Linear(input_size, hidden_size)
        self.key_projection = nn.Linear(input_size, hidden_size)
        self.value_projection = nn.Linear(input_size, hidden_size)
        
        # Выходная проекция
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Вектор для интерпретации (одна "важная" голова)
        self.interpretation_vector = nn.Parameter(torch.randn(num_heads))
        
    def forward(self, query, key, value, mask=None):
        # query: [B, T_q, input_size]
        # key: [B, T_k, input_size]
        # value: [B, T_v, input_size]
        # mask: [B, T_q, T_k] (опционально)
        
        batch_size = query.size(0)
        
        # Проекции
        Q = self.query_projection(query)  # [B, T_q, hidden_size]
        K = self.key_projection(key)      # [B, T_k, hidden_size]
        V = self.value_projection(value)  # [B, T_v, hidden_size]
        
        # Разделяем на головы
        Q = Q.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)  # [B, num_heads, T_q, head_size]
        K = K.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)  # [B, num_heads, T_k, head_size]
        V = V.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)  # [B, num_heads, T_v, head_size]
        
        # Вычисляем внимание
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_size)  # [B, num_heads, T_q, T_k]
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
            
        attention_weights = F.softmax(scores, dim=-1)  # [B, num_heads, T_q, T_k]
        attention_weights = self.dropout_layer(attention_weights)
        
        # Применяем внимание
        attended_values = torch.matmul(attention_weights, V)  # [B, num_heads, T_q, head_size]
        
        # Интерпретируемый механизм: взвешенная сумма голов
        interpretation_weights = torch.softmax(self.interpretation_vector, dim=0)  # [num_heads]
        interpreted_values = torch.sum(
            attended_values * interpretation_weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
            dim=1
        )  # [B, T_q, head_size]
        
        # Объединяем головы (в интерпретируемом варианте одна голова)
        # Но для совместимости сохраним размерность
        interpreted_values = interpreted_values.repeat(1, 1, self.num_heads)  # [B, T_q, hidden_size]
        
        # Выходная проекция
        output = self.output_projection(interpreted_values)  # [B, T_q, hidden_size]
        
        return output, attention_weights


class TemporalFusionDecoder(nn.Module):
    """
    Декодер Temporal Fusion Transformer.
    """
    def __init__(self, hidden_size, output_size, num_heads=8, dropout=0.0):
        super(TemporalFusionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_heads = num_heads
        
        # LSTM для обработки исторических данных
        self.historical_lstm = nn.LSTM(
            hidden_size, hidden_size, batch_first=True, bidirectional=False
        )
        
        # LSTM для обработки будущих данных (для декодера)
        self.future_lstm = nn.LSTM(
            hidden_size, hidden_size, batch_first=True, bidirectional=False
        )
        
        # GRN для обработки входных данных
        self.input_grn = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout
        )
        
        # Механизм внимания
        self.attention = InterpretableMultiHeadAttention(
            hidden_size, hidden_size, num_heads, dropout
        )
        
        # GRN для обработки контекста
        self.context_grn = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout
        )
        
        # Выходной слой
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, historical_features, future_features=None):
        # historical_features: [B, T_hist, hidden_size]
        # future_features: [B, T_future, hidden_size] (опционально)
        
        batch_size = historical_features.size(0)
        
        # Обработка исторических данных через LSTM
        historical_lstm_out, (hidden, cell) = self.historical_lstm(historical_features)
        # Используем последнее скрытое состояние
        historical_context = historical_lstm_out[:, -1:, :]  # [B, 1, hidden_size]
        
        # Если есть будущие признаки, обрабатываем их
        if future_features is not None:
            # Обработка будущих данных через LSTM с начальным состоянием
            future_lstm_out, _ = self.future_lstm(future_features, (hidden, cell))
            
            # Применяем GRN к будущим признакам
            future_grn_out = self.input_grn(future_lstm_out)
            
            # Применяем внимание между будущими признаками и историческим контекстом
            attended_features, attention_weights = self.attention(
                future_grn_out, historical_lstm_out, historical_lstm_out
            )
            
            # Объединяем attended_features с future_grn_out
            context_vector = attended_features + future_grn_out
            
        else:
            # Если будущих признаков нет, используем только исторический контекст
            context_vector = historical_context.repeat(1, 32, 1)  # Предполагаем 32 шага прогноза
            attention_weights = None
            
        # Обрабатываем контекст через GRN
        final_context = self.context_grn(context_vector)
        
        # Выходной слой
        output = self.output_layer(final_context)  # [B, T_future, output_size]
        
        return output, attention_weights


class TradingTFT(nn.Module):
    """
    Мощная архитектура Temporal Fusion Transformer для прогнозирования цен на фондовом рынке.
    """
    def __init__(
        self,
        feature_size=256,
        hidden_size=128,
        output_size=32,
        num_heads=8,
        dropout=0.1,
        static_context_size=64,
        use_variable_selection=True
    ):
        super(TradingTFT, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.use_variable_selection = use_variable_selection
        
        # Проекция входных признаков
        self.input_projection = nn.Linear(feature_size, hidden_size)
        
        # Статический контекст (если нужен)
        self.static_context_vector = nn.Parameter(torch.randn(static_context_size))
        self.static_context_grn = GatedResidualNetwork(
            static_context_size, hidden_size, hidden_size, dropout
        )
        
        # GRN для обработки входных данных
        self.input_grn = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout
        )
        
        # Сеть выбора переменных (если используется)
        if use_variable_selection:
            # Разделяем признаки на группы (простой подход)
            self.num_variables = 8
            self.variable_split_size = hidden_size // self.num_variables
            self.variable_selection = VariableSelectionNetwork(
                self.variable_split_size, self.num_variables, hidden_size, dropout,
                context_size=hidden_size
            )
        
        # LSTM для обработки временных зависимостей
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, batch_first=True, bidirectional=False
        )
        
        # GRN для обработки LSTM выхода
        self.lstm_grn = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout
        )
        
        # Декодер
        self.decoder = TemporalFusionDecoder(
            hidden_size, 1, num_heads, dropout
        )
        
        # Инициализация весов
        self._init_weights()
        
    def _init_weights(self):
        """Инициализация весов."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
                        
    def forward(self, src, tgt=None):
        """
        Прямой проход модели.
        
        Args:
            src (torch.Tensor): Исторические данные [B, T_hist, F_hist=256].
            tgt (torch.Tensor, optional): Целевые значения [B, T_pred, 1].
            
        Returns:
            torch.Tensor: Прогнозы [B, T_pred, 1].
        """
        batch_size, seq_len, _ = src.size()
        
        # Проекция входных признаков
        x = self.input_projection(src)  # [B, T_hist, hidden_size]
        
        # Обработка через GRN
        x = self.input_grn(x)  # [B, T_hist, hidden_size]
        
        # Статический контекст
        static_context = self.static_context_grn(self.static_context_vector)
        static_context = static_context.unsqueeze(0).repeat(batch_size, 1)  # [B, hidden_size]
        
        # Выбор переменных (если используется)
        if self.use_variable_selection:
            # Разделяем на переменные
            variables = torch.chunk(x, self.num_variables, dim=-1)
            x, variable_weights = self.variable_selection(variables, static_context)
        else:
            variable_weights = None
            
        # Обработка через LSTM
        lstm_out, (hidden, cell) = self.lstm(x)  # [B, T_hist, hidden_size]
        
        # Обработка LSTM выхода через GRN
        lstm_processed = self.lstm_grn(lstm_out)  # [B, T_hist, hidden_size]
        
        # Декодирование
        output, attention_weights = self.decoder(lstm_processed)
        
        return output


# Пример использования
if __name__ == "__main__":
    # Параметры модели
    B, T_hist, feature_size = 4, 256, 256
    T_pred = 32
    output_size = 1
    
    # Создание модели
    model = TradingTFT(
        feature_size=feature_size,
        hidden_size=512,
        output_size=T_pred,
        num_heads=8,
        dropout=0.1,
        static_context_size=64,
        use_variable_selection=True
    )
    
    # Примерные входные данные
    src = torch.randn(B, T_hist, feature_size)  # История после TradingProcessor
    tgt = torch.randn(B, T_pred, output_size)   # Целевые значения (опционально)
    
    # Прогон модели
    model.eval()
    with torch.no_grad():
        output = model(src, tgt)
        print(f"Выход модели: {output.shape}")  # [B, 32, 1]
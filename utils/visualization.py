import torch
import numpy as np
import matplotlib.pyplot as plt



def plot_dataset_sample(sample):
    """
    Отрисовывает один sample из полученного датасета.
    На первом графике отрисовываются цены.
    На втором графике отрисовывается объём.
    
    Args:
        sample (dict): Словарь, содержащий ключи 'history', 'target' и 'ticker'.
                       'history' - тензор формы [1, history_len, 5] (Open, High, Low, Close, Volume).
                       'target' - тензор формы [1, target_len, 1] (Close).
                       'ticker' - строка с названием тикера.
    """
    # Извлекаем данные и убираем размерность батча
    history = sample['history'].squeeze(0).numpy()  # [history_len, 5]
    target = sample['target'].squeeze(0).numpy()    # [target_len, 1]
    ticker = sample['ticker']

    # Разделяем исторические данные
    history_len = history.shape[0]
    history_prices = history[:, :4]  # Open, High, Low, Close
    history_volume = history[:, 4]   # Volume

    # Извлекаем цены закрытия из таргета
    target_len = target.shape[0]
    target_close = target[:, 0]      # Close prices for target

    # Создаем ось X для графиков
    history_x = np.arange(history_len)
    target_x = np.arange(history_len, history_len + target_len)

    # --- Первый график: Цены ---
    plt.figure(figsize=(15, 8))

    # Исторические цены
    plt.plot(history_x, history_prices[:, 0], color='gray', alpha=0.7, label='Open (History)') # Open
    plt.plot(history_x, history_prices[:, 1], color='green', alpha=0.7, label='High (History)') # High
    plt.plot(history_x, history_prices[:, 2], color='red', alpha=0.7, label='Low (History)') # Low
    plt.plot(history_x, history_prices[:, 3], color='black', alpha=0.8, label='Close (History)') # Close

    # Разделительная линия
    plt.axvline(x=history_len - 1, color='blue', linestyle='--', linewidth=1, alpha=0.7, label='History/Target Boundary')

    # Цены закрытия таргета
    plt.plot(target_x, target_close, color='black', linewidth=2, label='Close (Target)')

    # Координаты для линии: последняя точка Close из истории -> первая точка Close из таргета
    connect_x = [history_len - 1, history_len]
    connect_y = [history_prices[-1, 3], target_close[0]]
    plt.plot(connect_x, connect_y, color='orange', linewidth=1.5, linestyle='-', marker='o', markersize=4,
             label='Connection (Last Hist Close -> First Target Close)')

    # Настройки первого графика
    plt.title(f'Price History and Target for {ticker}')
    plt.xlabel('Time Steps')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.grid(True, alpha=0.5)
    # Устанавливаем пределы оси X для первого графика
    plt.xlim(0, history_len + target_len - 1)

    plt.tight_layout()
    plt.show()

    # --- Второй график: Объёмы ---
    plt.figure(figsize=(15, 4))

    # Исторические объёмы
    plt.bar(history_x, history_volume, width=1.0, color='lightblue', alpha=0.7, label='Volume (History)')
    
    # Настройки второго графика
    plt.title(f'Trading Volume History for {ticker}')
    plt.xlabel('Time Steps')
    plt.ylabel('Normalized Volume')
    plt.legend()
    plt.grid(True, alpha=0.5, axis='y')
    # Устанавливаем пределы оси X для второго графика (только история)
    plt.xlim(0, history_len + target_len - 1)

    plt.tight_layout()
    plt.show()



def plot_model_prediction(history, target, prediction, ticker_name="Unknown"):
    """
    Отрисовывает на одном графике исторические цены, таргет и прогноз модели.

    Args:
        history (torch.Tensor or np.ndarray): Исторические данные.
                                          Формат: [B, T_hist, 5] или [T_hist, 5].
                                          Предполагается, что последний канал - Close price (индекс 3).
        target (torch.Tensor or np.ndarray): Целевые значения.
                                       Формат: [B, T_pred, 1] или [T_pred, 1].
        prediction (torch.Tensor or np.ndarray): Прогнозы модели.
                                           Формат: [B, T_pred, 1] или [T_pred, 1].
        ticker_name (str, optional): Название тикера для заголовка графика. Defaults to "Unknown".
    """
    # --- 1. Обработка входных данных ---
    # Преобразуем в numpy, если нужно, и снимаем с градиентного графа
    if isinstance(history, torch.Tensor):
        history_np = history.detach().cpu().numpy()
    else:
        history_np = np.array(history)

    if isinstance(target, torch.Tensor):
        target_np = target.detach().cpu().numpy()
    else:
        target_np = np.array(target)

    if isinstance(prediction, torch.Tensor):
        prediction_np = prediction.detach().cpu().numpy()
    else:
        prediction_np = np.array(prediction)

    # Обработка размерностей (удаление размерности батча, если она 1 или равна 1)
    # Предполагаем, что если размерность 3, то первая - батч, и мы берем первый элемент.
    if history_np.ndim == 3:
        history_np = history_np[0] # [T_hist, 5]
    if target_np.ndim == 3:
        target_np = target_np[0]   # [T_pred, 1]
    if prediction_np.ndim == 3:
        prediction_np = prediction_np[0] # [T_pred, 1]

    # Проверка базовых форматов
    if history_np.ndim != 2 or history_np.shape[1] < 4:
        raise ValueError("history должен быть 2D массивом с минимум 4 колонками (последняя - Close)")
    if target_np.ndim != 2 or target_np.shape[1] != 1:
        raise ValueError("target должен быть 2D массивом с 1 колонкой")
    if prediction_np.ndim != 2 or prediction_np.shape[1] != 1:
        raise ValueError("prediction должен быть 2D массивом с 1 колонкой")
    if target_np.shape[0] != prediction_np.shape[0]:
        raise ValueError("target и prediction должны иметь одинаковую длину по временной оси")

    # --- 2. Извлечение данных для графика ---
    T_hist = history_np.shape[0]
    T_pred = target_np.shape[0]

    # Исторические цены закрытия
    historical_closes = history_np[:, 3] # Индекс 3 для Close

    # Таргет и прогноз (снимаем размерность канала)
    target_values = target_np.squeeze(-1) # [T_pred]
    predicted_values = prediction_np.squeeze(-1) # [T_pred]

    # Создаем ось X для каждого набора данных
    # История: от 0 до T_hist-1
    # Таргет и прогноз: от T_hist до T_hist + T_pred - 1
    # (предполагаем, что таргет и прогноз идут сразу после истории)
    x_history = np.arange(T_hist)
    x_target_pred = np.arange(T_hist, T_hist + T_pred)

    # --- 3. Построение графика ---
    plt.figure(figsize=(12, 6))

    # Исторические цены (черные)
    plt.plot(x_history, historical_closes, label='Historical Close', color='black', linewidth=1)

    # Таргет (зеленый)
    plt.plot(x_target_pred, target_values, label='Target', color='green', linewidth=1)

    # Прогноз (красный)
    plt.plot(x_target_pred, predicted_values, label='Prediction', color='red', linewidth=1)

    # Вертикальная линия, разделяющая историю и прогноз
    plt.axvline(x=T_hist-1, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    # Настройки графика
    plt.xlabel('Time Steps')
    plt.ylabel('Price (Close)')
    plt.title(f'Model Prediction vs Target - {ticker_name}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Отображение графика
    plt.tight_layout()
    plt.show()
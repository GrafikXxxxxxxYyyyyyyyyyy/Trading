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



def plot_denormalized_sample(sample):
    """
    Отрисовывает денормализованные исторические данные и таргет.
    
    Args:
        sample (dict): Словарь с ключами 'history', 'target', 'ticker', 'stats'.
                       'history' и 'target' должны быть numpy массивами или torch тензорами.
                       'stats' должен содержать статистики для денормализации.
    """
    # Извлекаем данные
    history = sample['history']
    target = sample['target']
    ticker = sample['ticker']
    stats = sample['stats']
    
    # Если данные в формате torch.Tensor, конвертируем в numpy
    if hasattr(history, 'numpy'):
        history = history.squeeze(0).numpy()
    if hasattr(target, 'numpy'):
        target = target.squeeze(0).numpy()
    
    # Денормализация
    denorm_history, denorm_target = denormalize_sample(history, target, stats)
    
    # Создаем временную шкалу
    history_len = denorm_history.shape[0]
    target_len = denorm_target.shape[0]
    history_time = np.arange(history_len)
    target_time = np.arange(history_len, history_len + target_len)
    
    # Создаем график
    plt.figure(figsize=(12, 6))
    
    # Отрисовка исторических данных (цены закрытия)
    plt.plot(history_time, denorm_history[:, 3], label='Historical Close Price', color='black', linewidth=1)
    plt.plot(history_time, denorm_history[:, 0], label='Historical Open Price', color='gray', linewidth=1)
    plt.plot(history_time, denorm_history[:, 1], label='Historical High Price', color='darkgreen', linewidth=1)
    plt.plot(history_time, denorm_history[:, 2], label='Historical Low Price', color='darkred', linewidth=1)
    
    # Отрисовка таргета (цены закрытия)
    plt.plot(target_time, denorm_target[:, 0], label='Target Close Price', color='red', linewidth=1)
    
    # Настройка графика
    plt.title(f'{ticker} - Denormalized Price Data')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Показываем график
    plt.tight_layout()
    plt.show()



def denormalize_sample(history, target, stats):
    """
    Денормализует исторические данные и таргет.
    
    Args:
        history (np.ndarray): Исторические данные формы [history_len, 5].
        target (np.ndarray): Таргет данных формы [target_len, 1].
        stats (dict): Статистики нормализации.
        
    Returns:
        tuple: (denorm_history, denorm_target)
    """
    # Копируем данные, чтобы не изменять оригинальные
    denorm_history = np.copy(history)
    denorm_target = np.copy(target)
    
    # Индексы: 0=Open, 1=High, 2=Low, 3=Close, 4=Volume
    price_indices = [0, 1, 2, 3]  # OHLC
    
    # Денормализация цен через лог-доходности
    for i in price_indices:
        # Получаем первую цену для восстановления
        first_price = stats.get(f'first_price_{i}', 1.0)
        
        # Восстанавливаем цены из лог-доходностей
        log_returns = denorm_history[:, i]
        prices = np.zeros_like(log_returns)
        prices[0] = first_price
        
        # Последовательное восстановление цен
        for t in range(1, len(log_returns)):
            # P_t = P_{t-1} * exp(r_t)
            prices[t] = prices[t-1] * np.exp(log_returns[t])
            
        denorm_history[:, i] = prices
    
    # Денормализация таргета (цены закрытия)
    first_close_price = stats.get('first_price_3', 1.0)
    
    # Для таргета нам нужно продолжить последовательность от последней цены в истории
    last_close_price = denorm_history[-1, 3]  # Последняя цена закрытия в истории
    target_log_returns = denorm_target[:, 0]
    target_prices = np.zeros_like(target_log_returns)
    target_prices[0] = last_close_price * np.exp(target_log_returns[0])
    
    for t in range(1, len(target_log_returns)):
        target_prices[t] = target_prices[t-1] * np.exp(target_log_returns[t])
        
    denorm_target[:, 0] = target_prices
    
    # Денормализация объема
    volume_mean = stats.get('volume_mean', 0.0)
    volume_std = stats.get('volume_std', 1.0)
    
    # Обратное z-score преобразование
    denorm_history[:, 4] = denorm_history[:, 4] * volume_std + volume_mean
    
    return denorm_history, denorm_target



def plot_model_prediction_denormalized(history, target, stats, predict):
    # Если данные в формате torch.Tensor, конвертируем в numpy
    if hasattr(history, 'numpy'):
        history = history.squeeze(0).numpy()
    if hasattr(target, 'numpy'):
        target = target.squeeze(0).numpy()
    if hasattr(predict, 'numpy'):
        predict = predict.squeeze(0).detach().cpu().numpy()

    # Денормализация
    denorm_history, denorm_target = denormalize_sample(history, target, stats)
    _, denorm_predict = denormalize_sample(history, predict, stats)

    # Создаем временную шкалу
    history_len = denorm_history.shape[0]
    target_len = denorm_target.shape[0]
    history_time = np.arange(history_len)
    target_time = np.arange(history_len, history_len + target_len)

    # Создаем график
    plt.figure(figsize=(12, 6))
    
    # Отрисовка исторических данных (цены закрытия)
    plt.plot(history_time, denorm_history[:, 3], label='Historical Close Price', color='black', linewidth=1)
    
    # Отрисовка таргета (цены закрытия)
    plt.plot(target_time, denorm_target[:, 0], label='Target Close Price', color='darkgreen', linewidth=1)
    plt.plot(target_time, denorm_predict[:, 0], label='Predict Close Price', color='darkred', linewidth=1)
    
    # Настройка графика
    plt.title(f'Denormalized Price Predict')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Показываем график
    plt.tight_layout()
    plt.show()
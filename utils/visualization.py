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
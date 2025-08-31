import torch
from typing import Optional
from models.model_wrapper import TradingModel



class TradingPipeline:
    """
    Пайплайн для выполнения инференса (предсказания) с использованием обученной TradingModel.
    """
    def __init__(self, device: str = "mps"):
        """
        Args:
            device (str): Устройство по умолчанию для выполнения инференса ('cpu', 'cuda', 'mps').
        """
        # Используем переданное устройство, а не хардкод 'mps'
        self.device = device 

    def __call__(
        self,
        model: TradingModel,
        history_prices: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Выполняет инференс модели на предоставленных исторических данных.

        Args:
            model (TradingModel): Экземпляр обученной TradingModel.
            history_prices (torch.FloatTensor, optional): 
                Исторические цены в формате [B, T_hist, 5] (Open, High, Low, Close, Volume).
                Если не предоставлены, модель должна иметь способ получения данных внутри,
                но для этого пайплайна требуется вход.

        Returns:
            torch.FloatTensor: Прогноз модели в формате [B, T_pred, 1].
                         T_pred определяется конфигурацией модели (обычно 32).
                         
        Raises:
            ValueError: Если history_prices не предоставлены или имеют неправильную форму.
            TypeError: Если model не является экземпляром TradingModel.
        """
        if not isinstance(model, TradingModel):
            raise TypeError("Аргумент 'model' должен быть экземпляром TradingModel.")

        if history_prices is None:
            raise ValueError("Аргумент 'history_prices' должен быть предоставлен.")

        if not isinstance(history_prices, torch.Tensor):
            raise TypeError("Аргумент 'history_prices' должен быть torch.Tensor.")

        # Проверка базовой формы (ожидаем 3D тензор)
        if history_prices.dim() != 3:
             raise ValueError(f"history_prices должен быть 3D тензором, получил {history_prices.dim()}D.")

        # Ожидаем, что последнее измерение равно 5 (OHLCV)
        if history_prices.shape[-1] != 5:
             raise ValueError(f"Последнее измерение history_prices должно быть 5 (OHLCV), получил {history_prices.shape[-1]}.")

        # 1. Убедимся, что модель в режиме оценки
        model.eval()

        # 2. Переместим входные данные на устройство модели
        #    Модель сама переместит обработанные данные через wrapper.__call__
        #    Но лучше переместить вход сразу
        history_prices = history_prices.to(model.device) # Используем device из модели

        # 3. Отключаем градиенты для инференса (опционально, но рекомендуется)
        with torch.no_grad():
            # 4. Вызываем модель через обёртку
            #    TradingModel.__call__ обрабатывает history_prices с помощью processor
            #    и передаёт результат в внутреннюю модель.
            #    Так как target=None, будет вызван режим инференса модели.
            predictions = model(history_prices, target=None) # target=None для инференса

        # 5. Возвращаем прогнозы
        return predictions


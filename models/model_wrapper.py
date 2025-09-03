import os
import json
import torch

from safetensors.torch import save_model, load_model
from models.feature_extractor import TradingFeatureExtractor
from src import (
    TradingFFN,
    TradingLSTM,
    TradingLSTMFFN,
    TradingTCN,
    # TradingCNN,
    # TradingTransformer,
    # TradingLSTM_FFN,
    # TradingBERT,
)



MODEL_REGISTRY = {
    "TradingFFN": TradingFFN,
    "TradingLSTM": TradingLSTM,
    "TradingLSTMFFN": TradingLSTMFFN,
    "TradingTCN": TradingTCN,
    # "TradingCNN": TradingCNN,
    # "TradingTransformer": TradingTransformer,
    # "TradingLSTM_FFN": TradingLSTM_FFN,
    # "TradingBERT": TradingBERT,
}



class TradingModel:
    def __init__(
        self, 
        device: str = "mps"
    ):
        self.device = device
        self.model: torch.nn.Module = None
        self.model_config: dict = None

        # Создаём предобученный экстрактор признаков
        self.feature_extractor = TradingFeatureExtractor(
            input_size=5,
            feature_size=256,
        )
        load_model(self.feature_extractor, 'pretrained-extractor/trading_feature_extractor.safetensors')
        self.feature_extractor.to(device)
        self.feature_extractor.eval()

    
    def eval(self):
        if self.model is not None:
            self.model.eval()
        else:
            raise ValueError("Модель не инициализирована. Вызовите from_config или from_pretrained.")


    def train(self):
        if self.model is not None:
            self.model.train()
        else:
            raise ValueError("Модель не инициализирована. Вызовите from_config или from_pretrained.")


    def parameters(self):
        if self.model is not None:
            return self.model.parameters()
        else:
            raise ValueError("Модель не инициализирована. Вызовите from_config или from_pretrained.")
        

    def to(self, device):
        self.device = device
        self.feature_extractor.to(device)
        if self.model is not None:
            self.model.to(device)
        return self
        

    def save_pretrained(self, dir_path="pretrained-models"):
        """
        Сохраняет модель и её конфигурацию в указанную базовую директорию.
        Создаёт поддиректорию с именем типа модели.
        Структура: {base_dir_path}/{model_type}/config.json и {model_type}.safetensors
        
        Args:
            base_dir_path (str): Базовый путь к директории для сохранения всех моделей.
                                 По умолчанию "pretrained-models".
        """
        if self.model is None:
            raise ValueError("Нет модели для сохранения. Инициализируйте модель сначала.")
        if self.model_config is None or "model_type" not in self.model_config:
             raise ValueError("Конфигурация модели не найдена или не содержит 'model_type'.")

        model_type = self.model_config["model_type"]
        model_dir_path = os.path.join(dir_path, model_type)
        os.makedirs(model_dir_path, exist_ok=True)

        config_path = os.path.join(model_dir_path, "config.json")
        try:
            with open(config_path, 'w') as f:
                json.dump(self.model_config, f, indent=2)
        except Exception as e:
            print(f"Ошибка при сохранении config.json: {e}")
            raise

        model_weights_path = os.path.join(model_dir_path, f"{model_type}.safetensors")
        try:
            save_model(self.model, model_weights_path)
        except Exception as e:
            print(f"Ошибка при сохранении весов модели: {e}")
            raise

        print(f"Модель {model_type} успешно сохранена в {model_dir_path}")

    
    @classmethod
    def from_pretrained(cls, dir_path, device=None):
        """
        Загружает модель и её конфигурацию из указанной директории.
        Предполагается структура: {dir_path}/config.json и {model_type}.safetensors
        
        Args:
            dir_path (str): Путь к директории с сохраненной моделью и конфигурацией.
            device (str, optional): Устройство для загрузки модели.
            
        Returns:
            TradingModel: Экземпляр загруженной модели.
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = "mps"
            else:
                device = "cpu"

        config_path = os.path.join(dir_path, "config.json")
        if not os.path.exists(config_path):
             raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")

        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        except Exception as e:
            print(f"Ошибка при загрузке config.json: {e}")
            raise

        model_type = config_dict.get("model_type")
        if model_type is None:
            raise ValueError("Конфигурация должна содержать ключ 'model_type'")
        
        # --- Используем словарь для получения класса модели ---
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Неизвестный тип модели: {model_type}. Доступные типы: {list(MODEL_REGISTRY.keys())}")
        
        model_class = MODEL_REGISTRY[model_type]
        # --- --- ---

        model_wrapper = cls(device=device)
        model_wrapper.model_config = config_dict

        # --- Создаем модель, используя класс из словаря ---
        try:
            # Фильтруем конфигурацию, исключая 'model_type'
            model_config = {k: v for k, v in config_dict.items() if k != "model_type"}
            # Создаем экземпляр модели
            model_wrapper.model = model_class(**model_config).to(device)
        except Exception as e:
            print(f"Ошибка при создании модели {model_type}: {e}")
            raise
        # --- --- ---

        model_weights_path = os.path.join(dir_path, f"{model_type}.safetensors")
        if not os.path.exists(model_weights_path):
             raise FileNotFoundError(f"Файл весов модели не найден: {model_weights_path}")

        try:
            load_model(model_wrapper.model, model_weights_path)
            model_wrapper.model.eval()
        except Exception as e:
            print(f"Ошибка при загрузке весов модели: {e}")
            raise

        print(f"Модель {model_type} успешно загружена из {dir_path}")

        return model_wrapper
    

    @classmethod
    def from_config(cls, config, device=None):
        """
        Загружает необученную модель из указанной конфигурации.
        config может быть путем к JSON файлу или словарем.
        Сохраняет конфигурацию в self.model_config.
        
        Args:
            config (dict or str): Конфигурация модели (словарь или путь к JSON файлу).
            device (str, optional): Устройство для загрузки модели.
            
        Returns:
            TradingModel: Экземпляр созданной модели.
        """
        if isinstance(config, str):
            with open(config, 'r') as f:
                config_dict = json.load(f)
        elif isinstance(config, dict):
            config_dict = config
        else:
            raise TypeError("config должен быть dict или str (путь к файлу)")
            
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = "mps"
            else:
                device = "cpu"
        
        model_wrapper = cls(device=device)
        model_wrapper.model_config = config_dict.copy()
        
        model_type = config_dict.get("model_type")
        if model_type is None:
            raise ValueError("Конфигурация должна содержать ключ 'model_type'")
            
        # --- Используем словарь для получения класса модели ---
        if model_type not in MODEL_REGISTRY:
             raise ValueError(f"Неизвестный тип модели: {model_type}. Доступные типы: {list(MODEL_REGISTRY.keys())}")
        
        model_class = MODEL_REGISTRY[model_type]
        # --- --- ---

        # --- Создаем модель, используя класс из словаря ---
        try:
            model_config = {k: v for k, v in config_dict.items() if k != "model_type"}
            model_wrapper.model = model_class(**model_config).to(device)
        except Exception as e:
            print(f"Ошибка при создании модели {model_type}: {e}")
            raise
        # --- --- ---
            
        return model_wrapper
    

    def __call__(self, history, target=None):
        """
        Прогнозирование будущих цен.
        
        Args:
            history (Tensor): Исторические данные [B, 256, 5] (OHLCV)
            target (Tensor, optional): Целевые значения [B, 32, 1] (только Close)
        
        Returns:
            Tensor: Прогноз [B, 32, 1]
        """
        enriched_history = self.feature_extractor.extract_features(history)

        return self.model(enriched_history, target)
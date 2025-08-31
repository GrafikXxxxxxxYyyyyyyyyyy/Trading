import torch
import os
import json
from safetensors.torch import save_model, load_model

from utils.processor_xxl import TradingProcessor
from src import (
    TradingLSTM,
    TradingTCN,
    TradingCNN,
    TradingTransformer,
)


class TradingModel:
    def __init__(
        self, 
        device: str = "cpu" # Изменим значение по умолчанию на "cpu" для совместимости
    ):
        self.device = device
        self.processor = TradingProcessor()
        self.model: torch.nn.Module = None
        self.model_config: dict = None


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
        # Создаём путь к директории конкретной модели
        model_dir_path = os.path.join(dir_path, model_type)
        os.makedirs(model_dir_path, exist_ok=True)

        # 1. Сохраняем конфигурацию
        config_path = os.path.join(model_dir_path, "config.json")
        try:
            with open(config_path, 'w') as f:
                json.dump(self.model_config, f, indent=2)
        except Exception as e:
            print(f"Ошибка при сохранении config.json: {e}")
            raise

        # 2. Сохраняем веса модели с использованием safetensors
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
        # Определяем устройство
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

        # Загружаем конфигурацию
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        except Exception as e:
            print(f"Ошибка при загрузке config.json: {e}")
            raise

        model_type = config_dict.get("model_type")
        if model_type is None:
            raise ValueError("Конфигурация должна содержать ключ 'model_type'")

        # Создаем экземпляр TradingModel
        model_wrapper = cls(device=device)
        model_wrapper.model_config = config_dict # Сохраняем конфигурацию

        # Создаем соответствующую модель на основе типа
        try:
            if model_type == "TradingTCN":
                tcn_config = {k: v for k, v in config_dict.items() if k != "model_type"}
                model_wrapper.model = TradingTCN(**tcn_config).to(device)
                
            elif model_type == "TradingLSTM":
                lstm_config = {k: v for k, v in config_dict.items() if k != "model_type"}
                model_wrapper.model = TradingLSTM(**lstm_config).to(device)
                
            elif model_type == "TradingCNN":
                cnn_config = {k: v for k, v in config_dict.items() if k != "model_type"}
                model_wrapper.model = TradingCNN(**cnn_config).to(device)

            elif model_type == "TradingTransformer":
                transformer_config = {k: v for k, v in config_dict.items() if k != "model_type"}
                model_wrapper.model = TradingTransformer(**transformer_config).to(device)
                
            else:
                raise ValueError(f"Неизвестный тип модели: {model_type}")
        except Exception as e:
            print(f"Ошибка при создании модели {model_type}: {e}")
            raise

        # Загружаем веса модели
        model_weights_path = os.path.join(dir_path, f"{model_type}.safetensors")
        if not os.path.exists(model_weights_path):
             raise FileNotFoundError(f"Файл весов модели не найден: {model_weights_path}")

        try:
            load_model(model_wrapper.model, model_weights_path)
            model_wrapper.model.eval() # Переводим в режим оценки после загрузки
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
        # Если config - строка, считаем её путем к файлу
        if isinstance(config, str):
            with open(config, 'r') as f:
                config_dict = json.load(f)
        elif isinstance(config, dict):
            config_dict = config
        else:
            raise TypeError("config должен быть dict или str (путь к файлу)")
            
        # Определяем устройство
        if device is None:
            # Пытаемся определить оптимальное устройство
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = "mps"
            else:
                device = "cpu"
        
        # Создаем экземпляр TradingModel
        model_wrapper = cls(device=device)
        # Сохраняем конфигурацию
        model_wrapper.model_config = config_dict.copy() # Сохраняем копию
        
        # Получаем тип модели
        model_type = config_dict.get("model_type")
        if model_type is None:
            raise ValueError("Конфигурация должна содержать ключ 'model_type'")
            
        # Создаем соответствующую модель на основе типа
        if model_type == "TradingTCN":
            # Создаем TradingTCN, исключая model_type из параметров
            tcn_config = {k: v for k, v in config_dict.items() if k != "model_type"}
            model_wrapper.model = TradingTCN(**tcn_config).to(device)
            
        elif model_type == "TradingLSTM":
            # Создаем TradingLSTM, исключая model_type из параметров
            lstm_config = {k: v for k, v in config_dict.items() if k != "model_type"}
            model_wrapper.model = TradingLSTM(**lstm_config).to(device)
            
        elif model_type == "TradingCNN":
            cnn_config = {k: v for k, v in config_dict.items() if k != "model_type"}
            model_wrapper.model = TradingCNN(**cnn_config).to(device)

        elif model_type == "TradingTransformer":
            transformer_config = {k: v for k, v in config_dict.items() if k != "model_type"}
            model_wrapper.model = TradingTransformer(**transformer_config).to(device)
            
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")
            
        return model_wrapper


    def __call__(self, history, target=None):
        # Расширяем признаковое пространство [B, 256, 5] -> [B, 256, 128]
        processed_history = self.processor(history).to(self.device)

        return self.model(processed_history, target)

import torch
import os
import json
from safetensors.torch import save_file, load_file

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
        device: str = "mps"
    ):
        self.device = device
        self.processor = TradingProcessor()
        self.model: torch.nn.Module = None

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def parameters(self):
        return self.model.parameters()

    def save_pretrained(self, dir_path):
        """
        Сохраняет модель и связанные компоненты в указанную директорию в формате safetensors
        
        Args:
            dir_path (str): Путь к директории для сохранения модели
        """
        pass

    @classmethod
    def from_pretrained(cls, dir_path, device=None):
        """
        Загружает модель из указанной директории
        
        Args:
            dir_path (str): Путь к директорию с сохраненной моделью
            device (str, optional): Устройство для загрузки модели
            
        Returns:
            TradingModel: Экземпляр загруженной модели
        """
        pass

    @classmethod
    def from_config(cls, config, device=None):
        """
        Загружает необученную модель из указанной конфигурации.
        config может быть путем к JSON файлу или словарем.
        
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
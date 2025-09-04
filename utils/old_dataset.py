import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import glob


class TradingDataset(Dataset):
    def __init__(self, data_path, mode='train', transform=None):
        """
        Custom dataset for trading data.
        Assumes data is stored in CSV files with 5 columns: [Open, High, Low, Close, Volume].
        
        Args:
            data_path (str): path to data directory (e.g., 'data/')
            mode (str): 'train' or 'validation'
            transform (callable, optional): optional transform to be applied on a sample
        """
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        
        # Проверяем существование директории
        mode_path = os.path.join(data_path, mode)
        if not os.path.exists(mode_path):
            raise FileNotFoundError(f"Mode directory not found: {mode_path}")
        
        # Получаем все пути к тикерам
        self.ticker_paths = glob.glob(os.path.join(mode_path, '*'))
        
        if not self.ticker_paths:
            raise ValueError(f"No ticker directories found in {mode_path}")
        
        # Собираем все пары history-target файлов
        self.samples = []
        self._collect_samples()
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid history-target pairs found in {mode_path}")
        
        print(f"Found {len(self.samples)} samples for {mode} mode")
    
    
    def _collect_samples(self):
        """Собирает все доступные пары history-target файлов"""
        for ticker_path in self.ticker_paths:
            if not os.path.isdir(ticker_path):
                continue
                
            # Получаем все history файлы для данного тикера
            history_files = glob.glob(os.path.join(ticker_path, 'history_*.csv'))
            
            for history_file in history_files:
                # Проверяем, что history файл не пустой
                if not os.path.exists(history_file) or os.path.getsize(history_file) == 0:
                    continue
                    
                # Получаем индекс файла
                file_basename = os.path.basename(history_file)
                if file_basename.startswith('history_'):
                    file_index = file_basename.replace('history_', '').replace('.csv', '')
                    target_file = os.path.join(ticker_path, f'target_{file_index}.csv')
                    
                    # Проверяем существование и непустоту соответствующего target файла
                    if os.path.exists(target_file) and os.path.getsize(target_file) > 0:
                        self.samples.append({
                            'history_file': history_file,
                            'target_file': target_file,
                            'ticker': os.path.basename(ticker_path)
                        })
    
    
    def __len__(self):
        """Возвращает общее количество samples"""
        return len(self.samples)
    

    def __getitem__(self, idx):
        """
        Возвращает один sample по индексу.
        Returns:
            dict: {
                'history': torch.Tensor of shape [1, 256, 5],
                'target': torch.Tensor of shape [1, 32, 1], # Only Close prices
                'ticker': str
            }
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample_info = self.samples[idx]
        
        try:
            # Загружаем history данные
            # Ожидаем CSV с 5 столбцами: Open, High, Low, Close, Volume
            history_df = pd.read_csv(sample_info['history_file'], header=None)
            if history_df.empty or history_df.shape[1] != 5:
                raise ValueError(f"History file format error: {sample_info['history_file']}. "
                                 f"Expected 5 columns, got {history_df.shape[1] if not history_df.empty else 0}")
            # Преобразуем в numpy массив и проверяем форму
            history_data = history_df.values.astype(np.float32) # [256, 5]
            
            # Загружаем target данные
            target_df = pd.read_csv(sample_info['target_file'], header=None)
            if target_df.empty or target_df.shape[1] != 5:
                raise ValueError(f"Target file format error: {sample_info['target_file']}. "
                                 f"Expected 5 columns, got {target_df.shape[1] if not target_df.empty else 0}")
            target_data = target_df.values.astype(np.float32) # [32, 5]
            
            # Извлекаем только цены закрытия (индекс 3) для таргета
            target_close_prices = target_data[:, 3:4] # [32, 1]
            
            # Формируем sample с правильными размерностями
            sample = {
                'history': torch.from_numpy(history_data).unsqueeze(0),  # [1, 256, 5]
                'target': torch.from_numpy(target_close_prices).unsqueeze(0), # [1, 32, 1]
                'ticker': sample_info['ticker']
            }
            
            # Применяем transform если задан
            if self.transform:
                sample = self.transform(sample)
            
            return sample
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            print(f"History file: {sample_info['history_file']}")
            print(f"Target file: {sample_info['target_file']}")
            # Возвращаем None или raise исключение
            raise e
        

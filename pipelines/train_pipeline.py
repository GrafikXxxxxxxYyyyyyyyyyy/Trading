import os
import math
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from typing import Any, Optional, Union, Dict
from dataclasses import dataclass

# --- Добавлено для TensorBoard ---
from torch.utils.tensorboard import SummaryWriter
import time
# --- --- ---

from models.model_wrapper import TradingModel
from utils.dataset import TradingDataset



@dataclass
class TradingTrainingArgs:
    train_batch_size: int = 8
    output_dir: str = 'pretrained-models'
    num_train_epochs: int = 1
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-08
    adam_weight_decay: float = 1e-2
    dataloader_num_workers: int = 0
    save_steps: int = 500
    # --- Добавлено для TensorBoard ---
    tensorboard_log_dir: str = "runs/trading_experiment" # Директория для логов TensorBoard
    # --- --- ---


class TradingTrainer:
    def __init__(
        self, 
        model: TradingModel, 
        args: TradingTrainingArgs,
        train_dataset: TradingDataset,
    ):
        self.model = model
        self.args = args
        self.dataset = train_dataset
        # --- Добавлено для TensorBoard ---
        self.writer: Optional[SummaryWriter] = None
        # --- --- ---

    def train(self):
        # --- Добавлено для TensorBoard: Инициализация SummaryWriter ---
        if self.args.tensorboard_log_dir:
            try:
                # Создаём уникальное имя запуска на основе времени
                timestamp = str(int(time.time()))
                run_name = f"run_{timestamp}_{self.model.model_config['model_type']}"
                full_log_dir = os.path.join(self.args.tensorboard_log_dir, run_name)
                self.writer = SummaryWriter(log_dir=full_log_dir)
                print(f"Инициализирован TensorBoard логгер. Логи будут сохранены в {full_log_dir}")
            except Exception as e:
                print(f"Не удалось инициализировать TensorBoard: {e}. Логирование отключено.")
                self.writer = None
        # --- --- ---

        # 1. Создаём директорию проекта
        if self.args.output_dir is not None:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # 2. Включаем обучение параметров трансформера
        self.model.train()

        # 3. Initialize the optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )
        
        # 3.1 Создаём функцию для оценки
        # loss_function = torch.nn.MSELoss()
        loss_function = torch.nn.HuberLoss()


        # 4. DataLoaders creation:
        def collate_fn(example):
            histories = [item['history'] for item in example]
            targets = [item['target'] for item in example]
            tickers = [item['ticker'] for item in example]

            batch_histories = torch.cat(histories, dim=0)  
            batch_targets = torch.cat(targets, dim=0)      

            return {
                'history': batch_histories,
                'target': batch_targets,
                'ticker': tickers
            }

        train_dataloader = torch.utils.data.DataLoader(
            self.dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )

        # 5. Training loop
        progress_bar = tqdm(
            train_dataloader,
            desc="batch loss",
            total=len(train_dataloader) * self.args.num_train_epochs,
        )
        global_step = 0
        for epoch in range(self.args.num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                target = batch['target'].to(self.model.device)
                history = batch['history'].to(self.model.device)
                
                model_pred = self.model(history, target)

                # loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = loss_function(model_pred.float(), target.float())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                
                progress_bar.update(1)
                progress_bar.set_postfix({'loss': f'{loss.detach().item():.6f}'})

                # --- Добавлено для TensorBoard: Логирование метрики ---
                if self.writer is not None:
                    try:
                        self.writer.add_scalar("Loss/train", loss.detach().item(), global_step)
                    except Exception as log_e:
                        print(f"Ошибка при логировании в TensorBoard на шаге {global_step}: {log_e}")
                # --- --- ---

                if global_step > 0 and global_step % self.args.save_steps == 0:
                    self.model.save_pretrained(dir_path=self.args.output_dir)
                
        # --- Добавлено для TensorBoard: Закрытие SummaryWriter ---
        if self.writer is not None:
            try:
                self.writer.close()
                print(f"TensorBoard логгер закрыт. Логи сохранены в {self.args.tensorboard_log_dir}")
            except Exception as e:
                print(f"Ошибка при закрытии TensorBoard логгера: {e}")
        # --- --- ---

        # Сохраняем модель в конце обучения
        self.model.save_pretrained(dir_path=self.args.output_dir)
        print(f"Обучение завершено. Модель сохранена в {self.args.output_dir}")

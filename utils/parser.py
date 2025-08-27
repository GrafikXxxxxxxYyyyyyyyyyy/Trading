import os
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm



def normalize_chunk(chunk, history_len):
    # Берём последнюю в истории цену закрытия
    last_close_price = chunk[history_len-1, 3]
    new_chunk = np.zeros_like(chunk)
    # Нормализуем Open, High, Low, Close на последнюю цену
    new_chunk[:, :4] = (chunk[:, :4] / last_close_price) - 1
    # Отдельно нормализуем объем
    new_chunk[:, 4] = chunk[:, 4] / chunk[history_len-1, 4]

    return new_chunk



def split_into_chunks(data, chunk_len, step=1):
    chunks = []
    for i in range(0, len(data) - chunk_len + 1, step):
        chunk = data[i:i + chunk_len]
        chunks.append(chunk)

    return chunks


def parse_single_ticker(
    ticker, 
    path_to_save='data/', 
    timeframe='1d',
    start_date='2020-01-01',
    target_len=32, 
    history_len=256, 
    split_coef=0.1,
):
    # Получаем историю цен
    stock = yf.Ticker(ticker)
    history = stock.history(
        interval=timeframe, 
        start=start_date, 
        actions=False, 
        auto_adjust=True, 
        prepost=True
    )

    # Берем только цены закрытия и преобразуем в numpy массив
    data_values = history.values

    # Разбиваем историю на чанки длины history_len + price_len
    chunk_size = history_len + target_len
    chunks = split_into_chunks(data_values, chunk_size)

    # Разделяем чанки на тренировочный и валидационный наборы (хронологически)
    split_index = int(len(chunks) * (1 - split_coef))
    train_chunks = chunks[:split_index]
    val_chunks = chunks[split_index:]

    # Создаем директории для сохранения
    train_ticker_path = os.path.join(path_to_save, 'train', ticker)
    val_ticker_path = os.path.join(path_to_save, 'validation', ticker)
    
    os.makedirs(train_ticker_path, exist_ok=True)
    os.makedirs(val_ticker_path, exist_ok=True)

    # Обработка тренировочных чанков
    for i, chunk in enumerate(train_chunks):
        # Нормализация относительно последней цены в истории
        normalized_chunk = normalize_chunk(chunk, history_len)
        normalized_history = normalized_chunk[:history_len]
        normalized_target = normalized_chunk[history_len:]
        
        # Сохранение истории и таргета
        pd.DataFrame(normalized_history).to_csv(
            os.path.join(train_ticker_path, f'history_{i}.csv'), 
            index=False, header=False
        )
        pd.DataFrame(normalized_target).to_csv(
            os.path.join(train_ticker_path, f'target_{i}.csv'), 
            index=False, header=False
        )
    
    # Обработка валидационных чанков
    for i, chunk in enumerate(val_chunks):
        # Нормализация относительно последней цены в истории
        normalized_chunk = normalize_chunk(chunk, history_len)
        normalized_history = normalized_chunk[:history_len]
        normalized_target = normalized_chunk[history_len:]
        
        # Сохранение истории и таргета
        pd.DataFrame(normalized_history).to_csv(
            os.path.join(val_ticker_path, f'history_{i}.csv'), 
            index=False, header=False
        )
        pd.DataFrame(normalized_target).to_csv(
            os.path.join(val_ticker_path, f'target_{i}.csv'), 
            index=False, header=False
        )



def parse_snp500(
    path_to_save='data/', 
    timeframe='1d',
    start_date='2020-01-01',
    target_len=32, 
    history_len=256, 
    split_coef=0.1,
):
    # Проверяем, существует ли директория, и создаем её, если нет
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    # Если нет таблицы с тикерами всех акций, то её нужно спарсить
    if not os.path.exists(os.path.join(path_to_save, 'snp500_tickers.csv')):
        import requests
        # Используем requests с заголовком User-Agent для обхода блокировки
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36'}
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status() # Проверка на успешный статус ответа
            # Передаем текст HTML в read_html
            sp500_table = pd.read_html(response.text)[0]
            
            # Убедимся, что директория существует
            os.makedirs(path_to_save, exist_ok=True)
            
            # Сохраняем в csv таблицу матрицу ['Symbol', 'Security']
            sp500_table[['Symbol', 'Security']].to_csv(os.path.join(path_to_save, "snp500_tickers.csv"), index=False, header=False)
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при загрузке данных с Wikipedia: {e}")
            raise
        except Exception as e:
            print(f"Ошибка при парсинге таблицы: {e}")
            raise

    # Чтение списка тикеров из файла CSV в датафрейм
    tickers_df = pd.read_csv("data/snp500_tickers.csv", header=None, names=['Symbol', 'Security'])

    # Проходимся по всему списку тикеров
    for index, row in tqdm(tickers_df.iterrows()):
        parse_single_ticker(
            ticker=row['Symbol'],
            path_to_save=path_to_save,
            timeframe=timeframe,
            start_date=start_date,
            target_len=target_len,
            history_len=history_len,
            split_coef=split_coef,
        )

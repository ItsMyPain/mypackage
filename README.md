# weather_classifier

В этом проекте обучается модель, предсказывающа вероятность дождя завтра.

Конфигурации каждого этапа хранятся в папке `configs`.

## Инициализация проекта

```shell
poetry install
```

## Запуск севера MLFlow

```shell
poetry run mlflow server --host 127.0.0.1 --port 8080
```

## Запуск обучения

Для обучения необходимо указать адрес сервера `mlflow` в `configs/train.yaml`.

```shell
poetry run python3 weather_classifier/train.py
```

## Запуск тестирования

Для тестирования необходимо указать адрес сервера `mlflow` в `configs/test.yaml`.

```shell
poetry run python3 weather_classifier/infer.py
```

## Запуск сервера

Сервер запускается по адресу `http://localhost:5000/`.

```shell
poetry run python3 weather_classifier/run_server.py
```

### Обращение к серверу

Сервер принимает POST запросы по адресу `http://localhost:5000/predict`.

В тело запроса передаются данные в формате JSON.

Пример:

```
{"inputs": [[62932, 11.9, 41.6, 0, 7.6, 12.6, 43, 7, 31, 71, 19, 1015.2, 1011.8, 0, 1, 21.6, 38.7, 0]]}
```

Сервер возвращает ответ в формате JSON.
Пример:

```
{"outputs":[[0.9983733892440796,0.0016266102902591228]]}
```

Пример запроса:

```shell
curl http://127.0.0.1:5000/predict -H 'Content-Type: application/json' -d '{"inputs": [[62932, 11.9, 41.6, 0, 7.6, 12.6, 43, 7, 31, 71, 19, 1015.2, 1011.8, 0, 1, 21.6, 38.7, 0]]}'
```

```shell
curl http://127.0.0.1:5000/predict -H 'Content-Type: application/json' -d '{"inputs": [[62932, 11.9, 41.6, 0, 7.6, 12.6, 43, 7, 31, 71, 19, 1015.2, 1011.8, 0, 1, 21.6, 38.7, 0], [120383, 13.5, 20.8, 0.8, 3.4, 8.6, 50, 28, 30, 64, 48, 1012.3, 1010.9, 7, 7, 18.3, 19.3, 0]]}'
```

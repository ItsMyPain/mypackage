# weather_classifier

В этом проекте обучается модель, предсказывающа вероятность дождя завтра.

Конфигурации каждого этапа хранятся в папке `configs`.

## Инициализация проекта

```shell
poetry init
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

```shell
poetry run python3 weather_classifier/run_server.py
```

### Обращение к серверу

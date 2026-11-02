# День 01. Запрос к LLM через API

Консольная программа: ввод вопроса → запрос к LLM → вывод ответа.

**Стек:** Python 3.9+, requests, Cloudflare Workers AI.

**Модель:** `@cf/meta/llama-3.1-8b-instruct`.

## Установка

Из корня репозитория:

```bash
cd projects/day-01-llm-api
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Настройка

В [панели Cloudflare](https://dash.cloudflare.com/) откройте **Workers AI → Use REST API**, создайте токен по шаблону **Create a Workers AI API Token** и скопируйте **Account ID**. Для модели по умолчанию подходит план Workers Free.

Один раз скопируйте шаблон в папке задания:

```bash
cp -n .env.example .env
```

В `.env` заполните `CLOUDFLARE_API_TOKEN` и `CLOUDFLARE_ACCOUNT_ID`. Программа автоматически читает файл рядом с `main.py`, независимо от текущей папки терминала. Переменные окружения имеют приоритет над `.env`.

## Запуск

С активным `.venv`:

```bash
python main.py
```

Введите вопрос и дождитесь ответа. Каждый запуск отправляет один запрос.

`.env` исключён из Git; публикуется только `.env.example` без секретов. Необязательная переменная `CLOUDFLARE_MODEL` задаёт другую модель; клиент ожидает формат ответа `result.response`.

[Документация API](https://developers.cloudflare.com/workers-ai/get-started/rest-api/) · [Бесплатная квота](https://developers.cloudflare.com/workers-ai/platform/pricing/)

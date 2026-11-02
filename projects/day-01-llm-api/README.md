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

Задайте переменные в терминале macOS (zsh). После каждой команды `read` вставьте запрошенное значение и нажмите Enter; ввод токена скрыт.

```zsh
read -rs 'CLOUDFLARE_API_TOKEN?Cloudflare API token: '
export CLOUDFLARE_API_TOKEN
printf '\n'
read -r 'CLOUDFLARE_ACCOUNT_ID?Cloudflare Account ID: '
export CLOUDFLARE_ACCOUNT_ID
```

## Запуск

В том же терминале с активным `.venv`:

```bash
python main.py
```

Введите вопрос и дождитесь ответа. Каждый запуск отправляет один запрос.

Токен не добавляйте в Git. `.env` автоматически не загружается. Необязательная переменная `CLOUDFLARE_MODEL` задаёт другую модель; клиент ожидает формат ответа `result.response`.

[Документация API](https://developers.cloudflare.com/workers-ai/get-started/rest-api/) · [Бесплатная квота](https://developers.cloudflare.com/workers-ai/platform/pricing/)

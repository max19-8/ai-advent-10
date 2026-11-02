"""Один вопрос к Llama через REST API Cloudflare Workers AI, без SDK."""

import os
import re
import sys
from urllib.parse import quote

import requests


def main():
    token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    if not token or not account_id:
        print(
            "Ошибка: задайте CLOUDFLARE_API_TOKEN и CLOUDFLARE_ACCOUNT_ID.",
            file=sys.stderr,
        )
        return 1
    if not re.fullmatch(r"[a-fA-F0-9]{32}", account_id):
        print("Ошибка: Account ID должен содержать 32 шестнадцатеричных символа.", file=sys.stderr)
        return 1

    question = input("Введите вопрос: ").strip()
    if not question:
        print("Ошибка: вопрос не должен быть пустым.", file=sys.stderr)
        return 1

    model = os.environ.get("CLOUDFLARE_MODEL", "@cf/meta/llama-3.1-8b-instruct").strip()
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/"
        f"{quote(model, safe='/@')}"
    )
    print("Ожидание ответа…", flush=True)
    try:
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}"},
            json={
                "messages": [{"role": "user", "content": question}],
                "max_tokens": 512,
            },
            timeout=(10, 90),
        )
    except requests.Timeout:
        print("Ошибка: превышено время ожидания API.", file=sys.stderr)
        return 1
    except requests.RequestException:
        print("Ошибка соединения с Cloudflare API. Проверьте сеть.", file=sys.stderr)
        return 1

    try:
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Expected an object")
    except ValueError:
        print(f"Ошибка: API вернул не JSON-объект (HTTP {response.status_code}).", file=sys.stderr)
        return 1

    if not response.ok or data.get("success") is False or data.get("errors"):
        print(f"Ошибка Cloudflare API: HTTP {response.status_code}.", file=sys.stderr)
        for error in (data.get("errors") or [])[:3]:
            if isinstance(error, dict):
                message = str(error.get("message", "Неизвестная ошибка"))
                message = message.replace(token, "[TOKEN_HIDDEN]")
                print(f"Причина от Cloudflare: {message[:1000]}", file=sys.stderr)
        return 1

    try:
        answer = data["result"]["response"].strip()
    except (KeyError, AttributeError, TypeError):
        print("Ошибка: API вернул неожиданный формат ответа.", file=sys.stderr)
        return 1
    if not answer:
        print("Модель не вернула текст ответа.", file=sys.stderr)
        return 1

    print(f"\nОтвет:\n{answer}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (EOFError, KeyboardInterrupt):
        print("\nВвод отменён.", file=sys.stderr)
        sys.exit(130)

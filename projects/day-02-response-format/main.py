"""Четыре режима контроля ответа, по одному запросу в каждом, без SDK."""

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from jsonschema import Draft202012Validator


DIRECTORY = Path(__file__).resolve().parent
DEFAULT_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
FORMAT = (
    '\nВерни только JSON-объект с единственным обязательным полем answer '
    '(строка с ответом на вопрос). Других полей быть не должно. '
    'Не добавляй пояснения вне JSON и Markdown-обёртку.'
)
STOP = "<END>"
MAX_TOKENS = 50
MODES = {
    "schema": "Строгая схема",
    "baseline": "Без ограничений",
    "limited": "Лимит 50 токенов",
    "formatted": "Описание формата + stop sequence",
}
# Четыре варианта по одному запуску; каждый запрос независим.
SCHEDULE = tuple(MODES)


def build_payload(mode, model, question):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "temperature": 0.6,
        "stream": False,
    }
    if mode == "baseline":
        return payload
    if mode == "limited":
        payload["max_tokens"] = MAX_TOKENS
        return payload
    if mode not in ("formatted", "schema"):
        raise ValueError(f"Неизвестный режим: {mode}")
    if mode == "formatted":
        payload["messages"][0]["content"] += FORMAT
        payload["messages"][0]["content"] += (
            f"\nСразу после JSON напиши маркер {STOP}, затем закончи ответ."
        )
        payload["stop"] = [STOP]
    else:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "answer", "strict": True, "schema": SCHEMA},
        }
    return payload


def reject_constant(value):
    raise ValueError(f"Недопустимая JSON-константа: {value}")


def check_answer(answer):
    checks = {"json_ok": False, "schema_ok": False}
    try:
        # Никакого удаления Markdown, извлечения подстроки или исправления JSON.
        parsed = json.loads(answer, parse_constant=reject_constant)
    except ValueError as error:
        checks["error"] = f"JSON: {error}"
        return checks
    checks["json_ok"] = True
    errors = list(Draft202012Validator(SCHEMA).iter_errors(parsed))
    if errors:
        checks["error"] = "Схема: " + "; ".join(error.message for error in errors)
        return checks
    checks["schema_ok"] = True
    return checks


def request_once(url, token, payload):
    result = {"request": payload, "api_ok": False}
    try:
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
            timeout=(10, 90),
        )
    except requests.RequestException:
        result["error"] = "Ошибка сети или таймаут: запрос к Cloudflare не выполнен."
        result["network_error"] = True
        return result
    result["http_status"] = response.status_code
    result["raw_response"] = response.text
    try:
        data = response.json()
    except ValueError:
        result["error"] = "API вернул ответ не в формате JSON."
        return result
    if not response.ok or not isinstance(data, dict):
        result["error"] = f"Ошибка API: HTTP {response.status_code}."
        return result
    if data.get("error") or data.get("errors") or data.get("success") is False:
        result["error"] = "API сообщил об ошибке; подробности в raw_response."
        return result
    try:
        choice = data["choices"][0]
        answer = choice["message"]["content"]
        if not isinstance(answer, str):
            raise TypeError("Ответ не является строкой")
        result["finish_reason"] = choice.get("finish_reason")
        result["stop_reason"] = choice.get("stop_reason")
    except (KeyError, IndexError, TypeError, AttributeError):
        result["error"] = "API вернул неожиданный формат или ответ без текста."
        return result
    result.update(api_ok=True, answer=answer, usage=data.get("usage"))
    result["checks"] = check_answer(answer)
    return result


def run_experiment(url, token, model, question):
    records = []
    for mode in SCHEDULE:
        repeat = 1 + sum(record["mode"] == mode for record in records)
        print(f"[{len(records) + 1}/{len(SCHEDULE)}] {MODES[mode]}, запуск {repeat}…", flush=True)
        record = request_once(url, token, build_payload(mode, model, question))
        record.update(mode=mode, repeat=repeat)
        records.append(record)
        if record.get("network_error") or (mode == "schema" and not record["api_ok"]):
            print("Прогон остановлен: проверьте сеть или поддержку схемы в API.")
            break
    return records


def make_report(records, question):
    lines = []
    for index, record in enumerate(records, 1):
        payload = record.get("request", {})
        prompt = "\n\n".join(message["content"] for message in payload.get("messages", [])) or question
        conditions = []
        if record["mode"] in ("baseline", "limited"):
            conditions.append("Формат свободный.")
        else:
            conditions.append('Формат: JSON с единственным строковым полем answer.')
        if "max_tokens" in payload:
            conditions.append(f"max_tokens: {payload['max_tokens']}.")
        else:
            conditions.append("Явный лимит токенов не задан; используется лимит Cloudflare.")
        if "response_format" in payload:
            conditions.append("Строгая JSON Schema через API: answer обязателен, лишние поля запрещены.")
        if "stop" in payload:
            conditions.append("Stop sequence: " + json.dumps(payload["stop"], ensure_ascii=False) + ".")
        if "model" in payload:
            conditions.append(f"Модель: {payload['model']}.")
        if "temperature" in payload:
            conditions.append(f"temperature: {payload['temperature']}.")
        answer = record.get("answer", "Ответ LLM не получен.")
        lines += [
            f"## {index}. {MODES[record['mode']]}, попытка {record['repeat']}",
            "",
            "**Запрос:**", "",
            "````text", prompt, "````", "",
            "**Условия:**", "",
            "````text", "\n".join(conditions), "````", "",
            "**Вывод LLM:**", "",
            "````text", answer, "````", "",
        ]
    return "\n".join(lines)


def save_results(records, model, token, account_id, question):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = DIRECTORY / "results"
    directory.mkdir(exist_ok=True)
    result = {
        "model": model, "task": question, "schema": SCHEMA,
        "records": records,
    }
    # Заголовки авторизации не сохраняются; дополнительно скрываем секрет в теле ошибок.
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    report = make_report(records, question)
    for secret in (token, account_id):
        serialized = serialized.replace(secret, "[HIDDEN]")
        report = report.replace(secret, "[HIDDEN]")
    (directory / f"{stamp}.json").write_text(serialized + "\n", encoding="utf-8")
    path = directory / f"{stamp}.md"
    path.write_text(report, encoding="utf-8")
    return path, report


def main():
    load_dotenv(DIRECTORY / ".env", override=False)
    token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    model = os.environ.get("CLOUDFLARE_MODEL", DEFAULT_MODEL).strip()
    if not token or not re.fullmatch(r"[a-fA-F0-9]{32}", account_id) or not model:
        print("Ошибка: заполните токен, Account ID (32 hex-символа) и модель в .env.", file=sys.stderr)
        return 1
    question = input("Введите вопрос: ").strip()
    if not question:
        print("Ошибка: вопрос не должен быть пустым.", file=sys.stderr)
        return 1
    Draft202012Validator.check_schema(SCHEMA)
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions"
    records = run_experiment(url, token, model, question)
    path, report = save_results(records, model, token, account_id, question)
    print("\n" + report)
    print(f"Результаты сохранены: {path}")
    return 0 if len(records) == len(SCHEDULE) and all(record["api_ok"] for record in records) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (EOFError, KeyboardInterrupt):
        print("\nПрогон отменён.", file=sys.stderr)
        sys.exit(130)

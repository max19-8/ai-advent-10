"""Запуск: python main.py; техническая проверка: python main.py --check."""

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from experiment import DEFAULT_MODEL, MODELS, PROMPT_PROTOCOL, run_experiment
from report import save_results
from tasks import EASY_TASKS, HARD_TASKS, KNAPSACK_TASKS, ROUTE_TASKS, TASKS


DIRECTORY = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description="Четыре способа, reasoning low/high. По умолчанию две простые задачи. Отчёт сохраняется и печатается.")
    parser.add_argument("--check", action="store_true", help="Только два прямых запроса первой задачи выбранного набора: low и high. Проверка подключения.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--hard", action="store_true", help="Одна более сложная задача про мост и фонарь: 8 решений, 10 запросов.")
    selection.add_argument("--easy", action="store_true", help="Одна задача про два насоса и слив. 8 решений, 10 запросов.")
    selection.add_argument("--knapsack", action="store_true", help="Рюкзак: выбор предметов с максимальной ценностью при ограничении веса. 8 решений, 10 запросов.")
    selection.add_argument("--route", action="store_true", help="Простой маршрут: 4 вершины, одна обязательная остановка и купон. 8 решений, 10 запросов.")
    args = parser.parse_args(argv)
    suite = next((name for name in ("route", "knapsack", "easy", "hard") if getattr(args, name)), "basic")
    tasks = {"basic": TASKS, "easy": EASY_TASKS, "hard": HARD_TASKS, "knapsack": KNAPSACK_TASKS, "route": ROUTE_TASKS}[suite]
    if args.check:
        tasks = tasks[:1]
    load_dotenv(DIRECTORY / ".env", override=False)
    token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    model = os.environ.get("CLOUDFLARE_MODEL", DEFAULT_MODEL).strip()
    if not token or not re.fullmatch(r"[a-fA-F0-9]{32}", account_id):
        print("Ошибка: заполните CLOUDFLARE_API_TOKEN и CLOUDFLARE_ACCOUNT_ID (32 hex-символа) в .env рядом с main.py.", file=sys.stderr)
        return 1
    if model not in MODELS:
        print("Ошибка: выберите GPT-OSS с reasoning low/high: " + ", ".join(MODELS) + ". Проверьте также экспортированный CLOUDFLARE_MODEL.", file=sys.stderr)
        return 1
    now = datetime.now(timezone.utc)
    stem = now.strftime("%Y%m%dT%H%M%S%fZ") + ("-" + suite if suite != "basic" else "") + ("-check" if args.check else "")
    run = {
        "started_at": now.isoformat(), "model": model, "check": args.check,
        "state": "выполняется", "suite": suite,
        "prompt_protocol": PROMPT_PROTOCOL,
        "tasks": list(tasks), "records": [],
    }
    path = None
    report = ""

    def checkpoint():
        nonlocal path, report
        path, report = save_results(run, DIRECTORY / "results", stem, (token, account_id))

    print("Модель: " + model, flush=True)
    print("Проверка: 2 запроса." if args.check else f"Эксперимент: {len(tasks) * 8} решений, {len(tasks) * 10} запросов. Повторных попыток нет.", flush=True)
    code = 1
    try:
        checkpoint()
        complete = run_experiment(
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/responses",
            token, model, run["records"], checkpoint, check=args.check, tasks=tasks,
        )
        run["state"] = "завершён" if complete else "остановлен из-за ошибки"
        code = 0 if complete else 1
    except KeyboardInterrupt:
        run["state"] = "прерван пользователем"
        code = 130
    finally:
        checkpoint()
        print("\n" + report)
        print("Markdown-отчёт: " + str(path))
        print("Исходные данные: " + str(path.with_suffix(".json")))
    return code


if __name__ == "__main__":
    sys.exit(main())

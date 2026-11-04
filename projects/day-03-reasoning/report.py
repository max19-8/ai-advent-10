"""Один и тот же Markdown-отчёт для файла и терминала."""

import json
import re
from pathlib import Path

from experiment import METHODS


def token_counts(call):
    usage = call.get("usage")
    usage = usage if isinstance(usage, dict) else {}

    def count(value):
        return value if type(value) is int and value >= 0 else None

    incoming = count(usage.get("input_tokens"))
    outgoing = count(usage.get("output_tokens"))
    total = count(usage.get("total_tokens"))
    if total is None and incoming is not None and outgoing is not None:
        total = incoming + outgoing
    details = usage.get("output_tokens_details")
    reasoning = count(details.get("reasoning_tokens")) if isinstance(details, dict) else None
    return {"input": incoming, "output": outgoing, "total": total, "reasoning": reasoning}


def total_tokens(calls):
    counts = [token_counts(call)["total"] for call in calls]
    known = sum(value for value in counts if value is not None)
    if any(value is None for value in counts):
        return f"н/д (известно: {known})"
    return str(known)


def fenced(text):
    fence = "`" * max(3, 1 + max((len(run) for run in re.findall(r"`+", text)), default=0))
    return fence + "text\n" + text + "\n" + fence


def make_report(run):
    records = run["records"]
    tasks = run["tasks"]
    task_count = len(tasks)
    calls = [call for record in records for call in record["calls"]]
    limits = {call["request"].get("max_output_tokens") for call in calls}
    limit_description = "не задаётся; действует ограничение сервиса по умолчанию"
    if len(limits) == 1 and None not in limits:
        limit_description = str(next(iter(limits))) + " для каждого запроса"
    elif len(limits) > 1:
        limit_description = "различается между запросами; см. параметры запросов в JSON"
    labels = {"correct": "верно", "wrong": "неверно", "review": "ручная проверка", "error": "ошибка / не завершено"}
    lines = [
        "# День 3. Разные способы рассуждения", "",
        f"Запуск: {run['started_at']}. Модель: `{run['model']}`.",
        "Режим: " + ("проверка подключения (2 запроса, только первая задача выбранного набора и прямой способ)." if run["check"] else f"полный эксперимент: задач — {task_count}, способов — 4, уровней рассуждения — 2."),
        "Версия способа с составлением промпта: `" + run.get("prompt_protocol", "generated-prompt-v1") + "`.",
        "Состояние: " + run["state"] + ".",
        f"Выполнено обращений к API: {len(calls)}. Токены всего: {total_tokens(calls)}. Время запросов: {sum(call['elapsed_seconds'] for call in calls):.2f} с.",
        "", "## Условия", "",
        "Каждый вариант выполняется один раз, без истории и стриминга; порядок low → high, внутри — задача → способ.",
        f"`reasoning.effort`: low / high; `max_output_tokens`: {limit_description}; temperature не задаётся (настройка сервиса по умолчанию).",
        "low — минимальное рассуждение этой модели, high — максимальное; low не означает отключение рассуждения.",
        "Время — длительность HTTP-запросов по часам клиента, включая сеть и ожидание сервиса. Для составления промпта суммируются оба этапа.",
        "Токены — input + output из usage API. Reasoning tokens показываются отдельно, когда доступны, и повторно к общему числу не прибавляются. н/д означает отсутствие статистики, не нулевой расход.",
        "Правильность проверяется по последней строке «Ответ: <число>». Если формат не распознан, нужна ручная проверка. Незавершённый ответ не засчитывается.",
        "", "## Сравнение", "",
        f"Знаменатель {task_count} — число запланированных задач; столбцы справа показывают неполноту и непроверенные ответы.", "",
        f"| Способ | Рассуждение | Верно / {task_count} | Решений получено / {task_count} | Ручная проверка | Ошибки / не завершено | Токены, сумма | Время, с |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, title in METHODS.items():
        for effort in ("low", "high"):
            group = [r for r in records if r["method"] == method and r["effort"] == effort]
            if not group:
                lines.append(f"| {title} | {effort} | — | 0/{task_count} | 0 | 0 | — | — |")
                continue
            states = [r["grading"]["status"] for r in group]
            group_calls = [call for r in group for call in r["calls"]]
            done = sum(state != "error" for state in states)
            elapsed = sum(call["elapsed_seconds"] for call in group_calls)
            lines.append(f"| {title} | {effort} | {states.count('correct')}/{task_count} | {done}/{task_count} | {states.count('review')} | {states.count('error')} | {total_tokens(group_calls)} | {elapsed:.2f} |")
    lines += ["", "## Наблюдения по полученным результатам", ""]
    comparisons = 0
    if run["check"]:
        lines.append("Это техническая проверка; сравнение четырёх способов выполняется полным запуском.")
    else:
        def score(method, effort):
            group = [r for r in records if r["method"] == method and r["effort"] == effort]
            if len(group) != task_count or any(r["grading"]["status"] not in ("correct", "wrong") for r in group):
                return None
            return sum(r["grading"]["status"] == "correct" for r in group)

        for effort in ("low", "high"):
            baseline = score("direct", effort)
            if baseline is None:
                continue
            for method in tuple(METHODS)[1:]:
                value = score(method, effort)
                if value is not None:
                    delta = value - baseline
                    observation = "не увеличил число верных ответов" if delta <= 0 else f"дал на {delta} верный ответ больше"
                    lines.append(f"- {effort}: способ «{METHODS[method]}» {observation} относительно прямого запроса ({value}/{task_count} против {baseline}/{task_count}).")
                    comparisons += 1
        for method, title in METHODS.items():
            low, high = score(method, "low"), score(method, "high")
            if low is not None and high is not None:
                lines.append(f"- «{title}»: low → high — {low}/{task_count} → {high}/{task_count}.")
                comparisons += 1
        if not comparisons:
            lines.append("Недостаточно полностью проверенных результатов для сравнения; проверьте ошибки и ответы с пометкой ручной проверки.")
    lines += [
        "", f"Набор содержит задач: {task_count}. Один ответ на вариант не измеряет стабильность и не позволяет обобщать результат на другие задачи. Если все получили {task_count}/{task_count}, этот набор показывает различия в расходах, но не в точности.",
        "", "## Задачи и ответы", "",
    ]
    for task in tasks:
        lines += [f"### {task['title']}", "", task["question"], "", f"Эталон: {task['expected']} {task['unit']}. {task['explanation']}", ""]
        for record in (r for r in records if r["task_id"] == task["id"]):
            grading = record["grading"]
            lines += [f"#### {METHODS[record['method']]} · {record['effort']}", "", f"Проверка: **{labels[grading['status']]}**. Извлечённое число: {grading['value'] or '—'}.", ""]
            if record["method"] == "generated_prompt":
                lines += ["Оценивается результат выполнения промпта во втором запросе.", ""]
            for call in record["calls"]:
                stage = "Составление промпта" if call["stage"] == "generate_prompt" else "Решение"
                answer_label = "Ответ:"
                if record["method"] == "generated_prompt":
                    if call["stage"] == "generate_prompt":
                        answer_label = "**Сгенерированный промпт:**"
                    else:
                        stage = "Выполнение сгенерированного промпта"
                        answer_label = "**Результат выполнения промпта:**"
                counts = token_counts(call)
                lines += [
                    f"**Этап: {stage}.** Время: {call['elapsed_seconds']:.2f} с. Токены: {total_tokens([call])}; input: {counts['input'] if counts['input'] is not None else 'н/д'}, output: {counts['output'] if counts['output'] is not None else 'н/д'}, reasoning: {counts['reasoning'] if counts['reasoning'] is not None else 'н/д'}.", "",
                    "Запрос:", "", fenced(call["request"]["input"][0]["content"]), "",
                    answer_label, "", fenced(call.get("answer", "")), "",
                ]
                if call.get("error"):
                    lines += ["Ошибка: " + call["error"], ""]
    return "\n".join(lines).rstrip() + "\n"


def redact(value, secrets):
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[HIDDEN]")
        return value
    if isinstance(value, dict):
        return {key: redact(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item, secrets) for item in value]
    return value


def save_results(run, directory, stem, secrets):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cleaned = redact(run, secrets)
    report = make_report(cleaned)
    md_path = directory / (stem + ".md")
    for path, text in (
        (directory / (stem + ".json"), json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n"),
        (md_path, report),
    ):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    return md_path, report

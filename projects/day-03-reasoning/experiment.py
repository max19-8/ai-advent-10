"""Независимые запросы через Cloudflare Responses API, без SDK и повторов."""

from time import perf_counter

import requests

from tasks import FINAL_FORMAT, TASKS, grade_answer


DEFAULT_MODEL = "@cf/openai/gpt-oss-20b"
MODELS = (DEFAULT_MODEL, "@cf/openai/gpt-oss-120b")
PROMPT_PROTOCOL = "generated-prompt-v2"
EFFORTS = ("low", "high")
METHODS = {
    "direct": "Прямой ответ",
    "step_by_step": "Решай пошагово",
    "generated_prompt": "Составление промпта",
    "experts": "Группа экспертов",
}


def build_payload(model, effort, prompt):
    return {
        "model": model,
        "input": [{"role": "user", "content": prompt}],
        "reasoning": {"effort": effort},
        "stream": False,
    }


def build_prompt(task, method, instruction=None):
    question = task["question"]
    if method == "direct":
        return question + FINAL_FORMAT
    if method == "step_by_step":
        return question + "\nРешай пошагово." + FINAL_FORMAT
    if method == "experts":
        return (
            question + "\nПредставь группу из трёх экспертов: аналитик, инженер, критик. "
            "Каждый должен предложить своё решение задачи. Покажи решение каждого "
            "с указанием роли, затем сопоставь их и дай общий итог." + FINAL_FORMAT
        )
    if method == "generated_prompt":
        if instruction is None:
            return (
                "Твоя роль на этом этапе — автор промпта. Подготовь готовый промпт "
                "для другого решателя, который получит его вместе с исходной задачей. "
                "Создаваемый промпт должен поручать решить задачу, проверить решение "
                "и выдать окончательный численный ответ в единицах задачи. "
                "Сейчас верни только текст готового промпта, без предисловия. "
                "Вычисления и численный ответ оставь будущему решателю. "
                "Требование вернуть только текст промпта относится исключительно "
                "к твоему текущему этапу: не переноси его в создаваемую инструкцию. "
                "Инструкция для решателя должна требовать именно решение и итог, "
                "а не создание или повторение другого промпта.\n\nИсходная задача:\n"
                + question
            )
        return (
            "Твоя роль на этом этапе — решатель. Используй сгенерированный промпт "
            "ниже, чтобы решить исходную задачу.\n\nИсходная задача:\n" + question
            + "\n\nСгенерированный промпт:\n" + instruction
            + "\n\nТеперь выполни этот промпт и выдай фактическое решение с окончательным "
            "численным ответом. Требования к автору о подготовке только текста промпта "
            "относятся к предыдущему этапу; на текущем этапе требуется решение задачи."
            + FINAL_FORMAT
        )
    raise ValueError("Неизвестный способ: " + method)


def request_once(url, token, payload):
    record = {"request": payload, "api_ok": False, "http_timeout_seconds": None}
    started = perf_counter()
    try:
        response = requests.post(
            url, headers={"Authorization": "Bearer " + token},
            json=payload, timeout=None,
        )
    except requests.RequestException as error:
        record.update(error=f"Ошибка сети: {type(error).__name__}: {error}", network_error=True)
        return record
    finally:
        record["elapsed_seconds"] = perf_counter() - started
    record["http_status"] = response.status_code
    record["raw_response"] = response.text
    try:
        data = response.json()
    except ValueError:
        record["error"] = "API вернул не JSON."
        return record
    if not response.ok or not isinstance(data, dict):
        record["error"] = "Ошибка API: HTTP " + str(response.status_code)
        return record
    # Некоторые Cloudflare endpoints используют оболочку result.
    if data.get("success") is False or data.get("error") or data.get("errors"):
        record["error"] = "API сообщил об ошибке; подробности в raw_response."
        return record
    if isinstance(data.get("result"), dict):
        data = data["result"]
    if data.get("error"):
        record["error"] = "API сообщил об ошибке; подробности в raw_response."
        return record
    record["usage"] = data.get("usage")
    record["status"] = data.get("status")
    record["incomplete_details"] = data.get("incomplete_details")
    record["reasoning"] = data.get("reasoning")
    if data.get("reasoning") is not None and not isinstance(data["reasoning"], dict):
        record["error"] = "API вернул неожиданный формат reasoning."
        return record
    output = data.get("output", [])
    if not isinstance(output, list):
        record["error"] = "API вернул неожиданный формат output."
        return record
    texts = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            record["error"] = "API вернул неожиданный формат message.content."
            return record
        for part in content:
            if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
    record["answer"] = "\n".join(texts)
    returned_effort = (data.get("reasoning") or {}).get("effort")
    if returned_effort is not None and returned_effort != payload["reasoning"]["effort"]:
        record["error"] = "API вернул другой уровень reasoning.effort."
    elif data.get("status") == "incomplete":
        record["error"] = "Ответ не завершён: " + str(data.get("incomplete_details"))
    elif data.get("status") in ("failed", "cancelled"):
        record["error"] = "API не завершил ответ: " + data["status"]
    elif not record["answer"].strip():
        record["error"] = "API не вернул итоговый текст ответа."
    else:
        record["api_ok"] = True
    return record


def run_experiment(url, token, model, records, checkpoint, *, check=False, tasks=TASKS):
    """Сохраняет каждый этап; при сбое прекращает новые запросы."""
    methods = ("direct",) if check else tuple(METHODS)
    selected_tasks = tasks[:1] if check else tasks
    total = len(selected_tasks) * len(methods) * len(EFFORTS)
    for effort in EFFORTS:
        for task in selected_tasks:
            for method in methods:
                print(f"[{len(records) + 1}/{total}] {task['title']} · {METHODS[method]} · {effort}", flush=True)
                record = {
                    "task_id": task["id"], "method": method, "effort": effort,
                    "calls": [], "grading": {"status": "error", "value": None},
                }
                records.append(record)
                if method == "generated_prompt":
                    call = request_once(url, token, build_payload(model, effort, build_prompt(task, method)))
                    call["stage"] = "generate_prompt"
                    record["calls"].append(call)
                    checkpoint()
                    if not call["api_ok"]:
                        return False
                    instruction = call["answer"]
                else:
                    instruction = None
                call = request_once(url, token, build_payload(model, effort, build_prompt(task, method, instruction)))
                call["stage"] = "solve"
                record["calls"].append(call)
                if call["api_ok"]:
                    record["grading"] = grade_answer(call["answer"], task)
                checkpoint()
                if not call["api_ok"]:
                    return False
    return True

"""Проверки без сети и расхода квоты Cloudflare."""

import contextlib
import copy
import hashlib
import heapq
import io
import itertools
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import experiment
import main
from report import make_report, save_results, token_counts, total_tokens
from tasks import EASY_TASKS, HARD_TASKS, KNAPSACK_TASKS, ROUTE_TASKS, TASKS, grade_answer


class ExperimentTests(unittest.TestCase):
    def response(self, data, status=200):
        response = Mock(status_code=status, ok=status < 400)
        response.json.return_value = data
        response.text = json.dumps(data, ensure_ascii=False)
        return response

    def successful_post(self, url, **kwargs):
        payload = kwargs["json"]
        prompt = payload["input"][0]["content"]
        answer = "12" if "Все дороги двусторонние." in prompt else "34" if "Рюкзак выдерживает" in prompt else "9" if "Насос A заполняет" in prompt else "29" if "Пять человек находятся" in prompt else "5"
        text = "Проверь условия и единицы измерения, реши задачу и выдай численный итог." if prompt.startswith("Твоя роль на этом этапе — автор промпта.") else "Проверка вычислений.\nОтвет: " + answer
        return self.response({
            "status": "completed", "reasoning": payload["reasoning"],
            "output": [
                {"type": "reasoning", "summary": []},
                {"type": "message", "content": [{"type": "output_text", "text": text}]},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30,
                      "output_tokens_details": {"reasoning_tokens": 8}},
        })

    def run_sample(self, check=False):
        records = []
        with patch("experiment.requests.post", side_effect=self.successful_post) as post:
            with patch("experiment.perf_counter", side_effect=itertools.count()):
                with contextlib.redirect_stdout(io.StringIO()):
                    complete = experiment.run_experiment("https://example.invalid", "token-secret", experiment.DEFAULT_MODEL, records, lambda: None, check=check)
        return complete, records, post

    def run_document(self, records, check=False):
        return {"started_at": "2026-11-04", "model": experiment.DEFAULT_MODEL,
                "prompt_protocol": experiment.PROMPT_PROTOCOL,
                "check": check, "state": "завершён", "records": records, "tasks": list(TASKS)}

    def test_generated_prompt_has_separate_author_and_solver_instructions(self):
        author_prompt = experiment.build_prompt(HARD_TASKS[0], "generated_prompt")
        self.assertIn("Твоя роль на этом этапе — автор промпта.", author_prompt)
        self.assertIn("выдать окончательный численный ответ", author_prompt)
        self.assertIn("не переноси его в создаваемую инструкцию", author_prompt)
        instruction = "Построй граф состояний, найди минимум и выдай численный ответ."
        solver_prompt = experiment.build_prompt(HARD_TASKS[0], "generated_prompt", instruction)
        self.assertIn(instruction, solver_prompt)
        self.assertIn(HARD_TASKS[0]["question"], solver_prompt)
        self.assertIn("Твоя роль на этом этапе — решатель.", solver_prompt)
        self.assertIn("Теперь выполни этот промпт", solver_prompt)
        self.assertNotIn("Не решай задачу", author_prompt)

    def test_report_shows_generated_prompt_and_its_execution_result_separately(self):
        _, records, _ = self.run_sample()
        report = make_report(self.run_document(records))
        self.assertIn("**Сгенерированный промпт:**\n\n```text\nПроверь условия", report)
        self.assertIn("**Результат выполнения промпта:**\n\n```text\nПроверка вычислений.\nОтвет: 5", report)
        self.assertIn("Оценивается результат выполнения промпта во втором запросе.", report)
        self.assertIn(experiment.PROMPT_PROTOCOL, report)

    def test_grading_uses_only_explicit_final_answer_and_expected_units(self):
        cases = [
            ("Вместе 110, разница 100.\nОтвет: 5", "correct"),
            ("Ответ: 10", "wrong"),
            ("**Ответ: 5,00 рублей.**", "correct"),
            ("Ответ: 0.05 рублей", "wrong"),
            ("Цена 5 рублей, тетрадь 105.", "review"),
            ("Ответ: 5 или 10", "review"),
            ("Ответ: 5\nНо ответ может быть другим", "review"),
            ("Ответ: 5 минут", "review"),
            ("", "review"),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(grade_answer(text, TASKS[0])["status"], expected)
        self.assertEqual(grade_answer("Ответ: 5 минут", TASKS[1])["status"], "correct")
        self.assertEqual(grade_answer("Ответ: 9 часов", EASY_TASKS[0])["status"], "correct")
        self.assertEqual(grade_answer("Ответ: 7 часов", EASY_TASKS[0])["status"], "wrong")
        self.assertEqual(grade_answer("Ответ: 9 минут", EASY_TASKS[0])["status"], "review")
        self.assertEqual(grade_answer("Ответ: 34 балла", KNAPSACK_TASKS[0])["status"], "correct")
        self.assertEqual(grade_answer("Ответ: 32", KNAPSACK_TASKS[0])["status"], "wrong")
        self.assertEqual(grade_answer("Ответ: 12 рублей", ROUTE_TASKS[0])["status"], "correct")
        self.assertEqual(grade_answer("Ответ: 13", ROUTE_TASKS[0])["status"], "wrong")
        self.assertEqual(grade_answer("Ответ: 12 минут", ROUTE_TASKS[0])["status"], "review")

    def test_full_experiment_has_16_solutions_and_20_independent_calls(self):
        complete, records, post = self.run_sample()
        self.assertTrue(complete)
        self.assertEqual(len(records), 16)
        self.assertEqual(post.call_count, 20)
        self.assertTrue(all(call.kwargs.get("timeout", "not-set") is None for call in post.call_args_list))
        self.assertTrue(all(r["grading"]["status"] == "correct" for r in records))
        self.assertEqual(len({(r["task_id"], r["method"], r["effort"]) for r in records}), 16)
        for record in records:
            with self.subTest(record=(record["task_id"], record["method"], record["effort"])):
                self.assertEqual(len(record["calls"]), 2 if record["method"] == "generated_prompt" else 1)
                for call in record["calls"]:
                    payload = call["request"]
                    self.assertEqual(payload["reasoning"]["effort"], record["effort"])
                    self.assertEqual(len(payload["input"]), 1)
                    self.assertNotIn("previous_response_id", payload)
                    self.assertNotIn("max_output_tokens", payload)
                    self.assertNotIn("max_tokens", payload)
                if record["method"] == "generated_prompt":
                    instruction = record["calls"][0]["answer"]
                    self.assertIn(instruction, record["calls"][1]["request"]["input"][0]["content"])

    def test_meta_prompt_generation_is_checkpointed_before_solution(self):
        snapshots = []
        records = []
        with patch("experiment.requests.post", side_effect=self.successful_post):
            with contextlib.redirect_stdout(io.StringIO()):
                experiment.run_experiment("url", "token", "model", records, lambda: snapshots.append(copy.deepcopy(records)))
        self.assertTrue(any(snapshot[-1]["method"] == "generated_prompt" and len(snapshot[-1]["calls"]) == 1 for snapshot in snapshots))

    def test_check_mode_uses_only_two_requests_and_both_efforts(self):
        complete, records, post = self.run_sample(check=True)
        self.assertTrue(complete)
        self.assertEqual(post.call_count, 2)
        self.assertEqual([(r["task_id"], r["method"], r["effort"]) for r in records], [("notebook", "direct", "low"), ("notebook", "direct", "high")])

    def test_usage_counts_include_both_stages_without_double_counting_reasoning(self):
        _, records, _ = self.run_sample()
        calls = [call for record in records for call in record["calls"]]
        self.assertEqual(total_tokens(calls), "600")
        self.assertEqual(token_counts(calls[0]), {"input": 10, "output": 20, "total": 30, "reasoning": 8})
        self.assertEqual(total_tokens([{"usage": {"input_tokens": 3, "output_tokens": 7}}]), "10")
        self.assertEqual(total_tokens([calls[0], {"usage": None}]), "н/д (известно: 30)")
        report = make_report(self.run_document(records))
        self.assertIn("| Составление промпта | low | 2/2 | 2/2 | 0 | 0 | 120 | 4.00 |", report)
        self.assertIn("`max_output_tokens`: не задаётся", report)
        for call in calls:
            call["request"]["max_output_tokens"] = 8192
        self.assertIn("`max_output_tokens`: 8192 для каждого запроса", make_report(self.run_document(records)))

    def test_errors_stop_the_run_without_retry_or_false_wrong_answer(self):
        for failure in (self.response({"error": {"message": "unsupported reasoning"}}, 400), experiment.requests.ConnectionError()):
            with self.subTest(failure=type(failure).__name__):
                records = []
                kwargs = {"side_effect": failure} if isinstance(failure, Exception) else {"return_value": failure}
                with patch("experiment.requests.post", **kwargs) as post:
                    with contextlib.redirect_stdout(io.StringIO()):
                        complete = experiment.run_experiment("url", "token", "model", records, lambda: None)
                self.assertFalse(complete)
                self.assertEqual(post.call_count, 1)
                self.assertEqual(records[0]["grading"]["status"], "error")

    def test_incomplete_text_and_effort_mismatch_are_not_scored(self):
        cases = (
            {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
            {"status": "completed", "reasoning": {"effort": "medium"}},
        )
        for changes in cases:
            data = self.successful_post("url", json=experiment.build_payload("model", "low", "question")).json()
            data.update(changes)
            with self.subTest(changes=changes), patch("experiment.requests.post", return_value=self.response(data)):
                result = experiment.request_once("url", "token", experiment.build_payload("model", "low", "question"))
            self.assertFalse(result["api_ok"])
            self.assertEqual(result["answer"], "Проверка вычислений.\nОтвет: 5")
            self.assertEqual(result["usage"]["total_tokens"], 30)

    def test_malformed_api_fields_are_reported_without_crashing(self):
        for changes in ({"output": [{"type": "message", "content": None}]}, {"reasoning": "low"}):
            data = self.successful_post("url", json=experiment.build_payload("model", "low", "question")).json()
            data.update(changes)
            with self.subTest(changes=changes):
                with patch("experiment.requests.post", return_value=self.response(data)):
                    result = experiment.request_once("url", "token", experiment.build_payload("model", "low", "question"))
                self.assertFalse(result["api_ok"])
                self.assertIn("error", result)

    def test_report_preserves_answers_and_redacts_credentials_in_json_and_markdown(self):
        _, records, _ = self.run_sample(check=True)
        records[0]["calls"][0]["answer"] += "\n```` token-secret account-secret"
        records[0]["calls"][0]["raw_response"] = "token-secret account-secret"
        with tempfile.TemporaryDirectory() as directory:
            path, report = save_results(self.run_document(records, check=True), directory, "sample", ("token-secret", "account-secret"))
            self.assertEqual(path.read_text(), report)
            data = path.with_suffix(".json").read_text()
            for text in (report, data):
                self.assertNotIn("token-secret", text)
                self.assertNotIn("account-secret", text)
            self.assertEqual(json.loads(data)["records"][0]["calls"][0]["raw_response"], "[HIDDEN] [HIDDEN]")
            self.assertIn("`````text", report)
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_incomplete_or_ungraded_run_does_not_claim_prompting_has_no_benefit(self):
        _, records, _ = self.run_sample()
        for record in records:
            if record["method"] == "experts":
                record["grading"] = {"status": "review", "value": None}
        report = make_report(self.run_document(records))
        self.assertNotIn("способ «Группа экспертов» не увеличил", report)
        self.assertIn("| Группа экспертов | high | 0/2 | 2/2 | 2 | 0 |", report)

    def test_cli_prints_the_same_report_it_writes(self):
        credentials = {"CLOUDFLARE_API_TOKEN": "token-secret", "CLOUDFLARE_ACCOUNT_ID": "a" * 32}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(main, "DIRECTORY", Path(directory)), patch("main.load_dotenv"):
                with patch.dict(os.environ, credentials, clear=True), patch("experiment.requests.post", side_effect=self.successful_post):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        code = main.main([])
            self.assertEqual(code, 0)
            files = list((Path(directory) / "results").glob("*.md"))
            self.assertEqual(len(files), 1)
            self.assertIn(files[0].read_text(), output.getvalue())
            run = json.loads(files[0].with_suffix(".json").read_text())
            self.assertEqual(run["state"], "завершён")
            self.assertEqual(len(run["records"]), 16)

    def test_cli_api_failure_keeps_partial_report_and_returns_nonzero(self):
        credentials = {"CLOUDFLARE_API_TOKEN": "token-secret", "CLOUDFLARE_ACCOUNT_ID": "a" * 32}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(main, "DIRECTORY", Path(directory)), patch("main.load_dotenv"):
                with patch.dict(os.environ, credentials, clear=True), patch("experiment.requests.post", side_effect=experiment.requests.Timeout):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        code = main.main([])
            self.assertEqual(code, 1)
            path = next((Path(directory) / "results").glob("*.md"))
            self.assertIn(path.read_text(), output.getvalue())
            data = json.loads(path.with_suffix(".json").read_text())
            self.assertEqual(len(data["records"]), 1)
            self.assertEqual(data["state"], "остановлен из-за ошибки")

    def test_cli_interrupt_keeps_completed_work(self):
        credentials = {"CLOUDFLARE_API_TOKEN": "token-secret", "CLOUDFLARE_ACCOUNT_ID": "a" * 32}
        response = self.successful_post("url", json=experiment.build_payload("model", "low", "question"))
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(main, "DIRECTORY", Path(directory)), patch("main.load_dotenv"):
                with patch.dict(os.environ, credentials, clear=True), patch("experiment.requests.post", side_effect=[response, KeyboardInterrupt]):
                    with contextlib.redirect_stdout(io.StringIO()):
                        code = main.main([])
            self.assertEqual(code, 130)
            data = json.loads(next((Path(directory) / "results").glob("*.json")).read_text())
            self.assertEqual(data["records"][0]["grading"]["status"], "correct")
            self.assertEqual(data["state"], "прерван пользователем")

    def test_bridge_reference_is_the_shortest_path_over_all_legal_states(self):
        times = (1, 3, 6, 8, 12)
        goal = (1 << len(times)) - 1
        queue = [(0, 0, False)]
        settled = set()
        minimum = None
        while queue:
            minutes, right_mask, torch_on_right = heapq.heappop(queue)
            state = (right_mask, torch_on_right)
            if state in settled:
                continue
            settled.add(state)
            if right_mask == goal and torch_on_right:
                minimum = minutes
                break
            available = [i for i in range(len(times)) if bool(right_mask & (1 << i)) == torch_on_right]
            for size in (1, 2):
                for group in itertools.combinations(available, size):
                    next_mask = right_mask
                    for person in group:
                        next_mask ^= 1 << person
                    heapq.heappush(queue, (minutes + max(times[i] for i in group), next_mask, not torch_on_right))
        self.assertEqual(str(minimum), HARD_TASKS[0]["expected"])

    def test_hard_cli_produces_8_solutions_10_calls_and_one_task_denominators(self):
        credentials = {"CLOUDFLARE_API_TOKEN": "token-secret", "CLOUDFLARE_ACCOUNT_ID": "a" * 32}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(main, "DIRECTORY", Path(directory)), patch("main.load_dotenv"):
                with patch.dict(os.environ, credentials, clear=True), patch("experiment.requests.post", side_effect=self.successful_post) as post:
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        code = main.main(["--hard"])
            self.assertEqual(code, 0)
            self.assertEqual(post.call_count, 10)
            path = next((Path(directory) / "results").glob("*-hard.json"))
            run = json.loads(path.read_text())
            self.assertEqual(run["tasks"], list(HARD_TASKS))
            self.assertEqual(len(run["records"]), 8)
            self.assertTrue(all(r["grading"]["status"] == "correct" for r in run["records"]))
            report = path.with_suffix(".md").read_text()
            self.assertIn("| Прямой ответ | low | 1/1 | 1/1 |", report)
            self.assertNotIn("/2", report)
            self.assertNotIn("Тетрадь и ручка", report)
            self.assertIn(report, output.getvalue())

    def test_easy_cli_runs_all_methods_with_both_efforts_and_no_client_limits(self):
        credentials = {"CLOUDFLARE_API_TOKEN": "token-secret", "CLOUDFLARE_ACCOUNT_ID": "a" * 32}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(main, "DIRECTORY", Path(directory)), patch("main.load_dotenv"):
                with patch.dict(os.environ, credentials, clear=True), patch("experiment.requests.post", side_effect=self.successful_post) as post:
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        code = main.main(["--easy"])
            self.assertEqual(code, 0)
            self.assertEqual(post.call_count, 10)
            path = next((Path(directory) / "results").glob("*-easy.json"))
            run = json.loads(path.read_text())
            self.assertEqual(run["suite"], "easy")
            self.assertEqual(run["tasks"], list(EASY_TASKS))
            self.assertEqual(len(run["records"]), 8)
            self.assertEqual({r["effort"] for r in run["records"]}, {"low", "high"})
            self.assertTrue(all(r["grading"] == {"status": "correct", "value": "9"} for r in run["records"]))
            for call in post.call_args_list:
                self.assertIsNone(call.kwargs["timeout"])
                self.assertNotIn("max_output_tokens", call.kwargs["json"])
                self.assertNotIn("max_tokens", call.kwargs["json"])
            self.assertIn(path.with_suffix(".md").read_text(), output.getvalue())

    def test_report_uses_task_metadata_from_the_run_instead_of_current_defaults(self):
        run = self.run_document([])
        run["tasks"] = [{"id": "archived", "title": "Архивное условие", "question": "Другой вопрос", "expected": "42", "unit": "", "explanation": "Архивное объяснение"}]
        report = make_report(run)
        self.assertIn("Архивное условие", report)
        self.assertIn("Другой вопрос", report)
        self.assertIn("Знаменатель 1", report)
        self.assertNotIn("Тетрадь и ручка", report)

    def test_knapsack_reference_is_verified_by_all_64_subsets(self):
        items = (("A", 6, 13), ("B", 5, 12), ("C", 4, 8), ("D", 3, 7), ("E", 2, 4), ("F", 7, 15))
        feasible = []
        for size in range(len(items) + 1):
            for group in itertools.combinations(items, size):
                if sum(item[1] for item in group) <= 15:
                    feasible.append((sum(item[2] for item in group), tuple(item[0] for item in group)))
        optimum = max(value for value, _ in feasible)
        self.assertEqual(str(optimum), KNAPSACK_TASKS[0]["expected"])
        self.assertEqual([names for value, names in feasible if value == optimum], [("B", "D", "F")])

    def test_knapsack_cli_passes_the_new_question_through_all_8_variants(self):
        credentials = {"CLOUDFLARE_API_TOKEN": "token-secret", "CLOUDFLARE_ACCOUNT_ID": "a" * 32}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(main, "DIRECTORY", Path(directory)), patch("main.load_dotenv"):
                with patch.dict(os.environ, credentials, clear=True), patch("experiment.requests.post", side_effect=self.successful_post) as post:
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        code = main.main(["--knapsack"])
            self.assertEqual(code, 0)
            self.assertEqual(post.call_count, 10)
            path = next((Path(directory) / "results").glob("*-knapsack.json"))
            run = json.loads(path.read_text())
            self.assertEqual(run["suite"], "knapsack")
            self.assertEqual(run["tasks"], list(KNAPSACK_TASKS))
            self.assertEqual(len({(r["method"], r["effort"]) for r in run["records"]}), 8)
            self.assertTrue(all(r["grading"] == {"status": "correct", "value": "34"} for r in run["records"]))
            self.assertIn(path.with_suffix(".md").read_text(), output.getvalue())
            for call in post.call_args_list:
                self.assertIsNone(call.kwargs["timeout"])
                self.assertNotIn("max_output_tokens", call.kwargs["json"])
                self.assertNotIn("max_tokens", call.kwargs["json"])

    def test_saved_runs_and_manual_reviews_are_consistent_and_unchanged(self):
        path = Path(__file__).resolve().parent / "results.json"
        bundle = json.loads(path.read_text())
        expected = {
            "20261104T174305505120Z": (16, 14),
            "20261104T180347308741Z-hard": (4, 2),
            "20261104T185145962691Z-easy": (8, 8),
            "20261104T191138483636Z-knapsack": (8, 8),
            "20261104T194048352468Z-route": (8, 7),
        }
        self.assertEqual({saved["id"] for saved in bundle["runs"]}, set(expected))
        for saved in bundle["runs"]:
            with self.subTest(run=saved["id"]):
                encoded = (json.dumps(saved["run"], ensure_ascii=False, indent=2) + "\n").encode()
                self.assertEqual(hashlib.sha256(encoded).hexdigest(), saved["source_sha256"])
                run = copy.deepcopy(saved["run"])
                for annotation in saved["review"]["overrides"]:
                    record = next(r for r in run["records"] if all(r[key] == annotation[key] for key in ("task_id", "method", "effort")))
                    self.assertEqual(record["grading"], annotation["original_grading"])
                    self.assertIn(annotation["evidence"], record["calls"][-1]["answer"])
                    record["grading"] = annotation["reviewed_grading"]
                outcomes = [r for r in run["records"] if r["grading"]["status"] in ("correct", "wrong")]
                self.assertEqual((len(outcomes), sum(r["grading"]["status"] == "correct" for r in outcomes)), expected[saved["id"]])
                for observation in saved["review"].get("observations", []):
                    record = next(r for r in run["records"] if all(r[key] == observation[key] for key in ("task_id", "method", "effort")))
                    self.assertEqual(record["grading"], observation["grading"])

    def test_route_reference_is_the_shortest_path_with_visits_and_single_coupon(self):
        edges = (("S", "A", 6), ("S", "B", 10), ("A", "B", 2), ("A", "T", 14), ("B", "T", 8))
        graph = {node: [] for node in ("S", "A", "B", "T")}
        for start, end, cost in edges:
            graph[start].append((end, cost))
            graph[end].append((start, cost))
        queue = [(0, "S", 0, False)]
        settled = set()
        minimum_twice = None
        while queue:
            cost_twice, node, visited, coupon_used = heapq.heappop(queue)
            state = (node, visited, coupon_used)
            if state in settled:
                continue
            settled.add(state)
            if node == "T" and visited == 1:
                minimum_twice = cost_twice
                break
            for next_node, weight in graph[node]:
                next_visited = visited | (1 if next_node == "A" else 0)
                heapq.heappush(queue, (cost_twice + 2 * weight, next_node, next_visited, coupon_used))
                if not coupon_used:
                    heapq.heappush(queue, (cost_twice + weight, next_node, next_visited, True))
        self.assertEqual(minimum_twice, 2 * int(ROUTE_TASKS[0]["expected"]))

    def test_route_cli_runs_all_8_variants_and_reports_the_new_task(self):
        credentials = {"CLOUDFLARE_API_TOKEN": "token-secret", "CLOUDFLARE_ACCOUNT_ID": "a" * 32}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(main, "DIRECTORY", Path(directory)), patch("main.load_dotenv"):
                with patch.dict(os.environ, credentials, clear=True), patch("experiment.requests.post", side_effect=self.successful_post) as post:
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        code = main.main(["--route"])
            self.assertEqual(code, 0)
            self.assertEqual(post.call_count, 10)
            path = next((Path(directory) / "results").glob("*-route.json"))
            run = json.loads(path.read_text())
            self.assertEqual(run["suite"], "route")
            self.assertEqual(run["tasks"], list(ROUTE_TASKS))
            self.assertEqual(len({(r["method"], r["effort"]) for r in run["records"]}), 8)
            self.assertTrue(all(r["grading"] == {"status": "correct", "value": "12"} for r in run["records"]))
            self.assertIn(path.with_suffix(".md").read_text(), output.getvalue())
            for call in post.call_args_list:
                self.assertIsNone(call.kwargs["timeout"])
                self.assertNotIn("max_output_tokens", call.kwargs["json"])
                self.assertNotIn("max_tokens", call.kwargs["json"])


if __name__ == "__main__":
    unittest.main()

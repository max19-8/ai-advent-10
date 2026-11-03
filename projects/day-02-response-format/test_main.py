"""Локальные проверки без обращений к Cloudflare."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import main


QUESTION = "Чем список отличается от кортежа в Python?"
ANSWER = {"answer": "Список изменяемый, кортеж — нет."}


class ExperimentTests(unittest.TestCase):
    def test_different_questions_work_with_the_same_schema(self):
        for question in (QUESTION, 'Назови столицу Франции.', 'Объясни код:\nprint("Привет")'):
            for mode in main.MODES:
                with self.subTest(question=question, mode=mode):
                    payload = main.build_payload(mode, "model", question)
                    prompt = payload["messages"][0]["content"]
                    self.assertTrue(prompt.startswith(question))
                    if mode in ("baseline", "limited", "schema"):
                        self.assertEqual(prompt, question)
                    else:
                        self.assertIn("answer", prompt)

    def test_answer_classification(self):
        valid = json.dumps(ANSWER)
        cases = [
            (valid, (True, True)),
            ('Вот JSON: ' + valid, (False, False)),
            ('```json\n' + valid + '\n```', (False, False)),
            ('{"answer":"Список', (False, False)),
            ('{}', (True, False)),
            ('{"answer": 28}', (True, False)),
            ('{"answer": ["список"]}', (True, False)),
            (json.dumps(dict(ANSWER, extra=1)), (True, False)),
            ('{"answer":NaN}', (False, False)),
        ]
        for answer, expected in cases:
            with self.subTest(answer=answer):
                checks = main.check_answer(answer)
                self.assertEqual(
                    tuple(checks[key] for key in ("json_ok", "schema_ok")), expected,
                )

    def response(self, data, status=200):
        response = Mock(status_code=status, ok=status < 400)
        response.json.return_value = data
        response.text = json.dumps(data, ensure_ascii=False)
        return response

    @patch("main.requests.post")
    def test_full_run_has_four_independent_requests(self, post):
        post.return_value = self.response({
            "choices": [{"message": {"content": json.dumps(ANSWER)},
                         "finish_reason": "stop"}],
        })
        with contextlib.redirect_stdout(io.StringIO()):
            records = main.run_experiment("https://example.invalid", "secret", "model", QUESTION)
        self.assertEqual(len(records), 4)
        for mode in main.MODES:
            self.assertEqual(sum(record["mode"] == mode for record in records), 1)
        self.assertEqual(post.call_count, 4)
        self.assertTrue(all(record["checks"]["schema_ok"] for record in records))
        payloads = [call.kwargs["json"] for call in post.call_args_list]
        self.assertTrue(all(len(payload["messages"]) == 1 for payload in payloads))
        self.assertTrue(all(payload["messages"][0]["content"].startswith(QUESTION) for payload in payloads))
        baseline = payloads[1]
        self.assertNotIn("max_tokens", baseline)
        self.assertNotIn("stop", baseline)
        self.assertNotIn("response_format", baseline)
        self.assertEqual(payloads[2]["max_tokens"], 50)
        self.assertEqual(payloads[2]["messages"][0]["content"], QUESTION)
        self.assertNotIn("stop", payloads[2])
        self.assertNotIn("response_format", payloads[2])
        self.assertEqual(payloads[3]["stop"], [main.STOP])
        self.assertNotIn("max_tokens", payloads[3])
        self.assertNotIn("response_format", payloads[3])
        self.assertEqual(payloads[0]["messages"][0]["content"], QUESTION)
        self.assertNotIn("max_tokens", payloads[0])
        self.assertNotIn("stop", payloads[0])
        self.assertEqual(payloads[0]["response_format"]["json_schema"]["schema"], main.SCHEMA)


    @patch("main.requests.post")
    def test_schema_rejection_stops_after_one_request(self, post):
        post.return_value = self.response({"error": {"message": "unsupported schema"}}, 400)
        with contextlib.redirect_stdout(io.StringIO()):
            records = main.run_experiment("https://example.invalid", "secret", "model", QUESTION)
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]["api_ok"])
        self.assertNotIn("checks", records[0])
        self.assertEqual(post.call_count, 1)

    @patch("main.requests.post", side_effect=main.requests.ConnectionError)
    def test_network_failure_stops_after_one_attempt(self, post):
        with contextlib.redirect_stdout(io.StringIO()):
            records = main.run_experiment("https://example.invalid", "secret", "model", QUESTION)
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["network_error"])
        self.assertEqual(post.call_count, 1)

    @patch("main.requests.post")
    def test_invalid_model_json_is_a_result_not_api_failure(self, post):
        post.return_value = self.response({
            "choices": [{"message": {"content": '{"answer":'}, "finish_reason": "length"}],
        })
        record = main.request_once("https://example.invalid", "secret", {})
        self.assertTrue(record["api_ok"])
        self.assertFalse(record["checks"]["json_ok"])
        self.assertEqual(record["finish_reason"], "length")

    def test_saved_results_hide_credentials_and_remain_json(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(main, "DIRECTORY", Path(directory)):
                path, _ = main.save_results([
                    {"mode": "schema", "repeat": 1, "api_ok": False,
                     "raw_response": "token-secret account-secret"},
                ], "model", "token-secret", "account-secret", QUESTION)
            text = path.with_suffix(".json").read_text()
            self.assertNotIn("token-secret", text)
            self.assertNotIn("account-secret", text)
            self.assertEqual(json.loads(text)["records"][0]["raw_response"], "[HIDDEN] [HIDDEN]")
            self.assertEqual(json.loads(text)["task"], QUESTION)
            self.assertIn(QUESTION, path.read_text())

    def test_report_contains_actual_conditions_prompt_and_unmodified_answer(self):
        payload = main.build_payload("formatted", "model", QUESTION)
        answer = '{"answer":"Список'
        report = main.make_report([{
            "mode": "formatted", "repeat": 1, "api_ok": True,
            "request": payload, "answer": answer, "checks": main.check_answer(answer),
            "finish_reason": "length", "stop_reason": None,
            "usage": {"completion_tokens": 30},
        }], QUESTION)
        self.assertIn('Явный лимит токенов не задан', report)
        self.assertIn(payload["messages"][0]["content"], report)
        self.assertIn("````text\n" + answer + "\n````", report)
        self.assertLess(report.index("**Запрос:**"), report.index("**Условия:**"))
        self.assertLess(report.index("**Условия:**"), report.index("**Вывод LLM:**"))
        self.assertNotIn("Что произошло", report)
        self.assertNotIn("Что это означает", report)
        self.assertNotIn("Проверка схемы", report)
        self.assertNotIn("## Сравнение", report)
        self.assertNotIn("Границы вывода", report)

    def test_report_shows_each_of_the_four_answers_without_analysis(self):
        records = [
            {"mode": mode, "repeat": index // 4 + 1, "api_ok": True,
             "request": main.build_payload(mode, "model", QUESTION),
             "answer": f"Ответ номер {index}"}
            for index, mode in enumerate(main.SCHEDULE)
        ]
        report = main.make_report(records, QUESTION)
        self.assertEqual(report.count("**Запрос:**"), 4)
        self.assertEqual(report.count("**Условия:**"), 4)
        self.assertEqual(report.count("**Вывод LLM:**"), 4)
        for record in records:
            self.assertIn(record["answer"], report)
        self.assertNotIn("JSON читается", report)


if __name__ == "__main__":
    unittest.main()

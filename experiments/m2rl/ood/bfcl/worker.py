"""One BFCL v3 case with the official handler and checker; generation is delegated over stdin/stdout.

Started by run.py inside the BFCL environment (gorilla commit cd9429c, `berkeley-function-call-leaderboard`
installed). Protocol, one JSON object per line:
  stdin  {"question_id", "category"}                       first line
  stdout {"type": "generate", "step", "prompt"}            the handler's formatted prompt
  stdin  {"type": "response", "step", "text", "input_tokens", "output_tokens"}
  stdout {"type": "complete", "question_id", "category", "score", ...}
Generated tool calls are executed by BFCL's own multi-turn environment; run inside a sandbox.
"""
import contextlib
import json
import os
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from types import MethodType, SimpleNamespace

SOURCE = Path(os.environ["BFCL_ROOT"])  # .../gorilla/berkeley-function-call-leaderboard at cd9429c
os.environ.setdefault("BFCL_PROJECT_ROOT", tempfile.mkdtemp(prefix="bfcl-"))
sys.path.insert(0, str(SOURCE))


def emit(value):
    sys.__stdout__.write(json.dumps(value, ensure_ascii=False) + "\n")
    sys.__stdout__.flush()


def main():
    request = json.loads(sys.stdin.readline())
    with contextlib.redirect_stdout(sys.stderr):
        from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
        from bfcl_eval._llm_response_generation import process_multi_turn_test_case
        from bfcl_eval.eval_checker.eval_runner import ast_file_runner, relevance_file_runner, multi_turn_runner
        from bfcl_eval.constants.category_mapping import TEST_FILE_MAPPING
        from bfcl_eval.utils import load_file, make_json_serializable
        category, qid = request["category"], request["question_id"]
        entry = next(r for r in load_file(SOURCE / "bfcl_eval/data" / TEST_FILE_MAPPING[category]) if r["id"] == qid)
        gold_path = SOURCE / "bfcl_eval/data/possible_answer" / TEST_FILE_MAPPING[category]
        gold = next((r for r in load_file(gold_path) if r["id"] == qid), None) if gold_path.exists() else None
        handler = QwenFCHandler("Qwen/Qwen3-4B-FC", temperature=0.6)
        step = 0

        def query(self, inference_data):
            nonlocal step
            prompt = self._format_prompt(inference_data["message"], inference_data["function"])
            inference_data["inference_input_log"] = {"formatted_prompt": prompt}
            emit(dict(type="generate", question_id=qid, step=step, prompt=prompt))
            started = time.monotonic()
            response = json.loads(sys.stdin.readline())
            if response.get("type") != "response" or response["step"] != step:
                raise ValueError("RPC step mismatch")
            step += 1
            usage = SimpleNamespace(prompt_tokens=response["input_tokens"], completion_tokens=response["output_tokens"])
            return SimpleNamespace(choices=[SimpleNamespace(text=response["text"])], usage=usage), time.monotonic() - started

        handler._query_prompting = MethodType(query, handler)
        prepared = process_multi_turn_test_case([deepcopy(entry)])[0]
        record = handler._multi_threaded_inference(prepared, True, False)
        if "traceback" in record:
            trace = record["traceback"]
            # A malformed tool-call envelope makes the official Qwen handler raise; the case scores 0.
            known = ("TypeError: string indices must be integers, not 'str'" in trace
                     and any(k in trace for k in ('tool_call["name"]', 'tool_call["function"]'))) \
                or ("KeyError: 'arguments'" in trace and 'tool_call["arguments"]' in trace)
            if not known or "qwen_fc.py" not in trace:
                raise RuntimeError("unexpected handler failure: " + trace)
        score_dir = Path(tempfile.mkdtemp(prefix="bfcl-score-"))
        if category.startswith("multi_turn"):
            accuracy, count = multi_turn_runner(handler, [deepcopy(record)], [deepcopy(entry)], [gold], handler.model_name, category, score_dir)
        elif category in ("irrelevance", "live_irrelevance", "live_relevance"):
            accuracy, count = relevance_file_runner(handler, [deepcopy(record)], [deepcopy(entry)], handler.model_name, category, score_dir)
        else:
            language = {"java": "Java", "javascript": "JavaScript"}.get(category, "Python")
            accuracy, count = ast_file_runner(handler, [deepcopy(record)], [deepcopy(entry)], [gold], language, category, handler.model_name, score_dir)
        if count != 1 or accuracy not in (0, 1):
            raise ValueError("invalid per-case score")
        emit(dict(type="complete", question_id=qid, category=category, score=accuracy, model_queries=step,
                  handler_error="traceback" in record, record=make_json_serializable(record)))


if __name__ == "__main__":
    main()

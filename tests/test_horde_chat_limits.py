"""Capture embedded Horde worker declarations without loading a model or running generation."""

import argparse
import os
import sys
import threading
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import koboldcpp


class HordeChatLimitsTests(unittest.TestCase):
    def capture_worker(self, engine_ctx, horde_ctx, horde_gen, job=None):
        options = argparse.Namespace(hordeurl="https://horde.example", hordechat=True, jinja=False, defaultgenamt=64)
        declarations = []
        submissions = []
        pending = [job] if job else []
        submitted = threading.Event()

        class InlineThread:
            def __init__(self, target, args):
                self.target = target
                self.args = args

            def start(self):
                self.target(*self.args)

        clock = SimpleNamespace(now=mock.Mock(side_effect=[datetime(2026, 1, 1), datetime(2026, 1, 1) + timedelta(seconds=1)]))

        def request(url, data, method="POST", headers=None, timeout=600, error_info=None):
            if url.endswith("/api/v1/info/version"):
                return {"result": "ok"}
            if url.endswith("/api/v2/generate/text/pop"):
                declarations.append(data.copy())
                if pending:
                    return pending.pop()
                koboldcpp.exitcounter = 10
                return {"id": None}
            if url.endswith("/api/v2/generate/text/submit"):
                submissions.append(data.copy())
                submitted.set()
                return {"reward": 0}
            raise AssertionError("An oversized Horde chat job must not reach local generation: " + url)

        with (
            mock.patch.multiple(
                koboldcpp,
                args=options,
                maxctx=engine_ctx,
                maxhordectx=horde_ctx,
                maxhordelen=horde_gen,
                friendlymodelname="koboldcpp/example",
                exitcounter=0,
                punishcounter=0,
                last_non_horde_req_time=0,
                has_vision_support=False,
                modelbusy=threading.Lock(),
                session_jobs=0,
                session_kudos_earned=0,
                rewardcounter=0,
            ),
            mock.patch.object(koboldcpp, "get_my_epurl", return_value="http://localhost:5001"),
            mock.patch.object(koboldcpp, "make_url_request", side_effect=request),
            mock.patch.object(koboldcpp.time, "sleep"),
            mock.patch.object(koboldcpp.threading, "Thread", InlineThread),
            mock.patch.object(koboldcpp, "datetime", clock),
            mock.patch.object(koboldcpp, "print_with_time"),
            mock.patch.object(koboldcpp, "transform_genparams", return_value={"prompt": "short"}),
            mock.patch.object(koboldcpp, "tokenize_ids", return_value=[1, 2]),
        ):
            with self.assertRaises(SystemExit):
                koboldcpp.run_horde_worker(options, "key", "test worker", 0, 1)
            if job:
                self.assertTrue(submitted.wait(2), "The oversized job was not faulted")
        return declarations[0], submissions

    def test_advertisement_respects_operator_context_and_generation_limits(self):
        declaration, _ = self.capture_worker(32768, 4096, 256)
        self.assertEqual(declaration["chat_limits"], {"max_context_length": 4096, "max_length": 256})
        self.assertEqual(declaration["max_context_length"], 4096)
        self.assertEqual(declaration["max_length"], 256)

    def test_advertisement_caps_large_operator_limits_at_the_engine_limits(self):
        for horde_ctx in (0, 8192):
            with self.subTest(horde_ctx=horde_ctx):
                declaration, _ = self.capture_worker(4096, horde_ctx, 10000)
                self.assertEqual(declaration["chat_limits"], {"max_context_length": 4096, "max_length": 3277})

    def test_chat_jobs_exceeding_either_declared_limit_fault_before_generation(self):
        for context, generation in ((8192, 64), (4096, 257)):
            with self.subTest(context=context, generation=generation):
                job = {
                    "id": "oversized",
                    "chat": True,
                    "payload": {
                        "messages": [{"role": "user", "content": "short"}],
                        "max_context_length": context,
                        "max_length": generation,
                    },
                }
                _, submissions = self.capture_worker(32768, 4096, 256, job=job)
                self.assertEqual(submissions, [{"id": "oversized", "state": "faulted", "model_id": "koboldcpp/example", "fault_reason": "prompt_overflow"}])


if __name__ == "__main__":
    unittest.main()

import io
import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import koboldcpp


def post_stats(body_obj):
    """POST the body to /api/extra/generate/stats and return the decoded response."""
    body = json.dumps(body_obj).encode()
    server = object.__new__(koboldcpp.KcppServerRequestHandler)
    server.headers = {"content-length": str(len(body))}
    server.rfile = io.BytesIO(body)
    server.wfile = io.BytesIO()
    server.path = "/api/extra/generate/stats"
    server.secure_endpoint = Mock(return_value=True)
    server.send_response = Mock()
    server.send_header = Mock()
    server.end_headers = Mock()
    with patch.object(koboldcpp, "args", SimpleNamespace(maxrequestsize=32)):
        server.do_POST()
    server.send_response.assert_called_with(200)
    return json.loads(server.wfile.getvalue())


def stats_struct(**overrides):
    """A stats struct as the library returns it, with the given fields changed."""
    values = dict(
        status=1,
        state=2,
        slot=1,
        queue_position=-1,
        prompt_tokens=23,
        completion_tokens=7,
        max_length=10,
        init_seconds=0.1,
        process_seconds=0.2,
        generation_seconds=0.3,
        elapsed_seconds=0.6,
        finish_reason=-1,
        finished=0,
    )
    values.update(overrides)
    return koboldcpp.generation_stats_outputs(**values)


class GenerationStatsEndpointTests(unittest.TestCase):
    """The endpoint forwards the library's struct for the request the key belongs to, and nothing else."""

    def setUp(self):
        self.backend = Mock()
        self.backend.batch_generate_stats.return_value = stats_struct()
        self.backend.generate_stats.return_value = stats_struct(slot=-1)
        self.backend.get_generation_serial.return_value = 5
        for target, value in (
            ("handle", self.backend),
            ("batch_request_ids_by_genkey", {}),
            ("currentusergenkey", ""),
            ("currentusergenkey_batched", False),
            ("currentusergenkey_serial", 4),
            ("totalgens", 0),
            ("requestsinqueue", 0),
        ):
            patcher = patch.object(koboldcpp, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_invalid_or_unknown_keys_are_not_found(self):
        for key in ([1], {"x": 1}, [], {}, None, 7, "", "missing"):
            with self.subTest(key=key):
                self.assertEqual(post_stats({"genkey": key}), {"found": False})
        self.backend.batch_generate_stats.assert_not_called()
        self.backend.generate_stats.assert_not_called()

    def test_batched_request_is_read_by_its_own_id(self):
        """With two batched requests active, polling one key reads only that request's id."""
        koboldcpp.batch_request_ids_by_genkey["a"] = 42
        koboldcpp.batch_request_ids_by_genkey["b"] = 43
        stats = post_stats({"genkey": "b"})
        self.backend.batch_generate_stats.assert_called_once_with(43)
        self.assertTrue(stats["found"])
        self.assertTrue(stats["batched"])
        self.assertFalse(stats["finished"])
        self.assertEqual(stats["completion_tokens"], 7)
        self.assertEqual(stats["slot"], 1)
        self.assertAlmostEqual(stats["elapsed_seconds"], 0.6, places=5)
        self.assertNotIn("status", stats)

    def test_released_batched_request_is_not_found(self):
        """Once the library no longer knows the request, a poll reports it as not found."""
        koboldcpp.batch_request_ids_by_genkey["a"] = 42
        self.backend.batch_generate_stats.return_value = koboldcpp.generation_stats_outputs()
        self.assertEqual(post_stats({"genkey": "a"}), {"found": False})

    def test_non_batched_request_follows_the_check_key_rule(self):
        """Without batching, only the current key may read the counters, as with check."""
        koboldcpp.currentusergenkey = "live"
        koboldcpp.totalgens = 1
        self.assertEqual(post_stats({"genkey": "other"}), {"found": False})
        stats = post_stats({"genkey": "live"})
        self.backend.generate_stats.assert_called_once_with(4)
        self.assertTrue(stats["found"])
        self.assertFalse(stats["batched"])
        self.assertEqual(stats["slot"], -1)
        self.assertEqual(stats["completion_tokens"], 7)

    def test_serial_seen_at_submit_is_passed_to_the_library(self):
        """The library decides whether the counters belong to this request yet, from the serial the handler saw at submit."""
        koboldcpp.currentusergenkey = "live"
        koboldcpp.currentusergenkey_serial = 9
        koboldcpp.totalgens = 2
        self.backend.generate_stats.return_value = stats_struct(state=0, slot=-1, prompt_tokens=0, completion_tokens=0, elapsed_seconds=0.0)
        stats = post_stats({"genkey": "live"})
        self.backend.generate_stats.assert_called_once_with(9)
        self.assertTrue(stats["found"])
        self.assertFalse(stats["finished"])
        self.assertEqual(stats["state"], 0)
        self.assertEqual(stats["completion_tokens"], 0)

    def test_finished_flag_is_forwarded_as_a_boolean(self):
        koboldcpp.currentusergenkey = "live"
        koboldcpp.totalgens = 1
        self.backend.generate_stats.return_value = stats_struct(slot=-1, state=3, finish_reason=1, finished=1)
        stats = post_stats({"genkey": "live"})
        self.assertIs(stats["finished"], True)
        self.assertEqual(stats["state"], 3)

    def test_keyless_poll_matches_check(self):
        """As with check, a client with no genkey may read only a keyless generation with an empty queue."""
        koboldcpp.totalgens = 1
        self.assertTrue(post_stats({"genkey": ""})["found"])
        koboldcpp.requestsinqueue = 1
        self.assertEqual(post_stats({"genkey": ""}), {"found": False})
        koboldcpp.requestsinqueue = 0
        koboldcpp.currentusergenkey = "live"
        self.assertEqual(post_stats({"genkey": ""}), {"found": False})

    def test_released_batched_key_does_not_read_the_shared_counters(self):
        """A batched request's key stays the current key after release; it must not fall through to the non-batched counters."""
        koboldcpp.currentusergenkey = "b"
        koboldcpp.currentusergenkey_batched = True
        koboldcpp.totalgens = 2
        self.assertEqual(post_stats({"genkey": "b"}), {"found": False})
        self.backend.generate_stats.assert_not_called()

    def test_nothing_is_readable_before_the_first_generation(self):
        koboldcpp.currentusergenkey = "live"
        koboldcpp.totalgens = 0
        self.assertEqual(post_stats({"genkey": "live"}), {"found": False})


if __name__ == "__main__":
    unittest.main()

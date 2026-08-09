from __future__ import annotations

import json
import unittest
from unittest import mock

from helicopter_cli import eval_run


class EndpointModelValidationTests(unittest.TestCase):
    def test_accepts_the_model_advertised_by_the_endpoint(self) -> None:
        response = mock.Mock()
        response.read.return_value = json.dumps(
            {"data": [{"id": "rwkv7-g1h-2.9b-20260710-ctx10240"}]}
        ).encode()
        response.__enter__ = mock.Mock(return_value=response)
        response.__exit__ = mock.Mock(return_value=False)

        with mock.patch.object(eval_run, "urlopen", return_value=response):
            self.assertEqual(
                eval_run.validate_endpoint_model(
                    "http://127.0.0.1:19329/v1",
                    "rwkv7-g1h-2.9b-20260710-ctx10240",
                ),
                ("rwkv7-g1h-2.9b-20260710-ctx10240",),
            )

    def test_rejects_a_2p9b_task_sent_to_a_1p5b_endpoint(self) -> None:
        response = mock.Mock()
        response.read.return_value = json.dumps(
            {"data": [{"id": "rwkv7-g1h-1.5b-20260710-ctx10240"}]}
        ).encode()
        response.__enter__ = mock.Mock(return_value=response)
        response.__exit__ = mock.Mock(return_value=False)

        with mock.patch.object(eval_run, "urlopen", return_value=response):
            with self.assertRaisesRegex(SystemExit, "model endpoint mismatch"):
                eval_run.validate_endpoint_model(
                    "http://127.0.0.1:19316/v1",
                    "rwkv7-g1h-2.9b-20260710-ctx10240",
                )

    def test_explicit_escape_hatch_keeps_proxy_compatibility(self) -> None:
        response = mock.Mock()
        response.read.return_value = json.dumps(
            {"data": [{"id": "proxy-model"}]}
        ).encode()
        response.__enter__ = mock.Mock(return_value=response)
        response.__exit__ = mock.Mock(return_value=False)

        with mock.patch.object(eval_run, "urlopen", return_value=response):
            self.assertEqual(
                eval_run.validate_endpoint_model(
                    "http://127.0.0.1:19316/v1",
                    "rwkv7-g1h-2.9b-20260710-ctx10240",
                    allow_mismatch=True,
                ),
                ("proxy-model",),
            )


if __name__ == "__main__":
    unittest.main()

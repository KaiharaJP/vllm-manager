"""LiteLLM リクエスト追跡の公開行サニタイズテスト。"""

import unittest

from app.litellm_request_track import _extract_request_meta, _public_row


class LiteLLMRequestTrackSecurityTests(unittest.TestCase):
    def test_request_summary_excludes_prompt_preview(self):
        meta = _extract_request_meta(
            {
                "model": "vllm-local",
                "messages": [{"role": "user", "content": "これは秘密のプロンプトです"}],
                "max_tokens": 64,
            }
        )
        self.assertIn("1 msg", meta["request_summary"])
        self.assertNotIn("秘密", meta["request_summary"])
        self.assertEqual(meta["_preview"], "これは秘密のプロンプトです")

    def test_public_row_strips_private_fields(self):
        row = {
            "id": "track-1",
            "request_summary": "1 msg, ~10 chars",
            "_preview": "secret",
            "_messages": [{"role": "user", "content": "secret"}],
        }
        public = _public_row(row)
        self.assertNotIn("_preview", public)
        self.assertNotIn("_messages", public)
        self.assertNotIn("secret", public["request_summary"])


if __name__ == "__main__":
    unittest.main()

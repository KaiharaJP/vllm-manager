"""監査ログ永続化のユニットテスト。"""

import importlib
import json
import os
import shutil
import tempfile
import unittest

_TEST_DATA_ENV = "VLLM_MANAGER_DATA_DIR"


def _reload_audit():
    import app.audit_log as audit_log

    importlib.reload(audit_log)
    return audit_log


class AuditLogTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="vllm-audit-test-")
        self._prev = os.environ.get(_TEST_DATA_ENV)
        os.environ[_TEST_DATA_ENV] = self._tmpdir
        self.audit = _reload_audit()

    def tearDown(self):
        if self._prev is None:
            os.environ.pop(_TEST_DATA_ENV, None)
        else:
            os.environ[_TEST_DATA_ENV] = self._prev
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_append_and_read(self):
        self.audit.append_audit(action="user_updated", actor="admin", message="created user")
        entries = self.audit.read_recent(limit=10)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["action"], "user_updated")
        self.assertEqual(entries[0]["actor"], "admin")

    def test_should_audit_event_skips_metrics(self):
        self.assertFalse(self.audit.should_audit_event("metrics"))
        self.assertTrue(self.audit.should_audit_event("server_job"))

    def test_audit_log_file_is_jsonl(self):
        self.audit.append_audit(action="server_job", actor="admin")
        lines = self.audit.AUDIT_LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        payload = json.loads(lines[0])
        self.assertEqual(payload["action"], "server_job")

    def test_rotation_moves_old_log_and_enforces_backup_count(self):
        # 1エントリ書き込むだけで次回ローテーションが起きるくらい小さい上限にする
        self.audit.MAX_AUDIT_LOG_BYTES = 1
        self.audit.AUDIT_LOG_BACKUP_COUNT = 2

        for i in range(4):
            self.audit.append_audit(action="server_job", actor="admin", message=f"entry-{i}")

        self.assertTrue(self.audit.AUDIT_LOG_FILE.exists())
        backup_1 = self.audit.DATA_DIR / "audit.log.1"
        backup_2 = self.audit.DATA_DIR / "audit.log.2"
        self.assertTrue(backup_1.exists())
        self.assertTrue(backup_2.exists())
        # 保持世代数（2）を超えた古いバックアップは残らない
        overflow = self.audit.DATA_DIR / "audit.log.3"
        self.assertFalse(overflow.exists())
        # 最新エントリは現行ファイルに残る
        current_payload = json.loads(
            self.audit.AUDIT_LOG_FILE.read_text(encoding="utf-8").strip().splitlines()[-1]
        )
        self.assertEqual(current_payload["message"], "entry-3")

    def test_read_recent_falls_back_to_backup_after_rotation(self):
        self.audit.MAX_AUDIT_LOG_BYTES = 1
        self.audit.AUDIT_LOG_BACKUP_COUNT = 2

        for i in range(3):
            self.audit.append_audit(action="server_job", actor="admin", message=f"entry-{i}")

        # 各エントリごとにローテーションするため、現行ファイルには最新1件しかない。
        # read_recent は直近のバックアップ世代からも補完して返す。
        entries = self.audit.read_recent(limit=2)
        messages = [e["message"] for e in entries]
        self.assertEqual(messages, ["entry-1", "entry-2"])


if __name__ == "__main__":
    unittest.main()

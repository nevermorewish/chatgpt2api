from __future__ import annotations

import unittest
from unittest.mock import patch

from services import sub2api_service


class FakeResponse:
    def __init__(self, payload: object, ok: bool = True, status_code: int = 200, text: str = "") -> None:
        self._payload = payload
        self.ok = ok
        self.status_code = status_code
        self.text = text

    def json(self) -> object:
        return self._payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, **kwargs) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)

    def close(self) -> None:
        pass


class Sub2APIServiceTests(unittest.TestCase):
    def test_list_remote_accounts_accepts_redacted_access_token_status(self) -> None:
        session = FakeSession([
            FakeResponse({
                "code": 0,
                "data": {
                    "items": [
                        {
                            "id": 123,
                            "name": "pro account",
                            "platform": "openai",
                            "type": "oauth",
                            "credentials": {"email": "user@example.com", "plan_type": "pro"},
                            "credentials_status": {"has_access_token": True},
                            "status": "active",
                        }
                    ],
                    "total": 1,
                },
            })
        ])
        with patch.object(sub2api_service, "_auth_headers", return_value={"Authorization": "Bearer token"}), \
             patch.object(sub2api_service, "Session", return_value=session):
            accounts = sub2api_service.list_remote_accounts({"base_url": "http://sub2api"})

        self.assertEqual(len(accounts), 1)
        self.assertEqual(accounts[0]["id"], "123")
        self.assertEqual(accounts[0]["plan_type"], "pro")

    def test_list_remote_groups_uses_all_endpoint_first(self) -> None:
        session = FakeSession([
            FakeResponse({
                "code": 0,
                "data": [
                    {"id": 7, "name": "OpenAI Pro", "platform": "openai", "status": "active"}
                ],
            })
        ])
        with patch.object(sub2api_service, "_auth_headers", return_value={"Authorization": "Bearer token"}), \
             patch.object(sub2api_service, "Session", return_value=session):
            groups = sub2api_service.list_remote_groups({"base_url": "http://sub2api"})

        self.assertEqual(groups[0]["id"], "7")
        self.assertTrue(str(session.calls[0]["url"]).endswith("/api/v1/admin/groups/all"))

    def test_fetch_exported_openai_accounts_reads_raw_credentials(self) -> None:
        session = FakeSession([
            FakeResponse({
                "type": "sub2api-data",
                "accounts": [
                    {
                        "name": "pro account",
                        "platform": "openai",
                        "type": "oauth",
                        "credentials": {"access_token": "raw-token", "plan_type": "pro"},
                    },
                    {
                        "name": "claude account",
                        "platform": "anthropic",
                        "type": "oauth",
                        "credentials": {"access_token": "ignored"},
                    },
                ],
            })
        ])
        with patch.object(sub2api_service, "_auth_headers", return_value={"Authorization": "Bearer token"}), \
             patch.object(sub2api_service, "Session", return_value=session):
            accounts = sub2api_service._fetch_exported_openai_accounts({"base_url": "http://sub2api"}, ["42"])

        self.assertEqual(len(accounts), 1)
        self.assertEqual(accounts[0]["credentials"]["access_token"], "raw-token")
        self.assertIn("ids", session.calls[0]["params"])


if __name__ == "__main__":
    unittest.main()

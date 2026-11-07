import pytest
from unittest.mock import patch, MagicMock

from app.services.ai_integration import (
    ask_ai, _ask_openai, _ask_xai, _ask_anthropic,
    check_ai_providers, get_ai_cost_estimate, _check_api_key
)


@pytest.fixture
def mock_env_keys(monkeypatch):
    monkeypatch.setattr("app.services.ai_integration.OPENAI_API_KEY", "sk-test-openai-key")
    monkeypatch.setattr("app.services.ai_integration.XAI_API_KEY", "xai-test-key")
    monkeypatch.setattr("app.services.ai_integration.ANTHROPIC_API_KEY", "sk-ant-test-key")


class TestAPIKeyValidation:
    def test_check_api_key_valid(self):
        assert _check_api_key("sk-test-openai-key", "openai") is True
        assert _check_api_key("xai-test-key", "xai") is True

    def test_check_api_key_invalid(self):
        assert _check_api_key("sk-your-openai-key-here", "openai") is False
        assert _check_api_key("YOUR_XAI_API_KEY", "xai") is False

    def test_check_api_key_missing(self):
        assert _check_api_key("", "openai") is False
        assert _check_api_key(None, "openai") is False


class TestCostEstimate:
    def test_get_ai_cost_estimate_openai(self):
        query = "Test query with some length to calculate tokens."
        res = get_ai_cost_estimate(query, "openai")
        assert "estimated_tokens" in res
        assert "estimated_cost_usd" in res
        assert "estimated_cost_rub" in res

    def test_get_ai_cost_estimate_xai(self):
        res = get_ai_cost_estimate("Test", "xai")
        assert "estimated_cost_usd" in res

    def test_get_ai_cost_estimate_anthropic(self):
        res = get_ai_cost_estimate("Test", "anthropic")
        assert "estimated_cost_usd" in res


class TestCheckProviders:
    @patch("app.services.ai_integration.ask_ai")
    def test_check_ai_providers_all_success(self, mock_ask_ai):
        mock_ask_ai.return_value = "OK"

        res = check_ai_providers()
        assert res["openai"]["available"] is True
        assert res["xai"]["available"] is True
        assert res["anthropic"]["available"] is True

    @patch("app.services.ai_integration.ask_ai")
    def test_check_ai_providers_mixed(self, mock_ask_ai):
        def side_effect(query, provider):
            if provider == "xai":
                raise Exception("API Error")
            return "OK"
        mock_ask_ai.side_effect = side_effect

        res = check_ai_providers()
        assert res["openai"]["available"] is True
        assert res["xai"]["available"] is False
        assert res["anthropic"]["available"] is True


class TestAskAI:
    @patch("app.services.ai_integration._ask_openai")
    def test_ask_ai_primary_success(self, mock_oai, mock_env_keys):
        mock_oai.return_value = "OpenAI response"
        
        res = ask_ai("Test", provider="openai")
        assert res == "OpenAI response"
        mock_oai.assert_called_once()

    @patch("app.services.ai_integration._ask_openai")
    @patch("app.services.ai_integration._ask_xai")
    @patch("app.services.ai_integration._ask_anthropic")
    def test_ask_ai_fallback(self, mock_ant, mock_xai, mock_oai, mock_env_keys):
        mock_oai.side_effect = Exception("OpenAI down")
        mock_xai.return_value = "xAI response"
        
        res = ask_ai("Test", provider="openai")
        assert res == "xAI response"
        mock_oai.assert_called_once()
        mock_xai.assert_called_once()

    @patch("app.services.ai_integration._ask_openai")
    @patch("app.services.ai_integration._ask_xai")
    @patch("app.services.ai_integration._ask_anthropic")
    def test_ask_ai_all_fail(self, mock_ant, mock_xai, mock_oai, mock_env_keys):
        mock_oai.side_effect = Exception("OpenAI down")
        mock_xai.side_effect = Exception("xAI down")
        mock_ant.side_effect = Exception("Anthropic down")
        
        with pytest.raises(Exception, match="Все ИИ провайдеры недоступны"):
            ask_ai("Test", provider="openai")


class TestProviderImplementations:
    @patch("requests.post")
    def test_ask_openai_success(self, mock_post, mock_env_keys):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "GPT response"}}]
        }
        mock_post.return_value = mock_response

        res = _ask_openai("Hello")
        assert res == "GPT response"
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_ask_xai_success(self, mock_post, mock_env_keys):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Grok response"}}]
        }
        mock_post.return_value = mock_response

        res = _ask_xai("Hello")
        assert res == "Grok response"
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_ask_anthropic_success(self, mock_post, mock_env_keys):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"text": "Claude response"}]
        }
        mock_post.return_value = mock_response

        res = _ask_anthropic("Hello")
        assert res == "Claude response"
        mock_post.assert_called_once()


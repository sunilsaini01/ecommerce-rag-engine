from unittest.mock import MagicMock

import httpx
import pytest
from groq import APIConnectionError

from src.backend.services.llm import LLMService, GENERATION_FAILED_MESSAGE


def _mock_groq_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return response


@pytest.mark.asyncio
class TestLLMService:
    async def test_successful_generation(self):
        service = LLMService()
        service.client.chat.completions.create = MagicMock(
            return_value=_mock_groq_response("The Widget costs $9.99.")
        )

        answer, ok = await service.generate_answer("widget price", "- Title: Widget | Price: $9.99")

        assert ok is True
        assert answer == "The Widget costs $9.99."

    async def test_generation_failure_after_retries_returns_safe_message(self):
        service = LLMService()
        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        service.client.chat.completions.create = MagicMock(
            side_effect=APIConnectionError(message="connection failed", request=request)
        )

        answer, ok = await service.generate_answer("widget price", "- Title: Widget | Price: $9.99")

        assert ok is False
        assert answer == GENERATION_FAILED_MESSAGE
        # tenacity should have retried up to 3 attempts total
        assert service.client.chat.completions.create.call_count == 3

    async def test_no_result_handling_upstream_uses_no_match_message(self):
        from src.backend.services.llm import NO_MATCH_MESSAGE
        assert "cannot find a product matching" in NO_MATCH_MESSAGE.lower()

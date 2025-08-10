"""
OpenRouter client with model fallback and retry/backoff for LLM-enhanced reporting
"""

import os
import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class OpenRouterClient:
    def __init__(self,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 models: Optional[List[str]] = None,
                 timeout_seconds: float = 25.0,
                  max_retries: int = 3,
                  max_tokens: int = 512):
        self.api_key = api_key or settings.OPEN_ROUTER_KEY or os.getenv("OPEN_ROUTER_KEY", "")
        self.base_url = base_url or settings.OPENROUTER_BASE_URL
        self.models = models or settings.OPENROUTER_MODELS
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_tokens = max_tokens

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Optional OpenRouter headers for usage attribution
            "HTTP-Referer": os.getenv("OPENROUTER_REFERRER", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_TITLE", "Thermal Eye Reports"),
        }

    @staticmethod
    def _redact(text: str) -> str:
        """Redact potential PII (emails/phone numbers) from prompts."""
        try:
            import re
            text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "<redacted_email>", text)
            text = re.sub(r"\b(?:\+?\d[\s-]?)?(?:\(?\d{3}\)?[\s-]?)?\d{3}[\s-]?\d{4}\b", "<redacted_phone>", text)
        except Exception:
            pass
        return text

    @staticmethod
    def _strip_code_fence(s: str) -> str:
        s = s.strip()
        if s.startswith("```") and s.endswith("```"):
            s = s.strip("`\n ")
            if s.lower().startswith("json"):
                s = s[4:].lstrip()
        return s

    def _post(self, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        start = time.perf_counter()
        with httpx.Client(base_url=self.base_url, timeout=self.timeout_seconds) as client:
            resp = client.post("/chat/completions", headers=self._headers(), data=json.dumps(payload))
            duration = time.perf_counter() - start
            resp.raise_for_status()
            return resp.json(), duration

    def generate_json(self, prompt: str, system: str = "You are an expert thermal inspection analyst.",
                      schema_hint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        last_error = None
        for model in self.models:
            for attempt in range(1, self.max_retries + 1):
                try:
                    payload = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": self._redact(prompt)},
                        ],
                        "temperature": 0.2,
                        "max_tokens": self.max_tokens,
                    }
                    data, duration = self._post(payload)
                    content = data["choices"][0]["message"]["content"]
                    content_str = self._strip_code_fence(content)
                    try:
                        parsed = json.loads(content_str)
                    except Exception:
                        parsed = {"summary_text": content_str}
                    logger.info(f"LLM success model={model} attempt={attempt} latency_ms={int(duration*1000)}")
                    return parsed
                except Exception as e:
                    last_error = e
                    # backoff for 429/5xx
                    sleep_s = min(2 ** attempt, 10)
                    logger.warning(f"OpenRouter model={model} attempt={attempt} error={e}; retrying in {sleep_s}s")
                    time.sleep(sleep_s)
            logger.info(f"Falling back from model {model}")
        raise RuntimeError(f"All OpenRouter models failed: {last_error}")


def build_llm_prompt(analysis: Any, detections: List[Any]) -> str:
    # concise structured prompt; models asked to return JSON with keys
    prompt = {
        "task": "Generate a technical thermal inspection summary and recommendations",
        "inputs": {
            "overall_risk_level": analysis.overall_risk_level,
            "max_temperature_detected": analysis.max_temperature_detected,
            "critical_hotspots": analysis.critical_hotspots,
            "potential_hotspots": analysis.potential_hotspots,
            "quality_score": analysis.quality_score,
            "detections": [
                {
                    "component_type": d.component_type,
                    "confidence": d.confidence,
                    "risk_level": d.risk_level,
                    "hotspot_classification": d.hotspot_classification,
                    "max_temperature": d.max_temperature,
                }
                for d in detections or []
            ],
        },
        "output_schema": {
            "summary_text": "str",
            "recommended_actions": ["str"],
            "risk_breakdown": {"critical": "int", "potential": "int", "normal": "int"},
            "root_cause_hypotheses": ["str"],
        },
        "constraints": [
            "Be specific, use Tata Power EHV context",
            "Use short sentences",
            "Do not hallucinate temperatures; use provided values",
        ],
    }
    return json.dumps(prompt)


# Singleton
openrouter_client = OpenRouterClient()


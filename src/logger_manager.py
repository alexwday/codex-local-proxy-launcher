"""Request/response logging and usage tracking for codex-local-proxy-launcher."""

import time
import logging
from typing import Any, Dict, List, Optional
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class UsageStats:
    """Track token usage statistics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: int = 0
    total_cost_usd: float = 0.0
    session_start: float = field(default_factory=time.time)
    by_model: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def avg_latency_ms(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests

    @property
    def session_duration_seconds(self) -> float:
        return time.time() - self.session_start

    def to_dict(self) -> dict:
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': round(self.success_rate, 1),
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'avg_latency_ms': round(self.avg_latency_ms, 0),
            'session_duration_seconds': round(self.session_duration_seconds, 0),
            'total_cost_usd': round(self.total_cost_usd, 4),
            'by_model': {
                model: {
                    **usage,
                    'cost_usd': round(float(usage.get('cost_usd', 0.0)), 4),
                }
                for model, usage in sorted(self.by_model.items())
            },
        }


class LoggerManager:
    """Manages API call logging and usage statistics."""

    def __init__(self, max_logs: int = 100):
        self.max_logs = max_logs
        self.api_calls: deque = deque(maxlen=max_logs)
        self.server_events: deque = deque(maxlen=max_logs)
        self.usage = UsageStats()

    def log_api_call(
        self,
        method: str,
        path: str,
        status: int,
        duration_ms: int,
        request_data: Optional[Dict] = None,
        response_data: Optional[Dict] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        public_model: Optional[str] = None,
        target_model: Optional[str] = None,
        codex_reasoning_effort: Optional[str] = None,
        upstream_reasoning_effort: Optional[str] = None,
        reasoning_effort_removed: bool = False,
    ):
        """Log an API call with optional request/response data."""
        codex_model = public_model or self._extract_model(request_data) or self._extract_model(response_data)
        codex_effort = codex_reasoning_effort or self._extract_reasoning_effort(request_data)
        upstream_effort = upstream_reasoning_effort or ""
        entry = {
            'timestamp': time.time(),
            'method': method,
            'path': path,
            'status': status,
            'duration_ms': duration_ms,
            'codex_model': codex_model or '',
            'target_model': target_model or '',
            'codex_reasoning_effort': codex_effort,
            'upstream_reasoning_effort': upstream_effort,
            'reasoning_effort_removed': bool(reasoning_effort_removed),
            'request': self._sanitize_for_log(request_data),
            'response': self._sanitize_for_log(response_data),
            'error_message': self._extract_error_message(response_data) if status >= 400 else '',
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost_usd': cost_usd,
        }

        self.api_calls.appendleft(entry)

        # Update usage stats
        self.usage.total_requests += 1
        if status < 400:
            self.usage.successful_requests += 1
            self.usage.total_latency_ms += duration_ms
        else:
            self.usage.failed_requests += 1

        # Token/cost usage may still occur on non-2xx outcomes (for example,
        # client-cancelled streams after partial generation), so always include
        # any reported values in usage totals.
        self.usage.total_input_tokens += input_tokens
        self.usage.total_output_tokens += output_tokens
        self.usage.total_cost_usd += cost_usd
        if codex_model:
            model_usage = self.usage.by_model.setdefault(
                codex_model,
                {
                    'requests': 0,
                    'successful_requests': 0,
                    'failed_requests': 0,
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'cost_usd': 0.0,
                },
            )
            model_usage['requests'] += 1
            if status < 400:
                model_usage['successful_requests'] += 1
            else:
                model_usage['failed_requests'] += 1
            model_usage['input_tokens'] += input_tokens
            model_usage['output_tokens'] += output_tokens
            model_usage['cost_usd'] += cost_usd

        # Log summary
        token_info = ""
        if input_tokens or output_tokens:
            cost_str = f"${cost_usd:.4f}" if cost_usd > 0 else ""
            token_info = f" | tokens: {input_tokens}+{output_tokens}" + (f" ({cost_str})" if cost_str else "")
        model_info = f" | model: {codex_model}->{target_model}" if codex_model and target_model else ""
        reasoning_info = ""
        if codex_effort:
            upstream_display = "removed" if reasoning_effort_removed else (upstream_effort or "-")
            reasoning_info = f" | reasoning: {codex_effort}->{upstream_display}"
        logger.info(f"{method} {path} -> {status} ({duration_ms}ms){model_info}{reasoning_info}{token_info}")

    def log_server_event(self, level: str, message: str, data: Optional[Dict] = None):
        """Log a server event."""
        entry = {
            'timestamp': time.time(),
            'level': level,
            'message': message,
            'data': data,
        }

        self.server_events.appendleft(entry)

        # Also log to standard logger
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(message)

    def get_api_calls(self, limit: int = 50) -> List[Dict]:
        """Get recent API calls."""
        return list(self.api_calls)[:limit]

    def get_server_events(self, limit: int = 50) -> List[Dict]:
        """Get recent server events."""
        return list(self.server_events)[:limit]

    def get_usage_stats(self) -> Dict:
        """Get current usage statistics."""
        return self.usage.to_dict()

    def get_model_usage(self) -> Dict:
        """Get current per-model usage keyed by Codex-facing model name."""
        return self.usage.to_dict()['by_model']

    def clear_logs(self):
        """Clear all logs (but preserve usage stats)."""
        self.api_calls.clear()
        self.server_events.clear()
        logger.info("Logs cleared")

    def reset_usage(self):
        """Reset usage statistics."""
        self.usage = UsageStats()
        logger.info("Usage statistics reset")

    def _extract_error_message(self, data: Optional[Dict]) -> str:
        """Extract a concise error message from common API error shapes."""
        if data is None:
            return ""
        if isinstance(data, str):
            return data
        if not isinstance(data, dict):
            return ""

        error = data.get('error')
        if isinstance(error, str):
            return error
        if isinstance(error, dict):
            for key in ('message', 'detail', 'error_description', 'code'):
                value = error.get(key)
                if value:
                    return str(value)

        for key in ('message', 'detail', 'error_description'):
            value = data.get(key)
            if value:
                return str(value)

        return ""

    def _extract_model(self, data: Optional[Dict]) -> str:
        """Extract model from common request/response payload shapes."""
        if not isinstance(data, dict):
            return ""
        model = data.get('model')
        return str(model) if model else ""

    def _extract_reasoning_effort(self, data: Optional[Dict]) -> str:
        """Extract reasoning effort from Responses or Chat Completions payloads."""
        if not isinstance(data, dict):
            return ""
        reasoning = data.get('reasoning')
        if isinstance(reasoning, dict) and reasoning.get('effort') is not None:
            return str(reasoning['effort'])
        effort = data.get('reasoning_effort')
        return str(effort) if effort is not None else ""

    def _sanitize_for_log(self, data: Optional[Dict]) -> Optional[Dict]:
        """Sanitize data for logging (truncate large content)."""
        if data is None:
            return None

        # Deep copy to avoid modifying original
        import copy
        sanitized = copy.deepcopy(data)

        # Truncate message content if too long
        if 'messages' in sanitized:
            for msg in sanitized['messages']:
                if isinstance(msg.get('content'), str) and len(msg['content']) > 500:
                    msg['content'] = msg['content'][:500] + '... [truncated]'
                elif isinstance(msg.get('content'), list):
                    for block in msg['content']:
                        if isinstance(block, dict):
                            if isinstance(block.get('text'), str) and len(block['text']) > 500:
                                block['text'] = block['text'][:500] + '... [truncated]'

        # Truncate response content
        if 'content' in sanitized:
            if isinstance(sanitized['content'], str) and len(sanitized['content']) > 500:
                sanitized['content'] = sanitized['content'][:500] + '... [truncated]'
            elif isinstance(sanitized['content'], list):
                for block in sanitized['content']:
                    if isinstance(block, dict):
                        if isinstance(block.get('text'), str) and len(block['text']) > 500:
                            block['text'] = block['text'][:500] + '... [truncated]'

        return sanitized

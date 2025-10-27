from __future__ import annotations

import asyncio
import json

import aiohttp

from src.config.settings import settings


class Notifier:
    def __init__(self) -> None:
        self.telegram_token = getattr(settings, "telegram_bot_token", None)
        self.telegram_chat_id = getattr(settings, "telegram_chat_id", None)
        self.slack_webhook = getattr(settings, "slack_webhook_url", None)

    async def send(self, text: str) -> None:
        # Enviar en paralelo a los destinos configurados
        tasks = []
        if self.telegram_token and self.telegram_chat_id:
            tasks.append(self._send_telegram(text))
        if self.slack_webhook:
            tasks.append(self._send_slack(text))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_telegram(self, text: str) -> None:
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {"chat_id": self.telegram_chat_id, "text": text}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=10) as resp:
                await resp.text()

    async def _send_slack(self, text: str) -> None:
        if not self.slack_webhook:
            return
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.slack_webhook,
                data=json.dumps({"text": text}),
                headers={"Content-Type": "application/json"},
                timeout=10,
            ) as resp:
                await resp.text()

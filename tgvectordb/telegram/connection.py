"""
manages the telethon client connection to telegram.

handles:
  - session login (phone number + code)
  - creating private channels for vector storage
  - keeping track of which channels belong to this db

the session file gets saved locally so you only have to
log in once. after that its automatic.
"""

import asyncio
from pathlib import Path
from typing import Optional

from telethon import TelegramClient
from telethon.tl.functions.channels import CreateChannelRequest
from telethon.tl.types import Channel

from tgvectordb.utils.config import DEFAULT_DATA_DIR, CHANNEL_PREFIX


class TelegramConnection:
    """
    wraps telethon client with our specific needs.
    """

    def __init__(
        self,
        api_id: int,
        api_hash: str,
        phone: str,
        db_name: str,
        data_dir: Path = None,
    ):
        self.api_id = api_id
        self.api_hash = api_hash
        self.phone = phone
        self.db_name = db_name
        self.data_dir = (data_dir or DEFAULT_DATA_DIR) / db_name
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.session_path = self.data_dir / "session"
        self.client = None
        self._connected = False

    async def connect(self):
        """connect and authenticate if needed."""
        if self._connected:
            return

        if self.client is None:
            self.client = TelegramClient(
                str(self.session_path), self.api_id, self.api_hash
            )

        await self.client.connect()

        if not await self.client.is_user_authorized():
            print(f"logging in with phone: {self.phone}")
            await self.client.send_code_request(self.phone)
            code = input("enter the code telegram sent you: ")
            try:
                await self.client.sign_in(self.phone, code)
            except Exception:
                # might need 2fa password
                password = input("2FA password required: ")
                await self.client.sign_in(password=password)
            print("logged in successfully!")

        self._connected = True

    async def disconnect(self):
        if self._connected:
            await self.client.disconnect()
            self._connected = False

    async def get_or_create_channel(self, suffix: str) -> Channel:
        """
        find an existing channel or create a new private one.
        suffix is like 'vectors' or 'index' - gets combined with db name.
        """
        channel_title = f"{CHANNEL_PREFIX}-{self.db_name}-{suffix}"

        # look through existing channels to find ours
        async for dialog in self.client.iter_dialogs():
            if dialog.is_channel and dialog.title == channel_title:
                return dialog.entity

        # didnt find it, create new private channel
        print(f"creating channel: {channel_title}")
        result = await self.client(CreateChannelRequest(
            title=channel_title,
            about=f"TgVectorDB storage for '{self.db_name}' - do not delete",
            megagroup=False,  # regular channel, not a supergroup
        ))

        # the channel entity is in the result
        channel = result.chats[0]
        print(f"channel created: {channel_title} (id: {channel.id})")
        return channel

    async def get_channel_id_str(self, channel: Channel) -> str:
        """get a string representation of channel id for storage."""
        return str(channel.id)

    def get_client(self) -> TelegramClient:
        return self.client

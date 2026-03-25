from pathlib import Path

from telethon import TelegramClient
from telethon.tl.functions.channels import CreateChannelRequest
from telethon.tl.types import Channel

from tgvectordb.utils.config import CHANNEL_PREFIX, DEFAULT_DATA_DIR


class TelegramConnection:
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
        session_path = self.data_dir / "session"
        self.client = TelegramClient(str(session_path), self.api_id, self.api_hash)
        self._connected = False

    async def connect(self):
        if self._connected:
            return
        await self.client.connect()
        if not await self.client.is_user_authorized():
            print(f"logging in with phone: {self.phone}")
            await self.client.send_code_request(self.phone)
            code = input("enter the code telegram sent you: ")
            try:
                await self.client.sign_in(self.phone, code)
            except Exception:
                password = input("2FA password required: ")
                await self.client.sign_in(password=password)
            print("logged in successfully!")
        self._connected = True

    async def disconnect(self):
        if self._connected:
            await self.client.disconnect()
            self._connected = False

    async def get_or_create_channel(self, suffix: str) -> Channel:
        channel_title = f"{CHANNEL_PREFIX}-{self.db_name}-{suffix}"
        async for dialog in self.client.iter_dialogs():
            if dialog.is_channel and dialog.title == channel_title:
                return dialog.entity
        print(f"creating channel: {channel_title}")
        result = await self.client(
            CreateChannelRequest(
                title=channel_title,
                about=f"TgVectorDB storage for '{self.db_name}' - do not delete",
                megagroup=False,
            )
        )
        channel = result.chats[0]
        print(f"channel created: {channel_title} (id: {channel.id})")
        return channel

    async def get_channel_id_str(self, channel: Channel) -> str:
        return str(channel.id)

    def get_client(self) -> TelegramClient:
        return self.client

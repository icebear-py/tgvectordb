import asyncio

from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.tl.types import Channel

from tgvectordb.utils.config import (
    TG_FETCH_BATCH_SIZE,
    TG_SEND_BURST_PAUSE,
    TG_SEND_BURST_SIZE,
    TG_SEND_DELAY,
)
from tgvectordb.utils.serialization import unpack_vector_message


async def _send_one_message(client, channel, text, retry_count=0):
    try:
        message = await client.send_message(channel, text)
        return message
    except FloodWaitError as e:
        wait = e.seconds
        if retry_count >= 3:
            raise RuntimeError(
                f"hit FLOOD_WAIT 3 times in a row (last wait: {wait}s). "
                f"telegram really doesnt want us sending right now. "
                f"try again in a few minutes, or use a different account."
            )
        print(f"  ⏳ flood wait: {wait}s (attempt {retry_count + 1}/3)")
        await asyncio.sleep(wait + 5)
        return await _send_one_message(client, channel, text, retry_count + 1)


async def send_vector_message(
    client: TelegramClient,
    channel: Channel,
    message_text: str,
) -> int:
    message = await _send_one_message(client, channel, message_text)
    await asyncio.sleep(TG_SEND_DELAY)
    return message.id


async def send_vector_messages_batch(
    client: TelegramClient,
    channel: Channel,
    messages: list,
    progress_callback=None,
) -> list:
    message_ids = []
    total = len(messages)
    burst_count = 0
    estimated_secs = (total * TG_SEND_DELAY) + (
        (total // TG_SEND_BURST_SIZE) * TG_SEND_BURST_PAUSE
    )
    estimated_mins = estimated_secs / 60
    print(
        f"  sending {total} messages (estimated {estimated_mins:.1f} min at safe rate)"
    )
    for i, message_text in enumerate(messages):
        message = await _send_one_message(client, channel, message_text)
        message_ids.append(message.id)
        burst_count += 1
        await asyncio.sleep(TG_SEND_DELAY)
        if burst_count >= TG_SEND_BURST_SIZE:
            burst_count = 0
            if i < total - 1:
                print(f"  sent {i + 1}/{total} — pausing {TG_SEND_BURST_PAUSE}s...")
                await asyncio.sleep(TG_SEND_BURST_PAUSE)
        if progress_callback and (i + 1) % 5 == 0:
            progress_callback(i + 1, total)
    return message_ids


async def fetch_messages_by_ids(
    client: TelegramClient,
    channel: Channel,
    message_ids: list,
) -> dict:
    results = {}
    batches = []
    for i in range(0, len(message_ids), TG_FETCH_BATCH_SIZE):
        batch = message_ids[i : i + TG_FETCH_BATCH_SIZE]
        batches.append(batch)
    for batch_ids in batches:
        try:
            messages = await client.get_messages(channel, ids=batch_ids)
        except FloodWaitError as e:
            print(f"  ⏳ rate limited on fetch, waiting {e.seconds}s")
            await asyncio.sleep(e.seconds + 2)
            messages = await client.get_messages(channel, ids=batch_ids)
        for message in messages:
            if message is None:
                continue
            if not message.text:
                continue
            try:
                parsed = unpack_vector_message(message.text)
                results[message.id] = parsed
            except (ValueError, KeyError):
                pass
        if len(batches) > 1:
            await asyncio.sleep(0.3)
    return results


async def fetch_all_messages(
    client: TelegramClient,
    channel: Channel,
    progress_callback=None,
) -> dict:
    results = {}
    count = 0
    async for message in client.iter_messages(channel):
        if not message.text:
            continue
        try:
            parsed = unpack_vector_message(message.text)
            results[message.id] = parsed
            count += 1
        except (ValueError, KeyError):
            pass
        if progress_callback and count % 100 == 0:
            progress_callback(count)
    return results


async def delete_messages(
    client: TelegramClient,
    channel: Channel,
    message_ids: list,
):
    for i in range(0, len(message_ids), 100):
        batch = message_ids[i : i + 100]
        try:
            await client.delete_messages(channel, batch)
        except FloodWaitError as e:
            await asyncio.sleep(e.seconds + 1)
            await client.delete_messages(channel, batch)
        await asyncio.sleep(0.5)


async def upload_file_to_channel(
    client: TelegramClient,
    channel: Channel,
    file_path: str,
    caption: str = "",
) -> int:
    try:
        message = await client.send_file(channel, file_path, caption=caption)
    except FloodWaitError as e:
        print(f"  ⏳ flood wait on file upload: {e.seconds}s")
        await asyncio.sleep(e.seconds + 2)
        message = await client.send_file(channel, file_path, caption=caption)
    return message.id


async def download_latest_file(
    client: TelegramClient,
    channel: Channel,
    save_path: str,
) -> bool:
    async for message in client.iter_messages(channel):
        if message.file:
            await client.download_media(message, file=save_path)
            return True
    return False

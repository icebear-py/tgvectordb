"""
sends and fetches vector messages from telegram channels.

each vector is one message. we use channels.getMessages with
specific IDs for fast fetching (not scanning through history).

important: this uses MTProto directly through telethon, NOT the bot API.
MTProto is faster because message text comes inline in the response -
no separate file download step needed.
"""

import asyncio
import time
from typing import Optional

from telethon import TelegramClient
from telethon.tl.types import Channel
from telethon.errors import FloodWaitError

from tgvectordb.utils.config import (
    TG_FETCH_BATCH_SIZE,
    TG_SEND_DELAY,
)
from tgvectordb.utils.serialization import unpack_vector_message


async def send_vector_message(
    client: TelegramClient,
    channel: Channel,
    message_text: str,
) -> int:
    """
    send a single vector message to the channel.
    returns the message id.

    includes a small delay to stay under rate limits.
    """
    try:
        msg = await client.send_message(channel, message_text)
        await asyncio.sleep(TG_SEND_DELAY)
        return msg.id
    except FloodWaitError as e:
        # telegram is telling us to slow down
        wait_secs = e.seconds
        print(f"rate limited! waiting {wait_secs} seconds...")
        await asyncio.sleep(wait_secs + 1)
        # retry once
        msg = await client.send_message(channel, message_text)
        return msg.id


async def send_vector_messages_batch(
    client: TelegramClient,
    channel: Channel,
    messages: list,
    progress_callback=None,
) -> list:
    """
    send multiple vector messages. returns list of message ids.
    tries to be nice about rate limits.
    """
    msg_ids = []
    total = len(messages)

    for i, msg_text in enumerate(messages):
        try:
            msg = await client.send_message(channel, msg_text)
            msg_ids.append(msg.id)
        except FloodWaitError as e:
            print(f"flood wait: sleeping {e.seconds}s (sent {i}/{total} so far)")
            await asyncio.sleep(e.seconds + 2)
            # retry this one
            msg = await client.send_message(channel, msg_text)
            msg_ids.append(msg.id)

        # throttle ourselves
        await asyncio.sleep(TG_SEND_DELAY)

        if progress_callback and (i + 1) % 10 == 0:
            progress_callback(i + 1, total)

    return msg_ids


async def fetch_messages_by_ids(
    client: TelegramClient,
    channel: Channel,
    message_ids: list,
) -> dict:
    """
    fetch specific messages by their ids. this is the fast path -
    telegram can look up messages by id directly without scanning.

    returns dict of {msg_id: parsed_data} where parsed_data has
    vector_int8, quant_params, metadata keys.

    fetches in batches of 100 (telegram limit per call).
    """
    results = {}

    # split into batches of 100
    batches = []
    for i in range(0, len(message_ids), TG_FETCH_BATCH_SIZE):
        batch = message_ids[i : i + TG_FETCH_BATCH_SIZE]
        batches.append(batch)

    for batch_ids in batches:
        try:
            messages = await client.get_messages(channel, ids=batch_ids)
        except FloodWaitError as e:
            print(f"rate limited on fetch, waiting {e.seconds}s")
            await asyncio.sleep(e.seconds + 1)
            messages = await client.get_messages(channel, ids=batch_ids)

        for msg in messages:
            if msg is None:
                # message was deleted or doesnt exist
                continue
            if not msg.text:
                continue

            try:
                parsed = unpack_vector_message(msg.text)
                results[msg.id] = parsed
            except (ValueError, KeyError) as e:
                # corrupted or wrong format message, skip it
                # print(f"warning: couldnt parse message {msg.id}: {e}")
                pass

    return results


async def fetch_all_messages(
    client: TelegramClient,
    channel: Channel,
    progress_callback=None,
) -> dict:
    """
    fetch ALL messages from a channel. used during reindexing.
    this can be slow for large channels - goes through the whole history.

    returns dict of {msg_id: parsed_data}
    """
    results = {}
    count = 0

    async for msg in client.iter_messages(channel):
        if not msg.text:
            continue

        try:
            parsed = unpack_vector_message(msg.text)
            results[msg.id] = parsed
            count += 1
        except (ValueError, KeyError):
            # not a vector message, skip
            pass

        if progress_callback and count % 100 == 0:
            progress_callback(count)

    return results


async def delete_messages(
    client: TelegramClient,
    channel: Channel,
    message_ids: list,
):
    """delete specific messages from the channel."""
    # telegram supports deleting up to 100 at a time
    for i in range(0, len(message_ids), 100):
        batch = message_ids[i : i + 100]
        await client.delete_messages(channel, batch)
        await asyncio.sleep(0.1)


async def upload_file_to_channel(
    client: TelegramClient,
    channel: Channel,
    file_path: str,
    caption: str = "",
) -> int:
    """upload a file (like the index backup) to a channel. returns msg id."""
    msg = await client.send_file(channel, file_path, caption=caption)
    return msg.id


async def download_latest_file(
    client: TelegramClient,
    channel: Channel,
    save_path: str,
) -> bool:
    """
    download the most recent file from a channel.
    used for restoring index backup.
    returns True if found and downloaded, False if no files in channel.
    """
    async for msg in client.iter_messages(channel):
        if msg.file:
            await client.download_media(msg, file=save_path)
            return True
    return False

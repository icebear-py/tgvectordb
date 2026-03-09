"""
handles converting vectors + metadata into telegram message strings
and parsing them back out.

the format is simple json:
{
    "v": "<base64 encoded int8 vector>",
    "q": [min_val, scale],           # quantization params to reconstruct float32
    "m": {                            # metadata - whatever the user wants
        "text": "original chunk text",
        "src": "document.pdf",
        ...
    }
}
"""

import json
import base64
import numpy as np
from typing import Any


def pack_vector_message(
    vector_int8: np.ndarray,
    quant_params: tuple,  # (min_val, scale)
    metadata: dict,
    text: str = "",
) -> str:
    """
    take an int8 vector + metadata and turn it into a json string
    that fits in a telegram message.

    returns the json string ready to send as a message.
    """
    # encode the int8 vector as base64 - way more compact than json array
    vec_bytes = vector_int8.astype(np.uint8).tobytes()
    vec_b64 = base64.b64encode(vec_bytes).decode("ascii")

    min_val, scale = quant_params

    msg_data = {
        "v": vec_b64,
        "q": [float(min_val), float(scale)],
        "m": metadata,
    }

    # include text separately so its easy to grab
    if text:
        msg_data["m"]["text"] = text

    msg_str = json.dumps(msg_data, ensure_ascii=True, separators=(",", ":"))

    # sanity check - telegram has a hard 4096 char limit
    if len(msg_str) > 4096:
        # try truncating the text to make it fit
        overflow = len(msg_str) - 4090  # small buffer
        if text and len(text) > overflow:
            msg_data["m"]["text"] = text[: -(overflow + 3)] + "..."
            msg_str = json.dumps(msg_data, ensure_ascii=True, separators=(",", ":"))
        else:
            raise ValueError(
                f"message too big even after truncation ({len(msg_str)} chars). "
                f"reduce metadata size or text length"
            )

    return msg_str


def unpack_vector_message(msg_str: str) -> dict:
    """
    parse a telegram message back into vector + metadata.

    returns dict with keys:
        - vector_int8: numpy uint8 array
        - quant_params: (min_val, scale) tuple
        - metadata: dict with text, source, etc
    """
    try:
        data = json.loads(msg_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"couldnt parse message as json: {e}")

    # decode the vector
    vec_bytes = base64.b64decode(data["v"])
    vector_int8 = np.frombuffer(vec_bytes, dtype=np.uint8)

    min_val, scale = data["q"]
    quant_params = (min_val, scale)

    metadata = data.get("m", {})

    return {
        "vector_int8": vector_int8,
        "quant_params": quant_params,
        "metadata": metadata,
    }


def estimate_message_size(dims: int, metadata: dict) -> int:
    """rough estimate of how big the final message will be in chars.
    useful for checking if stuff will fit before we actually build it."""
    # base64 of int8 vector
    vec_chars = ((dims + 2) // 3) * 4  # base64 encoding overhead
    meta_chars = len(json.dumps(metadata, separators=(",", ":")))
    overhead = 30  # json keys, brackets etc

    return vec_chars + meta_chars + overhead

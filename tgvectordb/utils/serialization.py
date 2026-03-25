import base64
import json

import numpy as np


def pack_vector_message(
    vector_int8: np.ndarray,
    quant_params: tuple,
    metadata: dict,
    text: str = "",
) -> str:
    vector_bytes = vector_int8.astype(np.uint8).tobytes()
    vector_b64 = base64.b64encode(vector_bytes).decode("ascii")
    min_val, scale = quant_params
    message_data = {
        "v": vector_b64,
        "q": [float(min_val), float(scale)],
        "m": metadata,
    }
    if text:
        message_data["m"]["text"] = text
    message_string = json.dumps(message_data, ensure_ascii=True, separators=(",", ":"))
    if len(message_string) > 4096:
        overflow = len(message_string) - 4090
        if text and len(text) > overflow:
            message_data["m"]["text"] = text[: -(overflow + 3)] + "..."
            message_string = json.dumps(
                message_data, ensure_ascii=True, separators=(",", ":")
            )
        else:
            raise ValueError(
                f"message too big even after truncation ({len(message_string)} chars). "
                f"reduce metadata size or text length"
            )
    return message_string


def unpack_vector_message(message_string: str) -> dict:
    try:
        data = json.loads(message_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"couldnt parse message as json: {e}")
    vector_bytes = base64.b64decode(data["v"])
    vector_int8 = np.frombuffer(vector_bytes, dtype=np.uint8)
    min_val, scale = data["q"]
    quant_params = (min_val, scale)
    metadata = data.get("m", {})
    return {
        "vector_int8": vector_int8,
        "quant_params": quant_params,
        "metadata": metadata,
    }


def estimate_message_size(dims: int, metadata: dict) -> int:
    vector_chars = ((dims + 2) // 3) * 4
    meta_chars = len(json.dumps(metadata, separators=(",", ":")))
    overhead = 30
    return vector_chars + meta_chars + overhead

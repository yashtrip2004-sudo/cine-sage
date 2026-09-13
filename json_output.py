import json


def extract_json(text: str) -> str:
    cleaned = text.strip()

    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[1].strip()

    json_start = cleaned.find("{")
    if json_start == -1:
        raise ValueError("The model response did not contain a JSON object.")

    try:
        value, _ = json.JSONDecoder().raw_decode(cleaned[json_start:])
    except json.JSONDecodeError as error:
        raise ValueError("The model response contained incomplete or invalid JSON.") from error

    return json.dumps(value)
import torch
import base64
import io


def tensor_to_base64(tensor: torch.Tensor) -> str:
    if tensor is None:
        return ""
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_tensor(encoded_str: str) -> torch.Tensor | None:
    if not encoded_str:
        return None
    buffer = io.BytesIO(base64.b64decode(encoded_str.encode("utf-8")))
    return torch.load(buffer)


__all__ = ["tensor_to_base64", "base64_to_tensor"]

import hashlib


class SHA:
    @staticmethod
    def generate_sha256(msg: str) -> str:
        return hashlib.sha256(msg.encode("utf-8")).hexdigest()

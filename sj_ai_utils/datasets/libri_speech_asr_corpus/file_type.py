X = "flac"
Y = "trans"

FILE_TYPE = {
    X: {"file": "*.flac", "multiple": True, "explain": "Audio files in FLAC format"},
    Y: {
        "file": "*.trans.txt",
        "multiple": False,
        "explain": "Transcription file in text format",
    },
}

__all__ = [
    "X",
    "Y",
    "FILE_TYPE",
]

PROMPTS = [
    "a door",
    "a chair",
    "a table",
    "a corridor",
    "open floor",
    "a wall",
]

ACTION_MAP = {
    "a corridor": "forward",
    "open floor": "forward",
    "a chair": "turn",
    "a wall": "rotate",
    "a door": "approach",
    "a table": "avoid",
}

DEFAULT_ACTION = "stop"
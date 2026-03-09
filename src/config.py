PROMPTS = [
    "a door directly in front of the robot",
    "a chair blocking the robot path",
    "a table obstacle",
    "a long indoor corridor",
    "free open floor space",
    "a close wall directly blocking the robot",
]

ACTION_MAP = {
    "a long indoor corridor": "forward",
    "free open floor space": "forward",
    "a chair blocking the robot path": "turn",
    "a close wall directly blocking the robot": "rotate",
    "a door directly in front of the robot": "approach",
    "a table obstacle": "avoid",
}

DEFAULT_ACTION = "stop"
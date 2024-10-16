from enum import Enum, StrEnum
import mesop as me

class ChatBox(StrEnum):
    LEFT="Left"
    RIGHT="Right"

class Models(StrEnum):
    GPT_4_o="gpt-4o"
    GPT_4_o_mini="gpt-4o-mini"
    Gemini_1_5_Pro = "gemini-1.5-pro"


@me.stateclass
class LeftState:
    uploader_file: me.UploadedFile = None  # State for the first uploader
    model: str
    pattern:str

@me.stateclass
class RightState:
    uploader_file: me.UploadedFile = None  # State for the second uploader
    model:str
    pattern:str


def get_state(key:str) -> me.state:
    if key == ChatBox.LEFT:
        state = me.state(LeftState)  # Access LeftState
    elif key == ChatBox.RIGHT:
        state = me.state(RightState)  # Access RightState
    return state
#import debugpy
#debugpy.listen(5678)
#debugpy.wait_for_client()

import base64
from enum import StrEnum
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

@me.stateclass
class RightState:
    uploader_file: me.UploadedFile = None  # State for the second uploader


ROOT_BOX_STYLE = me.Style(
    background="#e7f2ff",
    height="100vh",
    font_family="Inter",
    display="flex",
    flex_direction="column",
    align_content="center",
    align_items="center",
    padding=me.Padding.all(16),
)

LEFT_BOX_STYLE = me.Style(
    background="#e7f2ff",
    height="100%",
    width="100%",
    font_family="Inter",
    display="flex",
    flex_direction="column",
    align_content="center",
    align_items="center",
    padding=me.Padding.all(16),
)

RIGHT_BOX_STYLE = me.Style(
    background="#e7f2ff",
    height="100%",
    width="100%",
    font_family="Inter",
    display="flex",
    flex_direction="column",
    align_content="center",
    align_items="center",
    padding=me.Padding.all(16),
)


BOTTOM_BOX_STYLE = me.Style(
    background="green",
    font_family="Inter",
    display="flex",
    flex_direction="row",
    align_content="start",
    align_items="center",
    width="100%",    
)

UPLOADERS_BOX_STYLE = me.Style(
    background="pink",
    font_family="Inter",
    display="flex",
    flex_direction="column",
    align_content="start",
    align_items="center",
    width="100%",
    margin=me.Margin(bottom=50)
)

CHAT_BOX_STYLE = me.Style(

)

@me.page(
    path="/",
    stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap"
    ],
)
def page():
    with me.box(style=ROOT_BOX_STYLE):
        header()
        with me.box(style=BOTTOM_BOX_STYLE):
            with me.box(style=LEFT_BOX_STYLE):
                with me.box(style=UPLOADERS_BOX_STYLE, key="box_with_uploader_1"):
                    uploader(state=LeftState, key=ChatBox.LEFT)
                box_with_chat(key=ChatBox.LEFT)
            with me.box(style=RIGHT_BOX_STYLE, key="box_with_uploader_2"):
                with me.box(style=UPLOADERS_BOX_STYLE, key="box_with_uploader_2"):
                    uploader(state=RightState, key=ChatBox.RIGHT)    
                box_with_chat(key=ChatBox.RIGHT)


def header():
    with me.box(
        style=me.Style(
            padding=me.Padding.all(10),
            margin=me.Margin(bottom=50)
        ),
    ):
        me.text(
            "DuoChat",
            style=me.Style(
                font_weight=500,
                font_size=24,
                color="#3D3929",
                letter_spacing="0.3px",
            ),
        )


def send_prompt(e: me.ClickEvent):
    pass

def chat_input(key):
    with me.box(
        style=me.Style(
            border_radius=16,
            padding=me.Padding.all(8),
            background="pink",
            display="flex",
            width="100%",
            flex_direction="row",
            # Icon aligned at bottom
            align_items="end"
        ),
        key=f"{key}_chat_input_box"
    ):
        with me.box(
            style=me.Style(
                flex_grow=1
            ), 
            key=f"{key}_text_area_box"):  # Flex-grow allows the textarea to expand
            me.native_textarea(
                placeholder="Enter a prompt",
                style=me.Style(
                    padding=me.Padding(top=16, left=16),
                    outline="none",
                    width="100%",  # Use 100% of the available space in this flex item
                    height="100px",
                    border=me.Border.all(me.BorderSide(style="none")),
                ),
                key=f"{key}_text_area"
            )
        with me.content_button(
            type="icon", on_click=send_prompt
        ):
            me.icon("send")


def box_with_chat(key):
    with me.box(style=BOTTOM_BOX_STYLE):
        with me.box(
            style=me.Style(
                width="100%",
                background="grey",
                margin=me.Margin.symmetric(
                    horizontal=36,
                    vertical=36,
                ),
            ),
            key=key
        ):
            with me.box(
                style=me.Style(
                    display="flex",
                    flex_direction="row",
                    justify_content="space-between"
                )
            ):
                me.text(
                    "Chat",
                    style=me.Style(
                        font_size=18,
                        margin=me.Margin.symmetric(
                            horizontal=10,
                            vertical=10,
                        ),
                    ),
                )
                me.select(
                    style=me.Style(
                        font_size=18,
                        margin=me.Margin.symmetric(
                            horizontal=10,
                            vertical=10,
                        ),
                    ),                 
                    label="Select model",
                    options=[
                        me.SelectOption(label=Models.GPT_4_o, value=Models.GPT_4_o),
                        me.SelectOption(label=Models.GPT_4_o_mini, value=Models.GPT_4_o_mini),
                        me.SelectOption(label=Models.Gemini_1_5_Pro, value=Models.Gemini_1_5_Pro),
                    ],
                    multiple=False,
                    appearance="fill",
                )
                me.select(
                    style=me.Style(
                        font_size=18,
                        margin=me.Margin.symmetric(
                            horizontal=10,
                            vertical=10,
                        ),
                    ),                    
                    label="Select patterns",
                    options=[
                        me.SelectOption(label="ReACT", value="value1"),
                        me.SelectOption(label="Critic", value="value2"),
                    ],
                    multiple=False,
                    appearance="fill",
                )                
            chat_input(key)
  

# Function to handle image removal (clicked)
def handle_remove_image(e: me.ClickEvent):
    if e.key == ChatBox.LEFT:
        state = me.state(LeftState)  # Access LeftState
        state.uploader_file = None   # Update the file for LeftState
    elif e.key == ChatBox.RIGHT:
        state = me.state(RightState)  # Access RightState
        state.uploader_file = None   # Update the file for RightState  


def handle_upload(e: me.UploadEvent):
    if e.key == ChatBox.LEFT:
        state = me.state(LeftState)  # Access LeftState
        state.uploader_file = e.file   # Update the file for LeftState
        print(e.file)
    elif e.key == ChatBox.RIGHT:
        state = me.state(RightState)  # Access RightState
        state.uploader_file = e.file    # Update the file for RightState    


def _convert_contents_data_url(file: me.UploadedFile) -> str:
  return (
    f"data:{file.mime_type};base64,{base64.b64encode(file.getvalue()).decode()}"
  )

def uploader(state, key):
    state = me.state(state)  # Use a unique state for each uploader
    with me.box(
        style=me.Style(
            width="100%",
            background="grey",
            display="flex",
            margin=me.Margin.symmetric(
                horizontal=36,
                vertical=36,
            ),
            flex_direction="column"
        ),
    ):  
        me.uploader(
            label="Upload Image",
            accepted_file_types=["image/jpeg", "image/png"],
            on_upload=handle_upload,
            type="flat",
            color="primary",
            style=me.Style(font_weight="bold", margin=me.Margin(right=24)),
            key=key
        )

        # Provide a default value of None if the file state has not been set yet
        file_state = getattr(state, "uploader_file", None)

        if file_state and hasattr(file_state, 'size') and file_state.size:
            with me.box(
                style=me.Style(
                    margin=me.Margin.all(10),
                    background="orange",
                )
            ):
                # "X" button to close/remove the box
                with me.content_button(type="icon", on_click=handle_remove_image, key=key):
                    me.icon("close")  # "X" icon or close button                
                me.image(
                    src=_convert_contents_data_url(file_state),
                )
                #me.text(f"File name: {file_state.name}")
                #me.text(f"File size: {file_state.size}")
                #me.text(f"File type: {file_state.mime_type}")

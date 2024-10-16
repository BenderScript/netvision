import mesop as me


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
    justify_content="space-between",
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
    justify_content="space-between",
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
    margin=me.Margin(bottom=20, top=20)
)
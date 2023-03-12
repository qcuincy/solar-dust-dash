from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from dash.dependencies import Input, Output
from keras.models import load_model
from ultralytics import YOLO
from scripts.tools import *
from dash import dcc, html
from PIL import Image
import dash_bootstrap_components as dbc
import base64, io, os
import numpy as np
import dash

cls2name = {0:"clean", 1:"dirty"}

model_classification_cnn = load_model(os.path.join(os.getcwd(), "models\\jack_cnn.h5"), compile=False)

model_detection = YOLO(os.path.join(os.getcwd(), "models\\object_detection.pt"))

data_dir = os.path.join(os.getcwd(), "data")

vid_true = "https://player.vimeo.com/video/807300739?h=79a8d9bd19"
vid_pred = "https://player.vimeo.com/video/807300756?h=7a4e21681e"
report_pdf = "https://drive.google.com/file/d/1e6ssGJBBg-LgmdF6d0ImDFNdqC-U8J8k/preview"

dropdown_items = [
    {'label': 'Clean Panel 1', 'value': os.path.join(data_dir, "clean_panel_1.png")},
    {'label': 'Clean Panel 2', 'value': os.path.join(data_dir, "clean_panel_2.png")},
    {'label': 'Clean Panel 3', 'value': os.path.join(data_dir, "clean_panel_3.png")},
    {'label': 'Clean Panel 4', 'value': os.path.join(data_dir, "clean_panel_4.png")},
    {'label': 'Clean Panel 5', 'value': os.path.join(data_dir, "clean_panel_5.png")},
    {'label': 'Clean Panel 6', 'value': os.path.join(data_dir, "clean_panel_6.png")},
    {'label': 'Clean Panel 7', 'value': os.path.join(data_dir, "clean_panel_7.png")},
    {'label': 'Clean Panel 8', 'value': os.path.join(data_dir, "clean_panel_8.png")},
    {'label': 'Clean Panel 9', 'value': os.path.join(data_dir, "clean_panel_9.png")},
    {'label': 'Clean Panel 10', 'value': os.path.join(data_dir, "clean_panel_10.png")},
    {'label': 'Clean Panel 11', 'value': os.path.join(data_dir, "clean_panel_11.png")},
    {'label': 'Clean Panel 12', 'value': os.path.join(data_dir, "clean_panel_12.png")},
    {'label': 'Clean Panel 13', 'value': os.path.join(data_dir, "clean_panel_13.png")},
    {'label': 'Clean Panel 14', 'value': os.path.join(data_dir, "clean_panel_14.png")},
    {'label': 'Clean Panel 15', 'value': os.path.join(data_dir, "clean_panel_15.png")},
    {'label': 'Clean Panel 16', 'value': os.path.join(data_dir, "clean_panel_16.png")},
    {'label': 'Clean Panel 17', 'value': os.path.join(data_dir, "clean_panel_17.png")},
    {'label': 'Clean Panel 18', 'value': os.path.join(data_dir, "clean_panel_18.png")},
    {'label': 'Clean Panel 19', 'value': os.path.join(data_dir, "clean_panel_19.png")},
    {'label': 'Clean Panel 20', 'value': os.path.join(data_dir, "clean_panel_20.png")},
    {'label': 'Dirty Panel 1', 'value': os.path.join(data_dir, "dirty_panel_1.png")},
    {'label': 'Dirty Panel 2', 'value': os.path.join(data_dir, "dirty_panel_2.png")},
    {'label': 'Dirty Panel 3', 'value': os.path.join(data_dir, "dirty_panel_3.png")},
    {'label': 'Dirty Panel 4', 'value': os.path.join(data_dir, "dirty_panel_4.png")},
    {'label': 'Dirty Panel 5', 'value': os.path.join(data_dir, "dirty_panel_5.png")},
    {'label': 'Dirty Panel 6', 'value': os.path.join(data_dir, "dirty_panel_6.png")},
    {'label': 'Dirty Panel 7', 'value': os.path.join(data_dir, "dirty_panel_7.png")},
    {'label': 'Dirty Panel 8', 'value': os.path.join(data_dir, "dirty_panel_8.png")},
    {'label': 'Dirty Panel 9', 'value': os.path.join(data_dir, "dirty_panel_9.png")},
    {'label': 'Dirty Panel 10', 'value': os.path.join(data_dir, "dirty_panel_10.png")},
    {'label': 'Dirty Panel 11', 'value': os.path.join(data_dir, "dirty_panel_11.png")},
    {'label': 'Dirty Panel 12', 'value': os.path.join(data_dir, "dirty_panel_12.png")},
    {'label': 'Dirty Panel 13', 'value': os.path.join(data_dir, "dirty_panel_13.png")},
    {'label': 'Dirty Panel 14', 'value': os.path.join(data_dir, "dirty_panel_14.png")},
    {'label': 'Dirty Panel 15', 'value': os.path.join(data_dir, "dirty_panel_15.png")},
    {'label': 'Dirty Panel 16', 'value': os.path.join(data_dir, "dirty_panel_16.png")},
    {'label': 'Dirty Panel 17', 'value': os.path.join(data_dir, "dirty_panel_17.png")},
    {'label': 'Dirty Panel 18', 'value': os.path.join(data_dir, "dirty_panel_18.png")},
    {'label': 'Dirty Panel 19', 'value': os.path.join(data_dir, "dirty_panel_19.png")},
    {'label': 'Dirty Panel 20', 'value': os.path.join(data_dir, "dirty_panel_20.png")},
    {'label': 'Dirty Panel 21', 'value': os.path.join(data_dir, "dirty_panel_21.png")},
    {'label': 'Dirty Panel 22', 'value': os.path.join(data_dir, "dirty_panel_22.png")},
    {'label': 'Dirty Panel 23', 'value': os.path.join(data_dir, "dirty_panel_23.png")},
    {'label': 'Dirty Panel 24', 'value': os.path.join(data_dir, "dirty_panel_24.png")},
    {'label': 'Dirty Panel 25', 'value': os.path.join(data_dir, "dirty_panel_25.png")},
    {'label': 'Dirty Panel 26', 'value': os.path.join(data_dir, "dirty_panel_26.png")},
    {'label': 'Dirty Panel 27', 'value': os.path.join(data_dir, "dirty_panel_27.png")},
]
image_dropdown = dbc.DropdownMenu(
    id="image-dropdown",
    label="Clean Panel 1",
    children=[
        dbc.DropdownMenuItem(
            dropdown_items[i]["label"], href=dropdown_items[i]["value"], id=f"solar-panel-{i}"
        ) for i in range(len(dropdown_items))
    ],
    size="lg",
    style={"overflow-y":"auto", "max-height":"400px", "height":"400px"}

)

image_dropdown_detection = dbc.DropdownMenu(
    id="detection-image-dropdown",
    label="Clean Panel 1",
    children=[
        dbc.DropdownMenuItem(
            dropdown_items[i]["label"], href=dropdown_items[i]["value"], id=f"solar-panel-{i}"
        ) for i in range(len(dropdown_items))
    ],
    size="lg",
    style={"overflow-y":"auto", "max-height":"400px", "height":"400px"}
)

video_button = dbc.Button(
    "Show video", id="video-button"
)


selected_image = dbc.Card(
    children=[
        dbc.CardImg(src='data:image/jpeg;base64,{}'.format(base64.b64encode(open(os.path.join(data_dir, "clean_panel_1.png"), 'rb').read()).decode()), id="selected-image"),
        dbc.CardBody(
            html.P("", id="classification-output",className="card-text")
        ),
    ],
)

detection_selected_image = dbc.Card(
    children=[
        dbc.CardImg(src='data:image/jpeg;base64,{}'.format(base64.b64encode(open(os.path.join(data_dir, "clean_panel_1.png"), 'rb').read()).decode()), id="detection-selected-image"),
    ],
)

detection_output = dbc.Card(
    id="detection-output",
    children=[
        dbc.CardImg(src='data:image/jpeg;base64,{}'.format(base64.b64encode(open(os.path.join(data_dir, "clean_panel_1.png"), 'rb').read()).decode()))
    ]
)


image_row = dbc.Row(
    children=[
        dbc.Col(detection_selected_image, width="auto"),
        dbc.Col(detection_output, width="auto")
    ],
    id="media-row"
)


classification_layout = html.Div(
    [
        dbc.Row([dbc.Col(selected_image, width="auto")]),
        dbc.Row([image_dropdown]),
    ]
)

video = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(html.Iframe(src=vid_true, width="460", height="460", allow="autoplay; fullscreen"),width="auto"),
                dbc.Col(html.Iframe(src=vid_pred, width="460", height="460", allow="autoplay; fullscreen"),width="auto"),
            ]
        )
    ]
)



detection_image_layout = html.Div(
    [
        image_row,
        dbc.Row([image_dropdown_detection]),
    ]
)
detection_video_layout = html.Div(
    [
        video
    ]
)

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "margin-top":"2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Group 14", className="display-4"),
        html.Hr(),
        html.P(
            "Demonstration of our 'solar dust' classification and detection models", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Classification", href="/classification", active="exact"),
                dbc.NavLink("Detection (image)", href="/detection-image", active="exact"),
                dbc.NavLink("Detection (video)", href="/detection-video", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

home_layout = html.Div(
    [
        dbc.Row([
            dbc.Col([
                html.H2("The purpose of this dashboard is to show performance of the model's described in the research paper below:",className="display-5"),
                html.Hr(),
                html.Iframe(src=report_pdf, width="960", height="640", allow="autoplay")
            ])
        ])
    ]
)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])



@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return home_layout
    elif pathname == "/classification":
        return classification_layout
    elif pathname == "/detection-image":
        return detection_image_layout
    elif pathname == "/detection-video":
        return detection_video_layout
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )


@app.callback(
        Output("image-dropdown", "label"),
        [Input(f"solar-panel-{str(i)}", "n_clicks") for i in range(len(dropdown_items))]
)
def update_label(*args):
    id_lookup = {f"solar-panel-{i}":dropdown_items[i]["label"] for i in range(len(dropdown_items))}

    ctx = dash.callback_context

    vals = [1 if n is None else 0 for n in args]
    if sum(vals) == len(args) or not (ctx.triggered):
        return "Select image"
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    return id_lookup[button_id]
@app.callback(
        Output("detection-image-dropdown", "label"),
        [Input(f"solar-panel-{str(i)}", "n_clicks") for i in range(len(dropdown_items))]
)
def update_label_detection(*args):
    id_lookup = {f"solar-panel-{i}":dropdown_items[i]["label"] for i in range(len(dropdown_items))}

    ctx = dash.callback_context

    vals = [1 if n is None else 0 for n in args]
    if sum(vals) == len(args) or not (ctx.triggered):
        return "Select image"
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    return id_lookup[button_id]

@app.callback(
    Output('selected-image', 'src'),
    Input("image-dropdown", "label")
)
def update_image_src(image_label):
    image_filename = [item["value"] for item in dropdown_items if item["label"] == image_label][0]
    encoded_image = base64.b64encode(open(image_filename, 'rb').read()).decode()
    return 'data:image/jpeg;base64,{}'.format(encoded_image)

@app.callback(
    Output('detection-selected-image', 'src'),
    Input("detection-image-dropdown", "label")
)
def update_image_src_detection(image_label):
    image_filename = [item["value"] for item in dropdown_items if item["label"] == image_label][0]
    encoded_image = base64.b64encode(open(image_filename, 'rb').read()).decode()
    return 'data:image/jpeg;base64,{}'.format(encoded_image)

@app.callback(
    Output('classification-output', 'children'),
    Input('selected-image', 'src')
)
def predict_class(image_src):
    img = image.load_img(io.BytesIO(base64.b64decode(image_src.split(',')[1])), target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    preds = model_classification_cnn.predict(x)

    output = []
    for pred in preds:
        pred_acc = pred[0] if pred[0] > 0.5 else (1-pred[0])
        output.append(html.P(f'{cls2name[int(round(pred[0]))]}: {pred_acc:.2f}'))
    return output

@app.callback(
    Output('detection-output', 'children'),
    Input('detection-selected-image', 'src')
)
def predict_bbox(image_src):
    img = Image.open(io.BytesIO(base64.b64decode(image_src.split(',')[1])))

    x = np.asarray(img)

    orig_img = x.copy()
    result = model_detection(x)
    new_img = plot(orig_img, result[0], font_size=2)

    output = dbc.CardImg(id='bbox-image',className="image", src="data:image/jpeg;base64, " + pil_to_b64(Image.fromarray(new_img)))
    
    return output


if __name__ == "__main__":
    app.run_server(debug=False)
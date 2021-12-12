import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from lime_explainer import explainer, tokenizer, METHODS

app = dash.Dash(__name__)
app.title = "LIME explainer app for classification models"
# Serve scripts and CSS locally
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True
# Declare server name
server = app.server

# ======== App layout ======== 
app.layout = html.Div([
    html.H3('''
        LIME Explainer Dashboard for Hostility Detection
    ''', style={'text-align': 'center'}),
    html.Label('''
        1: Hostile 0: Non-Hostile
    ''', style={'text-align': 'center'}),
    html.Br(),
    html.Label('Enter your text:'),
    html.Div(
        dcc.Textarea(
            id='text-input',
            placeholder='Enter text to make class prediction and generate explanations for',
            className='textarea',
            style={
                'width': '100%', 'height': 50, 'verticalAlign': 'top',
                'fontFamily': 'Arial', 'fontColor': '#515151',
            }
        ),
        style={'display': 'flex', 'justifyContent': 'center'}
    ),
    html.Label('Number of samples to generate for LIME explainer (For really long input text, go up to 5000):'),
    dcc.Input(
        id='num-samples-input',
        type='number',
        value=1000,
    ),
    html.Label('Choose classifier:'),
    dcc.Dropdown(
        id='case-dropdown',
        options=[
            {'label': obj['name'], 'value': value} for value, obj in METHODS.items()
        ],
        value='fasttext',
    ),
    html.Br(),
    html.Div(
        [html.Button(id='submit-button', n_clicks=0, children='Explain results'),
         html.Button(id='reset-button', n_clicks=0, children='Reset', style={'backgroundColor': 'white', 'color': '#515151'})],
        style={'display': 'flex', 'justifyContent': 'center'}
    ),
    html.Br(),
    dcc.Loading(
        id='explainer-obj',
        type="default"
    ),
], style={'padding': '15px 15px 15px 15px',
          'marginLeft': 'auto', 'marginRight': 'auto', "width": "900px",
          'boxShadow': '0px 0px 5px 5px rgba(204,204,204,0.4)',
          'backgroundColor': 'rgba(0,0, 139, 0.05)'})


# ======== Callbacks ========

@app.callback(Output('text-input', 'value'),
              Input('reset-button', 'n_clicks'))
def clear_form(n_clicks):
    """Empty input textarea"""
    return ""


@app.callback(Output('explainer-obj', 'children'),
              Input('submit-button', 'n_clicks'),
              Input('reset-button', 'n_clicks'),
              State('case-dropdown', 'value'),
              State('num-samples-input', 'value'),
              State('text-input', 'value'))
def generate_explainer_html(submit_n_clicks, reset_n_clicks, case, n_samples, text):
    ctx = dash.callback_context  # Capture callback context to track button clicks
    empty_obj = html.Iframe(
        srcDoc='''<div>Enter input text to see LIME explanations.</div>''',
        width='100%',
        height='100px',
        style={'border': '2px #d3d3d3 solid'},
        hidden=True,
    )
    if not text or "reset" in ctx.triggered[0]["prop_id"]:
        # Return empty iFrame
        obj = empty_obj
    else:
        # Tokenize text using spaCy
        
        exp = explainer(case,
                        text=text,
                        )
        obj = html.Iframe(
            # Javascript is disabled from running in an IFrame for security reasons
            # Static HTML only!!!
            srcDoc=exp.as_html(),
            width='100%',
            height='800px',
            style={'border': '2px #d3d3d3 solid'},
        )
    return obj


if __name__ == '__main__':
    app.run_server(debug=True)
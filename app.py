import io
import base64
import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Scottish Exam Data Dashboard"

# Global variable to hold the uploaded data
global_df = pd.DataFrame()

# App Layout with Tabs
app.layout = html.Div([
    html.H1("Scottish Exam Data Dashboard", style={"textAlign": "center"}),

    # Tabs for different sections
    dcc.Tabs(id="tabs", value="data_exploration", children=[
        dcc.Tab(label="Data Exploration", value="data_exploration"),
    ]),

    # Content will be updated dynamically
    html.Div(id="tabs-content")
])

# Callback for Tabs Content
@app.callback(
    Output("tabs-content", "children"),
    Input("tabs", "value")
)
def render_tab_content(tab_name):
    if tab_name == "data_exploration":
        return html.Div([
            html.H3("Data Exploration"),
            html.P("Upload a CSV file to explore its content."),
            dcc.Upload(
                id="upload-data",
                children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
                style={
                    "width": "100%",
                    "height": "60px",
                    "lineHeight": "60px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "5px",
                    "textAlign": "center",
                    "margin": "10px",
                },
                multiple=False  # Single file upload
            ),
            html.Div(id="file-info"),
            html.Div(id="filter-controls", style={"margin": "20px 0"}),
            html.Div(id="table-container"),
            html.Div(id="stats-container", style={"marginTop": "30px"}),
        ])
    return None

# Callback to handle file upload
@app.callback(
    [Output("file-info", "children"), Output("filter-controls", "children"), Output("table-container", "children")],
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    State("upload-data", "last_modified")
)
def update_table(contents, filename, last_modified):
    global global_df
    if contents is not None:
        # Parse the contents of the uploaded file
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        decoded_file = io.StringIO(decoded.decode("utf-8"))

        try:
            global_df = pd.read_csv(decoded_file)
        except Exception as e:
            return html.Div(["There was an error processing this file: ", str(e)]), None, None

        # Display file info
        file_info = html.Div([
            html.H5(f"File: {filename}"),
            html.P(f"Last Modified: {last_modified}")
        ])

        # Create filter controls
        filter_controls = html.Div([
            html.Div([
                html.Label("Search:"),
                dcc.Input(
                    id="search-input",
                    type="text",
                    placeholder="Search for keywords...",
                    debounce=True,
                    style={"width": "100%"}
                )
            ], style={"marginBottom": "10px"}),
            html.Div([
                html.Label("Filter by Column:"),
                dcc.Dropdown(
                    id="column-filter",
                    options=[{"label": col, "value": col} for col in global_df.columns],
                    placeholder="Select a column to filter...",
                    style={"width": "100%"}
                )
            ])
        ])

        # Display table
        table = dash_table.DataTable(
            id="data-table",
            data=global_df.head(10).to_dict("records"),
            columns=[{"name": col, "id": col} for col in global_df.columns],
            page_size=10,
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"},
            style_cell={"textAlign": "left"},
        )
        return file_info, filter_controls, table

    return None, None, None

# Callback for Filtering and Searching
@app.callback(
    Output("data-table", "data"),
    [Input("search-input", "value"), Input("column-filter", "value")]
)
def update_filtered_table(search_value, column_filter):
    if global_df.empty:
        return []

    filtered_df = global_df.copy()

    # Apply search filter
    if search_value:
        filtered_df = filtered_df[
            filtered_df.apply(lambda row: search_value.lower() in row.to_string().lower(), axis=1)
        ]

    # Apply column filter (optional: could add dropdown for specific values)
    if column_filter:
        filtered_df = filtered_df[[column_filter]]

    return filtered_df.to_dict("records")

# Callback for Descriptive Statistics
@app.callback(
    Output("stats-container", "children"),
    Input("data-table", "data")
)
def update_stats_table(filtered_data):
    if not filtered_data:
        return html.Div(["No data to calculate statistics."])
    
    filtered_df = pd.DataFrame(filtered_data)

    # Compute statistics for numeric columns
    numeric_stats = filtered_df.describe().reset_index()

    # Display statistics
    stats_table = dash_table.DataTable(
        data=numeric_stats.to_dict("records"),
        columns=[{"name": col, "id": col} for col in numeric_stats.columns],
        page_size=10,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"},
        style_cell={"textAlign": "left"},
    )
    return html.Div([
        html.H4("Descriptive Statistics:"),
        stats_table
    ])

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)

import os
import pandas as pd
import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import dash_bootstrap_components as dbc
import numpy as np
import os
import pandas as pd
from dash import dcc, html, Input, Output, State, dash_table
import dash
import plotly.express as px

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True,  external_stylesheets=[dbc.themes.SOLAR])
app.title = "Scottish Examination Analysis Dashboard"

# Directory containing CSV files
DATA_DIR = "output"

# Parse directory for available options
def parse_directory(data_dir):
    directory_info = {}
    for year in os.listdir(data_dir):
        year_path = os.path.join(data_dir, year)
        if os.path.isdir(year_path):
            directory_info[year] = {}
            for level in os.listdir(year_path):
                level_path = os.path.join(year_path, level)
                if os.path.isdir(level_path):
                    directory_info[year][level] = [
                        os.path.splitext(file)[0] for file in os.listdir(level_path) if file.endswith(".csv")
                    ]
    return directory_info

directory_info = parse_directory(DATA_DIR)

# App layout
app.layout = html.Div([
    html.H1("Scottish Examination Analysis Dashboard", style={"textAlign": "center"}),

    # Filters
    html.Div([
        html.Label("Select Year (Optional):"),
        dcc.Dropdown(
            id="year-dropdown",
            options=[{"label": year, "value": year} for year in directory_info.keys()] + [{"label": "All Years", "value": "all"}],
            placeholder="Select a Year",
            style={"width": "90%"}
        ),
        html.Label("Select Level:"),
        dcc.Dropdown(
            id="level-dropdown",
            options=[{"label": level, "value": level} for level in set(l for levels in directory_info.values() for l in levels)],
            placeholder="Select a Level",
            style={"width": "90%"}
        ),
        html.Label("Select Subject:"),
        dcc.Dropdown(
            id="subject-dropdown",
            options=[{"label": subj, "value": subj} for subj in set(s for levels in directory_info.values() for subs in levels.values() for s in subs)],
            placeholder="Select a Subject",
            style={"width": "90%"}
        ),
    ], style={"marginBottom": "20px"}),

    # Tabs for different views
    dcc.Tabs(id="tabs", value="statistics", children=[
        dcc.Tab(label="Statistics", value="statistics"),
        dcc.Tab(label="Intent Trend", value="intent_trend"),
    ]),

    # Content for tabs
    html.Div(id="tabs-content")
])

@app.callback(
    Output("tabs-content", "children"),
    [
        Input("tabs", "value"),
        Input("year-dropdown", "value"),
        Input("level-dropdown", "value"),
        Input("subject-dropdown", "value"),
    ]
)
def render_tab_content(tab_name, selected_year, selected_level, selected_subject):
    if not (selected_level and selected_subject):
        return html.Div(["Please select at least Level and Subject to continue."])

    # Load data
    file_paths = []
    if selected_year and selected_year != "all":
        year_dir = directory_info.get(selected_year, {})
        if selected_level in year_dir and selected_subject in year_dir[selected_level]:
            file_paths.append(os.path.join(DATA_DIR, selected_year, selected_level, f"{selected_subject}.csv"))
    else:
        for year, levels in directory_info.items():
            if selected_level in levels and selected_subject in levels[selected_level]:
                file_paths.append(os.path.join(DATA_DIR, year, selected_level, f"{selected_subject}.csv"))

    if not file_paths:
        return html.Div(["No matching files found."])

    # Combine data
    dataframes = []
    for path in file_paths:
        try:
            df = pd.read_csv(path)
            year = os.path.basename(os.path.dirname(os.path.dirname(path)))  # Extract year
            df["year"] = year
            dataframes.append(df)
        except Exception as e:
            return html.Div([f"Error loading file {path}: {str(e)}"])

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df = combined_df.drop_duplicates()
    combined_df["year"] = pd.to_numeric(combined_df["year"], errors="coerce")

    if tab_name == "statistics":
        stats = combined_df.describe(include="all").reset_index()
        return html.Div([
            html.H4("Statistics Summary"),
            dash_table.DataTable(
                data=stats.to_dict("records"),
                columns=[{"name": col, "id": col} for col in stats.columns],
                style_table={"overflowX": "auto"},
                style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"},
                style_cell={"textAlign": "left"},
            )
        ])

    elif tab_name == "intent_trend":
        if "intent" not in combined_df.columns:
            return html.Div(["The selected data does not contain an 'intent' column."])
    
        # Determine number of files being read
        num_files = len(file_paths)
    
        if num_files == 1:
            # Single CSV: Show intent breakdown
            intent_breakdown = combined_df["intent"].value_counts().reset_index()
            intent_breakdown.columns = ["Intent", "Count"]
    
            fig = px.bar(
                intent_breakdown,
                x="Intent",
                y="Count",
                title="Intent Breakdown for Single Paper",
                labels={"Intent": "Question Intent", "Count": "Number of Questions"},
                text="Count",
            )
            fig.update_traces(textposition="outside")
    
            return html.Div([
                html.H4("Intent Breakdown"),
                dcc.Graph(figure=fig)
            ])
    
        else:
            # Multiple CSVs: Show intent trend
            unique_years = sorted(combined_df["year"].dropna().unique())
            unique_intents = sorted(combined_df["intent"].dropna().unique())
            full_index = pd.MultiIndex.from_product([unique_years, unique_intents], names=["year", "intent"])
    
            grouped = combined_df.groupby(["year", "intent"]).size()
            intent_trend = (
                grouped.reindex(full_index, fill_value=0)
                .reset_index(name="count")
            )
            intent_trend["proportion"] = intent_trend.groupby("year")["count"].transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
    
            toggle = dcc.RadioItems(
                id="yaxis-toggle-local",
                options=[
                    {"label": "Proportion", "value": "proportion"},
                    {"label": "Count", "value": "count"}
                ],
                value="proportion",  # Default choice
                inline=True
            )
    
            yaxis_choice = "proportion"  # Default to proportion if toggle state isn't managed globally
            fig = px.area(
                intent_trend,
                x="year",
                y=yaxis_choice,
                color="intent",
                title=f"{yaxis_choice.capitalize()} of Intent Over Time",
                labels={"year": "Year", yaxis_choice: f"{yaxis_choice.capitalize()} of Questions", "intent": "Intent"},
                line_group="intent",
                custom_data=["count"],
            )
            fig.update_traces(
                hovertemplate=(
                    "<b>Year:</b> %{x}<br>"
                    "<b>Proportion:</b> %{y:.2%}<br>"
                    "<b>Count:</b> %{customdata}<br>"
                )
            )
    
            return html.Div([
                toggle,
                html.H4("Intent Trend Analysis"),
                dcc.Graph(figure=fig)
            ])



def load_csv(selected_year, selected_level, selected_subject):
    if not (selected_level and selected_subject):
        return html.Div(["Please select at least Level and Subject to load file paths."])

    # Find all matching files
    file_paths = []
    if selected_year and selected_year != "all":
        # Specific year
        year_dir = directory_info.get(selected_year, {})
        if selected_level in year_dir and selected_subject in year_dir[selected_level]:
            file_paths.append(os.path.join(DATA_DIR, selected_year, selected_level, f"{selected_subject}.csv"))
    else:
        # Search across all years
        for year, levels in directory_info.items():
            if selected_level in levels and selected_subject in levels[selected_level]:
                file_paths.append(os.path.join(DATA_DIR, year, selected_level, f"{selected_subject}.csv"))

    if not file_paths:
        return html.Div(["No matching files found."])

    # Display file paths as a list
    return html.Div([
        html.H4("Loaded File Paths:"),
        html.Ul([html.Li(path) for path in file_paths])
    ])


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)

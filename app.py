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
from collections import Counter
import re
import dash
import dash_core_components as dcc
import dash_html_components as html
#from wordcloud import WordCloud
from dash_holoniq_wordcloud import DashWordcloud
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import re
import pandas as pd

from io import StringIO


# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True,  external_stylesheets=[dbc.themes.JOURNAL])

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.2/katex.min.css">
        <style>
            .math {
                font-size: 1.2rem; /* Adjust font size for visibility */
                font-weight: bold; /* Make the equation bold */
                color: #003366; /* Add a distinctive color (dark blue in this case) */
                background-color: #f2f4f7; /* Light background for contrast */
                padding: 5px 10px; /* Add padding around the equations */
                border-radius: 5px; /* Rounded corners for aesthetics */
                display: inline-block; /* Ensure equations are inline-block */
                margin: 10px 0; /* Add spacing above and below */
            }
        </style>

        <script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.2/katex.min.js"></script>
        <script>
            function renderKatex() {
                document.querySelectorAll('.math').forEach(function (el) {
                    katex.render(el.textContent, el, {throwOnError: false});
                });
            }
            document.addEventListener('DOMContentLoaded', renderKatex);
            document.addEventListener('DOMSubtreeModified', renderKatex);
        </script>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.title = "Scottish Examination Analysis Dashboard"
server = app.server

# Directory containing CSV files
DATA_DIR = "output"

#{"Year": , "event": ""},
notable_events = [
    {"year": 1918, "event": "Education (Scotland) Act introduced"},
    {"year": 1947, "event": "Education (Scotland) Act passed"},
    {"year": 1962, "event": "Introduction of O-Grades"},
    {"year": 1986, "event": "Standard Grades introduced"},
    {"year": 2013, "event": "Curriculum for Excellence introduced"},
]
           
readability_explanation = html.Div([
    html.H4("Understanding Readability Metrics in the Scottish Education System"),
    html.P("This section provides an explanation of the three readability indices used in the analysis: "
           "Coleman-Liau Index, Flesch-Kincaid Grade Level, and Gunning Fog Index. Each metric offers insight into text complexity, "
           "and you can analyse them either as raw values or as proportional contributions."),

    html.H5("1. Coleman-Liau Index (CLI)"),
    html.Div([
        html.Span("The Coleman-Liau Index estimates readability based on letter count per 100 words and sentence length. The formula is:"),
        html.Div("CLI = 0.0588 × L - 0.296 × S - 15.8", className="math"),
        html.Ul([
            html.Li("L: Average number of letters per 100 words"),
            html.Li("S: Average number of sentences per 100 words"),
        ]),
        html.P("A higher score indicates greater complexity, aligning with the Scottish school level required to understand the text.")
    ]),

    html.H5("2. Flesch-Kincaid Grade Level (FKGL)"),
    html.Div([
        html.Span("The Flesch-Kincaid Grade Level evaluates readability based on words per sentence and syllables per word. The formula is:"),
        html.Div("FKGL = 0.39 × (words/sentence) + 11.8 × (syllables/word) - 15.59", className="math"),
        html.P("A higher score suggests a more advanced education level is required to comprehend the text.")
    ]),

    html.H5("3. Gunning Fog Index (GFI)"),
    html.Div([
        html.Span("The Gunning Fog Index measures readability using sentence length and complex words. The formula is:"),
        html.Div("GFI = 0.4 × [(words/sentences) + 100 × (complex words/words)]", className="math"),
        html.P("A higher score indicates greater text complexity.")
    ]),
    
    html.Hr(),

    html.H5("Readability Score Interpretation in the Scottish Education System"),
    html.Table([
        html.Thead(html.Tr([
            html.Th("Score Range"), html.Th("Coleman-Liau Index (CLI)"), html.Th("Flesch-Kincaid Grade Level (FKGL)"), html.Th("Gunning Fog Index (GFI)"), html.Th("Scottish Education Level")
        ])),
        html.Tbody([
            html.Tr([html.Td("1 - 5"), html.Td("P5 - P7"), html.Td("P5 - P7"), html.Td("P5 - P7"), html.Td("Second Level")]),
            html.Tr([html.Td("6 - 8"), html.Td("S1 - S3"), html.Td("S1 - S3"), html.Td("S1 - S3"), html.Td("Third Level - Fourth Level (BGE)")]),
            html.Tr([html.Td("9 - 10"), html.Td("S4"), html.Td("S4"), html.Td("S4"), html.Td("National 4/5")]),
            html.Tr([html.Td("11 - 12"), html.Td("S5"), html.Td("S5"), html.Td("S5"), html.Td("Higher")]),
            html.Tr([html.Td("13+"), html.Td("S6 and beyond"), html.Td("S6 and beyond"), html.Td("S6 and beyond"), html.Td("Advanced Higher / University")])
        ])
    ], style={'width': '100%', 'border': '1px solid black', 'textAlign': 'center', 'marginTop': '20px'}),

    html.P("By using the readability scores above, you can estimate the education level required to understand the text."),
    
    
])



# Additional toggle explanation (conditionally included only for multiple papers)
toggle_explanation = html.Div([
    html.Hr(),
    html.Table([
        html.Thead(html.Tr([
            html.Th("Metric"), html.Th("What it Measures"), html.Th("Raw Score Interpretation (Scottish System)"), html.Th("Proportional Interpretation")
        ])),
        html.Tbody([
            html.Tr([
                html.Td("Coleman-Liau Index"),
                html.Td("Letter count & sentence length"),
                html.Td("Equivalent to Scottish school level (e.g., S3 for Fourth Level - BGE)"),
                html.Td("Contribution based on letter density")
            ]),
            html.Tr([
                html.Td("Flesch-Kincaid Grade Level"),
                html.Td("Words per sentence & syllables per word"),
                html.Td("Equivalent to Scottish school level (e.g., S6 for Advanced Higher)"),
                html.Td("Impact of syllabic complexity")
            ]),
            html.Tr([
                html.Td("Gunning Fog Index"),
                html.Td("Sentence length & complex words"),
                html.Td("Years of education needed (e.g., S4 for National 4/5)"),
                html.Td("Proportion of long/complex words")
            ])
        ])
    ], style={'width': '100%', 'border': '1px solid black', 'textAlign': 'left', 'marginTop': '20px'}),
    
    html.Hr(),
    
    html.H5("Interpreting the Toggle Options"),
    dcc.Markdown(r"""
        - **Raw Score Mode:** Displays the absolute readability levels directly in terms of the Scottish education system (e.g., S4 for National 4/5).
        - **Proportional Mode:** Shows how each metric contributes to the overall complexity, useful for comparing trends over time.

        By switching between these options, you can gain insights into which factors contribute most to a text’s difficulty over time.
    """),
    
    
])



def parse_directory(data_dir):
    directory_info = {}
    for year in os.listdir(data_dir):
        year_path = os.path.join(data_dir, year)
        if os.path.isdir(year_path):
            directory_info[year] = {}
            for level in os.listdir(year_path):
                level_path = os.path.join(year_path, level)
                if os.path.isdir(level_path):
                    directory_info[year][level] = {}
                    for paper in os.listdir(level_path):  # Handle paper subfolders
                        paper_path = os.path.join(level_path, paper)
                        if os.path.isdir(paper_path):
                            # Extract subject file names (without .csv extensions)
                            subjects = [
                                os.path.splitext(file)[0]
                                for file in os.listdir(paper_path)
                                if file.endswith(".csv")
                            ]
                            directory_info[year][level][paper] = subjects  # Keep "Paper 1", "Paper 2" as keys
    return directory_info


directory_info = parse_directory(DATA_DIR)

#=============================================================================
app.layout = dbc.Container(
    [
        # Store
        dcc.Store(id="selected-year", data=None),
        dcc.Store(id='combined-data', data=None),
          
        # Header
        dbc.Row(
            dbc.Col(
                html.H1(
                    "Scottish Exams - Linguistical Analysis Dashboard",
                    style={"textAlign": "center"}
                ),
                width=12
            ),
        ),
        
        # Main body with dropdowns and tabs
        dbc.Row(
            [
                # Left-hand side: Dropdown filters
                dbc.Col(
                    html.Div(
                        [
                            html.Label("Select Year (Optional):"),
                            dcc.Dropdown(
                                id="year-dropdown",
                                className="customDropdown",
                                options=[
                                    {"label": year, "value": year} for year in directory_info.keys()
                                ] + [{"label": "All Years", "value": "all"}],
                                placeholder="Select a Year",
                                style={"width": "100%"}
                            ),
                            html.Label("Select Level:"),
                            dcc.Dropdown(
                                id="level-dropdown",
                                className="customDropdown",
                                options=[
                                    {"label": level, "value": level}
                                    for level in set(l for levels in directory_info.values() for l in levels)
                                ],
                                placeholder="Select a Level",
                                style={"width": "100%"}
                            ),
                            html.Label("Select Subject:"),
                            dcc.Dropdown(
                                id="subject-dropdown",
                                className="customDropdown",
                                options=[
                                    {"label": subj, "value": subj}
                                    for subj in set(
                                        s
                                        for levels in directory_info.values()
                                        for papers in levels.values()
                                        for subjects in papers.values()
                                        for s in subjects
                                    )
                                ],
                                placeholder="Select a Subject",
                                style={"width": "100%"}
                            ),
                            html.Label("Select Paper: (Optional)"),
                            dcc.Dropdown(
                                id="paper-dropdown",
                                className="customDropdown",
                                options=[
                                    {"label": f"Paper {i}", "value": f"{i}"} for i in range(1, 6)
                                ] + [{"label": "All Papers", "value": "all"}],
                                placeholder="Select a Paper",
                                style={"width": "100%", "marginBottom": "15px"}
                            ),
                            
                            html.H4("Notable Events in Education"),
                            html.P("The numbers correspond to the yellow labels on any plots"),
                            html.Ul([html.Li(f"{l+1} = {event['year']}: {event['event']}") for l, event in enumerate(notable_events)]),
                        ],
                        className="vertical-tabs"
                    ),
                    # width=4,  # Adjust the width as needed
                ),
                
                # Right-hand side: Tabs
                dbc.Col(
                    html.Div(
                        [
                            dcc.Tabs(
                                id="tabs",
                                value="introduction",
                                children=[
                                    dcc.Tab(label="Introduction", value="introduction", className=".custom-tab"),
                                    dcc.Tab(label="Statistics", value="statistics", className=".custom-tab"),
                                    dcc.Tab(label="Intent Trend", value="intent_trend", className=".custom-tab"),
                                    dcc.Tab(label="Compound Sentiment Trend", value="sentiment_trend", className=".custom-tab"),
                                    dcc.Tab(label="Question Length Trend", value="sentence_length_trend", className=".custom-tab"),
                                    dcc.Tab(label="Question Topics", value="topics", className=".custom-tab"),
                                    dcc.Tab(label="Complexity Trends", value="complexity", className=".custom-tab"),
                                ],
                                style={
                                    "marginBottom": "0px",  # Remove gap between tabs and content
                                    "borderBottom": "none",  # Remove the bottom border of tabs
                                },
                            ),
                            html.Div(
                                id="tabs-content",
                                style={
                                    "border": "1px solid #ccc",  # Add border for a cleaner look
                                    "padding": "10px",  # Add padding inside the content area
                                    "marginTop": "0px",  # Ensure no gap between tabs and content
                                },
                            ),
                        ],
                    ),
                ),
            ],
            style={"marginBottom": "20px"},
        ),
    ],
    fluid=True,
    className="dashboard-container"   
)





#=============================================================================


@app.callback(
     [
        Output("tabs-content", "children"),
        Output("combined-data", "data"),  # Store combined data
    ],
    [
        Input("tabs", "value"),
        Input("year-dropdown", "value"),
        Input("level-dropdown", "value"),
        Input("subject-dropdown", "value"),
        Input("paper-dropdown", "value"),
    ]
)
def render_tab_content(tab_name, selected_year, selected_level, selected_subject, selected_paper):
       
    if tab_name == "introduction":
        
        subj_list = []
        for subj in set(
            s
            for levels in directory_info.values()
            for papers in levels.values()
            for subjects in papers.values()
            for s in subjects
        ):
            subj_list.append(subj)
            div_elements = [html.Li(_s) for _s in subj_list]
        
        levl_list = []
        for level in set(l for levels in directory_info.values() for l in levels):
            levl_list.append(level)
        div_elements_levels = [html.Li(_l) for _l in levl_list]
        
        # Collect subject and level counts
        level_counts = {level: 0 for levels in directory_info.values() for level in levels}
        subject_counts = {}
    
        for year, levels in directory_info.items():
            for level, papers in levels.items():
                level_counts[level] += sum(len(subjects) for subjects in papers.values())
                for subjects in papers.values():
                    for subject in subjects:
                        subject_counts[subject] = subject_counts.get(subject, 0) + 1
    
        # Prepare the introduction content
        div_elements = [html.Li(f"{subject}: {count}") for subject, count in subject_counts.items()]
        div_elements_levels = [html.Li(f"{level}: {count}") for level, count in level_counts.items()]
    


        # Data for exam grades progression
        exam_grades_data = [
            {"Era": "Pre-1947", "Grade": "Lower Grade", "Modern Equivalent": "National 4"},
            {"Era": "Pre-1947", "Grade": "Intermediate Grade", "Modern Equivalent": "National 5"},
            {"Era": "Pre-1947", "Grade": "Higher Grade", "Modern Equivalent": "Higher"},
            {"Era": "1947-1962", "Grade": "Lower Grade", "Modern Equivalent": "National 4"},
            {"Era": "1947-1962", "Grade": "Intermediate Grade", "Modern Equivalent": "National 5"},
            {"Era": "1947-1962", "Grade": "Higher Grade", "Modern Equivalent": "Higher"},
            {"Era": "1962-1986", "Grade": "O-Grade", "Modern Equivalent": "National 5"},
            {"Era": "1962-1986", "Grade": "Higher Grade", "Modern Equivalent": "Higher"},
            {"Era": "1962-1986", "Grade": "CSYS", "Modern Equivalent": "Advanced Higher"},
            {"Era": "1986-2013", "Grade": "Foundation Standard Grade", "Modern Equivalent": "National 3"},
            {"Era": "1986-2013", "Grade": "General Standard Grade", "Modern Equivalent": "National 4"},
            {"Era": "1986-2013", "Grade": "Credit Standard Grade", "Modern Equivalent": "National 5"},
            {"Era": "1986-2013", "Grade": "Higher Grade", "Modern Equivalent": "Higher"},
            {"Era": "1986-2013", "Grade": "Advanced Higher", "Modern Equivalent": "Advanced Higher"},
            {"Era": "2013-Present", "Grade": "National 1", "Modern Equivalent": "National 1"},
            {"Era": "2013-Present", "Grade": "National 2", "Modern Equivalent": "National 2"},
            {"Era": "2013-Present", "Grade": "National 3", "Modern Equivalent": "National 3"},
            {"Era": "2013-Present", "Grade": "National 4", "Modern Equivalent": "National 4"},
            {"Era": "2013-Present", "Grade": "National 5", "Modern Equivalent": "National 5"},
            {"Era": "2013-Present", "Grade": "Higher", "Modern Equivalent": "Higher"},
            {"Era": "2013-Present", "Grade": "Advanced Higher", "Modern Equivalent": "Advanced Higher"},
        ]

        return html.Div([
                    html.H4("Welcome to the Scottish Examination Analysis Dashboard"),
                    html.P(
                        """This dashboard allows for one to interact with analysis results on Scottish exam papers.
                               Currently a work in progress, at the time of writing there are a selection of papers from both
                               History and English available at three levels: Higher, National 4, and National 5. 'Old style'
                               levels have been converted to their modern CfE equivalent to simplify analysis for the end user."""
                    ),
                    html.P("At the moment we can analyse the following subjects in the database, with the following count per level:"),
                    html.H5("Subjects:"),
                    html.Ul(div_elements),
                    html.H5("Levels:"),
                    html.Ul(div_elements_levels),

                    html.P("And inspect the following metrics:"),
                    html.Ul([
                        html.Li("Statistics: View detailed statistics about the selected dataset."),
                        html.P(
                            """This tab shows a breakdown of the chosen paper, or papers, using various metrics."""
                        ),
                        html.Li("Intent Trend: Explore trends in question intents over time."),
                        html.P(
                            """Here, intent is what the question is asking of the reader. Is it asking them to
                            analyse something, or compare between two or more options, etc."""
                        ),
                        html.Li("Sentiment Trend: Analyse the sentiment scores of questions over time."),
                        html.P(
                            """Sentiment is a measurement of whether a given text is positive, neutral, or negative."""
                        ),
                        html.Li("Question Length Trend: Analyse the length of questions, in words, over time."),
                        html.P(
                            """The length of a question can be used as a measurement of its complexity."""
                        ),
                        html.Li("Question Topics: Named entities in the text and the number of times they appear."),
                        html.P(
                            """Placeholder formatting, will be explored later"""
                        ),
                        html.Li("Complexity Trends: Various metrics for establishign the compelxity of the text."),
                        html.P(
                            """How complex is the text, and how does this change over time?"""
                        )
                    ]),
                    html.P(
                        """To get started, select the desired year, level, subject, and paper using the dropdowns
                        above. Depending on your selection, the dashboard will update to reflect the
                        relevant data. If you leave the  year selection blank, or select the 'All Years' option,
                        it will analyse data from all of the available years. You MUST select both a subject and 
                        a level, however. If you leave paper blank or select "All papers" it will return every paper 
                        for that particular subject (for example paper 1, paper 2 for Higher History 2024)."""
                    ),
                    
                    html.H3("Scottish Exam Grades Through the Years"),
                        html.P(""""To make the filtering above easier for the user (that's you!) I first had to understand
                                    how levels in secondary schools had changed over the years so that I could effectively
                                    make a system to map any of the old-style levels to their modern equivalents under the current
                                    Curriculum for Excellence model.
                                    This table below shows the progression of Scottish exam grades and their modern equivalents."""),
                        dash_table.DataTable(
                            id='exam-grades-table',
                            columns=[
                                {"name": "Era", "id": "Era"},
                                {"name": "Grade", "id": "Grade"},
                                {"name": "Modern Equivalent", "id": "Modern Equivalent"},
                            ],
                            data=exam_grades_data,
                            style_table={'overflowX': 'auto'},
                            style_cell={
                                'textAlign': 'left',
                                'padding': '10px',
                                'fontFamily': 'Arial, sans-serif',
                                'fontSize': '14px',
                            },
                            style_header={
                                'backgroundColor': 'rgb(230, 230, 230)',
                                'fontWeight': 'bold'
                            },
                            style_data={
                                'border': '1px solid grey',
                            },
                        )
                ]), None

    if not (selected_level and selected_subject):
        return html.Div(["Please select at least Level and Subject to continue."]), None

#=============================================================================
    # Construct file paths
    file_paths = []
    if selected_year and selected_year != "all":
        # Handle a specific year
        year_dir = directory_info.get(selected_year, {})
        if selected_level in year_dir:
            if selected_paper and selected_paper != "all":
                # Load specific paper for the year
                paper_path = os.path.join(DATA_DIR, selected_year, selected_level, selected_paper, f"{selected_subject}.csv")
                if os.path.exists(paper_path):
                    file_paths.append(paper_path)
            else:
                # Load all papers for the year
                for paper in year_dir[selected_level]:
                    paper_path = os.path.join(DATA_DIR, selected_year, selected_level, paper, f"{selected_subject}.csv")
                    if os.path.exists(paper_path):
                        file_paths.append(paper_path)
    else:
        # Handle "all years"
        for year, levels in directory_info.items():
            for level, papers in levels.items():
                if level == selected_level:
                    if selected_paper and selected_paper != "all":
                        # Load specific paper across all years
                        paper_path = os.path.join(DATA_DIR, year, level, selected_paper, f"{selected_subject}.csv")
                        if os.path.exists(paper_path):
                            file_paths.append(paper_path)
                    else:
                        # Load all papers across all years
                        for paper, subjects in papers.items():
                            if selected_subject in subjects:
                                paper_path = os.path.join(DATA_DIR, year, level, paper, f"{selected_subject}.csv")
                                if os.path.exists(paper_path):
                                    file_paths.append(paper_path)

#=============================================================================

    if not file_paths:
        return html.Div(["No matching files found."]), None

    dataframes = []
    for path in file_paths:
        try: 
            df = pd.read_csv(path)
            #df = read_csv_from_s3(S3_BUCKET, path)
            # Extract paper and year
            paper_number = os.path.basename(os.path.dirname(path))  # Full "Paper 1", "Paper 2"
            df["paper"] = paper_number
            year = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))

            df["year"] = int(year)
            dataframes.append(df)
        except Exception as e:
            print(f"Error loading file {path}: {e}")  # Debugging
            return html.Div([f"Error loading file {path}: {e}"]), None
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df = combined_df.drop_duplicates()
   
    # Handle "Statistics" tab
    if tab_name == "statistics":
        num_files = len(file_paths)
        metrics = []
    
        if num_files == 1:
            # Single paper: Calculate metrics
            avg_word_count = combined_df["total_tokens"].mean()
            num_questions = len(combined_df)
            
            # Summary statistics for readability and sentiment
            stats = {}
            cols_to_describe = [
                "coleman_liau", "flesch_kincaid", "gunning_fog",
                "compound_sentiment_score", "total_tokens"
            ]
            stats = df[cols_to_describe].describe()
            
            metrics.append(f"Average Word Count per Question: {avg_word_count:.2f}")
            metrics.append(f"Total Number of Questions: {num_questions}")    
            #count mean std min 25% 50% 75% max
            metrics.append(f"Average (mean) Coleman-Liau score: {stats['coleman_liau']['mean']:.2f}")
            metrics.append(f"Minimum Coleman-Liau score: {stats['coleman_liau']['min']:.2f}")
            metrics.append(f"Maximum Coleman-Liau score: {stats['coleman_liau']['max']:.2f}")
            metrics.append(f"Standard deviation (std) Coleman-Liau score: {stats['coleman_liau']['std']:.2f}")
            metrics.append(f"Average (mean) Flesch-Kincaid score: {stats['flesch_kincaid']['mean']:.2f}")
            metrics.append(f"Minimum Flesch-Kincaid score: {stats['flesch_kincaid']['min']:.2f}")
            metrics.append(f"Maximum Flesch-Kincaid score: {stats['flesch_kincaid']['max']:.2f}")
            metrics.append(f"Standard deviation (std) Flesch-Kincaid score: {stats['flesch_kincaid']['std']:.2f}")
            metrics.append(f"Average (mean) Gunning Fog score: {stats['gunning_fog']['mean']:.2f}")
            metrics.append(f"Minimum Gunning Fog score: {stats['gunning_fog']['min']:.2f}")
            metrics.append(f"Maximum Gunning Fog score: {stats['gunning_fog']['max']:.2f}")
            metrics.append(f"Standard deviation (std) Gunning Fog score: {stats['gunning_fog']['std']:.2f}")
            metrics.append(f"Average (mean) Compound Sentiment score: {stats['compound_sentiment_score']['mean']:.2f}")
            metrics.append(f"Minimum Compound Sentiment score: {stats['compound_sentiment_score']['min']:.2f}")
            metrics.append(f"Maximum Compound Sentiment score: {stats['compound_sentiment_score']['max']:.2f}")
            metrics.append(f"Standard deviation (std) Compound Sentiment score: {stats['compound_sentiment_score']['std']:.2f}")
            metrics.append(f"Average (mean) token count: {stats['total_tokens']['mean']:.2f}")
            metrics.append(f"Minimum token count: {stats['total_tokens']['min']:.2f}")
            metrics.append(f"Maximum token count: {stats['total_tokens']['max']:.2f}")
            metrics.append(f"Standard deviation (std) token count: {stats['total_tokens']['std']:.2f}")
    
            return html.Div([
                html.H4("Statistics for the Selected Paper"),
                html.Ul([html.Li(metric) for metric in metrics])
            ]), combined_df.to_dict("records")
        else:
            # Multiple papers: Aggregate metrics
            avg_word_count = combined_df.groupby("year")["total_tokens"].mean().to_dict()
            total_questions = combined_df.groupby("year").size().to_dict()
            avg_coleman_liau = combined_df.groupby("year")["coleman_liau"].mean().to_dict()
            avg_flesch_kincaid = combined_df.groupby("year")["flesch_kincaid"].mean().to_dict()
            avg_gunning_fog = combined_df.groupby("year")["gunning_fog"].mean().to_dict()
            
            metrics.append("Average Coleman-Liau score by Year:")
            for year, count in avg_coleman_liau.items():
                metrics.append(f"  {year}: {count:.2f}")
                
            metrics.append("Average Flesch-Kincaid score by Year:")
            for year, count in avg_flesch_kincaid.items():
                metrics.append(f"  {year}: {count:.2f}")
    
            metrics.append("Average Gunning Fog score by Year:")
            for year, count in avg_gunning_fog.items():
                metrics.append(f"  {year}: {count:.2f}")

            metrics.append("Average Word Count per Question by Year:")
            for year, count in avg_word_count.items():
                metrics.append(f"  {year}: {count:.2f}")
            
            metrics.append("Total Number of Questions by Year:")
            for year, total in total_questions.items():
                metrics.append(f"  {year}: {total}")
    
            return html.Div([
                html.H4("Summary Statistics for Selected Papers"),
                html.Ul([html.Li(metric) for metric in metrics])
            ]), combined_df.to_dict("records")


    # Handle "Intent Trend" tab
    elif tab_name == "intent_trend":
        if "intent" not in combined_df.columns:
            return html.Div(["The selected data does not contain an 'intent' column."]), combined_df.to_dict("records")

        num_files = len(file_paths)

        if num_files == 1 or combined_df['year'].nunique() == 1:
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
                html.P(
                    """This plot shows the count for each type of question in the given exam paper. 
                    The intent of a question has been decided by training a Bidirectional and Auto-Regressive Transformer
                    model on a hand-crafted dataset of questions and their intent. Although early days, the results
                    show promise (however should not be taken as absolute until more testing has been done)."""
                ),
                dcc.Graph(figure=fig)
            ]), combined_df.to_dict("records")

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
                value="proportion",
                inline=True
            )

            return html.Div([
                toggle,
                html.H4("Intent Trend Analysis"),
                html.P(
                    """This plot shows the count for each type of question in the given exam paper. 
                    There are two options here; proportional, or count. Proportional ensures proportions are calculated 
                    for each year, making trends comparable despite varying question counts between papers, whereas Count 
                    shows the actual count of the question intent types in each paper. 
                    The intent of a question has been calculated by training a Bidirectional and Auto-Regressive Transformer
                    model on a hand-crafted dataset of questions and their intent. Although early days, the results
                    show promise (however should not be taken as absolute until more testing has been done)."""
                ),
                dcc.Graph(id="intent-trend-graph"),
                
                
                dcc.Store(id="intent-trend-data", data=intent_trend.to_dict("records"))
            ]), combined_df.to_dict("records")
        
    elif tab_name == "sentiment_trend":
       if "compound_sentiment_score" not in combined_df.columns:
           return html.Div(["The selected data does not contain a 'compound_sentiment_score' column."]), combined_df.to_dict("records")
   
       num_files = len(file_paths)
   
       if num_files == 1 or combined_df['year'].nunique() == 1:
           # Single CSV: Show trend over the course of questions
           combined_df = combined_df.sort_index()  # Ensure questions are in order
           avg_sentiment = combined_df["compound_sentiment_score"].mean()
   
           fig = px.line(
               combined_df,
               x=combined_df.index,
               y="compound_sentiment_score",
               title="Compound Sentiment Score Trend for Single Paper",
               labels={"index": "Question Index", "compound_sentiment_score": "Compound Sentiment Score"},
           )
           fig.update_traces(mode="lines+markers")
   
           # Add horizontal average line
           fig.add_hline(
               y=avg_sentiment,
               line_dash="dash",
               line_color="red",
               annotation_text=f"Average: {avg_sentiment:.2f}",
               annotation_position="top left",
           )
   
           return html.Div([
               html.H4("Sentiment Trend for Single Paper"),
               html.P(
                   """This plot shows the sentiment score for each question in the given exam paper.
                   Sentiment, in this sense, is a number between -1 and +1, with -1 being negative, +1 being 
                   positive, and 0 being neutral. Although it should be noted that sentiment does take into
                   account the length of a piece of text when calculating the sentiment score, thus, shorter (i.e. more
                   abrupt texts, can adjust the score into the negative. As modern texts do tend to be shorter for
                   accessibility reasons their sentiment score will be lower than their older equivalents. 
                   This doesn't necessarily mean they are negative, just that they are more direct."""
               ),
               dcc.Graph(figure=fig)
           ]), combined_df.to_dict("records")
   
       else:
           # Multiple CSVs: Show trend over time by averaging scores
           sentiment_trend = combined_df.groupby("year")["compound_sentiment_score"].mean().reset_index()
   
           fig = px.line(
               sentiment_trend,
               x="year",
               y="compound_sentiment_score",
               title="Average Compound Sentiment Score Trend Over Time",
               labels={"year": "Year", "compound_sentiment_score": "Average Compound Sentiment Score"},
           )
           fig.update_traces(mode="lines+markers")

           return html.Div([
               html.H4("Sentiment Trend Over Time"),
               html.P(
                   """This plot shows the overall average sentiment score for each question in the given exam papers over time.
                   Sentiment, in this sense, is a number between -1 and +1, with -1 being negative, +1 being 
                   positive, and 0 being neutral. Although it should be noted that sentiment does take into
                   account the length of a piece of text when calculating the sentiment score, thus, shorter (i.e. more
                   abrupt texts, can adjust the score into the negative. As modern texts do tend to be shorter for
                   accessibility reasons their sentiment score will be lower than their older equivalents. 
                   This doesn't necessarily mean they are negative, just that they are more direct."""
               ),
               dcc.Graph(figure=fig)
           ]), combined_df.to_dict("records")
                   
    elif tab_name == "sentence_length_trend":
        if "text" not in combined_df.columns:
            return html.Div(["The selected data does not contain a 'text' column."]), combined_df.to_dict("records")

        num_files = len(file_paths)

        if num_files == 1 or combined_df['year'].nunique() == 1:
            # Single Paper: Sentence length per question
            combined_df["sentence_length"] = combined_df["text"].apply(lambda x: len(x.split()))
            fig = px.line(
                combined_df,
                x=combined_df.index,
                y="sentence_length",
                title="Question Length Per Question for Single Paper",
                labels={"index": "Question Index", "sentence_length": "Question Length (words)"},
            )
            fig.update_traces(mode="lines+markers")

            return html.Div([
                html.H4("Question Length Trend for Single Paper"),
                html.P(
                    """This shows the length of each question (in words) throughout the paper."""
                ),
                dcc.Graph(figure=fig)
            ]), combined_df.to_dict("records")

        else:
            # Multiple Papers: Average sentence length per year
            combined_df["sentence_length"] = combined_df["text"].apply(lambda x: len(str(x).split()))
            sentence_length_trend = combined_df.groupby(["year", "paper"])["sentence_length"].mean().reset_index()

            fig = px.line(
                sentence_length_trend,
                x="year",
                y="sentence_length",
                title="Average Question Length Per Year",
                labels={"year": "Year", "sentence_length": "Average Question Length (words)", "paper": "Paper"},
            )
            fig.update_traces(mode="lines+markers")

            return html.Div([
                html.H4("Average Question Length Per Year"),
                html.P(
                    """This shows the average length of each question (in words) throughout each paper over the years."""
                ),
                dcc.Graph(figure=fig)
            ]), combined_df.to_dict("records")
        
    elif tab_name == "topics":
        
        
        
        all_entities = " ".join(combined_df["named_entities"].dropna())

        # Split the string into words and counts
        entries = re.findall(r'(\w+) \{(\d+)\}', all_entities)
        
        # Create a dictionary to aggregate counts for each word
        word_counts = {}
        for word, count in entries:
            if word in word_counts:
                word_counts[word] += int(count)
            else:
                word_counts[word] = int(count)

        # Convert the dictionary to a 2D array
        result = [[word, count] for word, count in word_counts.items()]
        
        #formatted_entries = [f"Named Entity: {word}: Count: {count}" for word, count in result]
           
        return html.Div([
            html.H4("Named Entities in the filtered results"),
            html.P(
                """This shows the frequency of named entities (people and places) for the filtered
                results. The larger a word appears in the wordcloud below, the more frequent it is.
                Hovering over a word will show the number of times it appears in the selected
                filters. Clicking a word will show a plot below the wordcloud that shows the frequency of
                that word over time."""
            ),
            html.Div([
                html.Div([
                    html.H4("Word Cloud of Named Entities"),
                    DashWordcloud(
                        id="cloud",
                        list=result,
                        width=600, height=400,
                        gridSize=10,
                        # weightFactor=2,
                        # origin=[90, 0],
                        # fontFamily='Sans, serif',
                        color='random-light',
                        backgroundColor='#ffffff',
                        shuffle=True,
                        rotateRatio=0.5,
                        shrinkToFit=False,
                        shape='circle',
                        weightFactor=8,
                        drawOutOfBound=False,
                        hover=True),
                ]),
            ], style={'display': 'flex', 'flex-direction': 'row'}),
            html.Div([
                html.H4("", id="report"),
                dcc.Graph(id="word-usage-plot"),
                ])    
        ]), combined_df.to_dict("records")
    
    elif tab_name == "complexity":
        if "coleman_liau" not in combined_df.columns:
            return html.Div(["The selected data does not contain a 'coleman_liau' column."]), combined_df.to_dict("records")
        if "flesch_kincaid" not in combined_df.columns:
            return html.Div(["The selected data does not contain a 'flesch_kincaid' column."]), combined_df.to_dict("records")
        if "gunning_fog" not in combined_df.columns:
            return html.Div(["The selected data does not contain a 'gunning_fog' column."]), combined_df.to_dict("records")
        
        num_files = len(file_paths)
        
        if num_files == 1 or combined_df['year'].nunique() == 1:
            # Single CSV: Show trend over the course of questions
            combined_df = combined_df.sort_index()  # Ensure questions are in order
        
            avg_coleman_liau = combined_df["coleman_liau"].mean()
            avg_flesch_kincaid = combined_df["flesch_kincaid"].mean()
            avg_gunning_fog = combined_df["gunning_fog"].mean()
        
            fig = px.line(
                combined_df,
                x=combined_df.index,
                y=["coleman_liau", "flesch_kincaid", "gunning_fog"],
                title="Readability Index Trend for Single Paper",
                labels={"index": "Question Index", "value": "Readability Index"},
                template="plotly_white",
            )
            fig.update_traces(mode="lines+markers")
    
            
            fig.add_hline(y=10, line_dash="dash", line_color="black", annotation_text="S4 Threshold", annotation_position="top left")
            fig.add_hline(y=12, line_dash="dash", line_color="black", annotation_text="S5 Threshold", annotation_position="top left")

            
        
            return html.Div([
                html.H4("Readability Trend for Single Paper"),
                html.P(
                    """This plot shows the readability indices for each question in the given exam paper. 
                    The Coleman-Liau Index, Flesch-Kincaid, and Gunning Fog indices are readability tests designed to gauge the complexity of a text.
                    The higher the index, the more difficult the text is to read."""
                ),
                dcc.Graph(figure=fig),
                
                readability_explanation,
                html.Script("renderKatex();"),
                
            ]), combined_df.to_dict("records")
        
        else:
            # Multiple CSVs: Show trend over time by averaging scores
            readability_trend = combined_df.groupby("year")[["coleman_liau", "flesch_kincaid", "gunning_fog"]].mean().reset_index()
        
            fig = px.line(
                readability_trend,
                x="year",
                y=["coleman_liau", "flesch_kincaid", "gunning_fog"],
                title="Average Readability Index Trend Over Time",
                labels={"year": "Year", "value": "Average Readability Index"},
                template="plotly_white",
            )
            fig.update_traces(mode="lines+markers")
            # Add enhancements
            #fig.update_traces(mode="lines+markers", hovertemplate="<b>Year:</b> %{x}<br><b>Value:</b> %{y:.2f}<extra></extra>")
            # fig.update_layout(
            #     font=dict(size=14),
            #     legend=dict(title="Metrics", orientation="h", x=0.5, xanchor="center"),
            #     xaxis=dict(tickangle=45),
            # )
            fig.add_hline(y=10, line_dash="dash", line_color="black", annotation_text="S4 Threshold", annotation_position="top left")
            fig.add_hline(y=12, line_dash="dash", line_color="black", annotation_text="S5 Threshold", annotation_position="top left")

            return html.Div([
                
                html.H4("Readability Trend Over Time"),
                html.P(
                    """This plot shows the readability indices (Coleman-Liau, Flesch-Kincaid, Gunning Fog) over time. """
                ),
                dcc.Graph(figure=fig),

                readability_explanation,   
                html.Script("renderKatex();"),
                
            ]), combined_df.to_dict("records")
       
    

@app.callback(
    Output(component_id='report', component_property='children'),
    Input(component_id='cloud', component_property='click')
)
def update_output_div(item):
    if item == None:
        return ''
    
    return f'Frequency of "{item[0]}" over time'.format(item)

    
@app.callback(
    Output("word-usage-plot", "figure"),
    [
        Input("cloud", "click"),
        State("combined-data", "data"),  # Retrieve stored data (list of dictionaries)
        Input("level-dropdown", "value"),
        Input("subject-dropdown", "value"),
    ]
)
def plot_word_usage(word, combined_data, selected_level, selected_subject):
    if not word or not combined_data:
        return {}

    # Convert the stored data back into a DataFrame
    #combined_df = pd.DataFrame(combined_data)

    combined_df = load_csv(selected_year = None, selected_paper = None, selected_level = selected_level, selected_subject = selected_subject)
    
    def count_word_in_year(named_entities, word):
        if not isinstance(named_entities, str):
            return 0
        # Use regex to find the word followed by its count in curly braces
        matches = re.findall(fr'\b{word}\b\s*\{{(\d+)\}}', named_entities.lower())
        # Convert counts to integers and sum them
        return sum(int(count) for count in matches)
    
    # Group by 'year' and aggregate 'named_entities', ignoring empty rows
    result = (
        combined_df.groupby("year", as_index=False)
        .agg({"named_entities": lambda rows: sum(count_word_in_year(row, word[0].lower()) for row in rows)})
        .rename(columns={"named_entities": "count"})
    )
    
    #print(result)
    
    # Create a line plot using the aggregated DataFrame
    fig = px.line(
        result,  # Use the DataFrame with 'year' and 'count'
        x="year",
        y="count",
        title=f"Occurrences of '{word[0]}' Over Time",
        labels={"year": "Year", "count": "Occurrences"},
    )
    
    # Add markers for better visualization
    fig.update_traces(mode="lines+markers")

    return fig


# Function to generate word cloud image
def generate_wordcloud(word_counts):
    wordcloud = DashWordcloud(
        list=word_counts,
        width=600, height=400,
        gridSize=16,
        # weightFactor=2,
        # origin=[90, 0],
        # fontFamily='Sans, serif',
        color='#f0f0c0',
        backgroundColor='#001f00',
        shuffle=False,
        rotateRatio=0.5,
        shrinkToFit=False,
        shape='circle',
        hover=True)
    return wordcloud


# Callback for dynamic toggle
@app.callback(
    Output("intent-trend-graph", "figure"),
    [
        Input("yaxis-toggle-local", "value"),
        Input("intent-trend-data", "data"),
    ]
)
def update_intent_trend_chart(yaxis_choice, intent_trend_data):
    intent_trend = pd.DataFrame(intent_trend_data)

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
    # Add annotations for notable events
    for l, event in enumerate(notable_events):
        fig.add_annotation(
            x=event["year"],
            y=0,
            text=str(l+1),
            showarrow=True,
            arrowhead=2,
            ax=-50,
            ay=-30,
            bgcolor="yellow"
        )
    return fig


def load_csv(selected_year, selected_level, selected_subject, selected_paper):
    if not (selected_level and selected_subject):
        return html.Div(["Please select at least Level and Subject to load file paths."])

    # Determine file paths
    file_paths = []
    if selected_year and selected_year != "all":
        year_dir = directory_info.get(selected_year, {})
        if selected_level in year_dir:
            if selected_paper and selected_paper != "all":
                file_paths = [
                    os.path.join(DATA_DIR, selected_year, selected_level, selected_paper, f"{selected_subject}.csv")
                ]
            else:
                # Load all papers
                for paper in year_dir[selected_level]:
                    paper_path = os.path.join(DATA_DIR, selected_year, selected_level, paper, f"{selected_subject}.csv")
                    file_paths.append(paper_path)
    else:
        # Load all years, levels, and papers
        for year, levels in directory_info.items():
            for level, papers in levels.items():
                if selected_level == level:
                    for paper, files in papers.items():
                        paper_path = os.path.join(DATA_DIR, year, level, paper, f"{selected_subject}.csv")
                        if (os.path.isfile(paper_path)):
                            file_paths.append(paper_path)
                        
    dataframes = []
    for path in file_paths:
        try:
            df = pd.read_csv(path)
            #print (path)
            # Extract paper and year
            paper_number = os.path.basename(os.path.dirname(path))  # Full "Paper 1", "Paper 2"
            df["paper"] = paper_number
            #print (paper_number)
            year = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))
            #print (year)
            df["year"] = int(year)
            dataframes.append(df)
        except Exception as e:
            print(f"Error loading file {path}: {e}")  # Debugging
            return html.Div([f"Error loading file {path}: {e}"]), None
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df = combined_df.drop_duplicates()       
    return combined_df

# =============================================================================
#     if not file_paths:
#         return html.Div(["No matching files found."])
# 
#     # Display file paths as a list
#     return html.Div([
#         html.H4("Loaded File Paths:"),
#         html.Ul([html.Li(path) for path in file_paths])
#     ])
# =============================================================================


# Run the app
if __name__ == "__main__":
    app.run_server(debug=False)
    #app.run(host='0.0.0.0', port=8050)

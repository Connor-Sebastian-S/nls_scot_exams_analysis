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
server = app.server

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
    
    dcc.Store(id="selected-year", data=None),
    
    html.H1("Scottish Exams - Linguistical Analysis Dashboard", style={"textAlign": "center"}),
    
    html.P("""Please first read the Introduction tab below so as to understand how to use the dashboard - thanks :)
           """, style={"textAlign": "center"}),
           
    

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
    dcc.Tabs(id="tabs", value="introduction", children=[
        dcc.Tab(label="Introduction", value="introduction"),
        dcc.Tab(label="Statistics", value="statistics"),
        dcc.Tab(label="Intent Trend", value="intent_trend"),
        dcc.Tab(label="Compound Sentiment Trend", value="sentiment_trend"),
        dcc.Tab(label="Question Length Trend", value="sentence_length_trend"),

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
       
    if tab_name == "introduction":
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
                               Currently a work in progress, at the time of writing there are a selection of paprs from both
                               History and English available at three levels: Higher, National 4, and National 5. 'Old style'
                               levels have been converted to their modern CfE equivalent to simplify analysis for the end user."""
                    ),
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
                        )
                    ]),
                    html.P(
                        """To get started, select the desired year, level, and subject using the dropdowns
                        above. Depending on your selection, the dashboard will update to reflect the
                        relevant data. If you leave the  year selection blank, or select the 'All Years' option,
                        it will analyse data from all of the available years. You MUST select both a subject and 
                        a level, however."""
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
                ])
    #if tab_name != "introduction":
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
    
    #if tab_name != "introduction":
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
            ])
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
            ])


    # Handle "Intent Trend" tab
    elif tab_name == "intent_trend":
        if "intent" not in combined_df.columns:
            return html.Div(["The selected data does not contain an 'intent' column."])

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
                html.P(
                    """This plot shows the count for each type of question in the given exam paper. 
                    The intent of a question has been decided by training a Bidirectional and Auto-Regressive Transformer
                    model on a hand-crafted dataset of questions and their intent. Although early days, the results
                    show promise (however should not be taken as absolute until more testing has been done)."""
                ),
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
            ])
        
    elif tab_name == "sentiment_trend":
       if "compound_sentiment_score" not in combined_df.columns:
           return html.Div(["The selected data does not contain a 'compound_sentiment_score' column."])
   
       num_files = len(file_paths)
   
       if num_files == 1:
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
           ])
   
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
           ])
                   
    elif tab_name == "sentence_length_trend":
        if "text" not in combined_df.columns:
            return html.Div(["The selected data does not contain a 'text' column."])

        num_files = len(file_paths)

        if num_files == 1:
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
            ])

        else:
            # Multiple Papers: Average sentence length per year
            combined_df["sentence_length"] = combined_df["text"].apply(lambda x: len(x.split()))
            sentence_length_trend = combined_df.groupby("year")["sentence_length"].mean().reset_index()

            fig = px.line(
                sentence_length_trend,
                x="year",
                y="sentence_length",
                title="Average Question Length Per Year",
                labels={"year": "Year", "sentence_length": "Average Question Length (words)"},
            )
            fig.update_traces(mode="lines+markers")

            return html.Div([
                html.H4("Average Question Length Per Year"),
                html.P(
                    """This shows the average length of each question (in words) throughout each paper over the years."""
                ),
                dcc.Graph(figure=fig)
            ])




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
    return fig


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
    app.run_server(debug=False)
    #app.run(host='0.0.0.0', port=8050)

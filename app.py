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
import plotly.graph_objs as go
from sklearn.decomposition import PCA
import seaborn as sns
from io import StringIO
from scipy.stats import linregress
from sklearn.cluster import KMeans


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
    html.H4("Understanding Readability Metrics in Scottish Exams", style={}),
    html.P("This section provides an explanation of the three readability indices used in the analysis: "
           "Coleman-Liau Index, Flesch-Kincaid Grade Level, and Gunning Fog Index."),

    html.H5("1. Coleman-Liau Index (CLI)", style={}),
    html.Div([
        html.Span("The Coleman-Liau Index estimates readability based on letter count per 100 words and sentence length. The formula is:"),
        html.Div("CLI = 0.0588 × L - 0.296 × S - 15.8", className="math"),
        html.Ul([
            html.Li("L: Average number of letters per 100 words"),
            html.Li("S: Average number of sentences per 100 words"),
        ]),
        html.P("A higher score indicates greater complexity, aligning with the Scottish school level required to understand the text.")
    ]),

    html.H5("2. Flesch-Kincaid Grade Level (FKGL)", style={}),
    html.Div([
        html.Span("The Flesch-Kincaid Grade Level evaluates readability based on words per sentence and syllables per word. The formula is:"),
        html.Div("FKGL = 0.39 × (words/sentence) + 11.8 × (syllables/word) - 15.59", className="math"),
        html.P("A higher score suggests a more advanced education level is required to comprehend the text.")
    ]),

    html.H5("3. Gunning Fog Index (GFI)", style={}),
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

    html.P("By using the readability scores above, you can estimate the education level required to understand the text.")   
])

intent_description = html.Div([
    html.H4("Intent Classification Model", style={}),

    html.P("""
        This tab showcases a fine-tuned DistilBERT model, designed to classify the intent behind exam questions.
        The system analyses each question's content and determines its intent category with high accuracy. Potential catagories
        are; "discuss", "describe", "compare", "explain", "argue", "reason", and "other".
    """, style={"lineHeight": "1.6"}),

    html.H5("How It Works", style={}),
    html.Ol([
        html.Li([
            html.Strong("Dataset Preparation:"),
            " A labelled dataset, consisting of questions and their corresponding intent categories, is used to train the model. ",
            "The dataset is split into training and validation sets to ensure robust performance."
        ]),
        html.Li([
            html.Strong("Text Tokenisation:"),
            " Each question is converted into a numerical format using a pre-trained tokenizer from the DistilBERT model. ",
            "This ensures that text data is formatted appropriately for processing by the machine learning model."
        ]),
        html.Li([
            html.Strong("Model Fine-Tuning:"),
            " The DistilBERT model is fine-tuned on the dataset, adapting its understanding of language to the specific task of intent classification. ",
            "Techniques like learning rate optimisation and regular evaluation maximise performance."
        ]),
        html.Li([
            html.Strong("Inference:"),
            " After training, the model can analyse unseen questions and predict their intent with precision."
        ])
    ]),

    html.H5("What This Model Does", style={}),
    html.P("""
        The model assigns an intent category to questions.
    """),

    html.H5("About DistilBERT", style={}),
    html.P("""
        DistilBERT is a lightweight version of the BERT model, designed to retain the language understanding capabilities of its predecessor while being faster and more efficient. 
        It is ideal for applications requiring limited computational resources or fast response times.
    """),

])
          
sentiment_description = html.Div([
    html.H4("Sentiment Analysis with NLTK", style={}),

    html.P("""
        Sentiment analysis is the process of evaluating the emotional tone of a piece of text. It helps to identify whether the text 
        conveys a positive, negative, or neutral sentiment. In this application, we used the NLTK library's SentimentIntensityAnalyzer 
        to perform this analysis.
    """, style={"lineHeight": "1.6"}),

    html.H5("What Is Sentiment?", style={"color": "#34495e"}),
    html.P("""
        Sentiment refers to the attitude, emotion, or tone conveyed by a piece of text. It can help us understand the writer's 
        perspective or the general mood of the content. For instance:
        """, style={"lineHeight": "1.6"}),
    html.Ul([
        html.Li("A positive sentiment might suggest happiness, satisfaction, or approval."),
        html.Li("A negative sentiment might indicate frustration, criticism, or sadness."),
        html.Li("A neutral sentiment represents text that is factual or without strong emotional undertones."),
    ]),

    html.H5("How It Works", style={"color": "#34495e"}),
    html.Ol([
        html.Li([
            html.Strong("Tokenisation:"),
            " The input text is split into smaller components (words, phrases) to analyse its structure and meaning."
        ]),
        html.Li([
            html.Strong("Scoring Sentiment:"),
            " The Sentiment Intensity Analyser assigns scores to the text based on the emotional impact of words and phrases. ",
            "It uses a lexicon of words with predefined sentiment values."
        ]),
        html.Li([
            html.Strong("Combining Scores:"),
            " The analyser computes an overall sentiment score by combining the individual word scores and taking into account contextual factors."
        ])
    ]),

    html.H5("Understanding Sentiment Analysis Results", style={"color": "#34495e"}),
    html.P("""
        The Sentiment Intensity outputs four key scores that provide insights into the emotional tone of the text:
    """, style={"lineHeight": "1.6"}),
    html.Ul([
        html.Li([html.Strong("Positive (pos): "), "A score representing the proportion of positive sentiment in the text."]),
        html.Li([html.Strong("Negative (neg): "), "A score indicating the proportion of negative sentiment."]),
        html.Li([html.Strong("Neutral (neu): "), "A score reflecting the proportion of neutral sentiment."]),
        html.Li([html.Strong("Compound: "), "A single score between -1 and 1 that summarises the overall sentiment of the text, where -1 is very negative, 1 is very positive, and 0 is neutral."]),
    ]),

    html.H5("Interpreting the Results", style={"color": "#34495e"}),
    html.P("""
        The compound score is the most useful for determining the overall sentiment. For example:
        """, style={"lineHeight": "1.6"}),
    html.Ul([
        html.Li("A compound score close to 1 indicates strong positive sentiment."),
        html.Li("A compound score near -1 indicates strong negative sentiment."),
        html.Li("A compound score around 0 suggests a neutral sentiment."),
    ]),
])

named_entity_description = html.Div([
    html.H4("Named Entities and Word Clouds", style={}),

    html.H5("What Is a Named Entity?"),
    html.P("""
        A named entity is a specific type of information that can be identified in text, such as the name of a person, 
        location, organisation, date, or other key elements. For example, in the sentence 'The Battle of Hastings took place in 1066,' 
        the named entities are 'The Battle of Hastings' (an event) and '1066' (a date).
    """, style={"lineHeight": "1.6"}),

    html.H5("How Are Named Entities Established?"),
    html.P("""
        Named Entity Recognition (NER) is a natural language processing technique used to identify and classify entities within text. 
        This is typically achieved using pre-trained language models or algorithms that analyse patterns in language and context 
        to detect entities. For instance:
    """, style={"lineHeight": "1.6"}),
    html.Ul([
        html.Li("Using context to determine if a word is a name or title."),
        html.Li("Matching terms against predefined categories like locations, dates, or organisations."),
        html.Li("Utilising probabilistic models trained on large datasets to identify entities in unseen text."),
    ]),

    html.H5("What Is a Word Cloud?"),
    html.P("""
        A word cloud is a visual representation of text data, where the size of each word corresponds to its frequency or relevance 
        within the dataset. In this application, the word cloud displays the named entities extracted from each question in the exam paper(s).
    """, style={"lineHeight": "1.6"}),

    html.H5("What Does the Word Cloud Show?"),
    html.P("""
        The word cloud highlights the named entities present in the exam paper or papers you upload. Larger words indicate entities 
        that appear more frequently across the questions. This allows you to quickly identify prominent topics, locations, names, 
        or themes within the dataset.
    """, style={"lineHeight": "1.6"}),

    html.H5("Tracking Word Usage Over Time"),
    html.P("""
        In addition to the word cloud, you can select a specific word (entity) from it to explore its usage across all exam papers 
        in the dataset, spanning multiple years. This is shown in a line plot, which provides insight into trends or changes in 
        the importance or focus on particular topics over time.
    """, style={"lineHeight": "1.6"}),

    html.P("""
        This feature provides valuable insights into the evolution of focus areas, topics, and themes in exam papers, helping you 
        identify patterns and trends effectively.
    """, style={"lineHeight": "1.6"})
])
   
question_length_description = html.Div([
    html.H4("Question Length as a Metric"),

    html.H5("What Is Question Length?"),
    html.P("""
        Question length refers to the number of words or tokens used in a question. 
        It is a simple yet powerful way to quantify the complexity or depth of a question.
    """, style={"lineHeight": "1.6"}),

    html.H5("Why Is Question Length Important?"),
    html.P("""
        The length of a question can reveal key insights about its structure and purpose. 
        Longer questions often indicate more complex ideas, requiring greater cognitive effort to understand and answer. 
        Shorter questions, on the other hand, may focus on straightforward or factual queries. 
        Analysing question length is valuable for several reasons:
    """, style={"lineHeight": "1.6"}),
    html.Ul([
        html.Li("It provides an indirect measure of question complexity."),
        html.Li("It can help assess the readability of exams by identifying overly lengthy or overly brief questions."),
        html.Li("Tracking average question length over time can highlight changes in the style or focus of exam papers.")
    ]),

    html.H5("How Is Question Length Calculated?"),
    html.P("""
        Question length is typically calculated as the total number of tokens in a question. Tokens are the smallest units of text, 
        such as words or punctuation, identified during text processing. By aggregating and analysing these lengths across a dataset, 
        we can gain insights into question patterns and trends.
    """, style={"lineHeight": "1.6"}),

    html.H5("Using Question Length in This Application"),
    html.P("""
        In this dashboard, question length is used to:
    """, style={"lineHeight": "1.6"}),
    html.Ul([
        html.Li("Calculate average word counts for individual papers or across years."),
        html.Li("Compare question lengths between different exam papers."),
        html.Li("Visualise trends in question length over time to track changes in exam design."),
    ]),

    html.P("""
        By incorporating question length as a metric, this tool provides an additional layer of understanding to educators, 
        helping them evaluate the balance, clarity, and fairness of their assessments.
    """, style={"lineHeight": "1.6"})
])

wcs_visualisation = html.Div([
    html.H4("Step-by-Step Breakdown of the Visualisation"),
    html.P("This section outlines the steps involved in preparing and visualising the weighted composite score over time, showing how readability, sentiment, and document size influence complexity."),
    
    html.H5("1. Data Preparation"),
    html.P("Before creating the plot, we must ensure that the data is properly prepared. The goal of the plot is to show the weighted composite score over time, aggregated by year. "
           "We start with a dataset containing various metrics for different papers, including readability scores (e.g., Coleman-Liau Index, Flesch-Kincaid Grade Level) "
           "and token counts (e.g., positive, negative, neutral tokens)."),
    
    html.H5("2. Normalisation"),
    html.P("To ensure that the metrics are on a comparable scale, we apply min-max normalisation to each column. "
           "This step is important because the metrics may differ in magnitude, and normalisation brings everything into the same range, typically [0, 1]."),

    html.H5("3. Weighting the Metrics"),
    html.P("Each of the metrics is given a weight based on its perceived importance. The weighted composite score is calculated by summing the weighted, "
           "normalised values for each metric. The weights are as follows:"),
    html.Ul([
        html.Li("Coleman-Liau: 0.25"),
        html.Li("Flesch-Kincaid: 0.25"),
        html.Li("Gunning-Fog: 0.25"),
        html.Li("Total Tokens: 0.15"),
        html.Li("Negative Tokens: 0.025"),
        html.Li("Positive Tokens: 0.025"),
        html.Li("Neutral Tokens: 0.025"),
        html.Li("Compound Sentiment Score: 0.025"),
    ]),
    html.P("The reasoning behind these weights is based on the relative importance of each metric in determining the complexity and sentiment of the text. "
           "The readability metrics (Coleman-Liau, Flesch-Kincaid, Gunning-Fog) are assigned equal weight because they each offer a different perspective on text complexity. "
           "Sentiment metrics contribute less due to their smaller role in overall complexity."),
    
    html.H5("4. Calculating the Composite Score"),
    html.P("Once the data is normalised and weighted, the composite score is calculated as a weighted sum of the normalised values for each metric."),
    html.P("This gives us a single score representing the combined impact of all metrics for each paper."),
    
    html.H5("5. Aggregating the Data"),
    html.P("The next step is to aggregate the data by year to show how the composite score changes over time. "
           "We group the dataset by year and calculate the average composite score for each year."),


    html.H5("6. Summary of the Steps"),
    html.P("To summarise the process for creating the plot:"),
    html.Ul([
        html.Li("Data Normalisation: Apply min-max normalisation to scale each metric between 0 and 1."),
        html.Li("Weight Assignment: Assign relative weights to each metric based on its perceived importance."),
        html.Li("Composite Score Calculation: Compute a weighted sum of the normalised values to get the composite score."),
        html.Li("Aggregation by Year: Calculate the average composite score for each year."),
        html.Li("Plotting: Visualise the trends in composite score over time using a line plot."),
    ]),
    html.P("This approach allows you to understand how different factors, such as readability and sentiment, influence the overall complexity of papers over time. "
           "It also highlights trends and relationships between these factors."),
    
    html.Hr(),
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
                    style={"textAlign": "center", }
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
                                style={"width": "100%", }
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
                                style={"width": "100%", }
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
                                style={"width": "100%", }
                            ),
                            html.Label("Select Paper: (Optional)"),
                            dcc.Dropdown(
                                id="paper-dropdown",
                                className="customDropdown",
                                options=[
                                    {"label": f"Paper {i}", "value": f"{i}"} for i in range(1, 6)
                                ] + [{"label": "All Papers", "value": "all"}],
                                placeholder="Select a Paper",
                                style={"width": "100%", "marginBottom": "15px", }
                            ),
                            
                            html.H4("Notable Events in Education", style = { }),
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
                                className="tab-container",
                                children=[
                                    dcc.Tab(label="Intro", value="introduction"),
                                    dcc.Tab(label="Stats", value="statistics"),
                                    dcc.Tab(label="Intent", value="intent_trend"),
                                    dcc.Tab(label="Sentiment", value="sentiment_trend"),
                                    dcc.Tab(label="Question Length", value="sentence_length_trend"),
                                    dcc.Tab(label="Topics", value="topics"),
                                    dcc.Tab(label="Complexity", value="complexity"),
                                    dcc.Tab(label="Questions", value="questions"),
                                    dcc.Tab(label="Comparative", value="comparative"),
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
    
    if selected_year and selected_level and selected_paper and selected_subject and selected_paper != "all" and selected_year != "all":
        # Handle specific year, level, paper, and subject
        paper_path = os.path.join(DATA_DIR, selected_year, selected_level, selected_paper, f"{selected_subject}.csv")
        if os.path.exists(paper_path):
            file_paths.append(paper_path)
        
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
        print(num_files)
        metrics = []
    
        if (selected_year and selected_level and selected_subject and selected_paper):
            # Single paper: Display descriptive statistics as text
            avg_word_count = combined_df["total_tokens"].mean()
            num_questions = len(combined_df)
    
            # Summary statistics for readability and sentiment
            stats = combined_df[[
                "coleman_liau", "flesch_kincaid", "gunning_fog",
                "compound_sentiment_score", "total_tokens"
            ]].describe()
    
            return html.Div([
                html.H4("Statistics for the Selected Paper", style={"color": "#2c3e50"}),
                html.P(f"Average Word Count per Question: {avg_word_count:.2f}"),
                html.P(f"Total Number of Questions: {num_questions}"),
                html.H5("Readability and Sentiment Statistics", style={"color": "#34495e"}),
                html.Ul([
                    html.Li(f"Average Coleman-Liau Score: {stats['coleman_liau']['mean']:.2f}"),
                    html.Li(f"Average Flesch-Kincaid Score: {stats['flesch_kincaid']['mean']:.2f}"),
                    html.Li(f"Average Gunning Fog Score: {stats['gunning_fog']['mean']:.2f}"),
                    html.Li(f"Average Compound Sentiment Score: {stats['compound_sentiment_score']['mean']:.2f}"),
                    html.Li(f"Average Token Count: {stats['total_tokens']['mean']:.2f}")
                ], style={"lineHeight": "1.8"})
            ]), combined_df.to_dict("records")
    
        else:
            # Multiple papers: Create plots for aggregated metrics
            avg_word_count = combined_df.groupby("year")["total_tokens"].mean().reset_index()
            total_questions = combined_df.groupby("year").size().reset_index(name="count")
            avg_coleman_liau = combined_df.groupby("year")["coleman_liau"].mean().reset_index()
            avg_flesch_kincaid = combined_df.groupby("year")["flesch_kincaid"].mean().reset_index()
            avg_gunning_fog = combined_df.groupby("year")["gunning_fog"].mean().reset_index()
    
            # Create individual plots
            plots = [
                dcc.Graph(
                    id="coleman-liau-plot",
                    figure={
                        "data": [go.Scatter(
                            x=avg_coleman_liau["year"],
                            y=avg_coleman_liau["coleman_liau"],
                            mode="lines+markers",
                            name="Coleman-Liau Score"
                        )],
                        "layout": go.Layout(
                            title="Coleman-Liau Score by Year",
                            xaxis={"title": "Year"},
                            yaxis={"title": "Coleman-Liau Score"},
                            template="plotly_white"
                        )
                    }
                ),
                dcc.Graph(
                    id="flesch-kincaid-plot",
                    figure={
                        "data": [go.Scatter(
                            x=avg_flesch_kincaid["year"],
                            y=avg_flesch_kincaid["flesch_kincaid"],
                            mode="lines+markers",
                            name="Flesch-Kincaid Score"
                        )],
                        "layout": go.Layout(
                            title="Flesch-Kincaid Score by Year",
                            xaxis={"title": "Year"},
                            yaxis={"title": "Flesch-Kincaid Score"},
                            template="plotly_white"
                        )
                    }
                ),
                dcc.Graph(
                    id="gunning-fog-plot",
                    figure={
                        "data": [go.Scatter(
                            x=avg_gunning_fog["year"],
                            y=avg_gunning_fog["gunning_fog"],
                            mode="lines+markers",
                            name="Gunning Fog Score"
                        )],
                        "layout": go.Layout(
                            title="Gunning Fog Score by Year",
                            xaxis={"title": "Year"},
                            yaxis={"title": "Gunning Fog Score"},
                            template="plotly_white"
                        )
                    }
                ),
                dcc.Graph(
                    id="word-count-plot",
                    figure={
                        "data": [go.Scatter(
                            x=avg_word_count["year"],
                            y=avg_word_count["total_tokens"],
                            mode="lines+markers",
                            name="Average Word Count"
                        )],
                        "layout": go.Layout(
                            title="Average Word Count per Question by Year",
                            xaxis={"title": "Year"},
                            yaxis={"title": "Average Word Count"},
                            template="plotly_white"
                        )
                    }
                ),
                dcc.Graph(
                    id="total-questions-plot",
                    figure={
                        "data": [go.Bar(
                            x=total_questions["year"],
                            y=total_questions["count"],
                            name="Total Questions"
                        )],
                        "layout": go.Layout(
                            title="Total Number of Questions by Year",
                            xaxis={"title": "Year"},
                            yaxis={"title": "Total Questions"},
                            template="plotly_white"
                        )
                    }
                )
            ]
    
            return html.Div([
                html.H4("Summary Statistics for Selected Papers", style={"color": "#2c3e50"}),
                html.Div(plots, style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "20px"})
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
                title="Intent Breakdown for Single Year",
                labels={"Intent": "Question Intent", "Count": "Number of Questions"},
                text="Count",
            )
            fig.update_traces(textposition="outside")

            return html.Div([
                html.H4("Intent Breakdown"),
                html.P(
                    """This plot shows the count for each type of question in the given exam paper. """
                ),
                dcc.Graph(figure=fig),
                intent_description
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
                    shows the actual count of the question intent types in each paper."""
                ),
                dcc.Graph(id="intent-trend-graph"),
                
                
                dcc.Store(id="intent-trend-data", data=intent_trend.to_dict("records")),
                
                intent_description
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
               title="Compound Sentiment Score Trend for Single Year",
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
               html.H4("Sentiment Trend for Single Year"),
               html.P(
                   """This plot shows the sentiment score for each question in the given exam paper."""
               ),
               dcc.Graph(figure=fig),
               sentiment_description
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
                   """This plot shows the overall average sentiment score for each question in the given exam papers over time."""
               ),
               dcc.Graph(figure=fig),
               sentiment_description
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
                title="Question Length Per Question for Single Year",
                labels={"index": "Question Index", "sentence_length": "Question Length (words)"},
            )
            fig.update_traces(mode="lines+markers")

            return html.Div([
                html.H4("Question Length Trend for Single Year"),
                html.P(
                    """This shows the length of each question (in words) throughout the paper."""
                ),
                dcc.Graph(figure=fig),
                question_length_description,
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
                dcc.Graph(figure=fig),
                question_length_description,
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
                dcc.Graph(id="word-usage-plot", style={"display": "none"}),
                named_entity_description,
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
            
            # Calculate the average of the three readability indices for each row
            combined_df['average'] = combined_df[["coleman_liau", "flesch_kincaid", "gunning_fog"]].mean(axis=1)

        
            fig = px.line(
                combined_df,
                x=combined_df.index,
                y=["coleman_liau", "flesch_kincaid", "gunning_fog", "average"],  # Include the average in the plot
                title="Readability Index Trend for Single Year",
                labels={"index": "Question Index", "value": "Readability Index"},
                template="plotly_white",
            )
            fig.update_traces(mode="lines+markers")
    
            
            fig.add_hline(y=10, line_dash="dash", line_color="black", annotation_text="S4 Threshold", annotation_position="top left")
            fig.add_hline(y=12, line_dash="dash", line_color="black", annotation_text="S5 Threshold", annotation_position="top left")

            
        
            return html.Div([
                html.H4("Readability Trend for Single Year"),
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
        
            # Calculate the average of the three readability indices for each year
            readability_trend['average'] = readability_trend[["coleman_liau", "flesch_kincaid", "gunning_fog"]].mean(axis=1)
            
            # Create the plot with the three existing indices and the new average line
            fig = px.line(
                readability_trend,
                x="year",
                y=["coleman_liau", "flesch_kincaid", "gunning_fog", "average"],  # Include the average in the plot
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
        
    elif tab_name == "questions":
        # Ensure all dropdowns are specified and none is "All"
        if not (selected_year and selected_level and selected_subject and selected_paper):
            return html.Div(["Please specify Year, Level, Subject, and Paper to view questions."]), None
        if selected_year == "all" or selected_level == "all" or selected_subject == "all" or selected_paper == "all":
            return html.Div(["Please refine your selection. The 'Questions' tab requires a specific paper."]), None
    
        # Construct the file path for the specified paper
        paper_path = os.path.join(DATA_DIR, selected_year, selected_level, selected_paper, f"{selected_subject}.csv")
        if not os.path.exists(paper_path):
            return html.Div(["The specified paper could not be found. Please check your selections."]), None
    
        # Load the paper data into a DataFrame
        try:
            df = pd.read_csv(paper_path)
    
            # Display a table of questions and metadata
            columns_to_display = [
                {"name": "Question Text", "id": "text"},
            ]
            data_to_display = df[["text"]].to_dict("records")
    
            # Create an HTML table for the questions data
            table_header = [
                html.Thead(html.Tr([
                    html.Th("Question Text")
                ]))
            ]
            
            # Table rows
            table_rows = [
                html.Tr([
                    html.Td(str(_+1) + ': ' + row["text"])
                ])
                for _, row in df.iterrows()
            ]
            
            table_body = html.Tbody(table_rows)

            # Return the table as part of the "Questions" tab content
            return html.Div([
                html.H4("Questions for the Selected Paper"),
                html.P("Below is each question in the selected paper:"),
                html.Table(table_header + [table_body], style={"width": "100%", "border": "1px solid black"})
            ]), None
    
        except Exception as e:
            return html.Div([f"Error loading file: {e}"]), None
        
    elif tab_name == "comparative":
        print("comparative")
        return html.Div([
            html.H4("Comparative Analysis"),
            html.P(
                """This plot visualizes the evolution of the composite scores over time to 
                identify patterns, significant changes, or correlations in complexity metrics."""
            ),
            dcc.Graph(id="ca_report", style={"display": "none"}),
            wcs_visualisation,
            html.Script("renderKatex();"),
            dcc.Graph(id="ca_linreg", style={"display": "none"}),
            #dcc.Graph(id="ca_clusters", style={"display": "none"}),
        ]), combined_df.to_dict("records")

                
@app.callback(
    Output("ca_report", "figure"),
    Output("ca_report", "style"),  
    Output("ca_linreg", "figure"),
    Output("ca_linreg", "style"),
    #Output("ca_clusters", "figure"),
    #Output("ca_clusters", "style"),
    [
    Input("level-dropdown", "value"),
    Input("subject-dropdown", "value"),
    Input("tabs", "value"),
    Input("paper-dropdown", "value"),
    ]
)
def get_papers_in_subject(selected_level, selected_subject, tab_name, selected_paper):
    if tab_name == "comparative":
        print(selected_level)
        print(selected_subject)

        combined_df = load_csv(selected_year = None, selected_paper = selected_paper, selected_level = selected_level, selected_subject = selected_subject)
        
        metrics_columns = [
            'coleman_liau',
            'flesch_kincaid',
            'gunning_fog',
            'total_tokens',
            'negative_tokens',
            'positive_tokens',
            'neutral_tokens',
            'compound_sentiment_score'
        ]
        
        # Calculate min and max for each metric
        min_values = combined_df[metrics_columns].min()
        max_values = combined_df[metrics_columns].max()
        
        # Normalize each column (metric) using min-max scaling
        normalized_df = (combined_df[metrics_columns] - min_values) / (max_values - min_values)
        
        # Define the weights for each metric
        weights = {
            'coleman_liau': 0.25,
            'flesch_kincaid': 0.25,
            'gunning_fog': 0.25,
            'total_tokens': 0.15,
            'negative_tokens': 0.025,
            'positive_tokens': 0.025,
            'neutral_tokens': 0.025,
            'compound_sentiment_score': 0.025
        }
        
        # Convert weights to a numpy array to facilitate multiplication
        weights_array = np.array([weights[col] for col in metrics_columns])
        
        # Apply the weights to each normalized column
        weighted_normalized_df = normalized_df * weights_array
        
        # Calculate the composite score by summing the weighted values for each row
        combined_df['composite_score'] = weighted_normalized_df.sum(axis=1)
        
        # Display the composite scores for verification
        #print("DataFrame with Composite Scores:\n", combined_df[['year', 'composite_score']].head())
        
        # Group by 'year' and calculate the average composite score for each year
        df_grouped_by_year = combined_df.groupby('year')['composite_score'].mean()
        
        # Display the average composite scores per year
        #print("Average Composite Scores by Year:\n", df_grouped_by_year)
        
        # Group by 'year' to calculate the mean of each metric
        df_grouped_by_year_metrics = combined_df.groupby('year')[metrics_columns].mean()
        
        # Calculate the correlation between the metrics over time
        correlation_matrix = df_grouped_by_year_metrics.corr()
        
        # Display the correlation matrix
        #print("Correlation Matrix:\n", correlation_matrix)
        
        # Calculate the weighted composite score for each year using the weights
        weighted_composite_score = (df_grouped_by_year_metrics * weights_array).sum(axis=1)
        
        # Display the weighted composite score by year
        #print("Weighted Composite Score by Year:\n", weighted_composite_score)

        # Convert the Series to a DataFrame
        weighted_composite_score_df = weighted_composite_score.reset_index()
        weighted_composite_score_df.columns = ['year', 'score']  # Rename columns for clarity

        # Plot using Plotly Express
        fig = px.line(
            weighted_composite_score_df,
            x='year',  # x-axis: year
            y='score',  # y-axis: score
            title="Weighted Composite Score Over Time",
            labels={"year": "Year", "score": "Weighted Composite Score"},
        )
        # Add markers for better visualization
        fig.update_traces(mode="lines+markers")
        
        #Compare Metrics
        #Analyze individual metrics alongside the composite scores to understand which metrics drive changes over time.
        #Use heatmaps for the correlation matrix to visualize relationships.
        # Heatmap for correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Between Metrics Over Time')
        plt.show()
        
        years = df_grouped_by_year.index.astype(int)
        values = df_grouped_by_year.values
        slope, intercept, r_value, p_value, std_err = linregress(years, df_grouped_by_year.values)
        print(f"Trend Line: y = {slope:.4f}x + {intercept:.4f} (R²={r_value**2:.4f}, p={p_value:.4e})")
        regression_line = slope * years + intercept

        # Create the plot
        lin_reg_fig = go.Figure()
        
        # Add scatter plot of the original data
        lin_reg_fig.add_trace(go.Scatter(
            x=years,
            y=values.flatten(),
            mode='markers',
            name='Original Data',
            marker=dict(size=8, color='blue')
        ))
        
        # Add regression line
        lin_reg_fig.add_trace(go.Scatter(
            x=years,
            y=regression_line,
            mode='lines',
            name='Regression Line',
            line=dict(color='red', width=2)
        ))
        
        # Add annotations for regression statistics
        lin_reg_fig.add_annotation(
            x=years[len(years) // 2],
            y=max(values.flatten()),
            text=f"Slope: {slope:.4f}<br>Intercept: {intercept:.4f}<br>R²: {r_value**2:.4f}",
            showarrow=False,
            font=dict(size=12, color="black"),
            align="left"
        )
        
        # Update layout
        lin_reg_fig.update_layout(
            title="Linear Regression Visualisation",
            xaxis_title="Year",
            yaxis_title="Value",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )


        return fig, {"display": "block"}, lin_reg_fig, {"display": "block"}

    #return None, {"display": "none"}
        
        


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
    Output("word-usage-plot", "style"),
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

    return fig, {"display": "block"}


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
    if selected_year and selected_level and selected_paper and selected_subject and selected_paper != "all" and selected_year != "all":
        # Handle specific year, level, paper, and subject
        paper_path = os.path.join(DATA_DIR, selected_year, selected_level, selected_paper, f"{selected_subject}.csv")
        if os.path.exists(paper_path):
            file_paths.append(paper_path)
        
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

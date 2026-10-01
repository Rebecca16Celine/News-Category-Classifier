"""
News Category Classifier - Main Application
"""

from nicegui import ui
import plotly.graph_objects as go

from model import NewsClassifierModel
from config import *


# ============================================================
# LOAD MODEL
# ============================================================

try:
    classifier = NewsClassifierModel()
    MODEL_LOADED = True
    print("Model loaded successfully!")
except Exception as e:
    MODEL_LOADED = False
    print(f"Warning: Model not loaded - {e}")


# ============================================================
# GLOBAL STATE
# ============================================================

recent_predictions = []
current_page = 'live'
main_container = None

# Container used to refresh the recent predictions section
recent_predictions_container = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_style(cat):
    """Get style information for a category."""
    return CATEGORY_STYLES.get(
        cat,
        {
            'color': '#666',
            'emoji': '📰',
            'display': cat
        }
    )


def create_text_input(
    placeholder='Type or paste news headline here...'
):
    """
    Create the standard textarea used throughout the app.
    """

    return (
        ui.textarea(
            placeholder=placeholder
        )
        .classes('w-full text-base news-textarea')
        .style(
            'height: 120px; '
            'min-height: 120px; '
            'max-height: 120px; '
            'background-color: #FFFFFF; '
            'border-radius: 12px;'
        )
        .props('outlined rounded')
    )


# ============================================================
# HEADER
# ============================================================

def create_header():
    """Create navigation header."""

    with ui.header():

        with ui.row().classes(
            'w-full items-center justify-between px-4 py-2'
        ):

            # Application name
            with ui.row().classes(
                'items-center gap-2 cursor-pointer'
            ).on(
                'click',
                lambda: navigate_to('live')
            ):

                ui.icon(
                    'article'
                ).classes(
                    'text-white text-2xl'
                )

                ui.label(
                    'NewsClassifier'
                ).classes(
                    'text-white text-lg font-bold'
                )

            # Navigation
            with ui.row().classes('gap-1'):

                for page, label, icon in [
                    ('live', 'Live Demo', 'psychology'),
                    ('eval', 'Evaluation', 'analytics'),
                    ('pipeline', 'Pipeline', 'account_tree'),
                    ('data', 'Data', 'insights'),
                ]:

                    active = ''

                    with ui.row().classes(
                        f'p-2 rounded cursor-pointer {active}'
                    ).on(
                        'click',
                        lambda p=page: navigate_to(p)
                    ):

                        ui.icon(
                            icon
                        ).classes(
                            'text-white text-sm'
                        )

                        ui.label(
                            label
                        ).classes(
                            'text-white text-sm cursor-pointer'
                        )


# ============================================================
# NAVIGATION
# ============================================================

def navigate_to(page):
    """Navigate between pages."""

    global current_page

    current_page = page

    main_container.clear()

    with main_container:

        if page == 'live':
            build_live_page()

        elif page == 'eval':
            build_eval_page()

        elif page == 'pipeline':
            build_pipeline_page()

        elif page == 'data':
            build_data_page()


# ============================================================
# LIVE DEMO PAGE
# ============================================================
def build_live_page():
    """Build the Live Demo page."""

    global recent_predictions_container

    # Reset the reference whenever Live Demo is rebuilt
    recent_predictions_container = None

    with ui.column().classes(
        'w-full max-w-7xl mx-auto p-6 gap-5'
    ):

        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        ui.label(
            '📰 News Category Classifier'
        ).classes(
            'text-3xl font-bold text-center'
        )

        ui.label(
            'Enter a news headline to predict its category'
        ).classes(
            'text-gray-600 text-center'
        )

        # ----------------------------------------------------
        # MAIN LIVE DEMO AREA
        # ----------------------------------------------------

        with ui.row().classes(
            'w-full gap-4 items-start'
        ):

            # =================================================
            # LEFT: SAMPLE HEADLINES
            # =================================================

            with ui.column().classes(
                'w-52 flex-shrink-0 gap-1'
            ):

                ui.label(
                    'Sample Headlines'
                ).classes(
                    'font-bold text-sm mb-1'
                )

                for cat, headlines in SAMPLE_HEADLINES.items():

                    style = get_style(cat)

                    with ui.expansion(
                        f'{style["emoji"]}  {style["display"]}'
                    ).classes(
                        'w-full sample-expansion'
                    ).props(
                        'dense'
                    ).style(
                        f'''
                        border: 2px solid {style["color"]};
                        border-radius: 7px;
                        margin-bottom: 2px;
                        min-height: 38px;
                        '''
                    ):

                        for headline in headlines:

                            ui.button(
                                headline,
                                on_click=lambda text=headline:
                                    text_input.set_value(text)
                            ).props(
                                'flat dense align-left'
                            ).classes(
                                'w-full text-left text-xs'
                            )

            # =================================================
            # RIGHT SIDE
            # =================================================

            with ui.row().classes(
                'flex-1 min-w-0 gap-4 items-start'
            ):

                # =============================================
                # LEFT SIDE: INPUT + RECENT PREDICTIONS
                # =============================================

                with ui.column().classes(
                    'flex-1 min-w-0 gap-4'
                ):

                    # -----------------------------------------
                    # ENTER NEWS TEXT
                    # -----------------------------------------

                    with ui.card().classes(
                        'w-full p-5'
                    ):

                        ui.label(
                            'Enter News Text'
                        ).classes(
                            'text-base font-bold mb-3'
                        )

                        text_input = create_text_input()

                        with ui.row().classes(
                            'gap-3 mt-4'
                        ):

                            ui.button(
                                'Classify Text',
                                on_click=lambda:
                                    do_classify(
                                        text_input,
                                        results_container
                                    )
                            ).classes(
                                'primary-btn'
                            )

                            ui.button(
                                'Clear',
                                on_click=lambda:
                                    text_input.set_value('')
                            ).props(
                                'flat'
                            )

                    # -----------------------------------------
                    # RECENT PREDICTIONS
                    # -----------------------------------------

                    with ui.card().classes(
                        'w-full p-4'
                    ):

                        ui.label(
                            'Recent Predictions'
                        ).classes(
                            'font-bold mb-2 text-sm'
                        )

                        recent_predictions_container = (
                            ui.column()
                            .classes('w-full gap-0')
                        )

                        render_recent_predictions(
                            recent_predictions_container
                        )

                # =============================================
                # RIGHT SIDE: PREDICTION RESULTS
                # =============================================

                results_container = ui.column().classes(
                    'w-80 flex-shrink-0 gap-3'
                )

# ============================================================
# RECENT PREDICTIONS
# ============================================================

def render_recent_predictions(container):
    """Render the recent predictions list."""

    container.clear()

    with container:

        if not recent_predictions:

            ui.label(
                'No predictions yet'
            ).classes(
                'text-gray-400 text-sm text-center py-3'
            )

            return

        for prediction in recent_predictions:

            style = get_style(
                prediction['cat']
            )

            with ui.row().classes(
                'items-center gap-2 py-2 '
                'border-b border-gray-100 w-full'
            ):

                ui.label(
                    style['emoji']
                ).classes(
                    'text-base'
                )

                with ui.column().classes(
                    'gap-0 flex-1'
                ):

                    ui.label(
                        style['display']
                    ).classes(
                        'font-semibold text-xs'
                    )

                    ui.label(
                        f"{prediction['conf'] * 100:.0f}% confidence"
                    ).classes(
                        'text-xs text-gray-500'
                    )


# ============================================================
# CLASSIFICATION
# ============================================================

def do_classify(text_input, results_container):
    """Perform classification."""

    try:

        text = text_input.value.strip()

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if len(text) < 10:

            ui.notify(
                'Please enter at least 10 characters',
                type='warning'
            )

            return

        if not MODEL_LOADED:

            ui.notify(
                'Model is not loaded.',
                type='negative'
            )

            return

        print(
            f"Classifying: {text[:50]}..."
        )

        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        pred, probs = classifier.predict(
            text,
            'svm'
        )

        print(
            f"Prediction: {pred}"
        )

        # ----------------------------------------------------
        # CLEAR PREVIOUS RESULT
        # ----------------------------------------------------

        results_container.clear()

        with results_container:

            style = get_style(pred)

            # ================================================
            # COMPACT PREDICTION CARD
            # ================================================

            with ui.card().classes(
                'w-full p-3'
            ).style(
                f'''
                background: {style["color"]}15;
                border: 2px solid {style["color"]};
                border-radius: 10px;
                '''
            ):

                with ui.row().classes(
                    'items-center gap-2'
                ):

                    ui.label(
                        style['emoji']
                    ).classes(
                        'text-2xl'
                    )

                    with ui.column().classes(
                        'gap-0'
                    ):

                        ui.label(
                            style['display']
                        ).classes(
                            'text-lg font-bold'
                        ).style(
                            f'color: {style["color"]}'
                        )

                        ui.label(
                            f'{probs[pred] * 100:.1f}% confidence'
                        ).classes(
                            'text-xs text-gray-600'
                        )

            # ================================================
            # COMPACT PROBABILITY DISTRIBUTION
            # ================================================

            with ui.card().classes(
                'w-full p-3'
            ):

                ui.label(
                    'Probability Distribution'
                ).classes(
                    'font-bold mb-2 text-sm'
                )

                for cat, prob in sorted(
                    probs.items(),
                    key=lambda x: x[1],
                    reverse=True
                ):

                    category_style = get_style(cat)

                    with ui.row().classes(
                        'items-center gap-1 w-full mb-1'
                    ):

                        ui.label(
                            f'{category_style["emoji"]} '
                            f'{category_style["display"]}'
                        ).classes(
                            'text-xs w-24'
                        )

                        ui.linear_progress(
                            prob
                        ).classes(
                            'flex-1'
                        ).props(
                            f'color={category_style["color"]} '
                            'size=10px'
                        )

                        ui.label(
                            f'{prob * 100:.0f}%'
                        ).classes(
                            'text-xs font-bold w-8 text-right'
                        )

        # ----------------------------------------------------
        # ADD TO RECENT PREDICTIONS
        # ----------------------------------------------------

        recent_predictions.insert(
            0,
            {
                'cat': pred,
                'conf': max(probs.values()),
                'text': text[:50]
            }
        )

        # Keep only the latest five
        recent_predictions[:] = recent_predictions[:5]

        # ----------------------------------------------------
        # REFRESH RECENT PREDICTIONS
        # ----------------------------------------------------

        if recent_predictions_container is not None:

            render_recent_predictions(
                recent_predictions_container
            )

        # ----------------------------------------------------
        # NOTIFICATION
        # ----------------------------------------------------

        ui.notify(
            'Prediction complete!',
            type='positive'
        )

    except Exception as e:

        print(
            f"Error: {e}"
        )

        ui.notify(
            f'Error: {str(e)}',
            type='negative'
        )


# ============================================================
# MODEL EVALUATION PAGE
# ============================================================

def build_eval_page():
    """Build evaluation page."""

    with ui.column().classes(
        'w-full max-w-6xl mx-auto p-6 gap-4'
    ):

        ui.label(
            '📊 Model Evaluation'
        ).classes(
            'text-2xl font-bold'
        )

        # ----------------------------------------------------
        # MODEL SELECTOR + SHOW RESULTS
        # ----------------------------------------------------

        with ui.row().classes(
            'w-full items-center justify-between'
        ):

            with ui.row().classes(
                'items-center gap-3'
            ):

                ui.label(
                    'Select Model:'
                ).classes(
                    'font-semibold text-sm'
                )

                model_select = ui.select(
                    MODEL_NAMES,
                    value='svm',
                    label='Model'
                ).classes(
                    'w-56'
                )

            ui.button(
                'Show Results',
                on_click=lambda:
                    show_eval(
                        model_select.value,
                        eval_container
                    )
            ).classes(
                'secondary-btn'
            )

        # ----------------------------------------------------
        # EMPTY RESULTS CONTAINER
        # ----------------------------------------------------
        #
        # Metrics and confusion matrix are NOT shown initially.
        # They appear only after Show Results is clicked.
        #

        eval_container = ui.column().classes(
            'w-full mt-4'
        )

        # ----------------------------------------------------
        # MODEL COMPARISON - ALWAYS VISIBLE
        # ----------------------------------------------------

        ui.label(
            'Model Comparison'
        ).classes(
            'text-xl font-bold mt-4'
        )

        with ui.card().classes(
            'w-full p-4'
        ):

            all_metrics = classifier.get_all_metrics()

            fig = go.Figure()

            for key, color, label in [
                (
                    'accuracy',
                    '#6B4FA0',
                    'Accuracy'
                ),
                (
                    'precision',
                    '#4A9E6E',
                    'Precision'
                ),
                (
                    'recall',
                    '#5B8DB8',
                    'Recall'
                ),
                (
                    'f1',
                    '#9B59B6',
                    'F1-Score'
                )
            ]:

                fig.add_trace(
                    go.Bar(
                        name=label,
                        x=[
                            MODEL_NAMES[m]
                            for m in all_metrics
                        ],
                        y=[
                            all_metrics[m][key]
                            for m in all_metrics
                        ],
                        marker_color=color,
                        text=[
                            f"{all_metrics[m][key]:.3f}"
                            for m in all_metrics
                        ],
                        textposition='auto'
                    )
                )

            fig.update_layout(
                barmode='group',
                height=300,
                yaxis_range=[0, 1]
            )

            ui.plotly(
                fig
            ).classes(
                'w-full'
            )


# ============================================================
# SHOW MODEL EVALUATION
# ============================================================

def show_eval(model_key, container):
    """Show evaluation results for selected model."""

    metrics = classifier.get_model_metrics(
        model_key
    )

    if metrics is None:

        ui.notify(
            'No evaluation results found for this model.',
            type='warning'
        )

        return

    container.clear()

    with container:

        # ----------------------------------------------------
        # METRIC CARDS
        # ----------------------------------------------------

        with ui.row().classes(
            'gap-3 w-full'
        ):

            for label, key, icon in [
                (
                    'Accuracy',
                    'accuracy',
                    '✓'
                ),
                (
                    'Precision',
                    'precision',
                    '🎯'
                ),
                (
                    'Recall',
                    'recall',
                    '📊'
                ),
                (
                    'F1',
                    'f1',
                    '⚖️'
                )
            ]:

                with ui.card().classes(
                    'flex-1 text-center p-3'
                ):

                    ui.label(
                        f'{icon} {label}'
                    ).classes(
                        'text-xs text-gray-600'
                    )

                    ui.label(
                        f"{metrics[key]:.3f}"
                    ).classes(
                        'text-xl font-bold'
                    ).style(
                        'color: #6B4FA0'
                    )

        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        with ui.card().classes(
            'w-full p-3'
        ):

            ui.label(
                'Confusion Matrix'
            ).classes(
                'font-bold mb-2 text-sm'
            )

            cm = metrics[
                'confusion_matrix'
            ]

            fig = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=[CATEGORY_STYLES[c]['display'] for c in classifier.categories],
                    y=[CATEGORY_STYLES[c]['display'] for c in classifier.categories],
                    colorscale='Purples',
                    text=cm,
                    texttemplate='%{text}',
                    textfont={
                        'size': 10
                    },
                    showscale=False
                )
            )

            fig.update_layout(
                height=280,
                margin=dict(
                    l=40,
                    r=20,
                    t=20,
                    b=40
                )
            )

            ui.plotly(
                fig
            ).classes(
                'w-full'
            )


# ============================================================
# NLP PIPELINE PAGE
# ============================================================

def build_pipeline_page():
    """
    Build NLP Pipeline page.

    Pipeline results remain hidden until the user clicks
    Show Pipeline.
    """

    with ui.column().classes(
        'w-full max-w-6xl mx-auto p-6 gap-4'
    ):

        ui.label(
            '🔬 NLP Pipeline'
        ).classes(
            'text-2xl font-bold'
        )

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        with ui.row().classes(
            'w-full items-center justify-between'
        ):

            ui.label(
                'Enter Text'
            ).classes(
                'font-bold text-sm'
            )

            ui.button(
                'Show Pipeline',
                on_click=lambda:
                    show_pipeline(
                        pipeline_input.value,
                        pipeline_container
                    )
            ).classes(
                'primary-btn'
            )

        # ----------------------------------------------------
        # TEXT INPUT
        # ----------------------------------------------------

        pipeline_input = create_text_input(
            placeholder='Enter text...'
        )

        pipeline_input.set_value(
            'The quick brown fox jumps over the lazy dog'
        )

        # ----------------------------------------------------
        # EMPTY PIPELINE RESULTS CONTAINER
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # There is intentionally NO call to show_pipeline()
        # here. Results appear only after button click.
        #

        pipeline_container = ui.column().classes(
            'w-full mt-6'
        )


# ============================================================
# SHOW NLP PIPELINE
# ============================================================

def show_pipeline(text, container):
    """Show NLP preprocessing steps."""

    if not text.strip():

        ui.notify(
            'Please enter some text.',
            type='warning'
        )

        return

    if not MODEL_LOADED:

        ui.notify(
            'Model is not loaded.',
            type='negative'
        )

        return

    try:

        steps = classifier.get_pipeline_steps(
            text
        )

        container.clear()

        with container:

            with ui.card().classes(
                'w-full p-4'
            ):

                ui.label(
                    'Preprocessing Steps'
                ).classes(
                    'font-bold mb-3 text-sm'
                )

                for step_name, step_value, color in [
                    (
                        'Raw Text',
                        steps['raw'],
                        '#6B4FA0'
                    ),
                    (
                        'Cleaned',
                        steps['cleaned'],
                        '#4A9E6E'
                    ),
                    (
                        'Tokenized',
                        steps['tokenized'],
                        '#5B8DB8'
                    ),
                    (
                        'No Stopwords',
                        steps['no_stopwords'],
                        '#9B59B6'
                    ),
                    (
                        'Stemmed',
                        steps['stemmed'],
                        '#2ECC71'
                    )
                ]:

                    with ui.column().classes(
                        'w-full mb-3'
                    ):

                        ui.label(
                            step_name
                        ).classes(
                            'font-semibold text-sm'
                        ).style(
                            f'color: {color}'
                        )

                        ui.label(
                            step_value
                        ).classes(
                            'text-base bg-gray-50 rounded p-2 w-full'
                        )

    except Exception as e:

        print(
            f"Pipeline error: {e}"
        )

        ui.notify(
            f'Error: {str(e)}',
            type='negative'
        )


# ============================================================
# DATA ANALYTICS PAGE
# ============================================================

def build_data_page():
    """Build data analytics page."""

    with ui.column().classes(
        'w-full max-w-6xl mx-auto p-6 gap-4'
    ):

        ui.label(
            '📈 Data Analytics'
        ).classes(
            'text-2xl font-bold'
        )

        info = classifier.get_dataset_info()

        # ----------------------------------------------------
        # SUMMARY CARDS
        # ----------------------------------------------------

        with ui.row().classes(
            'gap-3 w-full'
        ):

            for label, value in [
                (
                    'Total Samples',
                    info['total_samples']
                ),
                (
                    'Categories',
                    len(info['categories'])
                ),
                (
                    'Avg Length',
                    f"{info['avg_length']} chars"
                )
            ]:

                with ui.card().classes(
                    'flex-1 text-center p-3'
                ):

                    ui.label(
                        str(value)
                    ).classes(
                        'text-xl font-bold'
                    ).style(
                        'color: #6B4FA0'
                    )

                    ui.label(
                        label
                    ).classes(
                        'text-xs text-gray-600'
                    )

        # ----------------------------------------------------
        # CHARTS
        # ----------------------------------------------------

        with ui.row().classes(
            'gap-4 w-full'
        ):

            cats = info['categories']

            # ================================================
            # CLASS DISTRIBUTION
            # ================================================

            with ui.card().classes(
                'flex-1 p-3'
            ):

                ui.label(
                    'Class Distribution'
                ).classes(
                    'font-bold mb-2 text-sm'
                )

                fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=list(cats.keys()),
                            values=list(cats.values()),
                            marker=dict(
                                colors=[
                                    CATEGORY_STYLES[c]['color']
                                    for c in cats
                                ]
                            )
                        )
                    ]
                )

                fig.update_layout(
                    height=280,
                    margin=dict(
                        l=20,
                        r=20,
                        t=20,
                        b=20
                    )
                )

                ui.plotly(
                    fig
                ).classes(
                    'w-full'
                )

            # ================================================
            # SAMPLES PER CATEGORY
            # ================================================

            with ui.card().classes(
                'flex-1 p-3'
            ):

                ui.label(
                    'Samples per Category'
                ).classes(
                    'font-bold mb-2 text-sm'
                )

                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=list(cats.keys()),
                            y=list(cats.values()),
                            marker_color=[
                                CATEGORY_STYLES[c]['color']
                                for c in cats
                            ],
                            text=list(cats.values()),
                            textposition='auto'
                        )
                    ]
                )

                fig.update_layout(
                    height=280,
                    margin=dict(
                        l=20,
                        r=20,
                        t=20,
                        b=20
                    )
                )

                ui.plotly(
                    fig
                ).classes(
                    'w-full'
                )


# ============================================================
# MAIN PAGE
# ============================================================

@ui.page('/')
def main():

    global main_container

    # --------------------------------------------------------
    # LOAD CUSTOM CSS
    # --------------------------------------------------------

    try:

        with open(
            'style.css',
            'r'
        ) as f:

            ui.add_css(
                f.read()
            )

    except Exception:
        pass

    # --------------------------------------------------------
    # ADD APP-SPECIFIC CSS
    # --------------------------------------------------------

    ui.add_css(
        '''
        /* ================================================
           STANDARD TEXTAREA
           ================================================ */

        .news-textarea .q-field__control {
            background-color: #FFFFFF !important;
            border-radius: 12px !important;
        }

        .news-textarea .q-field__native {
            background-color: #FFFFFF !important;
        }

        .news-textarea textarea {
            background-color: #FFFFFF !important;
        }


        /* ================================================
           SAMPLE HEADLINE EXPANSIONS
           ================================================ */

        .sample-expansion .q-item {
            min-height: 36px !important;
            padding: 6px 8px !important;
        }

        .sample-expansion .q-item__section--main {
            font-size: 13px !important;
            font-weight: 600 !important;
        }

        .sample-expansion .q-item__section--side {
            padding-left: 4px !important;
        }

        .sample-expansion .q-expansion-item__content {
            padding: 2px 5px 5px 5px !important;
        }


        /* ================================================
           TEXTAREA CONSISTENCY
           ================================================ */

        .news-textarea {
            width: 100% !important;
        }


        /* ================================================
           COMPACT LINEAR PROGRESS
           ================================================ */

        .q-linear-progress {
            border-radius: 6px !important;
        }
        '''
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    create_header()

    # --------------------------------------------------------
    # MAIN CONTAINER
    # --------------------------------------------------------

    main_container = ui.column().classes(
        'w-full min-h-screen'
    )

    with main_container:

        build_live_page()


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == '__main__':

    ui.run(
        title='News Classifier',
        favicon='📰',
        port=8080,
        reload=False
    )
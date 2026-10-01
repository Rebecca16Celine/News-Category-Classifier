"""Configuration for News Classifier"""

COLORS = {
    'primary': '#6B4FA0',
    'secondary': '#4A9E6E',
    'background': '#F5F7FA',
}

CATEGORY_STYLES = {

    'business': {
        'color': '#4A9E6E',
        'emoji': '💼',
        'display': 'Business'
    },

    'entertainment': {
        'color': '#9B59B6',
        'emoji': '🎬',
        'display': 'Entertainment'
    },

    'food_drink': {
        'color': '#E67E22',
        'emoji': '🍔',
        'display': 'Food & Drink'
    },

    'health_wellness': {
        'color': '#1ABC9C',
        'emoji': '🩺',
        'display': 'Health & Wellness'
    },

    'parenting': {
        'color': '#F39C12',
        'emoji': '👨‍👩‍👧',
        'display': 'Parenting'
    },

    'politics': {
        'color': '#5B8DB8',
        'emoji': '🏛️',
        'display': 'Politics'
    },

    'science_technology': {
        'color': '#6B4FA0',
        'emoji': '🔬',
        'display': 'Science & Technology'
    },

    'sports': {
        'color': '#2ECC71',
        'emoji': '⚽',
        'display': 'Sports'
    },

    'style_beauty': {
        'color': '#E91E63',
        'emoji': '💄',
        'display': 'Style & Beauty'
    },

    'travel': {
        'color': '#3498DB',
        'emoji': '✈️',
        'display': 'Travel'
    },

}

MODEL_NAMES = {
    'naive_bayes': 'Naive Bayes',
    'logistic_regression': 'Logistic Regression',
    'svm': 'Linear SVM'
}

SAMPLE_HEADLINES = {

    'business': [
        "Global markets rally as companies report record profits",
        "Central bank announces new interest rate policy",
    ],

    'entertainment': [
        "Award-winning actor stars in new blockbuster film",
        "Music festival announces star-studded lineup",
    ],

    'food_drink': [
        "New restaurant brings innovative dishes to the city",
        "Chefs reveal the latest trends in healthy cooking",
    ],

    'health_wellness': [
        "Researchers discover new ways to improve heart health",
        "Experts share simple habits for better mental wellness",
    ],

    'parenting': [
        "Parents discover new strategies for supporting children's education",
        "Experts share advice for balancing work and family life",
    ],

    'politics': [
        "Election commission announces new voting regulations",
        "Parliament debates controversial healthcare bill",
    ],

    'science_technology': [
        "Scientists develop breakthrough artificial intelligence system",
        "Researchers announce major advances in quantum computing",
    ],

    'sports': [
        "United secures dramatic victory with last-minute goal",
        "Tennis champion advances to finals after thrilling match",
    ],

    'style_beauty': [
        "New fashion trends dominate the upcoming season",
        "Beauty experts reveal the latest skincare trends",
    ],

    'travel': [
        "Popular destination introduces new tourism attractions",
        "Travelers discover affordable ways to explore Europe",
    ]

}
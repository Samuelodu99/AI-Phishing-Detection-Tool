import os
import json
import re
import sqlite3
import logging
from datetime import datetime, timedelta
from flask import Flask, request, render_template, make_response, redirect, url_for, session, flash, jsonify
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import pdfplumber
from textblob import TextBlob
from Levenshtein import distance as levenshtein_distance
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Email, To, Mail, Content
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import numpy as np
import joblib
from functools import lru_cache
from flask import send_from_directory
import random
import string

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'Uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = 'static'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max file size
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# SendGrid Configuration
SENDGRID_API_KEY = os.environ.get('SENDGRID_API_KEY')
SENDER_EMAIL = os.environ.get('SENDER_EMAIL')
if not SENDGRID_API_KEY or not SENDER_EMAIL:
    raise ValueError("SENDGRID_API_KEY and SENDER_EMAIL must be set as environment variables.")
serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])

# Google Safe Browsing API key
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', "AIzaSyDKPAa9ieDrBMwf9mzPvXterfv73c593AA")

# Load dynamic thresholds from config.json
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    RISK_THRESHOLD = config.get('risk_threshold', 30)
    HIGH_RISK_THRESHOLD = config.get('high_risk_threshold', 80)
except FileNotFoundError:
    logger.error("config.json not found. Using default thresholds.")
    RISK_THRESHOLD = 30
    HIGH_RISK_THRESHOLD = 80

# Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__)).replace('final year project', 'FINAL_YEAR_PROJECT')

# In-memory phishing database
PHISHING_DB = {
    'urls': ['http://paypa1-security.com', 'https://fake-login.xyz', 'http://amaz0n-security.com'],
    'domains': ['paypa1.com', 'fake-login.xyz', 'amaz0n.com']
}

# Global variables for ML components
MODEL = None
VECTORIZER = None
TRAIN_DATA = []
TRAIN_LABELS = []
FEATURE_NAMES = []
EXPECTED_FEATURE_COUNT = 0
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')

# Initialize NLTK
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)
stop_words = list(stopwords.words('english'))

@lru_cache(maxsize=100)
def initialize_pipelines():
    global MODEL, VECTORIZER, FEATURE_NAMES, EXPECTED_FEATURE_COUNT
    global TRAIN_DATA, TRAIN_LABELS

    # Try loading saved model and vectorizer
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            MODEL = joblib.load(MODEL_PATH)
            VECTORIZER = joblib.load(VECTORIZER_PATH)
            # Load feature names and expected feature count from training
            with open(os.path.join(BASE_DIR, 'feature_names.json'), 'r') as f:
                FEATURE_NAMES = json.load(f)
            EXPECTED_FEATURE_COUNT = len(FEATURE_NAMES)
            logger.info("Loaded saved model and vectorizer.")
            return
    except Exception as e:
        logger.warning(f"Failed to load saved model or vectorizer: {str(e)}. Initializing new model.")

    # Load training data
    load_training_data()
    if not TRAIN_DATA or not TRAIN_LABELS:
        logger.warning("No training data available. Using default synthetic data.")
        default_phishing = [
            "Urgent: Verify your account at http://fake-login.com now!",
            "Your Accout is at risk! Click http://amaz0n-secure.com/login immediately to Verifiy.",
            "Your PayPal account is suspended. Click http://paypa1.com to fix.",
            "Act now! Your bank accout has issues at http://bank-secure-login.com.",
            "Suspended accout detected. Visit http://fakebank-login.com/secure to resolve now."
        ]
        default_legit = [
            "Welcome to our service! Your subscription is active.",
            "Thank you for your purchase from amazon.com!",
            "Your order from www.ebay.com has shipped.",
            "Meeting scheduled for tomorrow at 10 AM.",
            "Thank you for registering at www.trusted-site.com."
        ]
        TRAIN_DATA = default_phishing + default_legit
        TRAIN_LABELS = [1] * len(default_phishing) + [0] * len(default_legit)

    try:
        logger.info(f"Training data loaded: {len(TRAIN_DATA)} samples")
        X, y, feature_names = prepare_features_and_labels(TRAIN_DATA, TRAIN_LABELS)
        FEATURE_NAMES = feature_names
        EXPECTED_FEATURE_COUNT = X.shape[1]
        logger.debug(f"Expected feature count during training: {EXPECTED_FEATURE_COUNT}, Feature shape: {X.shape}")

        # Apply SMOTE if enough samples
        if len(np.unique(y)) > 1 and len(y) > 1:
            smote = SMOTE(random_state=42, k_neighbors=min(1, len(np.unique(y)) - 1))
            X_res, y_res = smote.fit_resample(X, y)
        else:
            logger.warning("Skipping SMOTE due to insufficient samples.")
            X_res, y_res = X, y

        # Add synthetic samples if dataset is small
        if len(y) < 10:
            logger.warning("Dataset too small. Adding synthetic samples.")
            synthetic_data = ["Example phishing text"] * 5 + ["Example legitimate text"] * 5
            synthetic_labels = [1] * 5 + [0] * 5
            X_synthetic, y_synthetic, _ = prepare_features_and_labels(synthetic_data, synthetic_labels)
            X_res = np.vstack((X_res, X_synthetic))
            y_res = np.hstack((y_res, y_synthetic))

        # Train model
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200]
        }
        model = XGBClassifier(eval_metric='logloss')
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_res, y_res)
        MODEL = grid_search.best_estimator_
        cv_scores = cross_val_score(MODEL, X_res, y_res, cv=5, scoring='f1')
        logger.info(f"Cross-validation F1 scores: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

        # Save model and vectorizer
        joblib.dump(MODEL, MODEL_PATH)
        joblib.dump(VECTORIZER, VECTORIZER_PATH)
        with open(os.path.join(BASE_DIR, 'feature_names.json'), 'w') as f:
            json.dump(FEATURE_NAMES, f)
        logger.info("Model and vectorizer saved successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize supervised model: {str(e)}")
        MODEL = XGBClassifier(eval_metric='logloss')
        VECTORIZER = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words=stop_words)
        EXPECTED_FEATURE_COUNT = 500 + 10  # Fallback feature count

def get_db_connection(db_name):
    try:
        db_path = os.path.join(BASE_DIR, db_name)
        logger.debug(f"Opening database at: {db_path}")
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error ({db_name}): {str(e)}")
        raise

def init_db():
    logger.info("Initializing databases...")
    with get_db_connection('results.db') as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS results
               (id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                filename TEXT,
                risk_score INTEGER,
                evidence TEXT,
                tip TEXT,
                timestamp TEXT,
                breakdown TEXT,
                feedback TEXT,
                created_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id))''')
        c.execute("SELECT COUNT(*) FROM results WHERE feedback IS NOT NULL")
        if c.fetchone()[0] == 0:
            initial_data = [
                (1, "initial_phishing_1.txt", 75, "Urgent: Verify your account at http://fake-login.com", "Do not click any links.", "2023-01-01 12:00:00", "Initial data", "correct"),
                (1, "initial_legitimate_1.txt", 0, "Welcome to our service! Your subscription is active.", "This appears to be legitimate.", "2023-01-01 12:01:00", "Initial data", "false_positive"),
                (1, "initial_phishing_2.txt", 90, "Your PayPal account is suspended. Click http://paypa1.com to verify.", "Do not click any links.", "2023-01-01 12:05:00", "Initial data", "correct"),
                (1, "initial_legitimate_2.txt", 0, "Thank you for your purchase from amazon.com!", "This appears to be legitimate.", "2023-01-01 12:03:00", "Initial data", "false_positive"),
            ]
            c.executemany("INSERT INTO results (user_id, filename, risk_score, evidence, tip, timestamp, breakdown, feedback) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", initial_data)
        conn.commit()
        logger.info("Results table initialized.")

    with get_db_connection('users.db') as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT UNIQUE,
                      email TEXT UNIQUE,
                      password TEXT,
                      name TEXT,
                      phone TEXT,
                      profile_picture TEXT,
                      reset_token TEXT,
                      reset_expires TIMESTAMP,
                      role TEXT DEFAULT 'user')''')
        conn.commit()
        logger.info("Users table initialized.")

def migrate_db():
    logger.info("Running database migrations...")
    with get_db_connection('users.db') as conn:
        c = conn.cursor()
        c.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in c.fetchall()]
        if 'name' not in columns:
            c.execute("ALTER TABLE users ADD COLUMN name TEXT")
        if 'phone' not in columns:
            c.execute("ALTER TABLE users ADD COLUMN phone TEXT")
        if 'profile_picture' not in columns:
            c.execute("ALTER TABLE users ADD COLUMN profile_picture TEXT")
        if 'reset_token' not in columns:
            c.execute("ALTER TABLE users ADD COLUMN reset_token TEXT")
        if 'reset_expires' not in columns:
            c.execute("ALTER TABLE users ADD COLUMN reset_expires TIMESTAMP")
        if 'role' not in columns:
            c.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")
        conn.commit()
    with get_db_connection('results.db') as conn:
        c = conn.cursor()
        c.execute("PRAGMA table_info(results)")
        columns = [col[1] for col in c.fetchall()]
        if 'breakdown' not in columns:
            c.execute("ALTER TABLE results ADD COLUMN breakdown TEXT")
        if 'feedback' not in columns:
            c.execute("ALTER TABLE results ADD COLUMN feedback TEXT")
        if 'created_at' not in columns:
            c.execute("ALTER TABLE results ADD COLUMN created_at TIMESTAMP")
            c.execute("UPDATE results SET created_at = timestamp WHERE created_at IS NULL")
        conn.commit()

def load_training_data():
    global TRAIN_DATA, TRAIN_LABELS
    try:
        with get_db_connection('results.db') as conn:
            c = conn.cursor()
            c.execute("SELECT evidence, feedback FROM results WHERE feedback IS NOT NULL")
            data = c.fetchall()
            TRAIN_DATA = []
            TRAIN_LABELS = []
            for evidence, feedback in data:
                if feedback in ['correct', 'false_negative']:  # Phishing
                    TRAIN_LABELS.append(1)
                elif feedback == 'false_positive':  # Legitimate
                    TRAIN_LABELS.append(0)
                else:
                    continue
                TRAIN_DATA.append(evidence if evidence else "")
            logger.debug(f"Loaded {len(TRAIN_DATA)} training samples from database.")
            if len(TRAIN_DATA) < 10:
                logger.warning("Insufficient training data. Adding synthetic samples.")
                synthetic_phishing = [
                    "Urgent: Verify your account at http://fake-login.com now!",
                    "Your PayPal account is suspended. Click http://paypa1.com to fix.",
                    "Your Accout is at risk! Click http://amaz0n-secure.com/login immediately to Verifiy.",
                    "Act now! Your bank accout has issues at http://bank-secure-login.com.",
                    "Suspended accout detected. Visit http://fakebank-login.com/secure to resolve now."
                ]
                synthetic_legit = [
                    "Welcome to our service! Your subscription is active.",
                    "Thank you for your purchase from amazon.com!",
                    "Your order from www.ebay.com has shipped.",
                    "Meeting scheduled for tomorrow at 10 AM.",
                    "Thank you for registering at www.trusted-site.com."
                ]
                for text in synthetic_phishing:
                    TRAIN_DATA.append(text)
                    TRAIN_LABELS.append(1)
                for text in synthetic_legit:
                    TRAIN_DATA.append(text)
                    TRAIN_LABELS.append(0)
                augment_data()
    except sqlite3.Error as e:
        logger.error(f"Database error loading training data: {str(e)}")
        TRAIN_DATA = []
        TRAIN_LABELS = []

def augment_data():
    global TRAIN_DATA, TRAIN_LABELS
    augmented_data = []
    augmented_labels = []
    for text, label in zip(TRAIN_DATA, TRAIN_LABELS):
        augmented_data.append(text)
        augmented_labels.append(label)
        typo_text = re.sub(r'(\w+)', lambda m: m.group(0) + random.choice(string.ascii_lowercase) if random.random() < 0.2 else m.group(0), text)
        augmented_data.append(typo_text)
        augmented_labels.append(label)
        if 'urgent' in text.lower():
            augmented_data.append(text.replace('urgent', 'immediate'))
            augmented_labels.append(label)
        if 'verify' in text.lower():
            augmented_data.append(text.replace('verify', 'confirm'))
            augmented_labels.append(label)
    TRAIN_DATA = augmented_data
    TRAIN_LABELS = augmented_labels

def extract_pos_features(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    return {tag for _, tag in pos_tags}

def extract_ner_features(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    chunked = ne_chunk(pos_tags)
    return {chunk.label() for chunk in chunked if hasattr(chunk, 'label')}

def extract_url_features(text):
    urls = re.findall(r'http[s]?://[^\s]+', text)
    features = {}
    for url in urls:
        features['has_https'] = int(url.startswith('https://'))
        features['domain_length'] = len(re.search(r'http[s]?://([^/]+)', url).group(1))
        features['has_suspicious'] = int(any(sus in url.lower() for sus in ['paypa1', '.xyz', '-security', '-secure', 'login']))
    return features

def prepare_features_and_labels(data, labels):
    global VECTORIZER
    if VECTORIZER is None:
        VECTORIZER = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words=stop_words)
    
    try:
        X_tfidf = VECTORIZER.fit_transform(data) if not hasattr(VECTORIZER, 'vocabulary_') else VECTORIZER.transform(data)
        feature_names = VECTORIZER.get_feature_names_out().tolist()
    except Exception as e:
        logger.error(f"Error in TF-IDF transformation: {str(e)}")
        raise ValueError(f"TF-IDF transformation failed: {str(e)}")
    
    pos_features = []
    ner_features = []
    url_features = []
    for d in data:
        try:
            pos_features.append(extract_pos_features(d))
            ner_features.append(extract_ner_features(d))
            url_features.append(extract_url_features(d))
        except Exception as e:
            logger.warning(f"Error extracting features for text: {str(e)}")
            pos_features.append(set())
            ner_features.append(set())
            url_features.append({})

    pos_possible = ['NN', 'VB', 'JJ', 'RB']
    ner_possible = ['PERSON', 'ORGANIZATION', 'GPE']
    url_possible = ['has_https', 'domain_length', 'has_suspicious']

    X_additional = []
    additional_feature_names = pos_possible + ner_possible + url_possible

    for i in range(len(data)):
        row = []
        pos_set = pos_features[i]
        for pos in pos_possible:
            row.append(1 if pos in pos_set else 0)
        ner_set = ner_features[i]
        for ner in ner_possible:
            row.append(1 if ner in ner_set else 0)
        url_dict = url_features[i]
        for url_feature in url_possible:
            row.append(url_dict.get(url_feature, 0))
        X_additional.append(row)

    try:
        X = np.hstack((X_tfidf.toarray(), np.array(X_additional)))
        feature_names.extend(additional_feature_names)
        logger.debug(f"Feature shape during training: {X.shape}")
        return X, np.array(labels), feature_names
    except Exception as e:
        logger.error(f"Error combining features: {str(e)}")
        raise ValueError(f"Feature combination failed: {str(e)}")

def prepare_features_for_prediction(text):
    global VECTORIZER, FEATURE_NAMES, EXPECTED_FEATURE_COUNT
    if VECTORIZER is None or not FEATURE_NAMES or EXPECTED_FEATURE_COUNT == 0:
        logger.error("Vectorizer, feature names, or expected feature count not initialized.")
        return None

    try:
        X_tfidf = VECTORIZER.transform([text]).toarray()
        logger.debug(f"TF-IDF feature shape: {X_tfidf.shape}")

        pos_set = extract_pos_features(text)
        ner_set = extract_ner_features(text)
        url_dict = extract_url_features(text)

        pos_possible = ['NN', 'VB', 'JJ', 'RB']
        ner_possible = ['PERSON', 'ORGANIZATION', 'GPE']
        url_possible = ['has_https', 'domain_length', 'has_suspicious']

        X_additional = []
        for pos in pos_possible:
            X_additional.append(1 if pos in pos_set else 0)
        for ner in ner_possible:
            X_additional.append(1 if ner in ner_set else 0)
        for url_feature in url_possible:
            X_additional.append(url_dict.get(url_feature, 0))

        X = np.hstack((X_tfidf, np.array([X_additional])))
        logger.debug(f"Combined feature shape before padding: {X.shape}")

        if X.shape[1] < EXPECTED_FEATURE_COUNT:
            padding = np.zeros((X.shape[0], EXPECTED_FEATURE_COUNT - X.shape[1]))
            X = np.hstack((X, padding))
            logger.debug(f"Padded features to match expected count: {EXPECTED_FEATURE_COUNT}")
        elif X.shape[1] > EXPECTED_FEATURE_COUNT:
            X = X[:, :EXPECTED_FEATURE_COUNT]
            logger.debug(f"Truncated features to match expected count: {EXPECTED_FEATURE_COUNT}")

        return X
    except Exception as e:
        logger.error(f"Error preparing features for prediction: {str(e)}")
        return None

def evaluate_model(X, y):
    try:
        cv_scores = cross_val_score(MODEL, X, y, cv=5, scoring='f1')
        y_pred = MODEL.predict(X)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        logger.info(f"Cross-validation F1: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
        logger.info(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
        logger.info(f"Confusion Matrix:\n{cm}")
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")

def retrain_model():
    global MODEL, VECTORIZER, TRAIN_DATA, TRAIN_LABELS, FEATURE_NAMES, EXPECTED_FEATURE_COUNT
    try:
        load_training_data()
        if not TRAIN_DATA or not TRAIN_LABELS:
            logger.warning("No training data available for retraining. Using synthetic data.")
            synthetic_phishing = [
                "Urgent: Verify your account at http://fake-login.com now!",
                "Your PayPal account is suspended. Click http://paypa1.com to fix.",
                "Your Accout is at risk! Click http://amaz0n-secure.com/login immediately to Verifiy.",
                "Act now! Your bank accout has issues at http://bank-secure-login.com.",
                "Suspended accout detected. Visit http://fakebank-login.com/secure to resolve now."
            ]
            synthetic_legit = [
                "Welcome to our service! Your subscription is active.",
                "Thank you for your purchase from amazon.com!",
                "Your order from www.ebay.com has shipped.",
                "Meeting scheduled for tomorrow at 10 AM.",
                "Thank you for registering at www.trusted-site.com."
            ]
            TRAIN_DATA = synthetic_phishing + synthetic_legit
            TRAIN_LABELS = [1] * len(synthetic_phishing) + [0] * len(synthetic_legit)
            augment_data()

        X, y, feature_names = prepare_features_and_labels(TRAIN_DATA, TRAIN_LABELS)
        FEATURE_NAMES = feature_names
        EXPECTED_FEATURE_COUNT = X.shape[1]

        if len(np.unique(y)) > 1 and len(y) > 1:
            smote = SMOTE(random_state=42, k_neighbors=min(1, len(np.unique(y)) - 1))
            X_res, y_res = smote.fit_resample(X, y)
        else:
            logger.warning("Skipping SMOTE due to insufficient samples.")
            X_res, y_res = X, y

        if len(y) < 10:
            logger.warning("Dataset too small. Adding synthetic samples.")
            synthetic_data = ["Example phishing text"] * 5 + ["Example legitimate text"] * 5
            synthetic_labels = [1] * 5 + [0] * 5
            X_synthetic, y_synthetic, _ = prepare_features_and_labels(synthetic_data, synthetic_labels)
            X_res = np.vstack((X_res, X_synthetic))
            y_res = np.hstack((y_res, y_synthetic))

        class_weights = compute_class_weight('balanced', classes=np.unique(y_res), y=y_res)
        MODEL = XGBClassifier(eval_metric='logloss')
        MODEL.fit(X_res, y_res, sample_weight=class_weights)
        evaluate_model(X_res, y_res)

        # Save model and vectorizer
        joblib.dump(MODEL, MODEL_PATH)
        joblib.dump(VECTORIZER, VECTORIZER_PATH)
        with open(os.path.join(BASE_DIR, 'feature_names.json'), 'w') as f:
            json.dump(FEATURE_NAMES, f)
        logger.info("Model retrained and saved successfully.")
    except Exception as e:
        logger.error(f"Error retraining model: {str(e)}")

def extract_text(file_path):
    try:
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        elif file_path.endswith('.pdf'):
            with pdfplumber.open(file_path) as pdf:
                text = ''
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                return text
        else:
            raise ValueError("Unsupported file type. Use .txt or .pdf")
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {str(e)}")
        return f"Error reading file: {str(e)}"

def preprocess_text(text):
    try:
        tokens = word_tokenize(text.lower())
        return [word for word in tokens if word.isalnum() and word not in stop_words]
    except Exception as e:
        logger.error(f"Error preprocessing text: {str(e)}")
        return []

def analyze_with_grok(text):
    initialize_pipelines()
    risk_score = 0
    evidence = []
    breakdown = []

    preprocessed_text = " ".join(preprocess_text(text))
    if not preprocessed_text:
        evidence.append("No text content to analyze after preprocessing.")
        breakdown.append("No text content: +0")

    header_patterns = {
        'from': r'^From:.*?<(.+?)>',
        'return_path': r'^Return-Path:.*?<(.+?)>',
        'received': r'^Received:.*?from\s+([^\s]+)'
    }
    headers = {}
    for line in text.split('\n'):
        for header, pattern in header_patterns.items():
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                headers[header] = match.group(1)

    if "To:" in text and not headers:
        risk_score += 5
        evidence.append("Email lacks raw headers; unable to verify sender authenticity.")
        breakdown.append("Missing headers: +5")

    if 'from' in headers and 'return_path' in headers:
        from_domain = headers['from'].split('@')[-1]
        return_domain = headers['return_path'].split('@')[-1]
        if from_domain != return_domain:
            risk_score += 15
            evidence.append(f"Header mismatch detected: From ({from_domain}) does not match Return-Path ({return_domain})")
            breakdown.append("Header mismatch: +15")

    legitimate_domains = ['paypal.com', 'gmail.com', 'amazon.com']
    legitimate_email_domains = ['paypal.com', 'gmail.com', 'outlook.com', 'yahoo.com']

    urgent_patterns = r'(urgent|now|immediately|suspended|at\s+risk|act\s+now)'
    has_urgent_language = False
    if re.search(urgent_patterns, text, re.IGNORECASE):
        risk_score += 20
        matches = re.findall(urgent_patterns, text, re.IGNORECASE)
        evidence.append(f"Urgent language detected: {', '.join(matches)}")
        breakdown.append("Urgent language: +20")
        has_urgent_language = True

    try:
        blob = TextBlob(preprocessed_text)
        sentiment = blob.sentiment.polarity
        if sentiment < -0.3 and not ("Phishing Detection Tool" in text and "Samuel Odu" in text):
            risk_score += 5
            evidence.append(f"Negative sentiment detected (polarity {sentiment:.2f})")
            breakdown.append(f"Negative sentiment: +5")
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {str(e)}")

    negative_intent_phrases = ['suspended', 'at risk', 'urgent action required']
    if any(phrase in preprocessed_text.lower() for phrase in negative_intent_phrases):
        risk_score += 3
        evidence.append("Negative intent detected (rule-based)")
        breakdown.append("Negative intent: +3")

    url_pattern = r'(http[s]?://[^\s]+)|(?:Reset\s+Password)'
    urls = re.findall(url_pattern, text)
    validated_urls = []
    validated_domains = []
    web_validation_hits = []

    if "Reset Password" in text and not any(url.startswith('http') for url in urls):
        if "Phishing Detection Tool" in text and "Samuel Odu" in text:
            evidence.append("Password reset link mentioned but no URL provided; likely embedded in a button (legitimate).")
        else:
            risk_score += 10
            evidence.append("Password reset link mentioned but no URL provided; possible phishing attempt.")
            breakdown.append("Missing reset URL: +10")

    try:
        safe_browsing = build('safebrowsing', 'v4', developerKey=GOOGLE_API_KEY, cache_discovery=False)
        http_urls = [url for url in urls if url.startswith('http')]
        if http_urls:
            body = {
                "client": {"clientId": "phishing-detection-tool", "clientVersion": "1.0"},
                "threatInfo": {
                    "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING"],
                    "platformTypes": ["ANY_PLATFORM"],
                    "threatEntryTypes": ["URL"],
                    "threatEntries": [{"url": url} for url in http_urls]
                }
            }
            response = safe_browsing.threatMatches().find(body=body).execute()
            if 'matches' in response:
                for match in response['matches']:
                    risk_score += 30
                    evidence.append(f"URL flagged by Google Safe Browsing: {match['threat']['url']} ({match['threatType']})")
                    breakdown.append("Google Safe Browsing: +30")
    except HttpError as e:
        evidence.append(f"Google Safe Browsing API error: {str(e)}.")

    for url in urls:
        if not url.startswith('http'):
            continue
        domain_match = re.search(r'http[s]?://([^/]+)', url)
        if domain_match:
            domain = domain_match.group(1)
            for legit_domain in legitimate_domains:
                if domain != legit_domain:
                    similarity = levenshtein_distance(domain.lower(), legit_domain.lower())
                    if 0 < similarity <= 2:
                        risk_score += 20
                        evidence.append(f"Visually similar domain detected: {domain} (similar to {legit_domain})")
                        breakdown.append(f"Visual similarity: +20")

        if any(sus in url.lower() for sus in ['paypa1', '.xyz', '-security', '-secure', 'login']):
            risk_score += 25
            evidence.append(f"Suspicious URL: {url}")
            breakdown.append(f"Suspicious URL: +25")
        if not url.startswith('https://'):
            risk_score += 5
            evidence.append(f"URL lacks HTTPS: {url}")
            breakdown.append(f"URL lacks HTTPS: +5")

    if re.search(r'^(hello|hi|dear)\b', text, re.IGNORECASE) and not re.search(r'hello,\s+\w+', text, re.IGNORECASE):
        risk_score += 5
        evidence.append("Generic greeting without personalization detected")
        breakdown.append("Generic greeting: +5")

    if 'paypal' in text.lower() and 'paypal.com' not in text.lower():
        risk_score += 5
        evidence.append("Possible spoofing: 'PayPal' mentioned but no official domain")
        breakdown.append("Possible spoofing: +5")

    email_pattern = r'(?:from|sender):?.*?@([\w.-]+)'
    emails = re.findall(email_pattern, text, re.IGNORECASE)
    for email_domain in emails:
        email_domain = email_domain.strip().lower()
        if any(sus in email_domain for sus in ['paypa1', '.xyz', '-', 'secure']):
            risk_score += 20
            evidence.append(f"Suspicious email domain: @{email_domain}")
            breakdown.append(f"Suspicious email domain: +20")
        elif email_domain not in legitimate_email_domains:
            risk_score += 5
            evidence.append(f"Unrecognized email domain: @{email_domain}")
            breakdown.append(f"Unrecognized email domain: +5")

    has_spelling_errors = False
    common_typos = r'(verifiy|accout|paymant|securty)'
    if re.search(common_typos, text, re.IGNORECASE):
        risk_score += 5
        matches = re.findall(common_typos, text, re.IGNORECASE)
        evidence.append(f"Spelling errors detected: {', '.join(matches)}")
        breakdown.append(f"Spelling errors: +5")
        has_spelling_errors = True

    if "Phishing Detection Tool" in text and "Samuel Odu" in text:
        evidence.append("Recognized legitimate footer; reducing risk score.")
        risk_score -= 5
        breakdown.append("Legitimate footer: -5")

    if risk_score > RISK_THRESHOLD:
        for url in urls:
            if not url.startswith('http'):
                continue
            validated_urls.append(url)
            if url.lower() in [u.lower() for u in PHISHING_DB['urls']]:
                web_validation_hits.append(f"URL {url} matches known phishing reports")
        for email_domain in emails:
            validated_domains.append(email_domain)
            if email_domain.lower() in [d.lower() for d in PHISHING_DB['domains']]:
                web_validation_hits.append(f"Domain @{email_domain} matches known phishing reports")
        if web_validation_hits:
            risk_score += 15
            evidence.append("; ".join(web_validation_hits))
            breakdown.append("Web validation: +15")

    if risk_score >= HIGH_RISK_THRESHOLD:
        for url in urls:
            if not url.startswith('http'):
                continue
            if url.lower() not in [u.lower() for u in PHISHING_DB['urls']]:
                PHISHING_DB['urls'].append(url)
                evidence.append(f"New phishing URL added to database: {url}")
        for email_domain in emails:
            if email_domain.lower() not in [d.lower() for d in PHISHING_DB['domains']]:
                PHISHING_DB['domains'].append(email_domain)
                evidence.append(f"New phishing domain added to database: @{email_domain}")

    risk_score = min(risk_score, 100)

    if MODEL is None or not hasattr(VECTORIZER, 'vocabulary_'):
        evidence.append("ML prediction failed: need to call fit or load_model beforehand")
        breakdown.append("ML prediction: +0 (prediction error)")
    elif preprocessed_text:
        X = prepare_features_for_prediction(preprocessed_text)
        if X is not None:
            try:
                ml_confidence = MODEL.predict_proba(X)[0][1]
                logger.debug(f"ML confidence score: {ml_confidence}")
                if ml_confidence > 0.7:
                    ml_score = int(ml_confidence * 30)
                    risk_score = min(risk_score + ml_score, 100)
                    evidence.append(f"ML model predicts phishing (confidence {ml_confidence:.2f})")
                    breakdown.append(f"ML prediction: +{ml_score}")
                else:
                    evidence.append(f"ML model confidence low (confidence {ml_confidence:.2f})")
                    breakdown.append("ML prediction: +0 (low confidence)")
            except Exception as e:
                logger.error(f"ML prediction failed: {str(e)}")
                evidence.append(f"ML prediction failed: {str(e)}")
                breakdown.append("ML prediction: +0 (prediction error)")
        else:
            evidence.append("ML prediction skipped: Feature preparation failed.")
            breakdown.append("ML prediction: +0 (feature preparation failed)")
    else:
        evidence.append("ML prediction skipped: No text available.")
        breakdown.append("ML prediction: +0 (skipped)")

    tip_parts = []
    if risk_score < RISK_THRESHOLD:
        tip_parts.append("This message appears safe, but remain vigilant.")
        tip_parts.append("Always verify the sender's email address.")
    else:
        tip_parts.append("Avoid clicking any links or providing personal information.")
        if 'paypal' in text.lower() or any('paypa1' in url.lower() for url in urls) or any('paypa1' in domain.lower() for domain in emails):
            tip_parts.append("Verify the email on PayPal's official website (www.paypal.com).")
            tip_parts.append("Report to PayPal at spoof@paypal.com.")
            additional_tips = []
            if has_urgent_language:
                additional_tips.append("urgent language")
            if has_spelling_errors:
                additional_tips.append("spelling errors")
            if additional_tips:
                tip_parts.append(f"Be cautious of {', '.join(additional_tips)}.")
            tip_parts.append("Tip: Check for misspellings like paypa1.com.")
        elif 'amazon' in text.lower() or any('amaz0n' in url.lower() for url in urls) or any('amaz0n' in domain.lower() for domain in emails):
            tip_parts.append("Verify on Amazon's official website (www.amazon.com).")
            tip_parts.append("Report to Amazon at stop-spoofing@amazon.com.")
            additional_tips = []
            if has_urgent_language:
                additional_tips.append("urgent language")
            if has_spelling_errors:
                additional_tips.append("spelling errors")
            if additional_tips:
                tip_parts.append(f"Be cautious of {', '.join(additional_tips)}.")
            tip_parts.append("Tip: Check for misspellings like amaz0n.com.")
        elif urls or any("Suspicious email domain" in e for e in evidence):
            tip_parts.append("Verify the sender and URLs with the official source.")
            tip_parts.append("Report suspicious emails to your email provider.")
            additional_tips = []
            if has_urgent_language:
                additional_tips.append("urgent language")
            if has_spelling_errors:
                additional_tips.append("spelling errors")
            if additional_tips:
                tip_parts.append(f"Be cautious of {', '.join(additional_tips)}.")
            if urls:
                tip_parts.append("Tip: Be cautious of links that lack HTTPS.")
            else:
                tip_parts.append("Tip: Check the sender's email domain.")
        else:
            tip_parts.append("Verify the sender directly with the official source.")
            tip_parts.append("Report suspicious emails to your email provider.")
            additional_tips = []
            if has_urgent_language:
                additional_tips.append("urgent language")
            if has_spelling_errors:
                additional_tips.append("spelling errors")
            if additional_tips:
                tip_parts.append(f"Be cautious of {', '.join(additional_tips)}.")
            tip_parts.append("Tip: Watch for red flags like urgent language.")

    tip = " ".join(tip_parts)

    return {
        "risk_score": risk_score,
        "evidence": "; ".join(evidence) if evidence else "No phishing indicators detected.",
        "tip": tip,
        "urls": validated_urls,
        "domains": validated_domains,
        "breakdown": breakdown
    }

def is_admin(user_id):
    try:
        with get_db_connection('users.db') as conn:
            c = conn.cursor()
            c.execute("SELECT role FROM users WHERE id = ?", (user_id,))
            user = c.fetchone()
            return user and user[0] == 'admin'
    except sqlite3.Error as e:
        logger.error(f"Database error checking admin status: {str(e)}")
        return False

def login_required(f):
    def wrap(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please log in to access this page.", "error")
            return redirect(url_for('login'))
        session.modified = True
        return f(*args, **kwargs)
    wrap.__name__ = f.__name__
    return wrap

def admin_required(f):
    def wrap(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please log in to access this page.", "error")
            return redirect(url_for('login'))
        if not is_admin(session['user_id']):
            flash("You do not have permission to access this page.", "error")
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    wrap.__name__ = f.__name__
    return wrap

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        name = request.form.get('name', '')
        phone = request.form.get('phone', '')

        if not all([username, email, password, confirm_password]):
            flash("Username, email, and password are required.", "error")
            return redirect(url_for('signup'))

        email_pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
        if not re.match(email_pattern, email):
            flash("Invalid email format.", "error")
            return redirect(url_for('signup'))

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password)

        try:
            with get_db_connection('users.db') as conn:
                c = conn.cursor()
                c.execute("INSERT INTO users (username, email, password, name, phone, role) VALUES (?, ?, ?, ?, ?, ?)",
                          (username, email, hashed_password, name, phone, 'user'))
                conn.commit()
                flash("Registration successful! Please log in.", "success")
                return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username or email already exists.", "error")
            return redirect(url_for('signup'))
        except sqlite3.Error as e:
            logger.error(f"Database error during signup: {str(e)}")
            flash("An error occurred during registration. Please try again.", "error")
            return redirect(url_for('signup'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        try:
            with get_db_connection('users.db') as conn:
                c = conn.cursor()
                c.execute("SELECT * FROM users WHERE username = ?", (username,))
                user = c.fetchone()

                if user and check_password_hash(user[3], password):
                    session['user_id'] = user[0]
                    session['username'] = user[1]
                    session.permanent = True
                    flash("Login successful!", "success")
                    return redirect(url_for('index'))
                else:
                    flash("Invalid username or password.", "error")
                    return redirect(url_for('login'))
        except sqlite3.Error as e:
            logger.error(f"Database error during login: {str(e)}")
            flash("An error occurred during login. Please try again.", "error")
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash("You have been logged out.", "success")
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    user_id = session.get('user_id')
    logger.debug(f"User ID from session: {user_id}")
    if user_id is None:
        logger.error("User ID is None in session; redirecting to login.")
        flash("Session error: Please log in again.", "error")
        return redirect(url_for('login'))

    is_admin_user = is_admin(user_id)
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file uploaded.", "error")
            return render_template('index.html', is_admin=is_admin_user)

        file = request.files['file']
        if file.filename == '':
            flash("No file selected.", "error")
            return render_template('index.html', is_admin=is_admin_user)

        allowed_extensions = {'txt', 'pdf'}
        ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if ext not in allowed_extensions:
            flash("Invalid file type. Use .txt or .pdf.", "error")
            return render_template('index.html', is_admin=is_admin_user)

        filename = secure_filename(file.filename)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        if file_size > app.config['MAX_CONTENT_LENGTH']:
            flash("File too large. Maximum size is 5MB.", "error")
            return render_template('index.html', is_admin=is_admin_user)
        file.seek(0)

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(file_path)
        except Exception as e:
            logger.error(f"Error saving file {filename}: {str(e)}")
            flash("Error saving file. Please try again.", "error")
            return render_template('index.html', is_admin=is_admin_user)

        text = extract_text(file_path)
        if "Error" in text:
            try:
                with get_db_connection('results.db') as conn:
                    c = conn.cursor()
                    c.execute("INSERT INTO results (user_id, filename, risk_score, evidence, tip, timestamp, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                              (user_id, filename, 0, text, "Invalid file type. Please upload a .txt or .pdf file.", datetime.now().strftime('%Y-%m-%d %H:%M:%S'), datetime.now()))
                    conn.commit()
            except sqlite3.Error as e:
                logger.error(f"Database error saving error result: {str(e)}")
                flash("Error saving result to database.", "error")
            flash(text, "error")
            return render_template('index.html', is_admin=is_admin_user)

        result = analyze_with_grok(text)
        breakdown = "; ".join(result['breakdown']) if result['breakdown'] else "No breakdown available."
        try:
            with get_db_connection('results.db') as conn:
                c = conn.cursor()
                c.execute("INSERT INTO results (user_id, filename, risk_score, evidence, tip, timestamp, breakdown, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                          (user_id, filename, result['risk_score'], result['evidence'], result['tip'], datetime.now().strftime('%Y-%m-%d %H:%M:%S'), breakdown, datetime.now()))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database error saving analysis result: {str(e)}")
            flash("Error saving result to database.", "error")
            return render_template('index.html', is_admin=is_admin_user)

        logger.info(f"Risk Score for {filename}: {result['risk_score']}")
        return render_template('index.html', result=result, filename=filename, is_admin=is_admin_user)
    return render_template('index.html', is_admin=is_admin_user)

@app.route('/past_results', methods=['GET', 'POST'])
@login_required
def past_results():
    user_id = session.get('user_id')
    logger.debug(f"User ID from session in past_results: {user_id}")
    if user_id is None:
        logger.error("User ID is None in session; redirecting to login.")
        flash("Session error: Please log in again.", "error")
        return redirect(url_for('login'))

    is_admin_user = is_admin(user_id)
    
    page = request.args.get('page', 1, type=int)
    per_page = 10
    offset = (page - 1) * per_page

    if request.method == 'POST' and request.form.get('action') == 'clear':
        try:
            with get_db_connection('results.db') as conn:
                c = conn.cursor()
                if is_admin_user:
                    c.execute("DELETE FROM results")
                else:
                    c.execute("DELETE FROM results WHERE user_id = ?", (user_id,))
                conn.commit()
            flash("Past results cleared successfully!", "success")
            retrain_model()
            return redirect(url_for('past_results'))
        except sqlite3.Error as e:
            logger.error(f"Database error clearing past results: {str(e)}")
            flash("Error clearing past results.", "error")
            return redirect(url_for('past_results'))

    try:
        with get_db_connection('results.db') as conn:
            c = conn.cursor()
            if is_admin_user:
                c.execute("SELECT COUNT(*) FROM results WHERE user_id IS NOT NULL")
            else:
                c.execute("SELECT COUNT(*) FROM results WHERE user_id = ? AND user_id IS NOT NULL", (user_id,))
            total_results = c.fetchone()[0]
            total_pages = (total_results + per_page - 1) // per_page

            if is_admin_user:
                c.execute("SELECT filename, risk_score, evidence, tip, timestamp, breakdown, id, feedback, user_id FROM results WHERE user_id IS NOT NULL ORDER BY timestamp DESC LIMIT ? OFFSET ?", (per_page, offset))
            else:
                c.execute("SELECT filename, risk_score, evidence, tip, timestamp, breakdown, id, feedback, user_id FROM results WHERE user_id = ? AND user_id IS NOT NULL ORDER BY timestamp DESC LIMIT ? OFFSET ?", (user_id, per_page, offset))
            results_raw = c.fetchall()
            results = [(r[0], int(r[1]), r[2], r[3], r[4], r[5], r[6], r[7], r[8]) for r in results_raw]
    except sqlite3.Error as e:
        logger.error(f"Database error fetching past results: {str(e)}")
        flash("Error fetching past results.", "error")
        return render_template('past_results.html', results=[], page=page, total_pages=1, risk_threshold=RISK_THRESHOLD, high_risk_threshold=HIGH_RISK_THRESHOLD, is_admin=is_admin_user)

    return render_template('past_results.html', results=results, page=page, total_pages=total_pages, risk_threshold=RISK_THRESHOLD, high_risk_threshold=HIGH_RISK_THRESHOLD, is_admin=is_admin_user)

@app.route('/submit_feedback/<int:result_id>', methods=['POST'])
@login_required
def submit_feedback(result_id):
    feedback = request.form.get('feedback')
    if feedback not in ['correct', 'false_positive', 'false_negative']:
        flash("Invalid feedback value.", "error")
        return redirect(url_for('past_results'))
    try:
        with get_db_connection('results.db') as conn:
            c = conn.cursor()
            c.execute("UPDATE results SET feedback = ? WHERE id = ? AND user_id = ?", (feedback, result_id, session['user_id']))
            if c.rowcount == 0:
                flash("Result not found or you do not have permission to provide feedback.", "error")
            else:
                conn.commit()
                retrain_model()
                flash("Feedback submitted successfully.", "success")
    except sqlite3.Error as e:
        logger.error(f"Database error submitting feedback: {str(e)}")
        flash("Error submitting feedback.", "error")
    return redirect(url_for('past_results'))

@app.route('/about')
def about():
    is_admin_user = session.get('user_id') and is_admin(session['user_id'])
    return render_template('about.html', is_admin=is_admin_user)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    is_admin_user = session.get('user_id') and is_admin(session['user_id'])
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        if not all([name, email, message]):
            flash("All fields are required.", "error")
            return render_template('contact.html', success=False, is_admin=is_admin_user)
        flash("Message sent successfully!", "success")
        return render_template('contact.html', success=True, is_admin=is_admin_user)
    return render_template('contact.html', success=False, is_admin=is_admin_user)

@app.route('/download_result/<filename>')
@login_required
def download_result(filename):
    try:
        with get_db_connection('results.db') as conn:
            c = conn.cursor()
            c.execute("SELECT risk_score, evidence, tip, breakdown FROM results WHERE filename = ? AND user_id = ? ORDER BY timestamp DESC LIMIT 1",
                      (filename, session['user_id']))
            result = c.fetchone()
            if not result:
                flash("Result not found or you do not have permission to access it.", "error")
                return redirect(url_for('past_results'))

        risk_score, evidence, tip, breakdown = result
        content = f"Phishing Detection Result\n\nFilename: {filename}\nRisk Score: {risk_score}\nEvidence: {evidence}\nBreakdown: {breakdown}\nTip: {tip}\n"
        response = make_response(content)
        response.headers['Content-Disposition'] = f'attachment; filename={filename}_result.txt'
        response.headers['Content-Type'] = 'text/plain'
        return response
    except sqlite3.Error as e:
        logger.error(f"Database error downloading result: {str(e)}")
        flash("Error downloading result.", "error")
        return redirect(url_for('past_results'))

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user_id = session['user_id']
    is_admin_user = is_admin(user_id)
    try:
        with get_db_connection('users.db') as conn:
            c = conn.cursor()
            c.execute("SELECT username, email, name, phone, profile_picture, password FROM users WHERE id = ?", (user_id,))
            user = c.fetchone()
            if not user:
                flash("User not found.", "error")
                return redirect(url_for('login'))
    except sqlite3.Error as e:
        logger.error(f"Database error fetching user profile: {str(e)}")
        flash("Error fetching profile.", "error")
        return redirect(url_for('index'))

    profile_picture = user[4]
    file_exists = False
    if profile_picture and os.path.exists(profile_picture):
        filename = os.path.basename(profile_picture)
        profile_picture = url_for('uploaded_file', filename=filename)
        file_exists = True
    else:
        profile_picture = None

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'update_profile':
            try:
                new_email = request.form.get('email')
                new_name = request.form.get('name')
                new_phone = request.form.get('phone')
                profile_picture_file = request.files.get('profile_picture')
                remove_picture = request.form.get('remove_picture') == 'on'

                if new_email and not re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', new_email):
                    flash("Invalid email format.", "error")
                    return redirect(url_for('profile'))

                updates = {}
                picture_path = user[4]

                if remove_picture and picture_path:
                    try:
                        if os.path.exists(picture_path):
                            os.remove(picture_path)
                    except Exception as e:
                        logger.error(f"Failed to remove profile picture: {str(e)}")
                        flash(f"Failed to remove profile picture: {str(e)}", "error")
                    updates['profile_picture'] = None
                    picture_path = None

                if profile_picture_file and profile_picture_file.filename and not remove_picture:
                    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
                    ext = profile_picture_file.filename.rsplit('.', 1)[1].lower() if '.' in profile_picture_file.filename else ''
                    if ext not in allowed_extensions:
                        flash("Invalid file type. Allowed types: PNG, JPG, JPEG, GIF.", "error")
                        return redirect(url_for('profile'))

                    profile_picture_file.seek(0, os.SEEK_END)
                    file_size = profile_picture_file.tell()
                    profile_picture_file.seek(0)
                    if file_size > 2 * 1024 * 1024:
                        flash("File too large. Maximum size is 2MB.", "error")
                        return redirect(url_for('profile'))

                    if picture_path and os.path.exists(picture_path):
                        try:
                            os.remove(picture_path)
                        except Exception as e:
                            logger.error(f"Failed to remove old profile picture: {str(e)}")
                            flash(f"Failed to remove old profile picture: {str(e)}", "warning")

                    filename = secure_filename(f"user_{user_id}_{profile_picture_file.filename}")
                    picture_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    profile_picture_file.save(picture_path)
                    updates['profile_picture'] = picture_path

                if new_email and new_email != user[1]:
                    updates['email'] = new_email
                if new_name is not None:
                    updates['name'] = new_name
                if new_phone is not None:
                    updates['phone'] = new_phone

                if updates:
                    with get_db_connection('users.db') as conn:
                        c = conn.cursor()
                        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
                        values = list(updates.values()) + [user_id]
                        c.execute(f"UPDATE users SET {set_clause} WHERE id = ?", values)
                        conn.commit()
                    flash("Profile updated successfully!", "success")
                else:
                    flash("No changes made to the profile.", "info")
            except sqlite3.IntegrityError as e:
                logger.error(f"Database error updating profile: {str(e)}")
                flash(f"Update failed: {str(e)}. Email may already be in use.", "error")
            except Exception as e:
                logger.error(f"Error updating profile: {str(e)}")
                flash(f"An error occurred: {str(e)}", "error")
            return redirect(url_for('profile'))

        elif action == 'change_password':
            try:
                current_password = request.form.get('current_password')
                new_password = request.form.get('new_password')
                confirm_password = request.form.get('confirm_password')

                if not check_password_hash(user[5], current_password):
                    flash("Current password is incorrect.", "error")
                    return redirect(url_for('profile'))

                if not new_password:
                    flash("New password cannot be empty.", "error")
                    return redirect(url_for('profile'))

                if new_password != confirm_password:
                    flash("New passwords do not match.", "error")
                    return redirect(url_for('profile'))

                if len(new_password) < 8:
                    flash("New password must be at least 8 characters long.", "error")
                    return redirect(url_for('profile'))

                hashed_password = generate_password_hash(new_password)
                with get_db_connection('users.db') as conn:
                    c = conn.cursor()
                    c.execute("UPDATE users SET password = ? WHERE id = ?", (hashed_password, user_id))
                    conn.commit()
                flash("Password updated successfully!", "success")
            except sqlite3.Error as e:
                logger.error(f"Database error updating password: {str(e)}")
                flash("Error updating password.", "error")
            except Exception as e:
                logger.error(f"Error updating password: {str(e)}")
                flash(f"An error occurred: {str(e)}", "error")
            return redirect(url_for('profile'))

    return render_template('profile.html',
                           username=user[0],
                           email=user[1],
                           name=user[2] or '',
                           phone=user[3] or '',
                           profile_picture=profile_picture,
                           file_exists=file_exists,
                           is_admin=is_admin_user)

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        try:
            with get_db_connection('users.db') as conn:
                c = conn.cursor()
                c.execute("SELECT id, email FROM users WHERE email = ?", (email,))
                user = c.fetchone()
                if user:
                    user_id = user[0]
                    token = serializer.dumps(user_id, salt='password-reset-salt')
                    reset_url = url_for('reset_password', token=token, _external=True)
                    sg = SendGridAPIClient(SENDGRID_API_KEY)
                    mail = Mail(
                        from_email=SENDER_EMAIL,
                        to_emails=email,
                        subject='Password Reset Request - Phishing Detection Tool'
                    )
                    mail.add_content(Content(
                        mime_type="text/html",
                        content=f"""
                            <!DOCTYPE html>
                            <html lang="en">
                            <head>
                                <meta charset="UTF-8">
                                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                                <title>Password Reset Request</title>
                            </head>
                            <body style="margin: 0; padding: 0; font-family: Arial, Helvetica, sans-serif; background-color: #f4f4f4; color: #333;">
                                <table width="100%" border="0" cellspacing="0" cellpadding="0" style="max-width: 600px; margin: 0 auto; background-color: #ffffff; border: 1px solid #e0e0e0;">
                                    <tr>
                                        <td style="background-color: #00DDEB; padding: 20px; text-align: center;">
                                            <h1 style="color: #ffffff; margin: 0; font-size: 24px;">Phishing Detection Tool</h1>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 20px;">
                                            <h2 style="color: #00DDEB; font-size: 20px; margin-top: 0;">Password Reset Request</h2>
                                            <p style="color: #333; font-size: 16px; line-height: 1.5;">Hello,</p>
                                            <p style="color: #333; font-size: 16px; line-height: 1.5;">We received a request to reset your password for the Phishing Detection Tool.</p>
                                            <p style="color: #333; font-size: 16px; line-height: 1.5;">Click the button below to reset your password:</p>
                                            <table border="0" cellspacing="0" cellpadding="0" style="text-align: center;">
                                                <tr>
                                                    <td>
                                                        <a href="{reset_url}" style="display: inline-block; background-color: #00DDEB; color: #ffffff; padding: 12px 25px; text-decoration: none; font-size: 16px; font-weight: bold; border-radius: 5px;">Reset Password</a>
                                                    </td>
                                                </tr>
                                            </table>
                                            <p style="color: #333; font-size: 16px; line-height: 1.5; margin-top: 20px;">This link will expire in 1 hour.</p>
                                            <p style="color: #333; font-size: 16px; line-height: 1.5;">If you did not request a password reset, please ignore this email.</p>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td style="background-color: #f4f4f4; padding: 10px; text-align: center; font-size: 12px; color: #777;">
                                            © 2025 Phishing Detection Tool | Final Year Project by Samuel Odu<br>
                                            <a href="mailto:support@phishingdetectiontool.com" style="color: #00DDEB; text-decoration: none;">Contact Us</a> | <a href="https://phishingdetectiontool.com" style="color: #00DDEB; text-decoration: none;">Unsubscribe</a>
                                        </td>
                                    </tr>
                                </table>
                            </body>
                            </html>
                        """
                    ))
                    try:
                        response = sg.send(mail)
                        logger.debug(f"Email sent: {response.status_code}")
                        flash("A password reset link has been sent to your email.", "success")
                    except Exception as e:
                        logger.error(f"Failed to send email: {str(e)}")
                        flash("Failed to send reset email. Please try again later.", "error")
                else:
                    flash("No account found with that email.", "error")
        except sqlite3.Error as e:
            logger.error(f"Database error during forgot password: {str(e)}")
            flash("An error occurred. Please try again.", "error")
        return redirect(url_for('forgot_password'))
    return render_template('forgot_password.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        user_id = serializer.loads(token, salt='password-reset-salt', max_age=3600)
    except SignatureExpired:
        flash("The password reset link has expired.", "error")
        return redirect(url_for('forgot_password'))
    except BadSignature:
        flash("Invalid password reset link.", "error")
        return redirect(url_for('forgot_password'))

    try:
        with get_db_connection('users.db') as conn:
            c = conn.cursor()
            c.execute("SELECT email FROM users WHERE id = ?", (user_id,))
            user = c.fetchone()
            if not user:
                flash("User not found.", "error")
                return redirect(url_for('forgot_password'))
    except sqlite3.Error as e:
        logger.error(f"Database error during reset password: {str(e)}")
        flash("An error occurred. Please try again.", "error")
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        new_password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        if new_password != confirm_password:
            flash("Passwords do not match.", "error")
            return redirect(url_for('reset_password', token=token))
        if not new_password:
            flash("Password cannot be empty.", "error")
            return redirect(url_for('reset_password', token=token))

        try:
            with get_db_connection('users.db') as conn:
                c = conn.cursor()
                c.execute("UPDATE users SET password = ? WHERE id = ?", (generate_password_hash(new_password), user_id))
                c.execute("UPDATE users SET reset_token = NULL, reset_expires = NULL WHERE id = ?", (user_id,))
                conn.commit()
            flash("Your password has been reset successfully. Please log in.", "success")
            return redirect(url_for('login'))
        except sqlite3.Error as e:
            logger.error(f"Database error resetting password: {str(e)}")
            flash("Error resetting password.", "error")
            return redirect(url_for('reset_password', token=token))

    return render_template('reset_password.html', token=token)

@app.route('/admin')
@admin_required
def admin_dashboard():
    users_page = request.args.get('users_page', 1, type=int)
    results_page = request.args.get('results_page', 1, type=int)
    per_page = 10

    user_search = request.args.get('user_search', '').strip()
    result_filter_risk = request.args.get('result_filter_risk', 'all')
    result_filter_user_id = request.args.get('result_filter_user_id', '')

    try:
        with get_db_connection('users.db') as conn:
            c = conn.cursor()
            user_query = "SELECT id, username, email, role FROM users"
            user_count_query = "SELECT COUNT(*) FROM users"
            user_params = []
            if user_search:
                user_query += " WHERE username LIKE ? OR email LIKE ?"
                user_count_query += " WHERE username LIKE ? OR email LIKE ?"
                user_params.extend([f"%{user_search}%", f"%{user_search}%"])

            c.execute(user_count_query, user_params)
            total_users = c.fetchone()[0]

            user_query += " ORDER BY id LIMIT ? OFFSET ?"
            user_params.extend([per_page, (users_page - 1) * per_page])
            c.execute(user_query, user_params)
            users = c.fetchall()

            total_users_pages = (total_users + per_page - 1) // per_page

            c.execute("SELECT id FROM users")
            user_ids = [str(row[0]) for row in c.fetchall()]
    except sqlite3.Error as e:
        logger.error(f"Database error fetching user data: {str(e)}")
        flash("Error accessing user data.", "error")
        return redirect(url_for('index'))

    try:
        with get_db_connection('results.db') as conn:
            c = conn.cursor()
            result_query = "SELECT id, user_id, filename, risk_score, evidence, tip FROM results"
            result_count_query = "SELECT COUNT(*) FROM results"
            result_params = []
            conditions = []

            if result_filter_risk != 'all':
                if result_filter_risk == 'high':
                    conditions.append("risk_score >= 70")
                elif result_filter_risk == 'medium':
                    conditions.append("risk_score BETWEEN 30 AND 69")
                elif result_filter_risk == 'low':
                    conditions.append("risk_score < 30")

            if result_filter_user_id:
                conditions.append("user_id = ?")
                result_params.append(result_filter_user_id)

            if conditions:
                result_query += " WHERE " + " AND ".join(conditions)
                result_count_query += " WHERE " + " AND ".join(conditions)

            c.execute(result_count_query, result_params)
            total_analyses = c.fetchone()[0]

            result_query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            result_params.extend([per_page, (results_page - 1) * per_page])
            c.execute(result_query, result_params)
            results = c.fetchall()

            total_results_pages = (total_analyses + per_page - 1) // per_page

            activity_data = []
            today = datetime.now()
            for i in range(30):
                day = today - timedelta(days=i)
                day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
                day_end = day_start + timedelta(days=1)
                c.execute("SELECT COUNT(*) FROM results WHERE created_at >= ? AND created_at < ?", (day_start, day_end))
                count = c.fetchone()[0]
                activity_data.append({'date': day.strftime('%Y-%m-%d'), 'count': count})
    except sqlite3.Error as e:
        logger.error(f"Database error fetching analysis data: {str(e)}")
        flash("Error accessing analysis data.", "error")
        return redirect(url_for('index'))

    return render_template('admin.html',
                          total_users=total_users,
                          total_analyses=total_analyses,
                          users=users,
                          results=results,
                          activity_data=activity_data,
                          users_page=users_page,
                          total_users_pages=total_users_pages,
                          results_page=results_page,
                          total_results_pages=total_results_pages,
                          per_page=per_page,
                          user_search=user_search,
                          result_filter_risk=result_filter_risk,
                          result_filter_user_id=result_filter_user_id,
                          user_ids=user_ids)

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@admin_required
def delete_user(user_id):
    try:
        with get_db_connection('users.db') as conn:
            c = conn.cursor()
            c.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
        with get_db_connection('results.db') as conn:
            c = conn.cursor()
            c.execute("DELETE FROM results WHERE user_id = ?", (user_id,))
            conn.commit()
        flash("User deleted successfully.", "success")
        retrain_model()
    except sqlite3.Error as e:
        logger.error(f"Database error deleting user: {str(e)}")
        flash("Error deleting user.", "error")
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/delete_result/<int:result_id>', methods=['POST'])
@admin_required
def delete_result(result_id):
    try:
        with get_db_connection('results.db') as conn:
            c = conn.cursor()
            c.execute("DELETE FROM results WHERE id = ?", (result_id,))
            conn.commit()
        flash("Analysis result deleted successfully.", "success")
        retrain_model()
    except sqlite3.Error as e:
        logger.error(f"Database error deleting result: {str(e)}")
        flash("Error deleting result.", "error")
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/update_role/<int:user_id>', methods=['POST'])
@admin_required
def update_role(user_id):
    new_role = request.form.get('role')
    if new_role not in ['user', 'admin']:
        flash("Invalid role.", "error")
        return redirect(url_for('admin_dashboard'))
    try:
        with get_db_connection('users.db') as conn:
            c = conn.cursor()
            c.execute("UPDATE users SET role = ? WHERE id = ?", (new_role, user_id))
            conn.commit()
        flash(f"User role updated to {new_role}.", "success")
    except sqlite3.Error as e:
        logger.error(f"Database error updating user role: {str(e)}")
        flash("Error updating user role.", "error")
    return redirect(url_for('admin_dashboard'))

@app.route('/get_result_details/<int:result_id>')
@admin_required
def get_result_details(result_id):
    try:
        with get_db_connection('results.db') as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM results WHERE id = ?", (result_id,))
            result = c.fetchone()
            if not result:
                return jsonify({'error': 'Result not found'}), 404

            result_dict = {
                'id': result[0],
                'user_id': result[1],
                'filename': result[2],
                'risk_score': result[3],
                'evidence': result[4],
                'tip': result[5],
                'timestamp': result[6],
                'breakdown': result[7],
                'feedback': result[8],
                'created_at': result[9]
            }
            return jsonify(result_dict)
    except sqlite3.Error as e:
        logger.error(f"Database error fetching result details: {str(e)}")
        return jsonify({'error': 'Database error'}), 500
@app.route('/create_admin')
def create_admin():
    try:
        with get_db_connection('users.db') as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE email = ?", ('admin@example.com',))
            admin_user = c.fetchone()
            if not admin_user:
                hashed_password = generate_password_hash('adminpassword')
                c.execute("INSERT INTO users (username, email, password, role) VALUES (?, ?, ?, ?)",
                          ('admin', 'admin@example.com', hashed_password, 'admin'))
                conn.commit()
                return "Admin user created!"
            return "Admin user already exists!"
    except sqlite3.Error as e:
        logger.error(f"Database error creating admin user: {str(e)}")
        return "Error creating admin user."
    
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    init_db()
    migrate_db()
    initialize_pipelines()
    app.run(debug=True, host='0.0.0.0', port=5000)
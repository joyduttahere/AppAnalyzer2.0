# BrandManager app.py
# FINAL, STABLE VERSION

import os
import ssl
import re
import time
import torch
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, jsonify
import json
import markdown2
from pyngrok import ngrok
import google.generativeai as genai

# SSL Configuration
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from dotenv import load_dotenv
load_dotenv()

import threading
from flask import Response
import time

from google_play_scraper import reviews, Sort, search
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Disable SSL warnings
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

# ==============================================================================
# 1. CONSTANTS AND CONFIGURATION
# ==============================================================================
PAIN_POINT_CATEGORIES = {
    "App Stability & Performance": {
        "Crashes & Stability": [
            "app crashes", "force close", "force closes", "app closes", "shuts down suddenly",
            "stops working", "won't start", "won't open", "can't open", "fails to launch",
            "black screen", "white screen", "blank screen", "frozen app", "app freezes",
            "hangs up", "hanging", "stuck loading", "unresponsive app", "not responding"
        ],
        "Bugs & Errors": [
            "software bug", "app bug", "buggy behavior", "glitch", "glitches", "glitchy",
            "error message", "error code", "runtime error", "system error", "fatal error",
            "malfunction", "broken feature", "feature broken", "corrupted data", "invalid response",
            "null pointer", "exception error", "404 error", "500 error", "server error"
        ],
        "Performance Issues": [
            "slow loading", "takes forever to load", "loading forever", "endless loading",
            "spinning wheel", "buffering constantly", "app lag", "lagging", "laggy performance",
            "sluggish response", "choppy animation", "stuttering", "frame drops", "delayed response",
            "timeout error", "times out", "slow performance", "performance issues"
        ],
        "Resource Usage": [
            "battery drain", "drains battery fast", "battery consumption", "overheating phone",
            "phone gets hot", "cpu usage high", "memory leak", "ram usage", "storage space",
            "takes up space", "disk space full", "data usage high", "mobile data consumption"
        ]
    },

    "Onboarding & Setup": {
        "Initial Setup Problems": [
            "setup failed", "installation incomplete", "setup wizard broken", "initial configuration error",
            "first time setup", "welcome screen stuck", "setup process confusing", "can't complete setup",
            "setup keeps crashing", "setup won't finish", "configuration failed", "initial setup error",
            "setup tutorial broken", "guided setup issues", "setup verification failed", "setup timeout"
        ],
        "Account Creation Issues": [
            "can't create account", "signup broken", "registration failed", "account creation error",
            "email already exists", "username taken", "password requirements unclear", "terms acceptance failed",
            "captcha not working", "verification email never came", "phone verification failed", "age verification issues",
            "country not supported", "region restrictions", "account creation timeout", "signup form broken"
        ],
        "Tutorial & First Experience": [
            "tutorial skipped accidentally", "tutorial too long", "tutorial confusing", "onboarding unclear",
            "first use confusing", "don't understand how to start", "tutorial broken", "walkthrough issues",
            "intro video won't play", "getting started guide missing", "first steps unclear", "tutorial won't load",
            "skip tutorial button missing", "tutorial navigation broken", "onboarding too complicated", "tutorial crashes"
        ],
        "Profile & Preferences Setup": [
            "profile setup incomplete", "preferences not saving", "avatar upload failed", "profile picture issues",
            "personal information error", "preference settings broken", "profile creation failed", "settings not applying",
            "default preferences wrong", "customization options limited", "profile validation failed", "bio won't save",
            "interest selection broken", "category preferences error", "notification setup failed", "privacy settings unclear"
        ],
        "Permission & Access Setup": [
            "permission denied", "camera access denied", "microphone permission issues", "location permission failed",
            "contacts access blocked", "storage permission required", "notification permission setup", "background app refresh",
            "permission popup keeps appearing", "can't grant permissions", "permission settings confusing", "access request failed",
            "privacy permission unclear", "permission explanation missing", "why does app need permission", "permission too intrusive"
        ],
        "Integration & Import Setup": [
            "can't import contacts", "social media login failed", "google signin broken", "facebook connect issues",
            "import from other app failed", "data migration error", "sync with existing account", "third party integration broken",
            "calendar sync setup failed", "email integration issues", "import csv failed", "backup restore during setup",
            "connect existing service", "link account failed", "oauth authentication error", "api connection timeout"
        ]
    },

    "Connectivity & Integration": {
        "Network & Sync": [
            "connection failed", "can't connect to server", "connection error", "network error",
            "server down", "server unavailable", "no internet connection", "offline issues",
            "sync failed", "sync error", "won't sync", "syncing problems", "cloud sync",
            "api timeout", "network timeout", "connection lost", "unstable connection"
        ],
        "Device Integration": [
            "bluetooth connection", "bluetooth pairing", "bluetooth issues", "camera not working",
            "microphone problems", "speaker issues", "gps not working", "location services",
            "fingerprint scanner", "face id problems", "touch id", "biometric authentication",
            "sensor issues", "accelerometer", "gyroscope problems", "hardware integration"
        ],
        "Platform Compatibility": [
            "not compatible", "incompatible device", "device not supported", "android version",
            "ios version", "operating system", "phone model", "tablet support", "screen resolution",
            "orientation issues", "landscape mode", "portrait mode", "version compatibility"
        ]
    },

    "User Experience": {
        "UI Design & Layout": [
            "ugly interface", "bad design", "poor layout", "cluttered screen", "messy design",
            "confusing layout", "hard to read text", "small text", "tiny buttons", "big buttons",
            "overlapping elements", "misaligned buttons", "cut off text", "weird fonts",
            "color scheme bad", "contrast issues", "dark mode issues", "theme problems"
        ],
        "Navigation & Usability": [
            "hard to navigate", "confusing navigation", "can't find button", "hidden options",
            "menu confusing", "too many steps", "complicated workflow", "not user friendly",
            "not intuitive", "steep learning curve", "hard to use", "difficult interface",
            "where is the setting", "how to use", "instructions unclear", "poor usability"
        ],
        "Accessibility": [
            "accessibility issues", "screen reader", "voice over", "contrast too low",
            "color blind friendly", "font size too small", "vision impaired", "hearing impaired",
            "disability support", "accessible design", "inclusive design", "ada compliance"
        ]
    },

    "Account & Authentication": {
        "Login & Security": [
            "can't login", "login failed", "login error", "password incorrect", "wrong password",
            "forgot password", "reset password", "password reset", "account locked", "locked out",
            "verification failed", "otp not received", "two factor authentication", "2fa issues",
            "authentication error", "signin problems", "signup failed", "email verification"
        ],
        "Account Management": [
            "account deleted", "profile missing", "account suspended", "banned account",
            "account recovery", "lost account", "can't access account", "account settings",
            "profile settings", "privacy settings", "data export", "account deletion"
        ],
        "Privacy & Security": [
            "privacy concerns", "data collection", "personal information", "location tracking",
            "data sharing", "third party access", "privacy policy", "data breach",
            "security concern", "not secure", "unsafe app", "suspicious activity",
            "account hacked", "unauthorized access", "identity theft", "data stolen"
        ]
    },

    "Business & Service": {
        "Customer Support": [
            "customer support", "customer service", "support team", "help desk", "contact support",
            "no response from support", "poor customer service", "rude support staff", "unhelpful support",
            "support chat", "support email", "support ticket", "complaint handling", "escalation needed",
            "supervisor", "manager", "support quality", "response time", "resolution time"
        ],
        "Billing & Payments": [
            "charged twice", "double charged", "billing error", "payment failed", "can't make payment",
            "payment declined", "credit card declined", "refund request", "money back", "overcharged",
            "billing issue", "subscription problem", "auto renewal", "cancel subscription", "billing cycle",
            "payment method", "paypal issues", "credit card problem", "transaction failed"
        ],
        "Product Quality": [
            "poor quality product", "low quality", "cheap quality", "defective item", "damaged product",
            "wrong item sent", "not as described", "misleading description", "fake product", "counterfeit",
            "not authentic", "not genuine", "not original", "brand quality", "quality control"
        ],
        "Delivery & Fulfillment": [
            "late delivery", "delayed shipping", "never arrived", "lost package", "missing order",
            "wrong address", "delivery issues", "courier problems", "tracking problems", "no tracking",
            "shipping cost", "delivery charges", "fast delivery", "same day delivery",
            "express shipping", "standard delivery", "out for delivery", "delivery notification"
        ]
    },

    "Features & Functionality": {
        "Missing Features": [
            "need this feature", "missing feature", "add this feature", "feature request", "should have",
            "wish it had", "would be nice", "suggestion for improvement", "enhancement request",
            "new feature needed", "update needed", "improvement needed", "missing functionality"
        ],
        "Broken Features": [
            "feature not working", "feature broken", "stopped working", "removed feature", "disabled feature",
            "limited functionality", "restricted access", "paywall feature", "premium only", "subscription required",
            "feature behind paywall", "locked feature", "unavailable feature", "feature missing"
        ],
        "Search & Discovery": [
            "search not working", "can't find items", "search results", "filter not working", "sort options",
            "browse categories", "recommendation engine", "suggested items", "algorithm issues",
            "no search results", "search function", "discovery features", "find products"
        ]
    },

    "Content & Information": {
        "Content Quality": [
            "poor content", "low quality content", "inaccurate information", "wrong information", "outdated content",
            "content not updated", "fresh content", "relevant content", "useful content", "useless content",
            "spam content", "inappropriate content", "offensive content", "content moderation"
        ],
        "Information Management": [
            "data backup", "data restore", "export data", "import data", "sync data", "cloud storage",
            "save progress", "lost data", "deleted information", "missing data", "recover data",
            "data recovery", "backup failed", "restore failed", "data corruption"
        ]
    },

    "Monetization & Advertising": {
        "Pricing & Subscriptions": [
            "too expensive", "overpriced", "price increase", "subscription cost", "premium price",
            "free version limited", "trial period", "subscription management", "pricing plans",
            "discount codes", "promotional offers", "payment plans", "cost comparison"
        ],
        "Advertisements": [
            "too many ads", "annoying ads", "popup ads", "video ads", "banner ads", "intrusive ads",
            "ad frequency", "ad blocker", "remove ads", "ad free version", "sponsored content",
            "advertising policy", "relevant ads", "targeted ads", "ad personalization"
        ],
        "In-App Purchases": [
            "in app purchase", "microtransaction", "buy credits", "purchase coins", "unlock features",
            "premium upgrade", "purchase failed", "receipt issues", "restore purchase", "refund purchase",
            "iap problems", "store issues", "payment processing", "purchase verification"
        ]
    }
}

# Critical issues that require immediate attention
CRITICAL_ISSUES = [
    # Security & Privacy
    "account hacked", "security breach", "data stolen", "identity theft", "credit card stolen",
    "money stolen", "unauthorized charges", "fraudulent activity", "scam", "phishing attempt",
    "malware detected", "virus", "suspicious activity", "data breach", "privacy violation",

    # Financial
    "charged without permission", "double billing", "can't cancel subscription", "unauthorized transaction",
    "payment error", "billing fraud", "money disappeared", "refund denied", "overcharged significantly",

    # Data Loss
    "lost all data", "data deleted permanently", "can't recover data", "backup failed completely",
    "account deleted", "profile disappeared", "history lost", "progress lost", "work lost",

    # Critical Functionality
    "emergency feature broken", "safety issue", "medical emergency", "urgent problem", "life threatening",
    "dangerous malfunction", "harmful content", "child safety", "inappropriate content for kids",

    # Critical Onboarding Issues
    "can't create account at all", "completely locked out during setup", "setup crashes repeatedly",
    "account creation fraud detected", "setup security warning", "onboarding data breach"
]

# Positive sentiment indicators
POSITIVE_INDICATORS = [
    "love this app", "great app", "awesome feature", "excellent service", "amazing experience",
    "fantastic update", "wonderful design", "perfect functionality", "best app ever", "outstanding quality",
    "brilliant idea", "superb performance", "impressed with", "highly recommend", "five stars",
    "thank you", "grateful for", "appreciate the", "very helpful", "extremely useful",
    "user friendly", "easy to use", "fast loading", "quick response", "smooth experience",
    "reliable app", "stable performance", "solid app", "works perfectly", "no issues",
    "smooth onboarding", "easy setup", "great first impression", "intuitive setup", "seamless registration"
]

# Negative sentiment indicators
NEGATIVE_INDICATORS = [
    "hate this app", "terrible experience", "awful service", "horrible design", "worst app ever",
    "completely useless", "total garbage", "absolute trash", "extremely disappointed", "very frustrating",
    "incredibly annoying", "really irritating", "makes me angry", "absolutely furious", "unacceptable quality",
    "disgusting behavior", "pathetic service", "ridiculous problems", "waste of time", "waste of money",
    "regret downloading", "sorry I installed", "big mistake", "never again", "avoid this app",
    "terrible onboarding", "confusing setup", "horrible first experience", "setup nightmare", "onboarding disaster"
]

# Enhanced feature request detection patterns
FEATURE_REQUEST_INDICATORS = [
    # Direct requests
    "need this feature", "add this feature", "feature request", "should have",
    "wish it had", "would be nice", "suggestion for improvement", "enhancement request",
    "new feature needed", "update needed", "improvement needed", "missing functionality",
    "would be great", "it would be nice", "please add", "can you add", "hope you add",
    "looking forward to", "waiting for", "expecting", "anticipating",
    
    # Comparative requests
    "other apps have", "competitor has", "like in other apps", "similar to", "compared to",
    "as good as", "better than", "improve upon", "upgrade from",
    
    # Conditional/wishful language
    "if you could", "would love", "would appreciate", "hoping for", "wish you would",
    "it should", "could use", "needs to", "ought to", "might want to",
    "consider adding", "think about", "what about", "how about",
    
    # Specific missing features (from your existing list)
    "widget option", "calendar grid layout", "birthdays with year", "important dates marked",
    "holidays marked", "festival dates", "shrink down", "month view", "year view"
]

# Pre-compile critical issues for faster matching
CRITICAL_ISSUES_SET = set(CRITICAL_ISSUES)
POSITIVE_INDICATORS_SET = set(POSITIVE_INDICATORS)
NEGATIVE_INDICATORS_SET = set(NEGATIVE_INDICATORS)

# Use PAIN_POINT_CATEGORIES as TOPIC_CATEGORIES for consistency
TOPIC_CATEGORIES = PAIN_POINT_CATEGORIES


# ==============================================================================
# 2. GLOBAL CACHES & MODEL LOADING
# ==============================================================================
MODEL_CACHE = {}
REVIEW_CACHE = {}
AI_RESULT_CACHE = {}
GEMINI_API_LIMIT_REACHED = False
GEMINI_API_CALL_COUNT = 0 

def get_classifier_model():
    """Optimized model loading with memory management."""
    global MODEL_CACHE
    if "classifier" in MODEL_CACHE: 
        return MODEL_CACHE["classifier"]
    
    print("--- LOADING SENTIMENT CLASSIFIER (OPTIMIZED) ---")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        
        # Move to device and optimize
        model = model.to(DEVICE)
        model.eval()  # Set to evaluation mode
        
        # Enable optimizations for better performance
        if DEVICE == "cuda":
            model = model.half()  # Use half precision on GPU
            print("✅ Using half precision on GPU")
        
        MODEL_CACHE["classifier"] = {'tokenizer': tokenizer, 'model': model, 'device': DEVICE}
        print("✅ Classifier model loaded and optimized.")
        return MODEL_CACHE["classifier"]
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to load sentiment classifier: {e}")
        raise

# Flask app initialization
app = Flask(__name__)

# ==============================================================================
# 3. UTILITY FUNCTIONS
# ==============================================================================
def force_close_connection(func):
    """Decorator to help prevent network errors during scraping."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        original_init = requests.Session.__init__
        def new_init(self, *a, **k):
            original_init(self, *a, **k)
            self.headers['Connection'] = 'close'
        requests.Session.__init__ = new_init
        try:
            return func(*args, **kwargs)
        finally:
            requests.Session.__init__ = original_init
    return wrapper

def generate_cache_key(app_id, country, dr1_str, dr2_str):
    """Generate a cache key that correctly includes the country."""
    # Add normalization to ensure consistency
    app_id = str(app_id).strip() if app_id else ""
    country = str(country).strip().lower() if country else "us"
    dr1_str = str(dr1_str).strip() if dr1_str else ""
    dr2_str = str(dr2_str).strip() if dr2_str else ""
    return f"{app_id}_{country}_{dr1_str}_{dr2_str}"

def parse_date_range_string(date_str):
    if not date_str or ' - ' not in date_str: return None, None
    try:
        start_str, end_str = date_str.split(' - ')
        return datetime.strptime(start_str, "%Y-%m-%d"), datetime.strptime(end_str, "%Y-%m-%d")
    except (ValueError, TypeError): return None, None

def truncate_review_content(content, max_chars=200):
    return content[:max_chars - 3] + "..." if len(content) > max_chars else content

# ==============================================================================
# 4. GEMINI API & AI SUMMARIZATION
# ==============================================================================
def generate_gemini_response(prompt_text):
    """Sends a prompt to the Gemini API, with caching and rate limit handling."""
    global AI_RESULT_CACHE, GEMINI_API_LIMIT_REACHED, GEMINI_API_CALL_COUNT # Add counter to globals
    if GEMINI_API_LIMIT_REACHED: return None
    cache_key = hash(prompt_text)
    if cache_key in AI_RESULT_CACHE: return AI_RESULT_CACHE[cache_key]
    try:
        gemini_api_key = os.environ.get('GEMINI_API_KEY')
        if not gemini_api_key: return None
        # --- INCREMENT THE COUNTER HERE ---
        GEMINI_API_CALL_COUNT += 1
        print(f"🚀 Making Gemini API call #{GEMINI_API_CALL_COUNT}...")
        # ---------------------------------
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt_text)
        result = response.text if response.parts else None
        AI_RESULT_CACHE[cache_key] = result
        return result
    except Exception as e:
        if 'quota' in str(e).lower(): GEMINI_API_LIMIT_REACHED = True
        print(f"❌ Gemini API error: {e}")
        AI_RESULT_CACHE[cache_key] = None
        return None

def generate_category_summary(reviews, category_name, is_positive=False):
    top_snippets = [f"• \"{truncate_review_content(r['content'], 200)}\"" for r in reviews[:2]]
    fallback_summary = f"Key themes mentioned by users include:\n" + "\n".join(top_snippets)
    review_texts = "\n".join([f"- {r['content']}" for r in reviews[:15]])
    sentiment_type = "positive feedback" if is_positive else "user complaints"
    prompt = f'Analyze these user {sentiment_type} about "{category_name}". Write a concise, one-paragraph summary (40-60 words) of the main themes.\n\nReviews:\n{review_texts}\n\nSummary:'
    ai_summary = generate_gemini_response(prompt)
    return ai_summary if ai_summary else fallback_summary

def summarize_with_llm(reviews):
    top_reviews = sorted(reviews, key=lambda r: r.get('sentiment_score', 0))[:3]
    fallback_brief = "#### Top Critical Issues Identified\n\n"
    for i, r in enumerate(top_reviews):
        fallback_brief += f"**{i+1}. Issue:** Users report problems such as: \"_{truncate_review_content(r['content'], 200)}_\"\n"
        fallback_brief += "**Recommendation:** Investigate reports on performance and stability.\n\n"
    review_texts = "\n".join([f"- {r['content']}" for r in top_reviews])
    prompt = f'You are a Product Analyst. Analyze these critical reviews and create a brief. Identify top 2-3 themes. For each, provide a one-sentence problem and two actionable recommendations.\n\nReviews:\n{review_texts}\n\nFormat in Markdown:\n**1. Theme Name**\n**Problem:** [Statement]\n- **Recommendations:**\n[Suggestions]'
    ai_brief = generate_gemini_response(prompt)
    return markdown2.markdown(ai_brief) if ai_brief else markdown2.markdown(fallback_brief)

# ==============================================================================
# 5. CORE ANALYSIS LOGIC
# ==============================================================================


def analyze_reviews_roberta(review_list):
    """Optimized analysis with progress tracking and smaller batches."""
    if not review_list: 
        return {'total_review_count': 0, 'attention_reviews': [], 'praise_reviews': [], 'pain_points': {}, 'praise_points': {}, 'topics': {}, 'avg_sentiment_score': 0}
    
    valid_reviews = [r for r in review_list if r and isinstance(r.get('content'), str) and r.get('content').strip()]
    if not valid_reviews:
        return {'total_review_count': 0, 'attention_reviews': [], 'praise_reviews': [], 'pain_points': {}, 'praise_points': {}, 'topics': {}, 'avg_sentiment_score': 0}
    
    # Limit review count to prevent timeouts
    MAX_REVIEWS = 3000  # Adjust based on your server capacity
    if len(valid_reviews) > MAX_REVIEWS:
        valid_reviews = valid_reviews[:MAX_REVIEWS]
        print(f"Limited reviews to {MAX_REVIEWS} for performance")
    
    print(f"Processing {len(valid_reviews)} reviews...")
    classifier = get_classifier_model()
    texts = [r['content'] for r in valid_reviews]
    sentiments = []
    
    # Smaller batch size to prevent timeouts
    batch_size = 16  # Reduced from 32
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch_num = (i // batch_size) + 1
        print(f"Processing batch {batch_num}/{total_batches}")
        
        batch = texts[i:i + batch_size]
        inputs = classifier['tokenizer'](
            batch, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=256  # Reduced from 512
        ).to(classifier['device'])
        
        with torch.no_grad():
            outputs = classifier['model'](**inputs)
            batch_sentiments = outputs.logits.softmax(dim=-1).cpu().numpy()
            sentiments.extend(batch_sentiments)
        
        # Clear GPU cache periodically
        if classifier['device'] == "cuda" and batch_num % 5 == 0:
            torch.cuda.empty_cache()
    
    # Continue with existing analysis logic
    for r, score in zip(valid_reviews, sentiments): 
        r['sentiment_score'] = score[2] - score[0]
    
    POSITIVE_THRESHOLD, NEGATIVE_THRESHOLD = 0.2, -0.2
    category_mentions = {main: {'sub_topics': {sub: {'pos': 0, 'neg': 0, 'negative_reviews': [], 'positive_reviews': []} for sub in subs}, 'main_pos': 0, 'main_neg': 0, 'summary': '', 'positive_summary': ''} for main, subs in TOPIC_CATEGORIES.items()}
    
    for r in valid_reviews:
        is_pos, is_neg = r['sentiment_score'] > POSITIVE_THRESHOLD, r['sentiment_score'] < NEGATIVE_THRESHOLD
        for main_cat, sub_topics in TOPIC_CATEGORIES.items():
            for sub_cat, keywords in sub_topics.items():
                if any(kw in r['content'].lower() for kw in keywords):
                    stats = category_mentions[main_cat]['sub_topics'][sub_cat]
                    if is_pos: stats['pos'] += 1; stats['positive_reviews'].append(r)
                    elif is_neg: stats['neg'] += 1; stats['negative_reviews'].append(r)
                    break
    
    for main_cat, data in category_mentions.items():
        data['main_pos'] = sum(sub['pos'] for sub in data['sub_topics'].values())
        data['main_neg'] = sum(sub['neg'] for sub in data['sub_topics'].values())
        neg_reviews = sorted([r for sub in data['sub_topics'].values() for r in sub['negative_reviews']], key=lambda r: r['sentiment_score'])
        pos_reviews = sorted([r for sub in data['sub_topics'].values() for r in sub['positive_reviews']], key=lambda r: r['sentiment_score'], reverse=True)
        if data['main_neg'] >= 2: data['summary'] = generate_category_summary(neg_reviews, main_cat, is_positive=False)
        if data['main_pos'] >= 2: data['positive_summary'] = generate_category_summary(pos_reviews, main_cat, is_positive=True)
        for sub_cat, sub_data in data['sub_topics'].items():
            sub_data['negative_reviews_for_display'] = sorted(sub_data['negative_reviews'], key=lambda r: r['sentiment_score'])[:3]
            sub_data['positive_reviews_for_display'] = sorted(sub_data['positive_reviews'], key=lambda r: r['sentiment_score'], reverse=True)[:3]
    
    pain_points = {k: v for k, v in category_mentions.items() if v['main_neg'] > v['main_pos'] and v['main_neg'] > 0}
    praise_points = {k: v for k, v in category_mentions.items() if v['main_pos'] > v['main_neg'] and v['main_pos'] > 0}
    total_sentiment = sum(r['sentiment_score'] for r in valid_reviews)
    avg_score = ((total_sentiment / len(valid_reviews)) + 1) * 2.5 if valid_reviews else 0

    return {'total_review_count': len(valid_reviews), 'attention_reviews': sorted([r for r in valid_reviews if r['sentiment_score'] < NEGATIVE_THRESHOLD], key=lambda r: r['sentiment_score']), 'praise_reviews': sorted([r for r in valid_reviews if r['sentiment_score'] > POSITIVE_THRESHOLD], key=lambda r: r['sentiment_score'], reverse=True), 'pain_points': dict(sorted(pain_points.items(), key=lambda i: i[1]['main_neg'], reverse=True)), 'praise_points': dict(sorted(praise_points.items(), key=lambda i: i[1]['main_pos'], reverse=True)), 'topics': category_mentions, 'avg_sentiment_score': round(avg_score, 2)}

def cleanup_memory():
    """Clean up memory periodically."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def find_proof_reviews(reviews, category, limit=3):
    """Finds sample reviews for a specific category to show as 'proof'."""
    proof, keywords = [], [kw for sub in TOPIC_CATEGORIES.get(category, {}).values() for kw in sub]
    for review in reviews:
        if any(kw in review['content'].lower() for kw in keywords):
            proof.append(review)
            if len(proof) >= limit: break
    return proof

# In section 5. CORE ANALYSIS LOGIC

def identify_feature_requests(review_list):
    """Finds and categorizes feature requests into simple themes."""
    themes = {
        "UI & Customization": [],
        "New Functionality": [],
        "Integration & Sync": []
    }
    ui_kws = ["design", "ui", "theme", "color", "widget", "customize"]
    integ_kws = ["sync", "integrate", "connect", "export", "backup"]
    req_kws = FEATURE_REQUEST_INDICATORS

    for review in review_list:
        if review.get('sentiment_score', 0) >= -0.1: # Only from non-negative reviews
            content_lower = review['content'].lower()
            if any(kw in content_lower for kw in req_kws):
                if any(kw in content_lower for kw in ui_kws):
                    themes["UI & Customization"].append(review)
                elif any(kw in content_lower for kw in integ_kws):
                    themes["Integration & Sync"].append(review)
                else:
                    themes["New Functionality"].append(review)
    return {k: v for k, v in themes.items() if v} # Return only non-empty themes

def generate_structured_insights(analysis_present, analysis_previous, all_present_reviews):
    """Generates the final insights object, with proof reviews and themed feature requests."""
    ai_summary = summarize_with_llm(analysis_present['attention_reviews'])
    pp_present, pp_previous = analysis_present['pain_points'], analysis_previous['pain_points']
    
    # Correctly find proof reviews for each insight category
    persisting = {cat: {'count': data['main_neg'], 'reviews': find_proof_reviews(analysis_present['attention_reviews'], cat)} for cat, data in pp_present.items() if cat in pp_previous}
    newly_surfaced = {cat: {'count': data['main_neg'], 'reviews': find_proof_reviews(analysis_present['attention_reviews'], cat)} for cat, data in pp_present.items() if cat not in pp_previous}
    resolved = {cat: {'prev': prev['main_neg'], 'curr': analysis_present['topics'].get(cat, {}).get('main_neg', 0), 'reviews': find_proof_reviews(analysis_present['praise_reviews'], cat)} for cat, prev in pp_previous.items() if cat not in pp_present}

    # Generate themed feature requests
    feature_themes_dict = identify_feature_requests(all_present_reviews)
    feature_ideas = {}
    for theme_name, requests in feature_themes_dict.items():
        feature_ideas[theme_name] = {
            'requests': requests[:3], # Show top 3 samples
            'summary': f"{len(requests)} users requested improvements related to {theme_name}.",
            'urgency': 'Medium',
            'count': len(requests)
        }

    return {
        "persisting_problems": persisting,
        "newly_surfaced_problems": newly_surfaced,
        "resolved_problems": resolved,
        "feature_ideas": feature_ideas,
        "ai_summary": ai_summary
    }
# ==============================================================================
# 6. DATA SCRAPING & CACHING
# ==============================================================================
@force_close_connection
def scrape_reviews_for_app(app_id, country, target_date):
    all_reviews, token, batch_num = [], None, 0
    while len(all_reviews) < 15000 and batch_num < 25:
        try:
            result, token = reviews(app_id, lang='en', country=country, sort=Sort.NEWEST, count=200, continuation_token=token)
            if not result or not token: break
            all_reviews.extend(result)
            batch_num += 1
            if result[-1]['at'] < target_date: break
        except Exception as e:
            print(f"Scraping error: {e}"); break
    return all_reviews

@force_close_connection
def search_google_play(brand_name, country):
    """
    FIXED: Searches Google Play and raises an exception on failure
    so the frontend receives a proper error status.
    """
    try:
        results = search(brand_name, n_hits=20, lang='en', country=country)
        return [{'id': r['appId'], 'text': f"{r['title']} (⭐ {r.get('score', 0):.1f})"} for r in results[:10]]
    except Exception as e:
        print(f"Error in search_google_play: {e}")
        # Re-raise the exception to be caught by the Flask route
        raise e

# ==============================================================================
# 7. FLASK ROUTES
# ==============================================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search-apps', methods=['POST'])
def handle_search_apps():
    """Enhanced search with better error handling."""
    try:
        data = request.json
        brand_name = data.get('brand_name', '').strip()
        country = data.get('country', 'us').strip().lower()
        
        if not brand_name:
            return jsonify({'error': 'Brand name is required.'}), 400
            
        print(f"Searching for: {brand_name} in {country}")
        results = search_google_play(brand_name, country)
        return jsonify(results)
    except Exception as e:
        print(f"Search error: {e}")
        return jsonify({'error': f'App search failed. Please try again later. Error: {str(e)}'}), 500

@app.route('/scrape-reviews', methods=['POST'])
def handle_scrape_reviews():
    global REVIEW_CACHE
    data = request.json
    
    # More robust parameter extraction
    app_id = data.get('app_id', '').strip()
    country = data.get('country', 'us').strip().lower()
    dr1 = data.get('date_range_1', '').strip()
    dr2 = data.get('date_range_2', '').strip()
    
    print(f"Scrape params: app_id={app_id}, country={country}, dr1={dr1}, dr2={dr2}")
    
    s1, e1 = parse_date_range_string(dr1)
    s2, e2 = parse_date_range_string(dr2)
    
    if not all([app_id, country, s1, e1, s2, e2]):
        return jsonify({'error': 'Invalid parameters. Please complete all fields.'}), 400

    cache_key = generate_cache_key(app_id, country, dr1, dr2)
    
    if cache_key in REVIEW_CACHE:
        cached = REVIEW_CACHE[cache_key]
        print(f"Cache hit! Returning cached data.")
        return jsonify({'from_cache': True, 'present_count': len(cached['present']), 'previous_count': len(cached['previous'])})

    try:
        all_reviews = scrape_reviews_for_app(app_id, country, min(s1, s2))
        present = [r for r in all_reviews if s1 <= r['at'].replace(tzinfo=None) <= e1.replace(hour=23, minute=59)]
        previous = [r for r in all_reviews if s2 <= r['at'].replace(tzinfo=None) <= e2.replace(hour=23, minute=59)]
        
        REVIEW_CACHE[cache_key] = {'present': present, 'previous': previous}
        return jsonify({'from_cache': False, 'present_count': len(present), 'previous_count': len(previous)})
    except Exception as e:
        print(f"Scraping error: {e}")
        return jsonify({'error': f'Scraping failed: {str(e)}'}), 500

def find_cache_entry(app_id, country, dr1, dr2):
    """Find cache entry with fuzzy matching if exact match fails."""
    exact_key = generate_cache_key(app_id, country, dr1, dr2)
    
    if exact_key in REVIEW_CACHE:
        return exact_key, REVIEW_CACHE[exact_key]
    
    # Try alternate key formats for backward compatibility
    alt_keys = [
        f"{app_id}_{dr1}_{dr2}",  # without country
        f"{app_id}_{country.upper()}_{dr1}_{dr2}",  # uppercase country
    ]
    
    for alt_key in alt_keys:
        if alt_key in REVIEW_CACHE:
            print(f"Found cache with alternate key: {alt_key}")
            return alt_key, REVIEW_CACHE[alt_key]
    
    return None, None

# 7. Add cache status endpoint for debugging
@app.route('/cache-status', methods=['GET'])
def cache_status():
    """Debug endpoint to check cache contents."""
    return jsonify({
        'cache_keys': list(REVIEW_CACHE.keys()),
        'cache_count': len(REVIEW_CACHE),
        'entries': {k: {'present_count': len(v['present']), 'previous_count': len(v['previous'])} 
                   for k, v in REVIEW_CACHE.items()}
    })

@app.route('/view-cache', methods=['POST'])
def handle_view_cache():
    data = request.json
    cache_key = generate_cache_key(data.get('app_id'), data.get('country'), data.get('date_range_1'), data.get('date_range_2'))
    
    # Add debugging
    print(f"Looking for cache key: {cache_key}")
    print(f"Available cache keys: {list(REVIEW_CACHE.keys())}")
    
    if cache_key not in REVIEW_CACHE:
        return jsonify({'error': 'No cached reviews. Please scrape first.'}), 404
    cached_data = REVIEW_CACHE[cache_key]

    def format_for_display(reviews):
        return [{'id': i, 'content': truncate_review_content(r['content']), 'full_content': r['content'], 'score': r.get('score', 0), 'at': r['at'].strftime('%Y-%m-%d'), 'userName': r.get('userName', 'N/A'), 'selected': True} for i, r in enumerate(reviews)]

    return jsonify({
        'present_reviews': format_for_display(cached_data['present']),
        'previous_reviews': format_for_display(cached_data['previous'])
    })


# Replace your existing /analyze route:
@app.route('/analyze', methods=['POST'])
def handle_analyze():
    global GEMINI_API_CALL_COUNT
    GEMINI_API_CALL_COUNT = 0
    data = request.json
    cache_key = generate_cache_key(data.get('app_id'), data.get('country'), data.get('date_range_1'), data.get('date_range_2'))
    
    if cache_key not in REVIEW_CACHE:
        return jsonify({'error': 'No reviews to analyze. Please scrape reviews first.'}), 400
    
    # Get ALL reviews from the master cache
    all_present_reviews = REVIEW_CACHE[cache_key]['present']
    all_previous_reviews = REVIEW_CACHE[cache_key]['previous']

    # Get the lists of selected IDs from the frontend
    selected_present_ids = data.get('selected_present_ids')
    selected_previous_ids = data.get('selected_previous_ids')

    # If selections were made, filter the lists for analysis
    present_to_analyze = [all_present_reviews[i] for i in selected_present_ids] if selected_present_ids is not None else all_present_reviews
    previous_to_analyze = [all_previous_reviews[i] for i in selected_previous_ids] if selected_previous_ids is not None else all_previous_reviews
    
    print(f"Analyzing {len(present_to_analyze)} present and {len(previous_to_analyze)} previous reviews.")

    try:
        start_time = time.time()
        
        analysis_present = analyze_reviews_roberta(present_to_analyze)
        print(f"Present analysis completed in {time.time() - start_time:.2f}s")
        
        analysis_previous = analyze_reviews_roberta(previous_to_analyze)
        print(f"Previous analysis completed in {time.time() - start_time:.2f}s")
        
        insights = generate_structured_insights(analysis_present, analysis_previous, present_to_analyze)
        
        html = render_template(
            'results.html',
            analysis_present=analysis_present,
            analysis_previous=analysis_previous,
            insights=insights,
            count_present=len(present_to_analyze),
            count_previous=len(previous_to_analyze),
            form_data=data,
            gemini_limit_reached=GEMINI_API_LIMIT_REACHED
        )
        
        total_time = time.time() - start_time
        print(f"Total analysis completed in {total_time:.2f}s")
        
        return jsonify({'html': html})
        
    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    finally:
        cleanup_memory()

# Add this configuration
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 minutes

# ==============================================================================
# 8. APPLICATION ENTRY POINT
# ==============================================================================
if __name__ == '__main__':
    try:
        ngrok_authtoken = os.environ.get("NGROK_AUTHTOKEN")
        if ngrok_authtoken: 
            ngrok.set_auth_token(ngrok_authtoken)
        
        print("Starting ngrok tunnel...")
        # THE FIX: Pass keyword arguments directly, not in an 'options' dictionary.
        public_url = ngrok.connect(5000, bind_tls=True)
        
        print("="*80 + f"\n✅ App is live at: {public_url}\n" + "="*80)
        
        # Run the Flask app. Using threaded=True and use_reloader=False is good for stability in Colab/local dev.
        app.run(port=5000, debug=False, threaded=True, use_reloader=False)
        
    except Exception as e:
        print(f"❌ Startup error: {e}")
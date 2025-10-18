
import os
import time
import re
import sqlite3

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
import altair as alt

# ===============================
# NLTK Setup
# ===============================
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
os.environ["NLTK_DATA"] = nltk_data_dir
os.makedirs(nltk_data_dir, exist_ok=True)

for resource in ["punkt", "stopwords", "wordnet"]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_dir, quiet=True)

# ----------------- Streamlit Page Config -----------------
st.set_page_config(page_title="Reviews Lab", layout="wide")

# ----------------- Load Models (Cached) -----------------
@st.cache_data
def load_models():
    model = joblib.load("final_model.pkl")
    w2v_model = Word2Vec.load("word2vec_model.model")
    tfidf = joblib.load("tfidf.pkl")
    return model, w2v_model, tfidf

model, w2v_model, tfidf = load_models()

# ----------------- Helpers -----------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'<.*?>', ' ', text.lower())
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]

def weighted_vector(tokens, w2v_model, tfidf_vectorizer):
    word2weight = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))
    vectors, weights = [], []
    for w in tokens:
        if w in w2v_model.wv and w in word2weight:
            vectors.append(w2v_model.wv[w])
            weights.append(word2weight[w])
    if not vectors:
        return np.zeros(w2v_model.vector_size)
    return np.average(vectors, axis=0, weights=weights)

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9.,!?\'" ]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# ----------------- SQLite DB for history -----------------
conn = sqlite3.connect("reviews.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_text TEXT,
    sentiment TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# ----------------- Load CSS -----------------
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ----------------- Top Nav & Hero -----------------
st.markdown("""
<div class="top-nav"><div class="logo">Reviews Lab</div></div>
<div class="hero">
  <h1><em>Know What Your Customers Feel</em></h1>
  <p>Automatically analyze reviews, track sentiment, and find growth opportunities — fast and easy.</p>
</div>
""", unsafe_allow_html=True)

# ----------------- Tabs -----------------
tabs = st.tabs(["ASIN Analysis", "Manual Review", "CSV Upload", "History"])

# ----------------- Load Reviews -----------------
@st.cache_data
def load_reviews():
    return pd.read_csv("all_reviews.csv")  # must have 'asin', 'review_text', 'rating'

df = load_reviews()
available_asins = df['asin'].unique().tolist()

# ----------------- Tab 1: ASIN Review -----------------
with tabs[0]:
    # Page title
    st.markdown('<h2>ASIN Analysis</h2>', unsafe_allow_html=True)

    # ASIN selection
    asin_selected = st.selectbox("Select Amazon ASIN", available_asins)
    
    # Analyze button
    if st.button("Analyze ASIN") and asin_selected:
        st.info(f"Processing reviews for ASIN: {asin_selected}...")
        time.sleep(1)
        
        # Filter reviews
        df_asin = df[df['asin'] == asin_selected].copy()
        if df_asin.empty:
            st.warning("No reviews found for this ASIN.")
        else:
            # Preprocess text
            df_asin['tokens'] = df_asin['review_text'].apply(preprocess_text)
            
            # Vectorize
            df_asin['vec'] = df_asin['tokens'].apply(lambda x: weighted_vector(x, w2v_model, tfidf))
            
            # Predict sentiment
            df_asin['pred'] = df_asin['vec'].apply(lambda v: model.predict(v.reshape(1, -1))[0])
            
            # Show sentiment summary
            pos_pct = (df_asin['pred'] == 1).mean() * 100
            st.success(f"Sentiment Analysis Complete: {pos_pct:.1f}% Positive, {100 - pos_pct:.1f}% Negative")
            
            # Ratings chart
            if 'rating' in df_asin.columns:
                st.bar_chart(df_asin['rating'].value_counts().sort_index())
            
            # Product info
            product_cols = ['product_title', 'category', 'price']
            if all(col in df_asin.columns for col in product_cols):
                st.subheader("Product Information")
                product_info = df_asin[product_cols].iloc[0]
                for col in product_cols:
                    st.write(f"**{col.replace('_',' ').title()}:** {product_info[col]}")
            
            # Review summary
            st.subheader("Review Summary")
            for label, revs in [("Positive", df_asin[df_asin['pred']==1]['review_text']),
                                ("Negative", df_asin[df_asin['pred']==0]['review_text'])]:
                if not revs.empty:
                    st.markdown(f"**{label} Reviews:**")
                    for rev in revs.sample(min(5, len(revs))):
                        st.write(f"- {clean_text(rev)}")


# ----------------- Tab 2: Manual Review -----------------
with tabs[1]:
    st.markdown('<h2>Manual Review</h2>', unsafe_allow_html=True)

    # --- User Input ---
    review_input = st.text_area("Enter review text:")

    # Initialize session state to prevent duplicate DB insertions
    if "review_saved" not in st.session_state:
        st.session_state.review_saved = False
        st.session_state.last_review = ""

    if st.button("Analyze Review", key="manual"):
        if not review_input.strip():
            st.warning("Please enter a review.")
        else:
            # --- Preprocess and vectorize ---
            tokens = preprocess_text(review_input)
            vec = weighted_vector(tokens, w2v_model, tfidf).reshape(1, -1)

            # --- Sentiment Prediction ---
            pred = model.predict(vec)[0]
            sentiment = "Positive 😊" if pred == 1 else "Negative 😞"
            st.markdown(f"<h3>Sentiment: {sentiment}</h3>", unsafe_allow_html=True)

            # --- Prediction Confidence ---
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(vec)[0]
                st.subheader("Prediction Confidence")
                st.write(f"Positive: {prob[1]*100:.1f}%, Negative: {prob[0]*100:.1f}%")

            # --- Save to DB only once per unique review ---
            if not st.session_state.review_saved or st.session_state.last_review != review_input:
                c.execute("INSERT INTO reviews (review_text, sentiment) VALUES (?, ?)", (review_input, sentiment))
                conn.commit()
                st.session_state.review_saved = True
                st.session_state.last_review = review_input


            # --- Tokens Display ---
            st.subheader("Tokens Considered by Model")
            st.write(tokens)

            # --- Feature Request Detection ---
            st.subheader(" Feature Request Detection")
            feature_phrases = [
                "wish it had", "would be better if", "should have",
                "needs to", "could improve", "would like",
                "it lacks", "it doesn’t have", "missing", "could be added"
            ]
            detected_phrases = [p for p in feature_phrases if p in review_input.lower()]
            if detected_phrases:
                st.success("Potential feature requests detected:")
                st.write(", ".join(detected_phrases))
            else:
                st.info("No obvious feature request phrases detected.")

            # --- Recent Manual Reviews ---
            st.subheader("Recent Manual Reviews")
            rows = c.execute("SELECT review_text, sentiment, timestamp FROM reviews ORDER BY id DESC LIMIT 5").fetchall()
            if rows:
                for txt, sent, ts in rows:
                    color = "#10b981" if "Positive" in sent else "#ef4444"
                    st.markdown(
                        f"<div style='border-left:5px solid {color};padding:10px;margin:5px 0;'>"
                        f"{sent} ({ts})<br>{txt}</div>", unsafe_allow_html=True
                    )
            else:
                st.info("No manual reviews yet.")

            # --- Optional: Download Result ---
            df_result = pd.DataFrame([[review_input, sentiment]], columns=["Review", "Sentiment"])
            st.download_button(
                "Download Result as CSV",
                df_result.to_csv(index=False),
                "review_result.csv"
            )


# ----------------- Tab 3: CSV Upload -----------------
with tabs[2]:
    st.markdown('<h2 id="csv-upload">CSV Upload</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV with reviews", type=["csv"])

    if uploaded_file:
        try:
            df_csv = pd.read_csv(uploaded_file, quotechar='"', on_bad_lines='skip')
            st.write("Preview of your CSV:")
            st.dataframe(df_csv.head())

            # --- Row selection ---
            max_rows = len(df_csv)
            slice_option = st.selectbox("Select which part of the CSV to analyze", ["Top rows", "Bottom rows"])
            n_rows = st.number_input("Enter number of rows to analyze", 1, max_rows, min(1000, max_rows))
            df_csv = df_csv.head(n_rows) if slice_option == "Top rows" else df_csv.tail(n_rows)

            # --- Column selection ---
            review_col = st.selectbox("Select review column", df_csv.columns)
            product_col = st.selectbox("Select product column (optional)", [None] + df_csv.columns.tolist())
            date_col = st.selectbox("Select date column (optional for trend tracking)", [None] + df_csv.columns.tolist())

            if st.button("Analyze CSV"):
                if not review_col:
                    st.warning("Please select a valid review column.")
                else:
                    with st.spinner(f"Analyzing {len(df_csv)} rows..."):
                        progress_bar = st.progress(0)  # create progress bar
                        tokens_list = []
                        
                        # --- Text Preprocessing ---
                        for i, r in enumerate(df_csv[review_col]):
                            tokens_list.append(preprocess_text(str(r)))
                            
                            # update progress
                            progress_bar.progress(int((i+1)/len(df_csv) * 100))
                        
                        df_csv['tokens'] = tokens_list
                        
                        # --- Sentiment Prediction ---
                        vectors = np.vstack([weighted_vector(t, w2v_model, tfidf) for t in df_csv['tokens']])
                        preds = model.predict(vectors)
                        df_csv['sentiment'] = np.where(preds == 1, 'Positive', 'Negative')

                        pos_pct = (preds == 1).mean() * 100
                        st.success(f"Analysis Complete: {pos_pct:.1f}% Positive, {100-pos_pct:.1f}% Negative")
                        st.dataframe(df_csv[[review_col, 'sentiment']].head(10))

                        # --- Product Sentiment Chart ---
                        if product_col and product_col in df_csv.columns:
                            st.subheader("Product-level Sentiment Bar Chart")
                            sentiment_counts = (
                                df_csv.groupby([product_col, 'sentiment'])
                                .size().reset_index(name='count')
                            )
                            sentiment_counts['pct'] = (
                                sentiment_counts.groupby(product_col)['count']
                                .transform(lambda x: 100 * x / x.sum())
                            ).round(1)

                            import altair as alt 
                            chart = alt.Chart(sentiment_counts).mark_bar().encode(
                                x='count:Q',
                                y=alt.Y(f'{product_col}:N', sort='-x'),
                                color='sentiment:N',
                                tooltip=[product_col, 'sentiment', 'count', alt.Tooltip('pct', format='.1f')]
                            )
                            st.altair_chart(chart, use_container_width=True)

                        # --- Sentiment Trend Chart ---
                        if date_col and date_col in df_csv.columns:
                            try:
                                df_csv[date_col] = pd.to_datetime(df_csv[date_col], errors='coerce')
                                trend_df = (
                                    df_csv.groupby(pd.Grouper(key=date_col, freq='M'))['sentiment']
                                    .apply(lambda x: (x == 'Positive').mean() * 100)
                                    .reset_index(name='positive_pct')
                                )
                                trend_chart = alt.Chart(trend_df).mark_line(point=True).encode(
                                    x=alt.X(f'{date_col}:T', title='Month'),
                                    y=alt.Y('positive_pct:Q', title='Positive Sentiment (%)'),
                                    tooltip=[f'{date_col}:T', alt.Tooltip('positive_pct:Q', format='.1f')]
                                ).properties(height=300)
                                st.subheader("Sentiment Trend Over Time")
                                st.altair_chart(trend_chart, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Could not generate trend chart: {e}")
                        else:
                            st.info("Select a date column to enable sentiment trend tracking.")

                        # --- Feature Request Detection ---
                        st.subheader("Feature Request Detection")
                        feature_phrases = [
                            "wish it had", "would be better if", "should have", "needs to",
                            "could improve", "would like", "it lacks", "it doesn’t have",
                            "missing", "could be added"
                        ]
                        df_feature_req = df_csv[df_csv[review_col].str.lower().apply(
                            lambda t: any(p in t for p in feature_phrases)
                        )]
                        if not df_feature_req.empty:
                            st.success(f"Detected {len(df_feature_req)} potential feature request(s).")
                            st.dataframe(df_feature_req[[review_col]].head(10))
                        else:
                            st.info("No clear feature request patterns found.")

        except Exception as e:
            st.error(f"Error reading CSV: {e}")


# ----------------- Tab 4: History -----------------
with tabs[3]:
    st.markdown('<h2 id="history">History</h2>', unsafe_allow_html=True)
    c.execute("SELECT review_text, sentiment, timestamp FROM reviews ORDER BY id DESC LIMIT 20")
    rows = c.fetchall()
    if rows:
        for txt, sent, ts in rows:
            color = "#10b981" if "Positive" in sent else "#ef4444"
            st.markdown(f"<div style='border-left:5px solid {color};padding:10px;margin:5px 0;'>{sent} ({ts})<br>{txt}</div>", unsafe_allow_html=True)
    else:
        st.info("No reviews yet.")

    # Clear history button
    if st.button("Clear History"):
        c.execute("DELETE FROM reviews")
        conn.commit()
        st.info("History cleared!")

# ----------------- Features Grid -----------------
st.markdown('<h2>Key Features</h2>', unsafe_allow_html=True)

feature_texts = [
    ("AI Sentiment Insights", "Detect emotions and tone across thousands of reviews with advanced NLP."),
    ("Feature Request Detection", "Identify trending customer requests to guide your next product update."),
    ("Competitor Comparison", "See how your product’s sentiment and features stack up against others."),
    ("Trend Tracking", "Monitor changes in sentiment and topics over time with clean visual charts."),
    ("Auto Reports & Export", "Generate shareable reports in PDF or CSV for quick insights."),
]

for i in range(0, len(feature_texts), 3):
    # Create a row of 3 columns
    cols = st.columns(3)
    for j, (title, desc) in enumerate(feature_texts[i:i+3]):
        cols[j].markdown(f'<div class="feature-card"><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)
    # Add space after each row
    st.markdown("<br>", unsafe_allow_html=True)

# ----------------- Connect With Me Section -----------------
st.markdown("""
<div style="text-align:center; padding:20px; border: 2px solid #f3f4f6; border-radius:12px; margin-bottom:40px;">
    <h3>Hello! 👋 I'm Faisal Khan</h3>
    <p>I'm passionate about data, machine learning, and building tools that turn insights into action. Connect with me!</p>
    <div style="display:flex; justify-content:center; gap:15px; margin-top:20px;">
        <a href="https://personal-portfolio-alpha-lake.vercel.app/" target="_blank" style="text-decoration:none; background-color:#e11d48; color:white; padding:10px 20px; border-radius:8px; font-weight:bold;">Portfolio</a>
        <a href="https://www.linkedin.com/in/faisal-khan23" target="_blank" style="text-decoration:none; background-color:#0e76a8; color:white; padding:10px 20px; border-radius:8px; font-weight:bold;">LinkedIn</a>
        <a href="https://github.com/Faisal-khann" target="_blank" style="text-decoration:none; background-color:#333; color:white; padding:10px 20px; border-radius:8px; font-weight:bold;">GitHub</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- Footer -----------------
st.markdown("""
<footer style="text-align:center; padding:28px 20px; font-size:13px; color:#6b7280; border-top:1px solid #f3f4f6; margin-top:34px;">
  © 2025 ReviewsLab — Built for product-led teams<br>
  Made with ❤️ using <b>Streamlit</b> & <b>Machine Learning</b><br>
  Developed by <b>Faisal Khan</b> | <a href="https://github.com/Faisal-khann" target="_blank">GitHub</a>
</footer>
""", unsafe_allow_html=True)
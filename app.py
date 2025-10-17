import streamlit as st
import pandas as pd
import numpy as np
import joblib
from gensim.models import Word2Vec
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import sqlite3
import time

# ----------------- Load NLTK resources -----------------
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ----------------- Streamlit Page Config -----------------
st.set_page_config(page_title="ReviewsLab - Amazon Reviews Analyzer", layout="wide")

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
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
    return tokens

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

# ----------------- Custom CSS -----------------
st.markdown("""
<style>
html, body, [class*="css"]  {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
  background: #fff;
  color: #111827;
}

/* Top navigation */
.top-nav{
  background: linear-gradient(180deg, #e0f0ff 0%, #ffffff 100%);
  box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  position: sticky;
  top: 0;
  z-index: 999;
  padding: 12px 40px;
  display:flex;
  justify-content:space-between;
  align-items:center;
  border-radius:6px;
  margin-bottom:18px;
}
.logo { font-weight:700; font-size:20px; color:#e11d48; }
.navlinks a {
  text-decoration: none;
  color: #fff;
  background: #e11d48;
  padding: 8px 18px;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 600;
  box-shadow: 0 3px 6px rgba(0,0,0,0.12);
  transition: background 0.3s, transform 0.2s, box-shadow 0.2s;
}
.navlinks a:hover {
  background: #be123c;
  transform: translateY(-2px);
  box-shadow: 0 6px 12px rgba(0,0,0,0.16);
}

/* Feature cards */
.feature-card {
  background: #fff;
  border: 2px solid #f3f4f6;
  border-radius: 10px;
  box-shadow: 0 2px 5px rgba(0,0,0,0.03);
  padding: 30px;
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  transition: transform 0.3s, box-shadow 0.3s;
  height: 100%;
}
.feature-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 10px 20px rgba(0,0,0,0.12);
}

/* Hide Streamlit default menu/footer */
#MainMenu {visibility: hidden;}
footer[data-testid="stFooter"] {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------- Top Nav -----------------
st.markdown("""
<div class="top-nav">
  <div class="logo">ReviewsLab</div>
  <div class="navlinks">
    <a href="#asin-analysis">ASIN Analysis</a>
    <a href="#manual-review">Manual Review</a>
    <a href="#csv-upload">CSV Upload</a>
    <a href="#history">History</a>
  </div>
</div>
""", unsafe_allow_html=True)

# ----------------- Hero Section -----------------
st.markdown("""
<div class="hero">
  <h1>Turn Amazon reviews into product intelligence</h1>
  <p>Analyze thousands of reviews in seconds — find trends, sentiment, and opportunities.</p>
</div>
""", unsafe_allow_html=True)

# ----------------- Tabs -----------------
tabs = st.tabs(["ASIN Analysis", "Manual Review", "CSV Upload", "History"])

# ----------------- Tab 1: ASIN Analysis -----------------
with tabs[0]:
    st.markdown('<h2 id="asin-analysis">ASIN Analysis</h2>', unsafe_allow_html=True)
    asin_input = st.text_input("Enter Amazon ASIN (e.g. B08XYZ123)")
    if st.button("Analyze ASIN", key="asin"):
        if asin_input.strip() == "":
            st.warning("Please enter a valid ASIN.")
        else:
            st.info("Processing reviews...")
            time.sleep(1)  # simulate loading
            # Load all reviews CSV
            df = pd.read_csv("all_reviews.csv")  # replace with your file
            df_asin = df[df['asin']==asin_input]
            if df_asin.empty:
                st.warning("No reviews found for this ASIN.")
            else:
                df_asin['tokens'] = df_asin['review_text'].apply(preprocess_text)
                df_asin['vec'] = df_asin['tokens'].apply(lambda x: weighted_vector(x, w2v_model, tfidf))
                df_asin['pred'] = df_asin['vec'].apply(lambda v: model.predict(v.reshape(1,-1))[0])
                pos_pct = (df_asin['pred']==1).mean()*100
                st.success(f"Sentiment Analysis Complete: {pos_pct:.1f}% Positive, {100-pos_pct:.1f}% Negative")

                # Ratings distribution chart
                if 'rating' in df_asin.columns:
                    st.bar_chart(df_asin['rating'].value_counts().sort_index())

                # Word cloud for positive reviews
                text = " ".join(df_asin[df_asin['pred']==1]['review_text'])
                wordcloud = WordCloud(width=800, height=400).generate(text)
                st.image(wordcloud.to_array(), use_column_width=True)

# ----------------- Tab 2: Manual Review -----------------
with tabs[1]:
    st.markdown('<h2 id="manual-review">Manual Review</h2>', unsafe_allow_html=True)
    review_input = st.text_area("Enter review text:")
    if st.button("Analyze Review", key="manual"):
        if review_input.strip() == "":
            st.warning("Please enter a review.")
        else:
            tokens = preprocess_text(review_input)
            vec = weighted_vector(tokens, w2v_model, tfidf).reshape(1,-1)
            pred = model.predict(vec)[0]
            sentiment = "Positive 😊" if pred==1 else "Negative 😞"
            st.markdown(f"<h3>Sentiment: {sentiment}</h3>", unsafe_allow_html=True)

            # Save to DB
            c.execute("INSERT INTO reviews (review_text, sentiment) VALUES (?,?)", (review_input, sentiment))
            conn.commit()

# ----------------- Tab 3: CSV Upload -----------------
# ----------------- Tab 3: CSV Upload -----------------
with tabs[2]:
    st.markdown('<h2 id="csv-upload">CSV Upload</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV with reviews", type=["csv"])
    
    if uploaded_file:
        try:
            df_csv = pd.read_csv(uploaded_file, quotechar='"', on_bad_lines='skip')
            st.write("Preview of your CSV:")
            st.dataframe(df_csv.head())

            # --- Row selection for faster processing ---
            max_rows = len(df_csv)
            slice_option = st.selectbox(
                "Select which part of the CSV to analyze",
                ["Top rows", "Bottom rows"]
            )
            n_rows = st.number_input(
                "Enter number of rows to analyze",
                min_value=1,
                max_value=max_rows,
                value=min(1000, max_rows),
                step=1
            )

            # Slice dataframe based on selection
            if slice_option == "Top rows":
                df_csv_slice = df_csv.head(n_rows)
            else:  # Bottom rows
                df_csv_slice = df_csv.tail(n_rows)

            # --- Column selection ---
            review_col = st.selectbox("Select the column containing reviews", df_csv.columns.tolist())
            product_col = st.selectbox(
                "Select the column containing product names (for sentiment bar chart)", 
                df_csv.columns.tolist()+[None]
            )

            # --- Analyze CSV ---
            if st.button("Analyze CSV"):
                if review_col.strip() == "":
                    st.warning("Please select a valid review column.")
                else:
                    with st.spinner(f"Analyzing {len(df_csv_slice)} rows..."):
                        # Preprocess reviews
                        tokens_list = []
                        progress = st.progress(0)
                        total = len(df_csv_slice)
                        for i, review in enumerate(df_csv_slice[review_col]):
                            tokens_list.append(preprocess_text(review))
                            progress.progress((i+1)/total)
                        df_csv_slice['tokens'] = tokens_list

                        # Batch vectorization & prediction
                        vectors = np.vstack([weighted_vector(t, w2v_model, tfidf) for t in df_csv_slice['tokens']])
                        preds = model.predict(vectors)
                        df_csv_slice['sentiment'] = np.where(preds==1, 'Positive', 'Negative')

                        # Summary and preview
                        st.success(
                            f"CSV Analysis Complete: {(preds==1).mean()*100:.1f}% Positive, {(preds==0).mean()*100:.1f}% Negative"
                        )
                        st.dataframe(df_csv_slice[[review_col,'sentiment']].head(10))

                        # --- Product-level bar chart with percentage tooltips ---
                        if product_col and product_col in df_csv_slice.columns:
                            st.subheader("Product-level Sentiment Bar Chart")
                            # Count reviews per product and sentiment
                            sentiment_counts = df_csv_slice.groupby([product_col, 'sentiment']).size().reset_index(name='count')
                            # Total reviews per product
                            total_counts = df_csv_slice.groupby(product_col).size().reset_index(name='total')
                            # Merge to calculate percentage
                            sentiment_counts = sentiment_counts.merge(total_counts, on=product_col)
                            sentiment_counts['pct'] = (sentiment_counts['count'] / sentiment_counts['total'] * 100).round(1)

                            import altair as alt
                            chart = alt.Chart(sentiment_counts).mark_bar().encode(
                                x=alt.X('count:Q', axis=alt.Axis(title='Review Count')),
                                y=alt.Y(f'{product_col}:N', sort='-x'),
                                color='sentiment:N',
                                tooltip=[
                                    alt.Tooltip(product_col, title="Product"),
                                    alt.Tooltip('sentiment', title="Sentiment"),
                                    alt.Tooltip('count', title='Review Count'),
                                    alt.Tooltip('pct', title='Percentage', format=".1f")
                                ]
                            )
                            st.altair_chart(chart, use_container_width=True)

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
cols = st.columns(3)
for i, (title, desc) in enumerate(feature_texts):
    col = cols[i % 3]
    col.markdown(f'<div class="feature-card"><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)

# ----------------- Footer -----------------
st.markdown("""
<footer style="text-align:center; padding:28px 20px; font-size:13px; color:#6b7280; border-top:1px solid #f3f4f6; margin-top:34px;">
  © 2025 ReviewsLab — Built for product-led teams<br>
  Made with ❤️ using <b>Streamlit</b> & <b>Machine Learning</b><br>
  Developed by <b>Faisal Khan</b> | <a href="https://github.com/yourusername" target="_blank">GitHub</a>
</footer>
""", unsafe_allow_html=True)

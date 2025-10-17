# Customer Sentiment Analyzer

<!--![Sentiment Analyzer Banner](banner_image_link_here)-->

<em>
A web application to analyze customer reviews on Amazon products. This tool leverages a full <strong>NLP pipeline</strong> combined with <strong>ML model building</strong> to understand customer sentiment both automatically and manually. It summarizes sentiments, performs trend analysis, and visualizes sentiment distribution using interactive charts. All manual reviews are stored in a database for historical tracking and further data cleaning.
</em>
  

---

## Table of Contents
- [Business Problem](#business-problem)
- [Features](#features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Business Problem

In the era of e-commerce, customers heavily rely on reviews before making a purchase. Amazon products often receive **hundreds or thousands of reviews**, making it challenging for:  

- **Customers** to quickly understand the overall sentiment of a product.  
- **Sellers** to monitor and analyze customer feedback efficiently.  
- **Businesses** to track sentiment trends over time and improve their products.  

Manual analysis of reviews is time-consuming, error-prone, and inefficient.  

**Our solution: Customer Sentiment Analyzer** solves this problem by:  

- Automatically extracting reviews using the product **ASIN**.  
- Classifying sentiment (positive, negative, neutral) for individual reviews.  
- Summarizing sentiment across multiple reviews.  
- Allowing users to **manually input reviews** and analyze sentiment.  
- Uploading **CSV files** containing reviews for batch analysis.  
- Performing **trend analysis and percentage calculation** for product sentiment.  
- Visualizing sentiment with **bar charts** for quick insights.  
- Storing all manual reviews in a **database** for historical tracking and cleaning.  

This allows both customers and businesses to make **data-driven decisions**, save time, and improve customer satisfaction.

---

## Features

- Extract sentiment from reviews using **ASIN**  
- Manual review sentiment analysis  
- CSV upload for bulk sentiment analysis  
- Sentiment trend analysis  
- Percentage of positive, negative, and neutral reviews  
- Bar chart visualization  
- Database to store historical manual reviews  
- Data cleaning options for stored reviews  

---

## How It Works

1. **ASIN Analysis**: Enter the Amazon product ASIN → App fetches reviews → Sentiment classification → Summary generated.  
2. **Manual Review Input**: Enter your own review → Sentiment is predicted instantly.  
3. **CSV Upload**: Upload a CSV of reviews → Sentiment is predicted for each row → Trend analysis and visualization.  
4. **Database Storage**: All manual reviews are stored in a database → Can be cleaned and reused for further analysis.  

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Customer_Sentiment_Analyzer.git

# Navigate into the project directory
cd Customer_Sentiment_Analyzer

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

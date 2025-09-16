# AppAnalyzer2.0
An AI-powered web dashboard that analyzes Google Play Store reviews using Google's Gemini API and Sentiment Analysis to provide actionable product insights.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joyduttahere/AppAnalyzer2.0/blob/main/Share_AppAnalyzer2_0.ipynb)
<br>
_Click the badge above to launch the interactive demo in Google Colab._

---

## Overview

**App Analyzer 2.0** is a web-based intelligence dashboard that leverages Google's Gemini API and advanced sentiment analysis to transform raw user reviews from the Google Play Store into actionable insights.

This tool is designed for product managers, marketers, and developers to quickly understand user feedback, track issues over time, and identify key areas for improvement. It features a sophisticated, interactive interface for scraping, filtering, and analyzing reviews in a seamless workflow.

**App Analyzer Screenshot:**
<img width="1002" height="898" alt="image" src="https://github.com/user-attachments/assets/f9bee2c3-ad8d-4041-94e5-4eb7200b715b" />

<img width="1105" height="703" alt="image" src="https://github.com/user-attachments/assets/d2828e73-00d7-4818-b985-b17e47680c2b" />

<img width="1756" height="876" alt="image" src="https://github.com/user-attachments/assets/d8f2e1b3-c559-472f-ab1c-c45e19276478" />

## Key Features

-   **Comparative Analysis:** Analyze and compare two distinct date ranges to track the evolution of user feedback.
-   **AI-Powered Summarization:** Utilizes the **Google Gemini API** for high-quality, context-aware summaries of user pain points, praise points, and critical issues.
-   **Accurate Sentiment Scoring:** Employs a RoBERTa-based model for precise sentiment classification of each review.
-   **Automated Topic Categorization:** Intelligently groups reviews into key topics like "App Stability," "User Experience," and "Features."
-   **Interactive Review Viewer:** A powerful modal to view, search, and filter scraped reviews by text, star rating, or username before analysis.
-   **Actionable Insights Dashboard:**
    -   **Persisting Problems:** Highlights issues that affect users across both time periods.
    -   **Newly Surfaced Problems:** Identifies new issues that have recently appeared.
    -   **Resolved Problems:** Validates bug fixes by showing which old complaints have disappeared.
    -   **Feature Request Theming:** Automatically groups constructive user suggestions into actionable themes.

## How to Run in Google Colab (Recommended)

This project is designed to be easily run in a free Google Colab environment with a GPU.

### Prerequisites

Before you start, you will need three free API keys:
1.  **Google Gemini API Key:** From [Google AI Studio](https://aistudio.google.com/).
2.  **Hugging Face Access Token:** From your [Hugging Face Tokens page](https://huggingface.co/settings/tokens) (role: "read").
3.  **ngrok Authtoken:** From your [ngrok Dashboard](https://dashboard.ngrok.com/get-started/your-authtoken).

### Quickstart Instructions

1.  **Click the "Open in Colab" Badge** at the top of this README.
2.  **Set the GPU Runtime:** In the Colab menu, go to **Runtime → Change runtime type** and select **T4 GPU**.
3.  **Add Your Secret Keys:** The notebook will prompt you to add your three keys (`GEMINI_API_KEY`, `HF_TOKEN`, `NGROK_AUTHTOKEN`) using the Colab Secrets Manager (🔑).
4.  **Run the Cells:** Execute the notebook cells from top to bottom. The code will set up the environment, install all dependencies, and launch the web app.
5.  **Access the App:** An `ngrok` URL will be displayed. Click it to open the App Analyzer dashboard in a new browser tab.

## Local Development Setup

To run this project on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/AppAnalyzer2.0.git
    cd AppAnalyzer2.0
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create a `.env` file** in the root directory and add your keys:
    ```
    GEMINI_API_KEY="your_gemini_key"
    HF_TOKEN="your_huggingface_token"
    NGROK_AUTHTOKEN="your_ngrok_authtoken"
    ```

5.  **Run the application:**
    ```bash
    python app.py
    ```

## Technology Stack

-   **Backend:** Flask (Python)
-   **Frontend:** HTML, CSS, JavaScript (with Litepicker.js)
-   **AI / ML:**
    -   Google Gemini API for Summarization
    -   Hugging Face Transformers (RoBERTa) for Sentiment Analysis
    -   PyTorch
-   **Data Scraping:** `google-play-scraper`
-   **Deployment:** Google Colab, Ngrok

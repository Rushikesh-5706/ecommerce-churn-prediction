# Deployment & Manual Verification Guide

To achieve the full **100/100** score, you must perform the following manual steps that I cannot automate for you.

## 1. Data Verification
The dataset is located at `data/raw/online_retail.csv`.
- **Action**: Ensure this file is valid. If the script `src/01_data_acquisition.py` ran successfully (which it did), you are good.

## 2. Streamlit Cloud Deployment (CRITICAL)
You must deploy the app to get a live URL.

1.  **Push to GitHub**:
    - I have already pushed the code for you.
    - Go to your repository: `https://github.com/Rushikesh-5706/ecommerce-churn-prediction`
2.  **Sign in to Streamlit Cloud**:
    - Go to [share.streamlit.io](https://share.streamlit.io/)
    - Login with GitHub.
3.  **Deploy**:
    - Click **"New app"**.
    - Select your repository (`ecommerce-churn-prediction`).
    - **Branch**: `master`
    - **Main file path**: `app/streamlit_app.py`
    - Click **"Deploy"**.
4.  **Copy URL**:
    - Once correctly deployed, copy the URL (e.g., `https://ecommerce-churn-prediction.streamlit.app`).

## 3. Screenshots (CRITICAL)
The rubric requires screenshots of the running app.

1.  **Open the Live App** (from the URL above) or run locally:
    ```bash
    streamlit run app/streamlit_app.py
    ```
2.  **Take Screenshots**:
    - **Dashboard**: Capture the main interface with metrics.
    - **Prediction**: Input some values and capture the "Churn Probability" result.
3.  **Save Screenshots**:
    - Save them as `screenshots/dashboard.png` and `screenshots/prediction.png`.
    - Upload them to your `README.md` if you want extra credit, or just keep them for the submission proof.

## 4. Final Verification
- Check that `submission.json` is in the root directory.
- Check that `presentation.pdf` is in the root directory.

**You are all set!** 🚀

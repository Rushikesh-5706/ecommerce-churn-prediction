# Deployment Guide

##  Overview

This guide provides step-by-step instructions for deploying the Customer Churn Prediction System to Streamlit Cloud.

**Live Application URL**: [To be added after deployment]

---

## Prerequisites

1. ✅ GitHub account
2. ✅ Streamlit Cloud account (free tier: share.streamlit.io)
3. ✅ All project files ready in GitHub repository
4. ✅ `requirements.txt` complete
5. ✅ Trained model saved as `models/best_model.pkl`

---

## Deployment Steps

### Step 1: Prepare GitHub Repository

```bash
# Initialize git (if not done already)
cd "/Users/rushikesh/Customer Churn Prediction System"
git init

# Add all files
git add .

# Commit
git commit -m "chore: Prepare for deployment"

# Create GitHub repository (via GitHub UI or CLI)
# Then push
git remote add origin https://github.com/YOUR_USERNAME/customer-churn-prediction.git
git branch -M main
git push -u origin main
```

**Important**: Ensure `.gitignore` excludes data files:
```.gitignore
data/
*.csv
*.pkl
models/*.pkl  
# Keep models/scaler.pkl for deployment
```

Note: You'll need to add dummy/sample model files for Streamlit Cloud deployment.

###Step 2: Create Model Artifacts for Deployment

Since models are .gitignored, create them in the deployed environment:

**Option A: Include lightweight models**
```bash
# Reduce model file size
import joblib
model = joblib.load('models/best_model.pkl')
joblib.dump(model, 'models/best_model_lite.pkl', compress=3)
```

**Option B: Download from cloud storage** (Recommended for production)
- Upload models to Google Drive / AWS S3
- Add download script in `app/streamlit_app.py` startup

### Step 3: Verify requirements.txt

Ensure all dependencies are listed:
```txt
pandas==2.2.3
numpy==2.3.0
scikit-learn==1.6.1
matplotlib==3.10.0
seaborn==0.13.2
plotly==5.24.1
streamlit==1.42.0
joblib==1.4.2
Pillow==11.1.0
imbalanced-learn==0.14.1
```

**Verify locally**:
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

### Step 4: Deploy to Streamlit Cloud

1. **Go to** [share.streamlit.io](https://share.streamlit.io)
2. **Sign in** with GitHub
3. **Click** "New app"
4. **Configure**:
   - Repository: `YOUR_USERNAME/customer-churn-prediction`
   - Branch: `main`
   - Main file path: `app/streamlit_app.py`
   - Python version: `3.12`
5. **Advanced Settings** (optional):
   - Secrets: Add any API keys if needed
   - Resources: Default (1 CPU, 800MB RAM)
6. **Click** "Deploy!"

### Step 5: Waitfor Deployment

- Initial build: 5-10 minutes
- Streamlit will install dependencies from `requirements.txt`
- Watch logs for errors

**Common Issues**:
- Model file not found → Add dummy model or download logic
- Memory error → Reduce model size or upgrade tier
- Import errors → Check `requirements.txt` versions

### Step 6: Test Deployed Application

Once live, test all features:
- ✅ Home page loads
- ✅ Single prediction works
- ✅ Batch prediction (upload CSV)
- ✅ Model dashboard displays
- ✅ No errors in logs

---

## Post-Deployment Checklist

### Functional Testing

- [ ] **Single Prediction**:
  - Enter customer features
  - Get churn probability & recommendation
- [ ] **Batch Prediction**:
  - Upload sample_customers.csv (10 rows)
  - Download predictions
- [ ] **Model Dashboard**:
  - Confusion matrix displays
  - Metrics visible
  - ROC curve loaded

### Performance Testing

- [ ] Page load time < 3 seconds
- [ ] Prediction latency < 500ms
- [ ] No memory warnings in logs

### Documentation Updates

- [ ] Update README.md with live URL
- [ ] Update submission.json with deployment URL
- [ ] Add screenshot of live app to README

---

## Configuration Options

### Environment Variables

Create `.streamlit/secrets.toml` (not committed to git):
```toml
[general]
app_name = "Customer Churn Prediction"
admin_email = "your-email@example.com"

[model]
version = "1.0.0"
roc_auc_threshold = 0.75
```

Access in code:
```python
import streamlit as st
app_name = st.secrets["general"]["app_name"]
```

### Custom Domain (Optional)

Streamlit Cloud free tier provides:
- Default: `https://YOUR_USERNAME-customer-churn-prediction-xyz123.streamlit.app`

For custom domain:
1. Upgrade to paid tier ($10/month)
2. Configure CNAME record
3. Update in Streamlit dashboard

---

## Monitoring & Maintenance

### Check Application Health

**Weekly**:
- Visit live URL to ensure uptime
- Check Streamlit Cloud dashboard for errors
- Review usage analytics

**Monthly**:
- Update dependencies (security patches)
- Rebuild app with latest `requirements.txt`

### Retraining Model

**Quarterly** (every 3 months):
1. Retrain model on latest data
2. Save new `best_model.pkl`
3. Push to GitHub
4. Streamlit auto-redeploys

Alternatively:
```bash
# Manual redeploy
git commit -m "chore: Update model (Q1 2026)"
git push
# Streamlit detects changes and redeploys automatically
```

### Debugging

**View Logs**:
- Streamlit Cloud Dashboard → Your App → View Logs
- Real-time error messages
- Print statements visible here

**Common Errors**:
| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError` | Missing dependency | Add to `requirements.txt` |
| `FileNotFoundError: models/best_model.pkl` | Model not in repo | Add download logic or commit lite version |
| `MemoryError` | App exceeds 800MB | Reduce model size or upgrade tier |
| `ImportError: pandas` | Wrong version | Pin exact version in `requirements.txt` |

---

## Local Development

### Running Locally

```bash
# Navigate to project
cd "/Users/rushikesh/Customer Churn Prediction System"

# Activate virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app/streamlit_app.py
```

Application opens at `http://localhost:8501`

### Hot Reload

Streamlit auto-reloads on file changes:
- Edit `app/streamlit_app.py`
- Save file
- Browser refreshes automatically

---

## Docker Deployment (Alternative)

If Streamlit Cloud is unavailable, use Docker:

### Build Image

```bash
docker build -t churn-prediction .
```

### Run Container

```bash
docker run -p 8501:8501 churn-prediction
```

### Docker Compose

```bash
docker-compose up
```

Application at `http://localhost:8501`

---

## Troubleshooting Guide

### Issue: Blank page after deployment

**Symptoms**: Streamlit shows "Please wait..." indefinitely

**Causes**:
1. Syntax error in `streamlit_app.py`
2. Missing required file
3. Import fails

**Solution**:
```bash
# Check logs in Streamlit Cloud dashboard
# Look for Python traceback
# Fix error in code
# Push update to GitHub
```

### Issue: Model predictions fail

**Symptoms**: "Error loading model" or predictions return NaN

**Causes**:
1. Model file corrupted
2. Feature mismatch (different feature count)
3. Scaler not found

**Solution**:
```python
# Add defensive checks in app
import os
if not os.path.exists('models/best_model.pkl'):
    st.error("Model file not found. Please retrain.")
    st.stop()
```

### Issue: Slow performance

**Symptoms**: Predictions take >5 seconds

**Causes**:
1. Large model file
2. Inefficient code
3. Free tier resource limits

**Solution**:
```python
# Cache model loading
@st.cache_resource
def load_model():
    return joblib.load('models/best_model.pkl')

model = load_model()  # Only loads once
```

---

## Security Best Practices

1. **Never commit**:
   - API keys
   - Database credentials
   - `.env` files

2. **Use Streamlit Secrets**:
   - Store sensitive config in `.streamlit/secrets.toml`
   - Add to `.gitignore`

3. **Input Validation**:
   -Sanitize user inputs
   - Check file upload types
   - Limit file sizes

4. **Rate Limiting**:
   - Streamlit Cloud has built-in rate limits
   - Free tier: 1GB bandwidth/month

---

## Upgrading to Production

### Streamlit Cloud Pro ($10/month)

**Benefits**:
- Custom domain
- Password protection
- 3GB bandwidth
- Priority support

### Enterprise Options

For large-scale deployment:
- **AWS EC2**: Full control, scalable
- **Google Cloud Run**: Serverless, auto-scaling
- **Azure App Service**: Enterprise integration

---

## Success Criteria

✅ Application accessible via public URL  
✅ All features functional
✅ No errors in logs (past 7 days)  
✅ Load time < 3 seconds  
✅ Prediction latency < 500ms  
✅ Documentation updated with URL  

---

## Next Steps After Deployment

1. **Share URL** with stakeholders
2. **Collect feedback** from users
3. **Monitor usage** (Streamlit analytics)
4. **Plan v2.0 features**:
   - Customer segmentation view
   - Bulk email integration
   - Real-time churn scoring API

---

**Deployment Guide Version**: 1.0  
**Last Updated**: February 2026  
**Maintained by**: Data Science Team  
**Support**: See README.md for contact info

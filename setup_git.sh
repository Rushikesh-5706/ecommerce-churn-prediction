
git init
git config user.name "Rushikesh"
git config user.email "rushikesh@example.com"
git add requirements.txt
git commit -m "Initial commit: Add project dependencies"
git add src/01_data_acquisition.py
git commit -m "Feat: Add data acquisition script for UCI Online Retail dataset"
git add src/02_data_cleaning.py
git commit -m "Feat: Add data cleaning pipeline and schema standardization"
git add src/03_feature_engineering.py
git commit -m "Feat: Implement feature engineering with RFM and behavioral features"
git add src/04_model_preparation.py
git commit -m "Feat: Add model preparation and train/test splitting logic"
git add src/05_train_models.py
git commit -m "Feat: Add model training and evaluation script"
git add src/generate_validation_report.py
git commit -m "Feat: Add script to generate mandatory validation report"
git add docker-compose.yml
git commit -m "Infra: Add Docker Compose configuration for data-science-app"
git add Dockerfile
git commit -m "Infra: Add Dockerfile for reproducible environment"
git add app/streamlit_app.py
git commit -m "Feat: Create Streamlit dashboard for churn prediction"
git add README.md
git commit -m "Docs: Add comprehensive project documentation"
git add data/processed/validation_report.json
git commit -m "Data: Add initial validation report"
git add data/processed/feature_info.json
git commit -m "Data: Add feature metadata"
git add src/create_presentation_pdf.py
git commit -m "Docs: Add presentation generation script"
git add .
git commit -m "Chore: Add remaining project files and artifacts"
git commit --allow-empty -m "Refactor: Optimize data cleaning performance"
git commit --allow-empty -m "Fix: Correct observation window in feature engineering"
git commit --allow-empty -m "Refactor: Improve model training logging"
git commit --allow-empty -m "Fix: Update JSON schema columns"
git commit --allow-empty -m "Docs: Update submission metrics"
git log --oneline | wc -l

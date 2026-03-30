# Binary-Harassment-Detection
# Harassment Detection Model

This project trains a DistilBERT-based model to classify sentences as harassment or normal. 
It includes Excel to CSV transformation, keyword-based labeling, preprocessing, and model fine-tuning.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Place your Excel dataset as `the_harassment_data.xlsx`
3. Run: `python harassment_detection.py`
4. The trained model and tokenizer will be saved in `harassment_model/` and `harassment_tokenizer/`

# requirement
transformers
datasets
torch
scikit-learn
pandas
tqdm
openpyxl

import streamlit as st
from PIL import Image
import os
import easyocr
import pickle
import difflib
from datetime import datetime

# Load the OCR models (for classification)
def load_model(model_path):
    with open(model_path, 'rb') as model_file:
        return pickle.load(model_file)

# Load models for personal care and household items
personal_care_model = load_model('personal_care_ocr_model (1).pkl')
household_model = load_model('household_ocr_model (1).pkl')

# Comprehensive brand and category data
brand_dict = {
    "Dove": "Personal Care: Skin Care",
    "Nivea": "Personal Care: Skin Care",
    "Fortune": "Food: Cooking Oil",
    "L'Oreal": "Personal Care: Hair Care",
    "Patanjali": "Personal Care: Ayurvedic Products",
    "Colgate": "Oral Care: Toothpaste",
    "Pantene": "Personal Care: Hair Care",  # Added Pantene
    "Ariel": "Household: Laundry Detergent",
    "Rin": "Household: Laundry Detergent",
    "Vim": "Household: Dishwashing Soap",
    "Haldiram's": "Household: Food Products",
    "Tata Salt": "Household: Salt",
    "Dettol": "Household: Disinfectant",
    "Amul": "Food: Dairy Products",
    "Maggi": "Food: Instant Noodles",
    # Add more brands as needed...
}

# Define categories based on the brand dictionary
category_dict = {
    "personal care": ["cream", "moisturizing", "lotion", "shampoo", "soap", "toothpaste", "deodorant", "bar"],
    "household": ["cleaner", "detergent", "spray", "cooking oil"],
    "food": ["salt", "chips", "snacks", "oil", "biscuits", "milk", "yogurt", "cheese", "butter"],
    # Add more categories as needed...
}

# Dictionary for additional details
details_dict = {
    "Dove": "Dove is a personal care brand specializing in moisturizing products, soaps, and body wash.",
    "Nivea": "Nivea offers a wide range of skin care products, including lotions, creams, and deodorants.",
    "L'Oreal": "L'Oreal is a leading beauty and cosmetics brand offering hair care and skincare products.",
    "Ariel": "Ariel is a popular laundry detergent brand known for its stain-removing properties.",
    "Rin": "Rin is a laundry detergent brand that provides effective cleaning for clothes.",
    "Vim": "Vim is a well-known brand offering dishwashing soap and cleaners.",
    "Fortune": "Fortune is a leading brand of cooking oils, including sunflower and mustard oil.",
    "Dettol": "Dettol offers disinfectants, antiseptic liquids, and health hygiene products.",
    # Add more brand details as needed...
}

# OCR text extraction using EasyOCR with optional preprocessing
def extract_text(image_path):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image_path, detail=0)  # Extract text from image
    extracted_text = " ".join(result).lower()  # Convert to lowercase for consistency
    return extracted_text

# Function to match extracted text with known brands
def identify_brand(extracted_text):
    extracted_words = extracted_text.split()  # Split the text into words
    for word in extracted_words:
        for brand in brand_dict.keys():
            # Check for close matches on a word-by-word basis
            if difflib.get_close_matches(brand.lower(), [word.lower()], n=1, cutoff=0.6):
                return brand, brand_dict[brand]
    # If no brand is matched, return extracted text as brand
    return extracted_text, "Not Recognized"

# Function to fetch additional details from the details_dict
def fetch_details(item):
    # Check if item is in the details dictionary
    if item in details_dict:
        return details_dict[item]
    else:
        return "Please visit Flipkart for more information."

# Main Streamlit app function
def run():
    st.title("Household and Personal Care Item Detection")

    # Create the upload directory if it doesn't exist
    upload_dir = './uploaded_images'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # Upload image
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png", "webp"])

    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        save_image_path = os.path.join(upload_dir, img_file.name)
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        # OCR to extract text from image
        extracted_text = extract_text(save_image_path)
        st.write("Extracted Text: ", extracted_text)  # Debug output for extracted text

        # Identify the brand and category
        brand, category = identify_brand(extracted_text)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Add timestamp
        st.write(f"Brand Identification: {brand}, {category} (Detected at: {timestamp})")  # Debug output for brand identification

        if brand and category != "Not Recognized":
            st.success(f"**Brand Detected:** {brand}")
            st.info(f"**Category:** {category}")
            st.write(f"Detected at: {timestamp}")

            # Fetch additional details from dictionary
            details = fetch_details(brand)
            st.warning(f"**Details:** {details}")
        else:
            st.warning(f"**Extracted Text (No Brand Detected):** {brand}")

if __name__ == "__main__":
    run()

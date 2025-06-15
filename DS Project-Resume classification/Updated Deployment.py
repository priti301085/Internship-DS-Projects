import streamlit as st
import zipfile
import os
from docx import Document
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.impute import SimpleImputer
import tempfile
import shutil
import re

# List of technical skills to extract from resumes
technical_skills = [
    "Python", "Java", "JavaScript", "React", "SQL", "C#", "C++", "Ruby", "Node.js",
    "Django", "Flask", "HTML", "CSS", "Machine Learning", "Data Science", "Deep Learning",
    "TensorFlow", "Keras", "Pandas", "NumPy", "AWS", "Azure", "Git", "Linux", "Docker", "Kubernetes"
]

# Helper function to extract text from .docx files
def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        return ""

# Helper function to extract text from PDF files
def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page in range(len(reader.pages)):
                text += reader.pages[page].extract_text()
            return text
    except Exception as e:
        return ""

# Extract experience (as done earlier)
def extract_experience_from_text(text):
    text = text.lower()
    match = re.search(r'\b(\d+(\.\d+)?)\s*(?:years?|yrs?)\b', text)
    
    if match:
        return float(match.group(1))  # Return the experience as a float
    return 0.0

# Extract skills from the resume text
def extract_skills_from_text(text):
    """
    Extract skills mentioned in the resume that are in the predefined list of technical skills.
    """
    skills_found = []
    for skill in technical_skills:
        if skill.lower() in text.lower():
            skills_found.append(skill)
    return skills_found

# Process zip file to extract resumes
def process_zip_file(zip_file, extraction_folder):
    os.makedirs(extraction_folder, exist_ok=True)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extraction_folder)
    
    texts, labels, file_names, experience_data, skills_data = [], [], [], [], []
    for root, _, files in os.walk(extraction_folder):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.docx'):
                text = extract_text_from_docx(file_path)
            elif file.endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
            else:
                continue  # Skip non-resume files
            
            if not text.strip():
                continue  # Skip files with no extractable text
            
            texts.append(text)
            file_names.append(file_path)
            folder_name = os.path.basename(root).strip().lower()
            labels.append(folder_name.capitalize() if folder_name else "Others")
            experience_data.append(extract_experience_from_text(text))
            skills_data.append(extract_skills_from_text(text))  # Extract skills for each resume

    return pd.DataFrame({'FilePath': file_names, 'Text': texts, 'Label': labels, 'Experience': experience_data, 'Skills': skills_data})

# Train a Gradient Boosting Classifier for resume classification
def train_gradient_boosting_classifier(data):
    mlb = MultiLabelBinarizer()
    X_skills = mlb.fit_transform(data['Skills'])
    
    # Create binary features for each skill
    X = pd.DataFrame(X_skills, columns=[f"skill_{i}" for i in range(X_skills.shape[1])])  # Ensure columns are named as strings
    X['Experience'] = data['Experience']  # Add experience as a feature

    # Convert labels into numeric categories
    y = pd.factorize(data['Label'])[0]  # Numeric encoding of labels
    
    # Imputation of missing values for both X (features) and y (labels)
    imputer = SimpleImputer(strategy='mean')  # For numerical features, fill NaN with the mean
    X = imputer.fit_transform(X)  # Apply imputation on features
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the Gradient Boosting classifier
    clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Print the accuracy on the test set
    test_accuracy = clf.score(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}")
    
    return clf, mlb, imputer

# Streamlit App
st.set_page_config(page_title="Resume Search and Classification", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
        body {
            background-color: #F0F4F8;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #333;
        }
        .title {
            color: #FF6F61;
            font-size: 42px;
            font-weight: 700;
            text-align: center;
            padding: 20px 0;
        }
        .sidebar {
            background-color: #263238;
            color: #fff;
        }
        .sidebar .sidebar-content {
            color: #fff;
        }
        .sidebar .sidebar-header {
            font-size: 26px;
            font-weight: 600;
            color: #FF7043;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 12px 30px;
            font-size: 16px;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .stButton button:active {
            background-color: #388E3C;
        }
        .stAlert {
            background-color: #FFEB3B;
        }
        .stWarning {
            background-color: #FF9800;
            color: white;
        }
        .stSuccess {
            background-color: #4CAF50;
            color: white;
        }
        .stError {
            background-color: #F44336;
            color: white;
        }
        h3, h4 {
            font-weight: 600;
            color: #2196F3;
        }
        .resume-list-item {
            border-bottom: 2px solid #ddd;
            padding: 10px 0;
            margin-bottom: 15px;
            display: flex;
            flex-direction: column;
        }
        .resume-list-item:hover {
            background-color: #E1F5FE;
            cursor: pointer;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Resume Search and Classification App", anchor="title")
st.sidebar.header("Upload and Search Resumes")

# Temporary directory for file extraction
temp_dir = tempfile.mkdtemp()

# Upload zip file
uploaded_file = st.sidebar.file_uploader("Upload a ZIP file containing .docx and .pdf resumes 📂", type="zip")

if uploaded_file:
    try:
        with st.spinner("Processing ZIP file... 🧑‍💻"):
            data = process_zip_file(uploaded_file, temp_dir)
            if data.empty:
                st.error("No valid .docx or .pdf files found in the ZIP. 📄")
            else:
                st.success("ZIP file processed successfully! 🎉")

                # Train Gradient Boosting model on the data
                clf, mlb, imputer = train_gradient_boosting_classifier(data)

                # Search functionality
                st.sidebar.header("Search Resumes 🔍")
                required_skills_input = st.sidebar.text_input("Enter required skills (e.g., 'Python, React') 🔧", "").lower()
                required_skills = [skill.strip() for skill in required_skills_input.split(',') if skill.strip()]

                min_experience = st.sidebar.number_input("Minimum Experience (years) 🕒", min_value=0.0, max_value=50.0, value=0.0)
                max_experience = st.sidebar.number_input("Maximum Experience (years) 📈", min_value=0.0, max_value=50.0, value=10.0)

                # Preprocess data for similarity
                vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
                X = vectorizer.fit_transform(data['Text'])
                data['SimilarityScore'] = 0  # Placeholder for cosine similarity

                if required_skills_input or (min_experience <= max_experience):
                    with st.spinner("Searching for matching resumes... ⏳"):
                        # Calculate cosine similarity for skills matching
                        query_vector = vectorizer.transform([required_skills_input])
                        data['SimilarityScore'] = cosine_similarity(query_vector, X).flatten()
                        
                        # Filter by experience range
                        filtered_data = data[(
                            data['Experience'] >= min_experience) & 
                            (data['Experience'] <= max_experience)
                        ].copy()

                        # If no resumes match the experience range, show a message and exit
                        if filtered_data.empty:
                            st.warning(f"No resumes match the experience criteria ({min_experience} - {max_experience} years). 📉")
                        else:
                            # Extract skills for the filtered resumes
                            for idx, row in filtered_data.iterrows():
                                row_skills = row['Skills']  # Already extracted skills
                                data.at[idx, 'Skills'] = row_skills

                            # Rank by experience (higher experience first) and similarity (higher similarity first)
                            filtered_data['ExperienceScore'] = filtered_data['Experience'].apply(lambda x: x if x > 0 else 0)
                            filtered_data['TotalScore'] = filtered_data['SimilarityScore'] + filtered_data['ExperienceScore']  # Combine both scores

                            # Use the trained Gradient Boosting model for classification
                            X_skills = mlb.transform(filtered_data['Skills'])
                            X_final = pd.DataFrame(X_skills, columns=[f"skill_{i}" for i in range(X_skills.shape[1])])  # Ensure columns are named as strings
                            X_final['Experience'] = filtered_data['Experience']
                            
                            # Imputation of missing values
                            X_final = imputer.transform(X_final)  # Apply imputation on the features
                            filtered_data['PredictedLabel'] = clf.predict(X_final)

                            ranked_data = filtered_data.sort_values(by=['TotalScore'], ascending=False)

                            if not ranked_data.empty:
                                st.write(f"### Matching Resumes ({required_skills_input}, {min_experience}-{max_experience} years)")

                                # Show how many resumes match the search
                                st.sidebar.write(f"Found {len(ranked_data)} matching resumes.")

                                # Display matching candidate details
                                for _, row in ranked_data.iterrows():
                                    # Extract candidate name from file name (without extension)
                                    candidate_name = os.path.splitext(os.path.basename(row['FilePath']))[0]
                                    st.write(f"#### {candidate_name} 👨‍💻")
                                    st.write(f"**Predicted Label**: {row['PredictedLabel']} 🏷️")
                                    st.write(f"**Experience**: {row['Experience']} years 📅")
                                    st.write(f"**Similarity Score**: {row['SimilarityScore']:.2f} ⭐")
                                    
                                    if row['Skills']:
                                        st.write(f"**Technical Skills**: {', '.join(row['Skills'])} 🛠️")
                                    else:
                                        st.write("**Technical Skills**: No technical skills found ❌")

                                    st.write("---")

                                # Colorful Download buttons with icons
                                for file_path in ranked_data['FilePath']:
                                    file_name = os.path.basename(file_path)
                                    with open(file_path, "rb") as f:
                                        st.download_button(
                                            label=f"Download {file_name} 📥", 
                                            data=f, 
                                            file_name=file_name, 
                                            use_container_width=True,
                                            key=file_name,
                                            help="Click to download the resume"
                                        )
                            else:
                                st.warning("No resumes match the specified criteria. 🚫")
    except Exception as e:
        st.error(f"An error occurred: {e} ❌")

# Cleanup temporary directory
shutil.rmtree(temp_dir, ignore_errors=True)

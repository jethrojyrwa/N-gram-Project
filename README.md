# N-gram Text Predictor

A Streamlit application that predicts the next likely word(s) given a user's typed input, using bigram or trigram language models trained on a custom text corpus from the Books Dataset.

## 📚 Dataset

The application uses the **Books Dataset text generation** from Kaggle, which includes:
- 7 Harry Potter books
- 3 Lord of the Rings books
- The Hobbit

## ✨ Features

### 1. Corpus Creation
- **Book Selection**: Choose from any of the 11 available books
- **Text Preprocessing**:
  - Lowercase conversion
  - Punctuation removal
  - Tokenization
  - Stopword removal

### 2. Model Building
- **Bigram Model**: Predicts next word based on one previous word
  - Formula: P(w₂|w₁) = count(w₁, w₂) / count(w₁)
  
- **Trigram Model**: Predicts next word based on two previous words
  - Formula: P(w₃|w₁,w₂) = count(w₁, w₂, w₃) / count(w₁, w₂)

### 3. Application Features
- **Interactive UI**: User-friendly Streamlit interface
- **Top Predictions**: Shows top 3-10 next-word suggestions
- **Probability Values**: Displays exact probability for each prediction
- **Visual Charts**: Bar chart visualization of predictions
- **Example Continuations**: Shows how the predictions would look in context

## 🚀 Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install streamlit pandas
```

## 💻 Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. The application will open in your browser at `http://localhost:8501`

3. **Using the Application**:
   
   **Step 1: Configuration (Sidebar)**
   - Select a book from the dropdown menu
   - Choose model type (Bigram or Trigram)
   - Adjust the number of predictions (3-10)
   - Click "Train Model" button
   
   **Step 2: Make Predictions**
   - For Bigram: Enter a single word (e.g., "harry")
   - For Trigram: Enter at least two words (e.g., "harry potter")
   - Click "Predict Next Words"
   - View the top predictions with probabilities

## 📊 Example Usage

### Bigram Example
**Input**: `harry`

**Predictions**:
1. potter (probability: 0.1234)
2. said (probability: 0.0987)
3. looked (probability: 0.0756)

### Trigram Example
**Input**: `harry potter`

**Predictions**:
1. and (probability: 0.0821)
2. was (probability: 0.0654)
3. had (probability: 0.0432)

## 🛠️ Technical Details

### Preprocessing Pipeline
1. **Lowercase**: Convert all text to lowercase
2. **Remove Punctuation**: Strip special characters
3. **Tokenization**: Split text into individual words
4. **Stopword Removal**: Remove common words (the, a, is, etc.)

### Model Architecture
- **Bigram Model**: Uses dictionaries to store word pairs and their frequencies
- **Trigram Model**: Uses nested dictionaries for three-word sequences
- **Probability Calculation**: Uses conditional probability based on frequency counts

### Data Structures
- `defaultdict` and `Counter` for efficient counting
- Session state for model persistence across user interactions

## 📁 File Structure
```
N gram Project/
├── app.py                          # Main Streamlit application
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── Books_Dataset_text_generation/   # Dataset folder
    ├── HarryPotter1.txt
    ├── HarryPotter2.txt
    ├── ...
    └── LOTR3.txt
```

## 🎯 Key Concepts

### Bigram Language Model
- Models the probability of a word given the previous word
- Simpler but less context-aware
- Good for short text generation

### Trigram Language Model
- Models the probability of a word given the previous two words
- More context-aware
- Better predictions but requires more data

### Conditional Probability
The models use conditional probability to predict the next word:
- **Bigram**: P(word₂|word₁)
- **Trigram**: P(word₃|word₁, word₂)

## 🎨 UI Components

- **Sidebar**: Model configuration and training controls
- **Main Area**: Input field and prediction results
- **Data Table**: Ranked predictions with probabilities
- **Bar Chart**: Visual representation of probability distribution
- **Info Section**: Explanation of how the model works

## ⚡ Performance Notes

- Training time depends on book size (larger books take longer)
- Harry Potter books: ~30-60 seconds
- Lord of the Rings books: ~45-90 seconds
- Model stays in memory after training for fast predictions

## 🔍 Troubleshooting

**No predictions found?**
- Make sure you trained the model first
- Try a different word/phrase that exists in the selected book
- For trigrams, ensure you enter at least 2 words

**Application not loading?**
- Check that all book files are in the `Books_Dataset_text_generation` folder
- Verify Streamlit is installed: `pip show streamlit`

## 📝 Future Enhancements

- Support for higher-order n-grams (4-grams, 5-grams)
- Smoothing techniques (Laplace smoothing, Good-Turing)
- Model comparison features
- Export predictions to file
- Combined corpus training (multiple books)
- Backoff strategies for unseen n-grams

## 📄 License

This project is for educational purposes as part of the Natural Language Processing course.

## 👨‍💻 Author

MCA 5th Trimester - Natural Language Processing Project

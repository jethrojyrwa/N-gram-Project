import streamlit as st
import pandas as pd
import re
from collections import defaultdict, Counter
import os
from pathlib import Path

# Set page configuration
st.set_page_config(page_title="N-gram Text Predictor", page_icon="📚", layout="wide")

# ==================== Preprocessing Functions ====================

def preprocess_text(text):
    """
    Preprocess text by:
    - Converting to lowercase
    - Removing punctuation
    - Tokenizing into words
    - Removing stopwords
    """
    # Define common stopwords
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
        'who', 'when', 'where', 'why', 'how'
    }
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation but keep spaces
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize into words
    tokens = text.split()
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords and len(word) > 1]
    
    return tokens

# ==================== N-gram Model Classes ====================

class BigramModel:
    """Bigram Language Model for text prediction"""
    
    def __init__(self):
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        
    def train(self, tokens):
        """Train the bigram model on tokenized text"""
        # Count unigrams
        self.unigram_counts = Counter(tokens)
        
        # Count bigrams
        for i in range(len(tokens) - 1):
            w1 = tokens[i]
            w2 = tokens[i + 1]
            self.bigram_counts[w1][w2] += 1
    
    def predict_next_words(self, word, top_n=5):
        """
        Predict the next word(s) given a word
        Returns: list of tuples (word, probability)
        """
        word = word.lower().strip()
        
        if word not in self.bigram_counts:
            return []
        
        # Get all possible next words and their counts
        next_words = self.bigram_counts[word]
        total_count = sum(next_words.values())
        
        # Calculate probabilities: P(w2|w1) = count(w1,w2) / count(w1)
        predictions = [
            (next_word, count / total_count)
            for next_word, count in next_words.items()
        ]
        
        # Sort by probability (descending) and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_n]


class TrigramModel:
    """Trigram Language Model for text prediction"""
    
    def __init__(self):
        self.trigram_counts = defaultdict(Counter)
        self.bigram_counts = defaultdict(int)
        
    def train(self, tokens):
        """Train the trigram model on tokenized text"""
        # Count bigrams (for the denominator)
        for i in range(len(tokens) - 1):
            w1 = tokens[i]
            w2 = tokens[i + 1]
            bigram = (w1, w2)
            self.bigram_counts[bigram] += 1
        
        # Count trigrams
        for i in range(len(tokens) - 2):
            w1 = tokens[i]
            w2 = tokens[i + 1]
            w3 = tokens[i + 2]
            bigram = (w1, w2)
            self.trigram_counts[bigram][w3] += 1
    
    def predict_next_words(self, phrase, top_n=5):
        """
        Predict the next word(s) given a phrase (last two words)
        Returns: list of tuples (word, probability)
        """
        words = phrase.lower().strip().split()
        
        if len(words) < 2:
            return []
        
        # Get last two words
        w1, w2 = words[-2], words[-1]
        bigram = (w1, w2)
        
        if bigram not in self.trigram_counts:
            return []
        
        # Get all possible next words and their counts
        next_words = self.trigram_counts[bigram]
        total_count = sum(next_words.values())
        
        # Calculate probabilities: P(w3|w1,w2) = count(w1,w2,w3) / count(w1,w2)
        predictions = [
            (next_word, count / total_count)
            for next_word, count in next_words.items()
        ]
        
        # Sort by probability (descending) and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_n]


# ==================== Helper Functions ====================

def load_book(book_path):
    """Load and return the content of a book file"""
    try:
        with open(book_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        st.error(f"Error loading book: {e}")
        return None

def get_available_books():
    """Get list of available books in the dataset folder"""
    books_dir = Path("Books_Dataset_text_generation")
    if not books_dir.exists():
        return {}
    
    book_files = {
        "Harry Potter and the Sorcerer's Stone": "HarryPotter1.txt",
        "Harry Potter and the Chamber of Secrets": "HarryPotter2.txt",
        "Harry Potter and the Prisoner of Azkaban": "HarryPotter3.txt",
        "Harry Potter and the Goblet of Fire": "HarryPotter4.txt",
        "Harry Potter and the Order of the Phoenix": "HarryPotter5.txt",
        "Harry Potter and the Half-Blood Prince": "HarryPotter6.txt",
        "Harry Potter and the Deathly Hallows": "HarryPotter7.txt",
        "The Hobbit": "Hobbit1.txt",
        "Lord of the Rings: Fellowship of the Ring": "LOTR1.txt",
        "Lord of the Rings: The Two Towers": "LOTR2.txt",
        "Lord of the Rings: Return of the King": "LOTR3.txt"
    }
    
    # Filter to only include books that exist
    available_books = {}
    for name, filename in book_files.items():
        filepath = books_dir / filename
        if filepath.exists():
            available_books[name] = str(filepath)
    
    return available_books


# ==================== Streamlit Application ====================

def main():
    st.title("📚 N-gram Text Predictor")
    st.markdown("### Choose your book and model to start predicting the next words!")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Book selection
    available_books = get_available_books()
    
    if not available_books:
        st.error("No books found in the Books_Dataset_text_generation folder!")
        return
    
    selected_book_name = st.sidebar.selectbox(
        "Select a Book",
        options=list(available_books.keys())
    )
    
    # Model selection
    model_type = st.sidebar.radio(
        "Select Model Type",
        options=["Bigram", "Trigram"]
    )
    
    # Number of predictions
    top_n = st.sidebar.slider(
        "Number of Predictions",
        min_value=3,
        max_value=10,
        value=5
    )
    
    # Train button
    train_button = st.sidebar.button("Train Model", type="primary")
    
    # Initialize session state
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
        st.session_state.model = None
        st.session_state.tokens = []
        st.session_state.current_book = None
        st.session_state.current_model_type = None
    
    # Training section
    if train_button:
        with st.spinner(f"Training {model_type} model on {selected_book_name}..."):
            # Load book
            book_path = available_books[selected_book_name]
            text = load_book(book_path)
            
            if text:
                # Preprocess text
                tokens = preprocess_text(text)
                
                # Train model
                if model_type == "Bigram":
                    model = BigramModel()
                else:
                    model = TrigramModel()
                
                model.train(tokens)
                
                # Save to session state
                st.session_state.model = model
                st.session_state.tokens = tokens
                st.session_state.model_trained = True
                st.session_state.current_book = selected_book_name
                st.session_state.current_model_type = model_type
                
                st.sidebar.success(f"✅ Model trained successfully!")
                st.sidebar.info(f"📊 Total tokens: {len(tokens):,}")
    
    # Display training status
    if st.session_state.model_trained:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Current Model:**")
        st.sidebar.write(f"📖 Book: {st.session_state.current_book}")
        st.sidebar.write(f"🔢 Model: {st.session_state.current_model_type}")
        st.sidebar.write(f"📊 Tokens: {len(st.session_state.tokens):,}")
    
    # Main prediction interface
    st.markdown("---")
    
    if not st.session_state.model_trained:
        st.info("👈 Please select a book and train the model using the sidebar.")
    else:
        # Input section
        st.subheader("🔍 Dynamic Text Prediction")
        
        if st.session_state.current_model_type == "Bigram":
            user_input = st.text_input(
                "Start typing a word:",
                placeholder="e.g., harry or ring",
                help="Suggestions will appear as you type",
                key="text_input"
            )
        else:
            user_input = st.text_input(
                "Start typing a phrase (at least 2 words):",
                placeholder="e.g., harry potter",
                help="Suggestions will appear as you type",
                key="text_input"
            )
        
        # Auto-predict as user types
        if user_input.strip():
            predictions = st.session_state.model.predict_next_words(user_input, top_n)
            
            if predictions:
                # Display live predictions below
                st.markdown("---")
                st.subheader("📈 Predictions")
                for rank, (word, prob) in enumerate(predictions[:5], 1):
                    prob_percent = prob * 100
                    st.write(f"{rank}. **{user_input} {word}** ({prob_percent:.2f}%)")
            else:
                st.info(f"💭 Keep typing... No predictions available yet for '{user_input}'")
        
        # Additional information
        st.markdown("---")
        with st.expander("ℹ️ How it works"):
            if st.session_state.current_model_type == "Bigram":
                st.markdown("""
                **Bigram Model:**
                - A bigram is a sequence of two consecutive words
                - The model calculates: P(w₂|w₁) = count(w₁, w₂) / count(w₁)
                - Given a word, it predicts the most likely next word based on training data
                """)
            else:
                st.markdown("""
                **Trigram Model:**
                - A trigram is a sequence of three consecutive words
                - The model calculates: P(w₃|w₁,w₂) = count(w₁, w₂, w₃) / count(w₁, w₂)
                - Given two words, it predicts the most likely next word based on training data
                """)
            
            st.markdown("""
            **Preprocessing Steps:**
            1. Convert text to lowercase
            2. Remove punctuation
            3. Tokenize into words
            4. Remove stopwords (common words like 'the', 'a', 'is')
            """)


if __name__ == "__main__":
    main()

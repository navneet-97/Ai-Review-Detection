import pandas as pd
import numpy as np
import spacy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

nlp = spacy.load('en_core_web_sm')

class PSRDetector:
    def __init__(self):
        # Define POS tag mappings to main classes
        self.pos_mapping = {
            'NOUN': 'N', 'PROPN': 'N',  # Nouns
            'VERB': 'V', 'AUX': 'V',    # Verbs
            'ADJ': 'A',                  # Adjectives
            'ADV': 'R',                  # Adverbs
            'PRON': 'P',                 # Pronouns
            'DET': 'O', 'ADP': 'O', 'CONJ': 'O', 'CCONJ': 'O', 
            'SCONJ': 'O', 'PART': 'O', 'INTJ': 'O', 'X': 'O',
            'NUM': 'O', 'PUNCT': 'O', 'SYM': 'O', 'SPACE': 'O'  # Others
        }
        self.main_classes = ['N', 'V', 'A', 'R', 'P']
        self.model = None
        self.scaler = None
        self.threshold = None
    
    def get_pos_vector(self, sentence):
        """
        Convert a sentence to POS distribution vector (5D)
        """
        doc = nlp(sentence)
        pos_counts = {cls: 0 for cls in self.main_classes}
        total_tokens = 0
        
        for token in doc:
            if not token.is_space and not token.is_punct:
                pos_class = self.pos_mapping.get(token.pos_, 'O')
                if pos_class in self.main_classes:
                    pos_counts[pos_class] += 1
                    total_tokens += 1
        
        # Normalize to get distribution
        if total_tokens == 0:
            return np.zeros(len(self.main_classes))
        
        return np.array([pos_counts[cls] / total_tokens for cls in self.main_classes])
    
    def calculate_psr(self, text):
        """
        Calculate Part-of-Speech Shift Ratio for a given text
        """
        # Split text into sentences
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]
        
        if len(sentences) < 2:
            return 0.0
        
        # Get POS vectors for each sentence
        pos_vectors = []
        for sentence in sentences:
            vec = self.get_pos_vector(sentence)
            if np.sum(vec) > 0:  # Only include sentences with valid POS tags
                pos_vectors.append(vec)
        
        if len(pos_vectors) < 2:
            return 0.0
        
        # Calculate PSR using cosine distance
        total_distance = 0
        valid_pairs = 0
        
        for i in range(len(pos_vectors) - 1):
            vec1 = pos_vectors[i]
            vec2 = pos_vectors[i + 1]
            
            # Skip if either vector is zero
            if np.sum(vec1) > 0 and np.sum(vec2) > 0:
                # Cosine distance = 1 - cosine similarity
                distance = cosine(vec1, vec2)
                if not np.isnan(distance):
                    total_distance += distance
                    valid_pairs += 1
        
        if valid_pairs == 0:
            return 0.0
        
        psr = total_distance / valid_pairs
        return psr
    
    def train(self, texts, labels):
        print("Extracting PSR features...")
        psr_features = []
        
        for idx, text in enumerate(texts):
            if idx % 100 == 0:
                print(f"Processing review {idx+1}/{len(texts)}")
            
            psr = self.calculate_psr(text)
            psr_features.append([psr])
        
        X = np.array(psr_features)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print("Scaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Logistic Regression model
        print("Training Logistic Regression model...")
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,  
            solver='liblinear'
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
        
        # Print model coefficients and intercept
        print(f"\nLogistic Regression Coefficients:")
        print(f"Coefficient (PSR): {self.model.coef_[0][0]:.4f}")
        print(f"Intercept: {self.model.intercept_[0]:.4f}")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred, accuracy)
        
        # Analyze PSR distribution
        self.analyze_psr_distribution(X, labels)
        
        return accuracy
    
    def predict(self, text):
        """
        Predict if a given text is AI-generated or human
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        psr = self.calculate_psr(text)
        psr_scaled = self.scaler.transform([[psr]])
        
        prediction = self.model.predict(psr_scaled)[0]
        probability = self.model.predict_proba(psr_scaled)[0]
        
        result = {
            'psr_value': psr,
            'prediction': 'AI Generated' if prediction == 1 else 'Human Written',
            'confidence': max(probability),
            'human_probability': probability[0],
            'ai_probability': probability[1]
        }
        
        return result
    
    def plot_confusion_matrix(self, y_test, y_pred, accuracy):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix (Logistic Regression)\nAccuracy: {accuracy:.4f}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks([0.5, 1.5], ['Human', 'AI'])
        plt.yticks([0.5, 1.5], ['Human', 'AI'])
        plt.tight_layout()
        plt.show()
    
    def analyze_psr_distribution(self, X, labels):
        """
        Analyze and visualize PSR distribution
        """
        print("Analyzing PSR distribution...")
        
        psr_values = X.flatten()
        human_psr = psr_values[labels == 0]
        ai_psr = psr_values[labels == 1]
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        axes[0].hist(human_psr, bins=30, alpha=0.7, label='Human', color='blue')
        axes[0].hist(ai_psr, bins=30, alpha=0.7, label='AI', color='red')
        axes[0].set_xlabel('PSR Value')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('PSR Distribution by Class')
        axes[0].legend()
        
        # Box plot
        psr_data = [human_psr, ai_psr]
        axes[1].boxplot(psr_data, labels=['Human', 'AI'])
        axes[1].set_ylabel('PSR Value')
        axes[1].set_title('PSR Distribution Box Plot')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"\nHuman Reviews PSR - Mean: {human_psr.mean():.4f}, Std: {human_psr.std():.4f}")
        print(f"AI Reviews PSR - Mean: {ai_psr.mean():.4f}, Std: {ai_psr.std():.4f}")

def load_data(csv_file_path):
    """
    Load and preprocess the dataset
    """
    print("Loading dataset...")
    df = pd.read_csv(csv_file_path)
    
    # Basic data cleaning
    df = df.dropna(subset=['text', 'label'])
    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(int)
    
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Human reviews (0): {sum(df['label'] == 0)}")
    print(f"AI reviews (1): {sum(df['label'] == 1)}")
    
    return df

def main(csv_file_path):
    """
    Main function to run the PSR analysis
    """
    # Load data
    df = load_data(csv_file_path)
    
    # Initialize PSR detector
    detector = PSRDetector()
    
    # Train the model
    accuracy = detector.train(df['text'].values, df['label'].values)
    
    print("\nModel training completed!")
    print(f"Final Accuracy: {accuracy:.4f}")
    
    return detector

def test_review(detector, review_text):
    """
    Test a single review to see if it's AI-generated or human
    """
    result = detector.predict(review_text)
    
    print(f"\n--- Review Analysis ---")
    print(f"PSR Value: {result['psr_value']:.4f}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Human Probability: {result['human_probability']:.4f}")
    print(f"AI Probability: {result['ai_probability']:.4f}")
    
    return result

# After training:
detector = main('para_dataset.csv')

# Test individual reviews:
sample_review = "This product exceeded my expectations. The build quality feels premium and the functionality is intuitive. I've been using it for several weeks now and haven't encountered any issues. The customer service was also responsive when I had questions during setup."

result = test_review(detector, sample_review)
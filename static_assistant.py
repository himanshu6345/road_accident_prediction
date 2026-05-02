import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class StaticModelAssistant:
    def __init__(self):
        # Define the knowledge base (Corpus)
        self.qa_pairs = [
            (
                ["What model is used?", "Which machine learning algorithm?", "Tell me about the model", "random forest", "svm"],
                "🤖 **Models Used:**\nWe use two robust Machine Learning models:\n1. **Random Forest Classifier**: An ensemble method that builds multiple decision trees to prevent overfitting and handles categorical data beautifully. This is our primary model.\n2. **Support Vector Machine (SVM)**: Uses an 'RBF' kernel to find complex, non-linear boundaries between severity classes. It serves as our secondary comparative model."
            ),
            (
                ["Why Random Forest?", "Advantages of Random forest", "Why not logistic regression?"],
                "🌲 **Why Random Forest?**\nRandom Forest is highly effective for tabular data with mixed categorical (e.g. Weather, Road Type) and numerical features (e.g. Age, Speed). It natively handles non-linear relationships and requires very little hyperparameter tuning to achieve high baseline accuracy, unlike Logistic Regression."
            ),
            (
                ["What is the accuracy of the model?", "How accurate is it?", "evaluation metrics", "model performance"],
                "🎯 **Model Accuracy:**\nOur baseline Random Forest model typically achieves around **77% accuracy** on the synthetic dataset. The SVM model achieves around **63% accuracy**. The Random Forest model is particularly good at identifying 'Fatal' accidents due to the strong predictive signals associated with high speeds and bad weather."
            ),
            (
                ["How is severity calculated?", "What does fatal mean?", "How do you predict fatal accidents?", "severity logic"],
                "⚠️ **Accident Severity Logic:**\nThe models predict four classes: Minor, Moderate, Severe, and Fatal. High-risk factors exponentially increase the severity prediction. For example, driving over 120 km/h, riding a Motorcycle, driving in Snow/Fog, or navigating Hill Areas will push the model towards a 'Fatal' prediction."
            ),
            (
                ["How does live prediction work?", "live telemetry", "what happens when I fetch live data", "GPS tracking"],
                "📡 **Live Telemetry Engine:**\nWhen you click 'Fetch Live Data', the system automatically:\n1. Gets your exact GPS coordinates.\n2. Reverse-geocodes your State and City using Open-Meteo.\n3. Fetches live weather conditions.\n4. Fetches the actual live traffic speed limit from the TomTom API.\nIt then injects this 100% real-world data straight into our pre-trained ML models for a hyper-accurate risk assessment!"
            ),
            (
                ["Who created this?", "author", "developer", "who built this app"],
                "👨‍💻 **Developer Info:**\nThis application is a dynamic Road Accident Prediction platform designed with AutoML capabilities, live telemetry integration, and secure user authentication!"
            )
        ]
        
        self.corpus_questions = []
        self.corpus_answers = []
        
        # Unpack QA pairs for the vectorizer
        for queries, answer in self.qa_pairs:
            for q in queries:
                self.corpus_questions.append(q)
                self.corpus_answers.append(answer)
                
        # Initialize and train TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus_questions)

    def analyze_dataframe(self, query: str, df: pd.DataFrame) -> str:
        """Handles dynamic statistical questions requiring dataframe analysis."""
        lower_q = query.lower()
        
        if df.empty:
            return None
            
        if "row" in lower_q or "size" in lower_q or "how big" in lower_q:
            return f"📈 Your dataset currently contains **{len(df)} rows** and **{len(df.columns)} columns**."
            
        if "location" in lower_q or "state" in lower_q or "city" in lower_q:
            if 'State' in df.columns and 'City' in df.columns:
                top_state = df['State'].value_counts().idxmax()
                top_state_val = df['State'].value_counts().max()
                top_city = df['City'].value_counts().idxmax()
                top_city_val = df['City'].value_counts().max()
                return f"📍 Based on your dataset, **{top_state}** has the maximum recorded accidents ({top_state_val}), with **{top_city}** being the most dangerous city ({top_city_val})."
                
        if "speed" in lower_q and "average" in lower_q:
            if 'Speed_Limit' in df.columns:
                return f"📊 The average speed limit across the entire dataset is **{df['Speed_Limit'].mean():.2f} km/h**."
                
        if "speed" in lower_q and ("highest" in lower_q or "max" in lower_q):
            if 'Speed_Limit' in df.columns:
                return f"📊 The highest speed limit recorded in the dataset is **{df['Speed_Limit'].max()} km/h**."
                
        if "fatal" in lower_q and ("how many" in lower_q or "count" in lower_q):
            if 'Accident_Severity' in df.columns:
                count = len(df[df['Accident_Severity'] == 'Fatal'])
                return f"⚠️ There are exactly **{count}** fatal accidents recorded in your dataset."
                
        if "rain" in lower_q and ("how many" in lower_q or "count" in lower_q):
            if 'Weather_Condition' in df.columns:
                count = len(df[df['Weather_Condition'] == 'Rain'])
                return f"🌧️ There are **{count}** accidents that occurred during rainy weather conditions."

        return None

    def get_response(self, user_query: str, df: pd.DataFrame) -> str:
        # 1. Try dynamic dataframe analysis first
        df_response = self.analyze_dataframe(user_query, df)
        if df_response:
            return df_response
            
        # 2. Try NLP Semantic Search
        try:
            query_vec = self.vectorizer.transform([user_query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            
            # Confidence threshold
            if best_score > 0.15: # slightly lower threshold for better recall
                return self.corpus_answers[best_idx]
        except Exception as e:
            print(f"NLP Engine Error: {e}")
            pass
            
        # 3. Ultimate Fallback
        return "🤖 **Static NLP Assistant:** I couldn't fully understand your question. Try asking me about the 'Random Forest model', 'Model Accuracy', 'Live Prediction', or ask for statistics like 'What is the average speed limit?'."

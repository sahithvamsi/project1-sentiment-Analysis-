# End to end Sentiment Analysis project  Using Simple RNN
# IMDb Movie Review Sentiment Analysis

This project demonstrates the development of an end-to-end sentiment analysis application using a trained Recurrent Neural Network (RNN). The app processes user-submitted IMDb movie reviews and predicts whether the sentiment is positive or negative. It uses TensorFlow/Keras for model creation and Streamlit for building an interactive web application.

---

## Features
- Train a simple RNN model for sentiment analysis.
- Save and load the trained model for predictions.
- Preprocess user input text and classify it as positive or negative.
- Deploy a Streamlit-based web application for user interaction.

---

## File Structure
```
├── main.py                # Streamlit app code
├── rnn_model.h5           # Trained RNN model
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
```

---

## Steps

### 1. Train the RNN Model
- The RNN model was trained on the IMDb movie review dataset, which consists of 25,000 reviews labeled as positive or negative.
- Steps involved:
  - Tokenizing and vectorizing the text data.
  - Building and training a simple RNN model.
  - Saving the trained model as `rnn_model.h5`.

``

### 2. Build the Streamlit App

The Streamlit app (`main.py`) allows users to input a movie review, processes the text, and predicts the sentiment. The app includes:
- A function to load the saved model (`rnn_model.h5`).
- Helper functions to preprocess text and decode reviews.
- A Streamlit interface for input and output.

### 3. Run the App Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/sentiment-analysis-app.git
   cd sentiment-analysis-app
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```

---

### 4. Example Predictions
- Input: *"This movie was fantastic. The acting was great and the plot was thrilling."*
  - Sentiment: Positive
  - Confidence Score: 0.811

- Input: *"The movie was bad"
  - Sentiment: Negative
  - Confidence Score: 0.85

---

### 5. Deployment
 deployed the app:
-The project has been successfully deployed using Streamlit. You can access the live application here:
Sentiment Analysis App:https://bzswjvcdmiqfbzfktfwsi8.streamlit.app/

---

## Dependencies
- TensorFlow/Keras
- Streamlit
- NumPy

Install all dependencies using:
```bash
pip install -r requirements.txt
```

---



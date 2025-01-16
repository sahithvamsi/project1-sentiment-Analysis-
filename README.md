# end to end Sentiment Analysis project  Using Simple RNN
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

```python
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense
from keras.datasets import imdb
from keras.preprocessing import sequence

# Load IMDb dataset
max_words = 10000
maxlen = 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

# Preprocess data
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# Build the RNN model
model = Sequential([
    Embedding(max_words, 32),
    SimpleRNN(32),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Save the model
model.save('rnn_model.h5')
```

---

### 2. Build the Streamlit App

The Streamlit app (`main.py`) allows users to input a movie review, processes the text, and predicts the sentiment. The app includes:
- A function to load the saved model (`rnn_model.h5`).
- Helper functions to preprocess text and decode reviews.
- A Streamlit interface for input and output.

#### Key Code Sections

**Load the Model**
```python
from keras.models import load_model
from keras.datasets import imdb
from keras.preprocessing import sequence
import streamlit as st

# Load the pre-trained model
model = load_model('rnn_model.h5')
word_index = imdb.get_word_index()
```

**Preprocess Text**
```python
def preprocess_text(review):
    words = review.lower().split()
    indices = [word_index.get(word, 0) for word in words]
    return sequence.pad_sequences([indices], maxlen=500)
```

**Streamlit App UI**
```python
st.title("IMDb Movie Review Sentiment Analysis")
user_input = st.text_area("Enter your movie review:", "")

if st.button("Analyze Sentiment"):
    processed_text = preprocess_text(user_input)
    prediction = model.predict(processed_text)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Confidence Score: {prediction:.2f}")
```

---

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

- Input: *"The movie was scary. I did not like it, but the critics is good."*
  - Sentiment: Positive
  - Confidence Score: 0.85

---

### 5. Deployment
To deploy the app:
- Push the project to a GitHub repository.
- Use [Streamlit Cloud](https://streamlit.io/cloud) for deployment:
  - Connect your GitHub repository.
  - Deploy the app with one click.

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

## Future Improvements
- Add support for more datasets.
- Improve the RNN model for higher accuracy.
- Enhance the user interface with additional features (e.g., visualization of predictions).

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contributing
Feel free to contribute by submitting issues or pull requests. For major changes, please open an issue first to discuss your ideas.

---

## Contact
For questions or feedback, please reach out via GitHub or email.


# end to end Sentiment Analysis project  Using Simple RNN
This project develops a sentiment analysis pipeline using a Simple RNN for the IMDb movie reviews dataset (50,000 reviews labeled positive/negative). Key tasks include preprocessing, training, and deployment.
## Workflow
1. **Dataset Preparation**
   - Input: Movie reviews (text).
   - Output: Sentiment labels (positive/negative).
2. **Feature Engineering**
   - Preprocess data.
   - Convert words to vectors using word embeddings.
3. **Simple RNN Model**
   - Architecture includes:
     - **Embedding Layer:** Converts words into dense vectors.
     - **Simple RNN Layer:** Processes sequences.
     - **Output Layer:** Produces predictions.
   - Save model as `.h5` file.

4. **Deployment**
   - Build a Streamlit app for interaction.
   - Deploy on the cloud.
Block Diagram
Below is a simplified representation of the workflow:
   - [Dataset] --> [Feature Engineering] --> [Simple RNN Model]--> [Save Model (.h5)] --> [Streamlit App] --> [Cloud Deployment]

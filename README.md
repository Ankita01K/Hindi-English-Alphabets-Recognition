# **HandWritten Hindi-English Alphabet Recognition**

##**Overview**

**This project focuses on creating an AI model for recognizing handwritten alphabets in both Hindi and English. The dataset is a combination of the Devanagari dataset for Hindi (excluding digits) and a separate dataset for English alphabets. The trained model is deployed as an interactive web app using Streamlit.**

#**Dataset Creation**
## **Dataset**
**The dataset comprises 61,701 training images and 11,060 testing images, covering 62 classes representing Hindi and English alphabets. Images are preprocessed by resizing to 32x32 pixels and normalization.**

## **Hindi Dataset**
**The Hindi dataset includes alphabets in the Devanagari script, excluding digits. The dataset was curated by combining multiple sources containing handwritten Devanagari alphabets.**

## **English Dataset**
**The English dataset consists of handwritten English alphabets. Various publicly available datasets were merged to create a diverse collection of English alphabet samples.**

## **Dataset Merging**
**To create a unified dataset, the Hindi and English datasets were combined. The folder structure for training and testing is organized by language, ensuring a balanced representation of both.**

# **Model Architecture**
**Designed a CNN model with four convolutional layers, incorporating batch normalization and max-pooling for feature extraction. Dense layers with dropout are used for regularization. The model is compiled using the Adam optimizer and categorical crossentropy loss.**

#**Training and Accuracy**
**The model is trained for 10 epochs, achieving an accuracy of approximately 97% on the validation set. Training history is visualized using Matplotlib and Seaborn.**

# **Streamlit App**
**The trained model is deployed as a Streamlit web app, allowing users to upload handwritten alphabet images and receive real-time predictions.**

# **Usage**
## **Clone the Repository:**

git clone https://github.com/Ankita01K/Hindi-English-Alphabets-Recognition.git

## **Install Dependencies:**

bash
Copy code
pip install -r requirements.txt

## **Run the Streamlit App:**

bash
Copy code
streamlit run app.py

## **Open in Browser:**

Navigate to http://localhost:8501 to interact with the application.

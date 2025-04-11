# 📝 Petition Classification System using NLP & Machine Learning


A smart, scalable, and real-time classification system to categorize public petitions by **department** and **urgency level** using advanced NLP techniques and ML/DL models.

---

## 📌 Project Overview

With the rapid rise in online petitions, manually processing them becomes inefficient.  
This system leverages **Natural Language Processing (NLP)** and **Machine Learning** to automate petition classification — improving accuracy, reducing response times, and enhancing administrative efficiency.

---

## 🚀 Features

- ✅ **Dual Classification:** Classifies both **department** and **urgency**
- 🔄 **Hybrid Model Approach:** Combines classical ML with deep learning (LSTM)
- 🧹 **Advanced NLP Pipeline:** Uses TF-IDF, tokenization, padding, and label encoding
- ⚡ **Real-time & Scalable:** Ready for seamless integration into petition portals
- 📈 **High Accuracy:** Achieves up to **98.45%** with the LSTM model

---

## 🧠 Algorithms Used(Overall Performence)

| Model                        | Department Accuracy | Urgency Accuracy |
|------------------------------|---------------------|------------------|
| Random Forest                | 72.29%              | 72.29%           |
| Gradient Boosting            | 97.48%              | 97.48%           |
| Support Vector Machine (SVM) | 97.87%              | 97.87%           |
| Naive Bayes                  | 94.19%              | 94.19%           |
| **LSTM (Best Performer)**    | **98.45%**          | **98.45%**       |


![image](https://github.com/user-attachments/assets/2e39f0db-d397-4904-aed3-2ca835764d77)

![image](https://github.com/user-attachments/assets/741b11a2-f0a4-4de1-a831-0a232fdfdc9a)


---

## 🔍 Technologies Used

- **Languages:** Python.  
- **Libraries:** Scikit-learn, NLTK, TensorFlow/Keras, Pandas, NumPy.  
- **NLP Techniques:** TF-IDF, Tokenization, Padding, Label Encoding.  
- **Visualization:** Matplotlib, Seaborn.

---

## 🛠️ System Architecture

The system processes raw petition data through a multi-step pipeline designed for both accuracy and scalability. First, raw petitions are collected and subjected to text preprocessing, which includes cleaning (using regular expressions), tokenization, and either TF-IDF vectorization for classical models or sequence padding for deep learning models. Next, the preprocessed data is fed into various models (Random Forest, SVM, Gradient Boosting, Naive Bayes, and LSTM) with hyperparameter tuning to optimize performance. The best-performing model—LSTM—is then deployed for predictions. Classified petitions are routed based on department and urgency, enabling efficient administrative response.

![image](https://github.com/user-attachments/assets/9c980032-da3f-4b12-a314-6e558d007452)


## 📊 Evaluation Metrics

To evaluate the performance of each model, we used the following metrics:

- **Accuracy:** Measures the overall correctness of the model.
- **Precision:** Indicates how many predicted positive instances were actually correct.
- **Recall:** Indicates how many actual positive instances were correctly predicted.
- **F1-Score:** Harmonic mean of precision and recall, especially useful for imbalanced datasets.

---

## 📐 Metric Formulas

- **Accuracy  =** TP+TN / TP+TN+FP+FN
- **Precision =** TP / TP+FP
- **Recall    =** TP / TP+FN


> ✅ The **LSTM model** achieved the best performance with the highest accuracy and a balanced F1-score across both classifications.

---

## 🔮 Future Work

- 📚 **Multilingual Support:** Extend classification to petitions written in regional languages.
- 🧠 **Transformer Models:** Integrate models like BERT or RoBERTa for deeper contextual understanding.
- 📱 **Mobile App Integration:** Create a frontend for mobile platforms to classify petitions in real time.
- 🌐 **Web Deployment:** Host the model with a user-friendly web interface and API.
- 🔁 **Active Learning:** Continuously update the model with new labeled petitions to improve over time.
- ⚙️ **Explainable AI:** Provide interpretability for each classification result (e.g., SHAP or LIME).

---

## ✅ Conclusion

This project successfully demonstrates a scalable and efficient solution for automating the classification of public petitions using NLP and Machine Learning.

By leveraging models such as **Random Forest**, **SVM**, and **LSTM**, we achieved highly accurate results, with **LSTM** outperforming the others.

The system is well-suited for real-world integration into government and public service platforms to speed up grievance redressal and ensure better administrative responses.

## 📸 Screenshots
![Screenshot 2025-04-03 215102](https://github.com/user-attachments/assets/df233e0e-7107-4ec9-804e-ecfd3349d336)
![Screenshot 2025-04-03 215417](https://github.com/user-attachments/assets/dd2bc6e7-77e5-4f78-89f4-d52d0685117c)
![Screenshot 2025-04-08 113455](https://github.com/user-attachments/assets/dcd43c8e-ec8a-415e-bf23-1ba05555972d)
![WhatsApp Image 2025-04-08 at 11 36 28_5603b294](https://github.com/user-attachments/assets/4186fbc4-abc3-4723-9b32-10a56be12b9e)






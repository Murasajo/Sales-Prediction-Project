## Sales Prediction and Visualization Project

This project demonstrates a machine learning pipeline for predicting sales based on advertising spend across different channels (TV, radio, and newspapers). It includes a Streamlit web application that visualizes both the original sales data and the predicted sales, providing insights into the model's performance.

### Features

- **Data Visualization**: Interactive bar charts and time series plots to explore the relationships between advertising channels and sales.
- **Sales Prediction**: Utilizes a pre-trained linear regression model to predict sales based on advertising spend.
- **Comparison of Predictions**: Side-by-side comparison of original and predicted sales to evaluate model accuracy.
- **Easy-to-Use Interface**: Streamlit app for intuitive and user-friendly interaction with the data and predictions.

### Usage

1. **Load Data**: Upload the advertising dataset (CSV format).
2. **Visualize Data**: View the original sales data and its relationship with advertising spend.
3. **Predict Sales**: Generate predictions using the pre-trained model.
4. **Compare Results**: Compare the original sales with predicted sales using interactive plots.

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Murasajo/Sales-Prediction-Project.git
    cd sales-prediction-app
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

### Dependencies

- Streamlit
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

### Acknowledgments

This project is based on a linear regression model to predict sales from advertising spend data. Special thanks to the open-source community for providing the tools and resources necessary for this project.

---

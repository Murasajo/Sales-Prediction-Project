import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the pre-trained model
@st.cache_resource
def load_model():
    with open('linear_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('advertising.csv')
    return df

def generate_predictions(model, df):
    features = df[['TV', 'radio', 'newspaper']]
    predictions = model.predict(features)
    df['Predicted_sales'] = predictions
    return df

# Plot the sales original data
def plot_sales_data(df):
    st.write("### Original Sales Data")
    st.bar_chart(df[['TV', 'radio', 'newspaper', 'sales']])
    fig, ax = plt.subplots()
    sns.lineplot(data=df[['sales']], ax=ax)
    st.pyplot(fig)


# PLot comparison of original and predicted sales
def plot_comparison(df):
    st.write("### Comparison of Original and Predicted Sales")
    fig, ax = plt.subplots()
    sns.lineplot(data=df[['sales', 'Predicted_sales']], ax=ax)
    st.pyplot(fig)

def main():
    st.title("Sales Prediction Visualization App")

    # Load data
    df = load_data()

    # Display raw data
    st.write("### Raw Data")
    st.write(df.head())

    # Visualize the original dataset
    plot_sales_data(df)

    # Load the model and generate predictions
    model = load_model()
    df_with_predictions = generate_predictions(model, df)

    # Display the predicted dataset
    st.write("### Data with Predictions")
    st.write(df_with_predictions.head())

    # Visualize the comparison
    plot_comparison(df_with_predictions)

if __name__ == "__main__":
    main()

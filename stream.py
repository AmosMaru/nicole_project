import streamlit as st
import pandas as pd
from joblib import load

# Load the Isolation Forest model
loaded_model = load('model.pkl')

# Streamlit app
def main():
    st.title('Anomaly Detection with Isolation Forest')

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)

        # Predict anomalies
        df['anomaly'] = loaded_model.predict(df)

        # Display the DataFrame with anomaly predictions
        st.write(df)

if __name__ == '__main__':
    main()

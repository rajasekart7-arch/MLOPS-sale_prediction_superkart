import streamlit as st
import pandas as pd
import requests
import io

BACKEND_API_URL = 'https://rajse-superkart-salespredictionbackend.hf.space'

st.set_page_config(page_title="SuperKart Predictor", layout="wide")
st.title('🛒 SuperKart Sales Prediction')

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    st.header('Single Entry')
    col1, col2 = st.columns(2)

    with col1:
        p_id = st.text_input('Product ID', 'FD6114')
        p_weight = st.number_input('Weight', value=12.6)
        p_sugar = st.selectbox('Sugar Content', ['Low Sugar', 'Regular'])
        p_area = st.number_input('Allocated Area', value=0.02)
        p_type = st.selectbox('Type', ['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household'])

    with col2:
        p_mrp = st.number_input('MRP', value=110.0)
        s_id = st.selectbox('Store ID', ['OUT049', 'OUT018', 'OUT010'])
        s_year = st.number_input('Est. Year', value=2009)
        s_size = st.selectbox('Store Size', ['Medium', 'Small', 'High'])
        s_city = st.selectbox('City Type', ['Tier 1', 'Tier 2', 'Tier 3'])
        s_type = st.selectbox('Store Type', ['Supermarket Type1', 'Supermarket Type2'])

    if st.button('Predict Single'):
        payload = {
            "Product_Id": p_id, "Product_Weight": p_weight, "Product_Sugar_Content": p_sugar,
            "Product_Allocated_Area": p_area, "Product_Type": p_type, "Product_MRP": p_mrp,
            "Store_Id": s_id, "Store_Establishment_Year": s_year, "Store_Size": s_size,
            "Store_Location_City_Type": s_city, "Store_Type": s_type
        }
        res = requests.post(f"{BACKEND_API_URL}/v1/sales", json=payload)
        if res.status_code == 200:
            st.metric("Predicted Sales", f"${res.json()['Predicted Price (in dollars)']}")
        else:
            st.error("Backend Error. Is the backend Space running?")

with tab2:
    st.header('Batch Prediction')
    st.write("Upload your 10-row CSV file below.")

    # We use a 'key' to ensure Streamlit tracks this specific component's state
    uploaded_file = st.file_uploader("Choose CSV", type="csv", key="batch_uploader")

    if uploaded_file is not None:
        st.success(f"✅ Ready to process: {uploaded_file.name}")

        try:
            preview_df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(preview_df.head(5))

            # Reset file pointer to the beginning after reading for preview
            uploaded_file.seek(0)

            if st.button('🚀 Run Batch Predictions', key="process_btn"):
                with st.spinner('Sending data to backend...'):
                    try:
                        # Prepare the multipart file upload
                        # We use getvalue() to get raw bytes - most stable for HF Proxy
                        files = {
                            'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/csv')
                        }

                        # API Call with a generous timeout
                        response = requests.post(
                            f"{BACKEND_API_URL}/v1/salebatch",
                            files=files,
                            timeout=120
                        )

                        if response.status_code == 200:
                            predictions = response.json()
                            # Convert dict to DataFrame
                            res_df = pd.DataFrame(
                                list(predictions.items()),
                                columns=['Product_Id', 'Predicted Sales']
                            )
                            st.subheader("Results")
                            st.dataframe(res_df)
                            st.download_button("📥 Download Predictions", res_df.to_csv(index=False), "results.csv")
                        else:
                            st.error(f"Backend Error {response.status_code}: {response.text}")

                    except requests.exceptions.ConnectionError:
                        st.error("Connection Error: Is the Backend Space Running?")
                    except Exception as e:
                        st.error(f"Unexpected Error: {e}")

        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    else:
        # This is what users see before they upload
        st.info("Upload your CSV file above to reveal the 'Process' button.")

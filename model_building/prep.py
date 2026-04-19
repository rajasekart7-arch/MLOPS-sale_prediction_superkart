
# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from huggingface_hub import hf_hub_download


# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))


file_path = hf_hub_download(
    repo_id="Rajse/Superkart-Dataset",
    filename="superkart.csv",
    repo_type="dataset"
)

dataset = pd.read_csv(file_path)
print("Dataset loaded successfully.")

# Define the target variable for the classification task
target = 'Product_Store_Sales_Total'

# Calculate the current year
current_year = datetime.now().year
# Calculate the current age of the store
dataset['Store_Current_Age'] = current_year - dataset['Store_Establishment_Year']

# Drop the 'Store_Establishment_Year' column (optional)
dataset.drop('Store_Establishment_Year', axis=1, inplace=True)

# Transform 'Product_Type' into 'Perishables' and 'Non Perishables'
perishables = [
    "Dairy",
    "Meat",
    "Fruits and Vegetables",
    "Breakfast",
    "Breads",
    "Seafood",
]

dataset['Product_Type'] = dataset['Product_Type'].apply(lambda x: 'Perishables' if x in perishables else 'Non Perishables')

# Drop 'Product_Id' as it is not useful for prediction and can cause overfitting
dataset.drop(["Product_Id"], axis=1, inplace=True)

X = dataset.drop(target, axis=1)
y = dataset[target]


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="Rajse/Superkart-Dataset",
        repo_type="dataset",
    )

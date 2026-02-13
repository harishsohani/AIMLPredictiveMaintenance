
# libraries for data manipulation
import pandas as pd
import sklearn

# libraries for creating a folder
import os

# libraries for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# libraries for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder

# libraries for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))

# store file path of data set (csv file) in variable
DATASET_PATH = "hf://datasets/harishsohani/AIMLPredictiveMaintenance/engine_data.csv"

# define random state for repeatibility
RANDOM_STATE = 42

# define test split size
TEST_SIZE = 0.30

# define report id and repo type
hf_repo_id        = "harishsohani/AIMLPredictiveMaintenance"

hf_data_repo_type = "dataset"



# read data from csv as data frame (Note: data is read from hugging face space)
df_hf = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Data cleaning

#===================================#
# 1.  Data type conversion - float  #
#===================================#
# get all integer columns
int_columns = df_hf.select_dtypes(include=['int64']).columns

# convert integer columns to float
df_hf[int_columns] = df_hf[int_columns].astype('float64')


#=====================================================#
# 2. Replace column names with space with Underscore  #
#=====================================================#
# replace space in column names with underscore (_)
df_hf.columns = df_hf.columns.str.replace(' ', '_')



# Define target variable
target_col = 'Engine_Condition'


# Split into X (features) and y (target)
X = df_hf.drop(columns=[target_col])
y = df_hf[target_col]

# Perform train-test split.
# Note 80% of data is used for training and 20% for testing
# Perform train-test split
# Since the data set is imbalanced in class - using stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# save tran and test data set as scv files
X_train.to_csv("Xtrain.csv",index=False)
X_test.to_csv("Xtest.csv",index=False)
y_train.to_csv("ytrain.csv",index=False)
y_test.to_csv("ytest.csv",index=False)

# define files list for uploading
files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

# copy train and test data (csv files)
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id=hf_repo_id,
        repo_type=hf_data_repo_type,
    )

print ("Uploaded Train and test data (csv) files to Hugging Face Space (Dataset) Successfully")

# rag-watsonx-demo

Create an account at IBM Cloud

https://dataplatform.cloud.ibm.com/wx/home?context=wx

Create an IAM API Key and get the Project ID, add them to the `.env` file.

The IBM Cloud API key is created at https://cloud.ibm.com/iam/apikeys

```
IBM_CLOUD_API_KEY=
WATSONX_PROJECT_ID=
```

## Run locally

```
python3 -m venv env 
source env/bin/activate
pip install -U pip
pip install -r requirements.txt
streamlit run main.py
``
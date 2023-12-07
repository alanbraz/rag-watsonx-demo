# rag-watsonx-demo

Create an account at IBM Cloud

https://dataplatform.cloud.ibm.com/wx/home?context=wx

Create an IAM API Key and get the Project ID, add them to the `.env` file

```
IBM_CLOUD_API_KEY=
WATSONX_PROJECT_ID=
```

## Run locally

```
python -m venv env 
source env/bin/activate
pip install -U pip
pip install -r requirements.txt
streamlit run main.py
``
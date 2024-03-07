from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

import streamlit as st

import os
from dotenv import load_dotenv

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods

load_dotenv()

@st.cache_resource
def load_pdf(pdf_name):
    loaders = [ PyPDFLoader(pdf_name)]
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2"),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=50)
    ).from_loaders(loaders)
    return index

# st.title("Ask Me")

def format_history():
    text = ""
    for message in st.session_state.messages:
        if message["content"].startswith("[LLM]") or message["role"] == "User":
            text += ("Humano" if message["role"] == "User" else "IA") +": "+ message["content"].lstrip("[LLM]").strip()+"\n" 
    return text

model = None
index = None
chain = None
rag_chain = None
watsonx_project_id = os.getenv("WATSONX_PROJECT_ID", None)

prompt_template = PromptTemplate(
    input_variables=["chat_history","user_input"], 
    template="""[INST]<<SYS>>\n
You are a helpful, respectful and honest assistant. 
Always answer as helpfully as possible, while being safe. 
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
Please ensure that your responses are socially unbiased and positive in nature.\n\n
If a question does not make any sense, or is not factually coherent, 
explain why instead of answering something not correct. 
If you don't know the answer to a question, please don't share false information.
Read the chat history to get context.\nTranslate the response to Brazilian portuguese.<</SYS>>
Chat History:\n\n{chat_history}\n\nUser: {user_input}\nAssistant: [/INST]"""
    )

prompt_template_br = PromptTemplate(
    input_variables=["context","question"], 
    template="""[INST]<<SYS>>\n
Você é um assistente prestativo, respeitoso e honesto.
Sempre responda da maneira mais prestativa possível, estando seguro.
Suas respostas não devem incluir conteúdo prejudicial, antiético, racista, sexista, tóxico, perigoso ou ilegal.
Certifique-se de que suas respostas sejam socialmente imparciais e de natureza positiva.\n\n
Se uma pergunta não fizer sentido ou não for factualmente coerente,
explique o porquê em vez de responder algo incorreto.
Se você não sabe a resposta a uma pergunta, não compartilhe informações falsas.
Leia o histórico do bate-papo para contextualizar.\n
Responda resumidamente em no máximo 2 parágrafos.
Traduza sempre a resposta para o português do Brasil.<</SYS>>
Histórico de conversa:\n\n{context}\n\nHumano: {question} Responda em Português brasileiro.\n
IA: [/INST]"""
    )

with st.sidebar: 
    st.title("watsonx RAG demo with Llama2 70B")        
    watsonx_api_key = st.text_input("Watsonx API Key", key="watsonx_api_key", value=os.getenv("IBM_CLOUD_API_KEY", None), type="password")
    if not watsonx_project_id:
        watsonx_project_id = st.text_input("Watsonx Project ID", key="watsonx_project_id")
    #TODO: change this to a select box with more than one model
    # watsonx_model = st.text_input("Model", key="watsonx_model", value="meta-llama/llama-2-70b-chat", type="default")   
    # watsonx_model = st.selectbox("Model", [ model.value for model in ModelTypes], index=6) 
    # watsonx_model_params = st.text_area("Params", key="watsonx_model_params", value='{"decoding_method":"sample", "max_new_tokens":200, "temperature":0.5}' )
    # language = st.sidebar.radio("Language", ("Português", "Inglês"))
    max_new_tokens = st.slider("Max output tokens", min_value=100, max_value=500, value=200, step=100)
    decoding_method = st.sidebar.radio("Decoding", (DecodingMethods.GREEDY.value, DecodingMethods.SAMPLE.value))
    parameters = {
        GenParams.DECODING_METHOD: decoding_method,
        GenParams.MAX_NEW_TOKENS: max_new_tokens,
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.TEMPERATURE: 0,
        GenParams.TOP_K: 50,
        GenParams.TOP_P: 1
    }
    st.info("Upload a PDF file to use RAG")
    uploaded_file = st.file_uploader("Upload file", accept_multiple_files=False)
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        st.write("filename:", uploaded_file.name)
        with open(uploaded_file.name, 'wb') as f:
            f.write(bytes_data)
        index = load_pdf(uploaded_file.name)

    model_name = "meta-llama/llama-2-70b-chat"

if watsonx_api_key and watsonx_project_id:
    st.info("Seting up {}...".format("watsonx"))

    my_credentials = { 
        "url"    : "https://us-south.ml.cloud.ibm.com", 
        "apikey" : watsonx_api_key
    }
    params = parameters #json.loads(watsonx_model_params)
    project_id  = watsonx_project_id #watsonx_project if len(watsonx_project) > 0 else None
    space_id    = None
    verify      = False
    model = WatsonxLLM(model=Model(model_name, my_credentials, params, project_id, space_id, verify))

if model:
    st.info("Model {} pronto.".format(model_name))
    chain = LLMChain(llm=model, prompt=prompt_template_br, verbose=True)

if chain:
    st.info("Chat pronto.")
    if index:
        rag_chain = RetrievalQA.from_chain_type(
                llm=model,
                chain_type="stuff",
                retriever=index.vectorstore.as_retriever(),
                # input_key="user_input",
                chain_type_kwargs={ "prompt": prompt_template_br },
                return_source_documents=False,
                verbose=True
            )
        st.info("Chat pronto com documento PDF.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

prompt = st.chat_input("Faça sua pergunta aqui", disabled=False if chain else True)

if prompt:
    st.chat_message("user").markdown(prompt)

    response = chain.run(question=prompt, 
                         context=format_history())

    st.session_state.messages.append({'role': 'User', 'content': prompt })
    st.chat_message("assistant").markdown("[LLM] "+response)
    st.session_state.messages.append({'role': 'Assistant', 'content': "[LLM] "+response })

    if rag_chain:
        response = rag_chain.run(prompt)
        st.chat_message("assistant").markdown("[DOC] "+ response)
        st.session_state.messages.append({'role': 'Assistant', 'content': "[DOC] "+response })

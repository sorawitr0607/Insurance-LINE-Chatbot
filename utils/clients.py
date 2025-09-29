## Import Library
import os
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI
# from google import genai
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi

load_dotenv()

## Setup Variable
_search_client: SearchClient | None = None
_service_search_client : SearchClient | None = None
_openai_client:  OpenAI       | None = None
# _gemini_client:  genai.Client | None = None
_line_api: MessagingApi | None = None


def get_search_client() -> SearchClient:
    global _search_client
    if _search_client is None:            
        _search_client = SearchClient(
            endpoint = os.getenv("AZURE_SEARCH_ENDPOINT"),
            credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY")),
            index_name = os.getenv("AZURE_SEARCH_INDEX"))
    return _search_client  

def get_service_search_client() -> SearchClient:
    global _service_search_client
    if _service_search_client is None:            
        _service_search_client = SearchClient(
            endpoint = os.getenv("AZURE_SEARCH_ENDPOINT"),
            credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY")),
            index_name = os.getenv("AZURE_SEARCH_INDEX_INSURANCE_SERVICE"))
    return _service_search_client               

def get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client

# def get_gemini() -> genai.Client:
#     global _gemini_client
#     if _gemini_client is None:
#         _gemini_client = genai.Client(
#             api_key=os.getenv("GEMINI_API_KEY"))
#     return _gemini_client

def get_line_api() -> MessagingApi:
    """Thread-safe singleton for the LINE Messaging API."""
    global _line_api
    if _line_api is None:
        configuration = Configuration(access_token=os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
        _line_api = MessagingApi(ApiClient(configuration))
    return _line_api

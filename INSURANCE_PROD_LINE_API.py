from openai import OpenAI
import os
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from pymongo import MongoClient
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import Configuration,ApiClient,MessagingApi,ReplyMessageRequest,TextMessage
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent,TextMessageContent
from datetime import datetime

load_dotenv()


# OpenAI setup
embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")
chat_model = os.getenv("OPENAI_CHAT_MODEL")

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Azure AI Search setup
search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
)

service_search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX_INSURANCE_SERVICE"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
)

# MongoDB
mongo_uri = os.getenv("COSMOS_MONGO_URI")
mongo_db = os.getenv("COSMOS_MONGO_DB")
mongo_table = os.getenv("COSMOS_MONGO_TABLE")
mongo_client = MongoClient(mongo_uri)
db = mongo_client[mongo_db]
conversations = db[mongo_table]

# LINE API
app = Flask(__name__)
configuration = Configuration(access_token=os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))

def embed_text(text: str):
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        input= text,
        model= embedding_model
    )
    return response.data[0].embedding

def print_results(results):
    answer = []
    for result in results:
        answer+=[f"Product Segment: {result['Product_Segment']}"]
        answer+=[f"Product Name: {result['Product_Name']}"]
        answer+=[f"Unique Point: {result['Unique_Pros']}"] 
        answer+=[f"Product Benefit: {result['Benefit']}"]
        answer+=[f"Product Condition: {result['Condition']}"]
        answer+=[f"Product Description: {result['Product_Description']}"]
        answer+=[f"URL: {result['Product_URL']}\n"]
    return answer

def print_results_service(results):
    answer = []
    for result in results:
        answer+=[f"Service Segment: {result['Service_Segment']}"]
        answer+=[f"Service Name: {result['Service_Name']}"]
        answer+=[f"Service Detail: {result['Service_Detail']}"]
        answer+=[f"URL: {result['Service_URL']}\n"]
    return answer
        

def retrieve_context(query, top_k=2):
    query_vector = embed_text(query)
    vector_query = VectorizedQuery(
        vector=query_vector, 
        k_nearest_neighbors=50, 
        fields="text_vector")
    results = search_client.search(
        search_text=None,
        vector_queries= [vector_query],
        select=["Product_Segment","Product_Name", "Unique_Pros", "Benefit","Condition","Product_Description","Product_URL"],
        top=top_k
    )
    return "\n\n".join(print_results(results))

def retrieve_insurance_service_context(query, top_k=2):
    query_vector = embed_text(query)
    vector_query = VectorizedQuery(
        vector=query_vector, 
        k_nearest_neighbors=50, 
        fields="text_vector"
    )
    results = service_search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["Service_Segment","Service_Name","Service_Detail","Service_URL"],
        top=top_k
    )
    return "\n\n".join(print_results_service(results))

def summarize_text(text: str, max_chars: int = 2000, user_id = None) -> str:

    if len(text) <= max_chars:
        return text
    
    system_prompt = "You are a helpful assistant. Summarize the user's conversation prioritize newer message"
    response = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    summary = response.choices[0].message.content.strip()
    timestamp = datetime.now()
    conversations.delete_many({"user_id": user_id})
    save_chat_history(user_id, "assistant", summary, timestamp)
    return summary

def generate_answer(query, context, chat_history=None):
        prompt = f"""
        You are a helpful expert insurance (ทั้งประกันชีวิตและประกันภัย) salesman agent assistant from 'Thai Group Holdings Public Company Limited.' or 'บริษัท อาคเนย์ประกันชีวิต' which have 2 BU 1.SE Life (อาคเนย์ประกันชีวิต) 2.INDARA (อินทรประกันภัย)
        Your goals:
            Answer the query using only Context (and conversation history) provided below to analyze and recommend insurance products or services.
    
        Constraints:
            - Respond **in Thai** unless absolutely necessary to reference specific names or URLs.
            - If you do not have sufficient information, respond that you are unsure or request clarification.
        Conversation History: {chat_history if chat_history else 'None'}
        Context: {context}
        Question: {query}
        Answer: """

        response = client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "system", "content": "You are a helpful expert insurance salesman agent assistant"},
            {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=700)
            
        return response.choices[0].message.content.strip()
    
def save_chat_history(user_id, sender, message, timestamp):
    conversations.insert_one({
        "user_id": user_id,
        "sender": sender,
        "message": message,
        "timestamp": timestamp
    })

def get_chat_history(user_id, limit=10):
    messages = list(conversations.find(
        {"user_id": user_id},
        sort=[("timestamp", -1)],
        limit=limit
    ))
    messages.reverse()
    history_text = "\n".join([f"{m['sender']}: {m['message']}" for m in messages])
    summary = summarize_text(history_text, max_chars=2000,user_id = user_id)
    return summary

def decide_search_path(user_query, chat_history=None):

    classification_prompt = f"""
    You are a classification model. Classify the following user query (consider together with conversion history provided below) into exactly one of these categories:
    1) RESET
    2) INSURANCE_SERVICE
    3) INSURANCE_PRODUCT
    4) CONTINUE CONVERSATION
    5) OFF-TOPIC

    Guidelines:
    - If the user explicitly asks or strongly implies wanting to reset the chat, choose "RESET".
    - If the query is about insurance services in particular (e.g., "กรอบระยะเวลาสำหรับการให้บริการ","ประกันกลุ่ม","ตรวจสอบผู้ขายประกัน","ดาวน์โหลดแบบฟอร์มต่างๆ","ค้นหาโรงพยาบาลคู่สัญญา","ค้นหาสาขา","บริการพิเศษ","บริการเรียกร้องสินไหมทดแทน","บริการด้านการพิจารณารับประกัน","บริการผู้ถือกรมธรรม์","บริการรับเรื่องร้องเรียน","ข้อแนะนำในการแจ้งอุบัติเหตุ","บริการตัวแทน - นายหน้า"), choose "INSURANCE_SERVICE".
    - If the query involves insurance product, or policy  (e.g., "แนะนำประกัน","ขอดูประกัน" etc.) but not specifically "Insurance Service," choose "INSURANCE_PRODUCT".
    - If the query is ask more detail of previous conversation that said in conversation history, choose "CONTINUE CONVERSATION".
    - Otherwise, choose "OFF-TOPIC".

    Return ONLY one of these category labels: RESET, INSURANCE_SERVICE, INSURANCE_PRODUCT, CONTINUE CONVERSATION, or OFF-TOPIC.

    User Query: {user_query}
    Conversation History: {chat_history if chat_history else 'None'}
    """

    response = client.chat.completions.create(
        model=chat_model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert text classification model. Respond with a single category label."
            },
            {
                "role": "user",
                "content": classification_prompt
            },
        ],
        temperature=0.1,
        max_tokens=10
    )

    # Extract classification from response
    path_decision = response.choices[0].message.content.strip().upper()

    # Validate output (in case the model returns something unexpected)
    valid_categories = ["RESET","INSURANCE_SERVICE","INSURANCE_PRODUCT","CONTINUE CONVERSATION","OFF-TOPIC"]
    if path_decision not in valid_categories:
        path_decision = "OFF-TOPIC"

    return path_decision

@app.route("/webhook", methods=['POST'])
def webhook():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.info("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_query = event.message.text
    user_id = event.source.user_id
    chat_history = get_chat_history(user_id)
    #print(len(chat_history))
    path_decision = decide_search_path(user_query,chat_history)
    #print(path_decision)
    
    if path_decision == "RESET":
        conversations.delete_many({"user_id": user_id})
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text="แชทของคุณถูกรีเซ็ตเรียบร้อยแล้ว")]
                )
            )
        return

    
    if path_decision == "INSURANCE_SERVICE":
        context = retrieve_insurance_service_context(user_query)
    elif path_decision == "INSURANCE_PRODUCT":
        context = retrieve_context(user_query)
    elif path_decision == "CONTINUE CONVERSATION":
        context = ""
    else:
        chat_history = None
        context = ""
        
    response = generate_answer(user_query, context, chat_history)

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=response)]
            )
        )

    timestamp = datetime.now()
    save_chat_history(user_id, "user", user_query, timestamp)
    save_chat_history(user_id, "assistant", response, timestamp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

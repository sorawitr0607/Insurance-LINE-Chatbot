from openai import OpenAI
import os
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
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
        

def retrieve_context(query, top_k=10 , skip_k=0):
    query_vector = embed_text(query)
    vector_query = VectorizedQuery(
        vector=query_vector, 
        k_nearest_neighbors=100, 
        fields="text_vector")
    results = search_client.search(
        search_text=None,
        vector_queries= [vector_query],
        select=["Product_Segment","Product_Name", "Unique_Pros", "Benefit","Condition","Product_Description","Product_URL"],
        top=top_k,
        skip = skip_k
        
    )
    return "\n\n".join(print_results(results))

def retrieve_insurance_service_context(query, top_k=3):
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

def summarize_text(text, max_chars, user_id):

    if len(text) <= max_chars:
        return text
    
    system_prompt = "You are a helpful assistant. Condense the user's conversation by selectively removing less important or redundant information. Prioritize preserving numeric details, specific names, exact wording, key facts, and recent messages. Avoid overly summarizing; keep the original details intact."
    response = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0.5,
        max_tokens=1000
    )
    summary = response.choices[0].message.content.strip()
    timestamp = datetime.now()
    from utils.chat_history_func import save_chat_history,del_chat_history,get_latest_decide
    del_chat_history(user_id)
    user_latest_decide = get_latest_decide(user_id)
    save_chat_history(user_id, "assistant", summary, timestamp,latest_decide=user_latest_decide)
    return summary

def summarize_context(new_question,chat_history):

    system_prompt = 'Summarize the following chat history and the latest user question into a concise, context-rich summary suitable for vector-based search and retrieval. Capture key concepts, specific instructions, queries, and responses. **Preserve all specific names, such as product names** mentioned in the conversation or question.'
    text = f'Chat History: {chat_history} Latest User Question: {new_question}'
    response = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0.3,
        max_tokens=200
    )
    summary = response.choices[0].message.content.strip()
    return summary


def decide_search_path(user_query, chat_history=None):

    # Build a short, direct prompt
    classification_prompt = f"""
You are a highly accurate text classification model. 
Determine which single label (from the set: RESET, INSURANCE_SERVICE, INSURANCE_PRODUCT, 
CONTINUE CONVERSATION, MORE, OFF-TOPIC) best fits the user's latest query. 

Guidelines:
- "RESET" if the user requests or strongly implies resetting the conversation.
- "INSURANCE_SERVICE" if the query is specifically about services (e.g., "กรอบระยะเวลาสำหรับการให้บริการ","ประกันกลุ่ม","ตรวจสอบผู้ขายประกัน","ดาวน์โหลดแบบฟอร์มต่างๆ","ค้นหาโรงพยาบาลคู่สัญญา","ค้นหาสาขา","บริการพิเศษ","บริการเรียกร้องสินไหมทดแทน","บริการด้านการพิจารณารับประกัน","บริการผู้ถือกรมธรรม์","บริการรับเรื่องร้องเรียน","ข้อแนะนำในการแจ้งอุบัติเหตุ","บริการตัวแทน - นายหน้า").
- "INSURANCE_PRODUCT" if the query is about insurance products (e.g., "I want to buy insurance", "Show me plans", etc.).
- "CONTINUE CONVERSATION" if the user is asking a follow-up about a previously conversation.
- "MORE" if the user wants additional product beyond previous discussion (e.g., 'show me more product','tell me more product').
- Otherwise, "OFF-TOPIC".

Return ONLY one label. Do not add explanations.

User Query: {user_query}
Conversation History: {chat_history if chat_history else 'None'}
""".strip()

    # Call the OpenAI Chat Completion endpoint
    response = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": "You are a classification model. Return only one label."},
            {"role": "user", "content": classification_prompt},
        ],
        temperature=0.0,
        max_tokens=10
    )

    # Extract classification
    path_decision = response.choices[0].message.content.strip().upper()

    valid_categories = [
        "RESET",
        "INSURANCE_SERVICE",
        "INSURANCE_PRODUCT",
        "CONTINUE CONVERSATION",
        "MORE",
        "OFF-TOPIC"
    ]

    # Validate output
    if path_decision not in valid_categories:
        path_decision = "OFF-TOPIC"

    return path_decision


def generate_answer(query, context, chat_history=None):
        prompt = (
        "You will only use the provided context,conversation history and user question to answer. (try to tell every core detail) "
        "If the user’s query is outside of the context or you lack sufficient info, "
        "politely state that you are not certain or ask for clarification. "
        "Always respond in Thai unless absolutely necessary to reference specific names or URLs.")
        user_prompt = f"""
    Conversation History: {chat_history if chat_history else 'None'}
    Context: {context}
    User Question: {query} """

        response = client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt}],
            temperature=0.7,
            max_tokens=700)
            
        raw_response = response.choices[0].message.content.strip()
        
        # print(raw_response)
    
        # guarded_answer = guard.parse(
        # text=raw_response,
        # reference_text=context,
        # llm_api=client.chat.completions.create)
        
        # print(guarded_answer)
        
        return raw_response

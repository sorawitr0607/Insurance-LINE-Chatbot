from openai import OpenAI
import os
import pickle
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from datetime import datetime
from zoneinfo import ZoneInfo
import bmemcached

load_dotenv()


memcached_endpoint = os.getenv("MEMCACHED_ENDPOINT")
memcached_port = os.getenv("MEMCACHED_PORT")
memcached_username = os.getenv("MEMCACHED_USERNAME")
memcached_password = os.getenv("MEMCACHED_PASSWORD")
  

mc_client = bmemcached.Client(
    (f"{memcached_endpoint}:{memcached_port}",), 
    memcached_username,
    memcached_password             
)

# OpenAI setup
embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")
chat_model = os.getenv("OPENAI_CHAT_MODEL")
classify_model = os.getenv("OPENAI_CLASSIFY_MODEL")


client = OpenAI(
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

EMBED_CACHE_TTL = int(24 * 3600)
SEARCH_CACHE_TTL = int(3600)


def embed_text(text: str):

    normalized = text.replace("\n", " ").strip()
    key = f"embed:{normalized}"
    # print(key)
    cached = mc_client.get(key)
    if cached:
        return pickle.loads(cached)

    response = client.embeddings.create(
        input=normalized,
        model= embedding_model
    )
    embedding = response.data[0].embedding
    mc_client.set(key, pickle.dumps(embedding),EMBED_CACHE_TTL)
    return embedding

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
        

def get_search_results(query: str, top_k: int, skip_k:int=0, service: bool = False):

    normalized = query.strip()
    key = f"search:{'svc' if service else 'prd'}:{normalized}|{top_k}|{skip_k}"
    # print(key)
    cached = mc_client.get(key)
    if cached:
        return pickle.loads(cached)

    vect = embed_text(query)
    vq = VectorizedQuery(
        vector=vect, 
        k_nearest_neighbors=100, 
        fields="text_vector"
    )
    client_to_use = service_search_client if service else search_client
    results = client_to_use.search(
        search_text=query,
        vector_queries=[vq],
        select=(
            ["Service_Segment","Service_Name","Service_Detail","Service_URL"] if service
            else ["Product_Segment","Product_Name","Unique_Pros","Benefit","Condition","Product_Description","Product_URL"]
        ),
        top=top_k,
        skip = skip_k
    )
    text = "=================\n".join(
        print_results_service(results) if service
        else print_results(results)
    )
    mc_client.set(key, pickle.dumps(text),SEARCH_CACHE_TTL)
    return text

def summarize_text(text, max_chars, user_id):

    if len(text) <= max_chars:
        return text
    
    system_prompt = "You are a helpful assistant. Condense the user's conversation by selectively removing less important or redundant information. Prioritize preserving numeric details, specific names, exact wording, key facts, and recent messages. Avoid overly summarizing; keep the original details intact.,Respond concisely and not exceed 1000 tokens."
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
    timestamp = datetime.now(ZoneInfo("Asia/Bangkok"))
    from utils.chat_history_func import save_chat_history,del_chat_history,get_latest_decide
    del_chat_history(user_id)
    user_latest_decide = get_latest_decide(user_id)
    save_chat_history(user_id, "assistant", summary, timestamp,user_latest_decide)
    return summary

def summarize_context(new_question,chat_history):

    system_prompt = (
        "You are an expert summarizer for a vector-based retrieval system. Your goal is "
        "to produce a concise, context-rich summary focused on the user's latest question. "
        "Include only details from the conversation history that are directly relevant "
        "to the new question. Omit irrelevant or off-topic content, and do not include URLs."
        "\n\n"
        "Ensure you preserve exact wording for any product names or special terms (including "
        "those in asterisks, e.g., *ProductName*). Keep it short but detailed enough that "
        "someone reading this summary can address the user's latest question accurately."
        "Respond concisely and within 180 tokens."
    )
    text = f"""
    Chat History: {chat_history}
    Latest User Question: {new_question}

    Instructions:
    - Focus on the user’s new question and only summarize the parts of the chat that are relevant.
    - If the new question refers to, for example, “the second insurance product,” then only include
      the details needed about that second product, ignoring the rest.
    - Preserve special terms or product names exactly as they appear (e.g., *ProductX*).
    - Exclude URLs or disclaimers unless the user specifically wants them.
    - Keep the summary concise but complete enough for follow-up vector-based retrieval.
    
    """.strip()
    response = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0.2,
        max_tokens=200
    )
    summary = response.choices[0].message.content.strip()
    # print(summary)
    return summary


def decide_search_path(user_query, chat_history=None):

    classification_prompt = f"""
You are a highly accurate text classification model. 
Determine which single label (from the set: INSURANCE_SERVICE, INSURANCE_PRODUCT, 
CONTINUE CONVERSATION, MORE, OFF-TOPIC) best fits this scenario, based on the User Query and the Conversation History.

Definitions and guidelines:

1. CONTINUE CONVERSATION
   - The user is clearly asking a follow-up question.
   - Or user references details that were already mentioned in the conversation history.
   - Example:
       - "Could you give me more information on insurance we talked about?"
       - "Clarify the cost you mentioned earlier."
       - "You said something about life coverage; can you elaborate?"
       - If the conversation history included "I want to buy insurance. Do you have life coverage?" 
         and the new user query says "tell me more about the first one," then it's classify to CONTINUE CONVERSATION.

2. INSURANCE_SERVICE
   - Specifically about insurance services such as "ติดต่อสอบถาม", "เอกสาร" , "โปรโมชั่น", "กรอบระยะเวลาสำหรับการให้บริการ","ประกันกลุ่ม","ตรวจสอบผู้ขายประกัน","ดาวน์โหลดแบบฟอร์มต่างๆ","ค้นหาโรงพยาบาลคู่สัญญา","ค้นหาสาขา","บริการพิเศษ","บริการเรียกร้องสินไหมทดแทน","บริการด้านการพิจารณารับประกัน","บริการผู้ถือกรมธรรม์","บริการรับเรื่องร้องเรียน","ข้อแนะนำในการแจ้งอุบัติเหตุ","บริการตัวแทน - นายหน้า", etc.

3. INSURANCE_PRODUCT
   - The user wants to buy, see, or compare insurance products such as life insurance or auto insurance policies.

4. MORE
   - The user specifically asks for additional products or variations beyond what was previously discussed.
   - Common triggers might be phrases like "Show me more products" or "What else do you have?"

5. OFF-TOPIC
   - Anything not covered above, or the user’s query is irrelevant to insurance.

Return ONLY one label. Do not add explanations.

------------------------------------
User Query: {user_query}
Conversation History: {chat_history if chat_history else 'None'}
"""

    response = client.chat.completions.create(
        model=classify_model,
        messages=[
            {
                "role": "system",
                "content": "You are a classification model. Return only one label: INSURANCE_SERVICE, INSURANCE_PRODUCT, CONTINUE CONVERSATION, MORE, OFF-TOPIC."
            },
            {
                "role": "user",
                "content": classification_prompt
            },
        ],
        temperature=0.3,  # Lower temperature to reduce random variations
        max_tokens=10,

    )

    # Extract classification
    path_decision = response.choices[0].message.content.strip().upper()

    valid_categories = [
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
        "You are a helpful expert insurance (ทั้งประกันชีวิตและประกันภัย) salesman agent assistant (Men)"
        "from 'Thai Group Holdings Public Company Limited' which has 2 business units: "
        "1) SE Life (อาคเนย์ประกันชีวิต) and 2) INSURE (อินทรประกันภัย). "
        "You must rely **only** on the provided context,provided conversation history and provided user question to answer. (try to tell every main detail) "
        "If the user question is outside the provided context or "
        "no provided context"
        "or user question is not related to insurance product/service, reply politely and very briefly (≤ 30 tokens)"
        "to say that you are uncertain or ask for clarification"
        "If User Question is Thai Always respond in Thai unless absolutely necessary to reference specific names or URLs."
        "Respond concisely and not exceed 680 tokens")
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
        
        return raw_response

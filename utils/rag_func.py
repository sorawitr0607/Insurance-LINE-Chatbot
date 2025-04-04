from openai import OpenAI
import os
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from datetime import datetime
from zoneinfo import ZoneInfo
# from guardrails import Guard, OnFailAction, configure
# from guardrails.hub import BespokeMiniCheck

load_dotenv()

# hub_token = os.getenv("GUARDRAILS_TOKEN")

# configure(
#     enable_metrics=True,
#     enable_remote_inferencing=True,
#     token=hub_token
# )

# # Instantiate Guard and use BespokeMiniCheck
# guard = Guard().use(
#     BespokeMiniCheck(
#         split_sentences=True,
#         threshold=0.5,
#         on_fail=OnFailAction.REASK,   # or OnFailAction.FIX, or OnFailAction.FIX_REASK
#     )
# )

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
        query_type="semantic",
        semantic_configuration_name="product-vector-paid-semantic-configuration",
        search_text=query,
        vector_queries= [vector_query],
        select=["Product_Segment","Product_Name", "Unique_Pros", "Benefit","Condition","Product_Description","Product_URL"],
        top=top_k,
        skip = skip_k
    )
    return "=================\n".join(print_results(results))

def retrieve_insurance_service_context(query, top_k=3):
    query_vector = embed_text(query)
    vector_query = VectorizedQuery(
        vector=query_vector, 
        k_nearest_neighbors=50, 
        fields="text_vector"
    )
    results = service_search_client.search(
        query_type="semantic",
        semantic_configuration_name="service-vector-plan-semantic-configuration",
        search_text=query,
        vector_queries= [vector_query],
        select=["Service_Segment","Service_Name","Service_Detail","Service_URL"],
        top=top_k
    )
    return "=================\n".join(print_results_service(results))

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
    timestamp = datetime.now(ZoneInfo("Asia/Bangkok"))
    from utils.chat_history_func import save_chat_history,del_chat_history,get_latest_decide
    del_chat_history(user_id)
    user_latest_decide = get_latest_decide(user_id)
    save_chat_history(user_id, "assistant", summary, timestamp,latest_decide=user_latest_decide)
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
Determine which single label (from the set: RESET, INSURANCE_SERVICE, INSURANCE_PRODUCT, 
CONTINUE CONVERSATION, MORE, OFF-TOPIC) best fits this scenario, based on the User Query and the Conversation History.

Definitions and guidelines:

1. RESET
   - If the user says "CHAT RESET" or explicitly requests to restart or reset the conversation.

2. CONTINUE CONVERSATION
   - The user is clearly asking a follow-up question.
   - Or user references details that were already mentioned in the conversation history.
   - Example:
       - "Could you give me more information on insurance we talked about?"
       - "Clarify the cost you mentioned earlier."
       - "You said something about life coverage; can you elaborate?"
       - If the conversation history included "I want to buy insurance. Do you have life coverage?" 
         and the new user query says "tell me more about the first one," then it's classify to CONTINUE CONVERSATION.

3. INSURANCE_SERVICE
   - Specifically about insurance services such as "ติดต่อสอบถาม", "เอกสาร" , "โปรโมชั่น", "กรอบระยะเวลาสำหรับการให้บริการ","ประกันกลุ่ม","ตรวจสอบผู้ขายประกัน","ดาวน์โหลดแบบฟอร์มต่างๆ","ค้นหาโรงพยาบาลคู่สัญญา","ค้นหาสาขา","บริการพิเศษ","บริการเรียกร้องสินไหมทดแทน","บริการด้านการพิจารณารับประกัน","บริการผู้ถือกรมธรรม์","บริการรับเรื่องร้องเรียน","ข้อแนะนำในการแจ้งอุบัติเหตุ","บริการตัวแทน - นายหน้า", etc.

4. INSURANCE_PRODUCT
   - The user wants to buy, see, or compare insurance products such as life insurance or auto insurance policies.

5. MORE
   - The user specifically asks for additional products or variations beyond what was previously discussed.
   - Common triggers might be phrases like "Show me more products" or "What else do you have?"

6. OFF-TOPIC
   - Anything not covered above, or the user’s query is irrelevant to insurance.

Return ONLY one label. Do not add explanations.

------------------------------------
User Query: {user_query}
Conversation History: {chat_history if chat_history else 'None'}
"""

    response = client.chat.completions.create(
        model=chat_model,
        messages=[
            {
                "role": "system",
                "content": "You are a classification model. Return only one label: RESET, INSURANCE_SERVICE, INSURANCE_PRODUCT, CONTINUE CONVERSATION, MORE, OFF-TOPIC."
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
        "You are a helpful expert insurance (ทั้งประกันชีวิตและประกันภัย) salesman agent assistant (Men)"
        "from 'Thai Group Holdings Public Company Limited' which has 2 business units: "
        "1) SE Life (อาคเนย์ประกันชีวิต) and 2) INDARA (อินทรประกันภัย). "
        "You will only use the provided context,provided conversation history and provided user question to answer. (try to tell every main detail) "
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

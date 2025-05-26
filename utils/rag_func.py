from openai import OpenAI
import os
import pickle
# import threading
# import time
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
from google import genai
from google.genai import types

load_dotenv()

# OpenAI setup
embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")
# chat_model = os.getenv("TYPHOON_CHAT_MODEL")
# classify_model = os.getenv("OPENAI_CHAT_MODEL")
summary_model = os.getenv("OPENAI_CHAT_MODEL")
openai_api = os.getenv("OPENAI_API_KEY")
# typhoon_api = os.getenv("TYPHOON_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

client_gemini = genai.Client(api_key=gemini_api_key)

client = OpenAI(
    api_key=openai_api,
)

# client_chat = OpenAI(
#     api_key=typhoon_api,
#     base_url="https://api.opentyphoon.ai/v1"
# )

DEFAULT_SAFETY_SETTINGS = {
    types.HarmCategory.HARM_CATEGORY_HARASSMENT: types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

safety_settings_list = [
    types.SafetySetting(category=category, threshold=threshold)
    for category, threshold in DEFAULT_SAFETY_SETTINGS.items()
]

# Configuration for Gemini API calls
classify_instruc = """You are a highly accurate text classification model.
Determine which single label (from the set: INSURANCE_SERVICE, INSURANCE_PRODUCT,
CONTINUE CONVERSATION, MORE, OFF-TOPIC) best fits this scenario, based on the User Query and the Conversation History.

Definitions and guidelines:

1. CONTINUE CONVERSATION
   - The user is clearly asking a follow-up question.
   - Or user references details that were already mentioned in the conversation history.

2. INSURANCE_SERVICE
   - Specifically about insurance services such as "ติดต่อสอบถาม", "เอกสาร" , "โปรโมชั่น", "กรอบระยะเวลาสำหรับการให้บริการ","ประกันกลุ่ม","ตรวจสอบผู้ขายประกัน","ดาวน์โหลดแบบฟอร์มต่างๆ","ค้นหาโรงพยาบาลคู่สัญญา","ค้นหาสาขา","บริการพิเศษ","บริการเรียกร้องสินไหมทดแทน","บริการด้านการพิจารณารับประกัน","บริการผู้ถือกรมธรรม์","บริการรับเรื่องร้องเรียน","ข้อแนะนำในการแจ้งอุบัติเหตุ","บริการตัวแทน - นายหน้า", etc.

3. INSURANCE_PRODUCT
   - The user wants to buy, see, or compare insurance products such as life insurance or auto insurance policies.

4. MORE
   - The user specifically asks for additional products or variations beyond what was previously discussed.

5. OFF-TOPIC
   - Anything not covered above, or the user’s query is irrelevant to insurance.

Return ONLY one label. Do not add explanations.
"""
generation_config_classify = types.GenerateContentConfig(
    temperature=0.3,
    max_output_tokens=20, # Slightly more buffer for classification labels
    system_instruction=classify_instruc
)

answer_instruc = ("You are 'Subsin', a helpful and professional male insurance assistant for Thai Group Holdings Public Company Limited, "
        "covering two business units: 1) ประกันชีวิต SE Life (อาคเนย์ประกันชีวิต) and 2) ประกันภัย INSURE (อินทรประกันภัย).\n\n"
        "### Guidelines ###\n"
        "- ONLY use information from the provided 'Context','Conversation History' and 'User Question' when answering. Do not use outside knowledge.\n"
        "- Always address all important points from the context if they relate to the question.\n"
        "- If the user question is outside the provided context or no provided context or user question is not related to insurance product/service, respond briefly (≤ 30 tokens) and politely indicate you are unsure or request clarification.\n"
        "- If the user’s question is in Thai, respond in Thai (unless referencing specific names, products, or URLs that require English).\n"
        "- Keep responses clear and concise. Do not exceed 680 tokens.\n"
        "- Never make up information or speculate.\n"
        "### End Guidelines ###\n\n"
        "Now, answer the User Question based on the Conversation History and the provided Context.")
generation_config_answer = types.GenerateContentConfig(
    temperature=0.7,
    max_output_tokens=700,
    system_instruction=answer_instruc,
    safety_settings=safety_settings_list
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

# # Rate limiter for Typhoon chat
# class RateLimiter:
#     def __init__(self, max_calls: int, period: float):
#         self.max_calls = max_calls
#         self.period = period
#         self.lock = threading.Lock()
#         self.calls = []  # timestamps

#     def acquire(self):
#         with self.lock:
#             now = time.time()
#             # drop stale entries
#             self.calls = [t for t in self.calls if t > now - self.period]
#             if len(self.calls) >= self.max_calls:
#                 to_wait = self.period - (now - self.calls[0])
#                 time.sleep(to_wait)
#             self.calls.append(time.time())

# # enforce both per-second and per-minute limits
# chat_limiter_sec = RateLimiter(5, 1)
# chat_limiter_min = RateLimiter(200, 60)
         

def embed_text(text: str):
    from utils.cache import get_memcache   
    mc_client = get_memcache()
    normalized = text.replace("\n", " ").strip()
    key = "embed:" + hashlib.md5(normalized.encode("utf-8")).hexdigest()
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
    from utils.cache import get_memcache   
    mc_client = get_memcache()
    normalized = query.strip()
    key = f"search:{'svc' if service else 'prd'}:"+hashlib.md5(normalized.encode("utf-8")).hexdigest()+f"|{top_k}|{skip_k}"
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
        model=summary_model,
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
        model=summary_model,
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

    prompt_content = f"""
User Query: {user_query}
Conversation History: {chat_history if chat_history else 'None'}
"""
# apply rate limits
    response = client_gemini.models.generate_content(
            model ='gemini-2.5-flash-preview-05-20',
            contents = prompt_content,
            config=generation_config_classify
        )
    if response.prompt_feedback and response.prompt_feedback.block_reason:
        print(f"Warning: Prompt was blocked in decide_search_path. Reason: {response.prompt_feedback.block_reason}")
        return "OFF-TOPIC" # Or handle as appropriate

    if not response.candidates or not response.text: # Check if .text is None or empty
        print("Warning: No text returned from Gemini in decide_search_path.")
        # Decide a default path or handle the error appropriately
        return "OFF-TOPIC" 
    raw_response = response.text.strip()
    path_decision = raw_response.strip().upper()
    return path_decision if path_decision in ["INSURANCE_SERVICE","INSURANCE_PRODUCT","CONTINUE CONVERSATION","MORE","OFF-TOPIC"] else "OFF-TOPIC"


def generate_answer(query, context, chat_history=None):
    gemini_prompt_parts = []
    if chat_history:
        gemini_prompt_parts.append(f"Conversation History:\n{chat_history}\n")

    gemini_prompt_parts.append(f"Context:\n{context if context else 'No specific context provided.'}\n")
    gemini_prompt_parts.append(f"User Question:\n{query}")

    full_prompt_for_gemini = "\n".join(gemini_prompt_parts)
    try:
        response = client_gemini.models.generate_content(
            model ='gemini-2.5-flash-preview-05-20',
            contents = full_prompt_for_gemini,
            config=generation_config_answer
        )

        # --- Best Practice: Check for prompt blocking ---
        if response.prompt_feedback.block_reason:
            raw_response = "ฉันขออภัย แต่ฉันไม่สามารถดำเนินการตามคำขอดังกล่าวได้เนื่องจากข้อจำกัดด้านเนื้อหา (I'm sorry, but I couldn't process that request due to content restrictions.)"
            return raw_response

        candidate = response.candidates[0]
        
        if candidate.safety_ratings:
            for rating in candidate.safety_ratings:
                # Assuming you want to block if probability is MEDIUM or HIGH
                if rating.probability >= types.HarmProbability.MEDIUM: 
                     raw_response = "ฉันขออภัย ฉันไม่สามารถให้คำตอบได้เนื่องจากหลักเกณฑ์ความปลอดภัยของเนื้อหา (I'm sorry, I cannot provide an answer to that due to content safety guidelines.)"
                     return raw_response
                
        raw_response = response.text.strip()


    except Exception as e:
        print(f"Error during Gemini answer generation API call: {e}")
        # raw_response remains the default error message

    return raw_response

import os
import pickle
# import threading
# import time
# from functools import lru_cache
from dotenv import load_dotenv
# import joblib
# from sentence_transformers import SentenceTransformer
from azure.search.documents.models import VectorizedQuery
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
from google.genai import types

from utils.clients import get_search_client, get_service_search_client,get_openai,get_gemini

# inside the function: 
client_gemini = get_gemini()
client = get_openai()
search_client = get_search_client()
service_search_client = get_service_search_client()

load_dotenv()

embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")
# chat_model = os.getenv("TYPHOON_CHAT_MODEL")
# classify_model = os.getenv("OPENAI_CHAT_MODEL")
chat_model = os.getenv("OPENAI_CHAT_MODEL")
summary_model = os.getenv("OPENAI_CHAT_MODEL")
openai_api = os.getenv("OPENAI_API_KEY")
# typhoon_api = os.getenv("TYPHOON_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")


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
classify_instruc = """You are a precise text classification model. Your primary task is to assign a single, most appropriate label to the user's query. **Crucially, you MUST consider the 'Conversation History' to understand the context.**

**TASK:**
Analyze the 'User Query' in the context of the 'Conversation History' and select ONE label from the following options:
- INSURANCE_SERVICE
- INSURANCE_PRODUCT
- CONTINUE CONVERSATION
- MORE
- OFF-TOPIC

**LABEL DEFINITIONS & GUIDELINES:**

1.  **CONTINUE CONVERSATION:**
    * The user is asking a direct follow-up question related to an item or topic explicitly mentioned or presented in the *immediately preceding turns* of the 'Conversation History'.
    * This includes instances where the bot has presented multiple options (e.g., "Product X", "Service Y", "Plan A", "Plan B").
    * **If the 'User Query' refers to these previously mentioned items using deictic terms ('อันนี้', 'อันนั้น', 'ตัวนี้', 'ตัวนั้น' - this one, that one) OR by their order of presentation (e.g., 'อันแรก', 'อันที่สอง', 'the first one', 'the second option'), and then asks a specific question ABOUT them (e.g., for details, price, comparison), label as CONTINUE CONVERSATION.**
        * Example (Deictic):
            * Bot: "We offer Plan A and Plan B."
            * User: "อันนี้มีรายละเอียดอะไรบ้าง" (What are the details for this one?) -> CONTINUE CONVERSATION
        * Example (Order/Comparison):
            * Bot: "Here are insurance options: Option Alpha (ประกันอัลฟ่า) and Option Beta (ประกันเบต้า)."
            * User: "ช่วยเปรียบเทียบ ประกันอันแรกกับอันที่สอง แบบสั้นๆหน่อย" (Please briefly compare the first and second one.) -> CONTINUE CONVERSATION
        * Example (Specific Question about ordered item):
            * Bot: "We have Product Shield and Product Guard."
            * User: "ประกันอันแรกคุ้มครองอะไรบ้าง" (What does the first one cover?) -> CONTINUE CONVERSATION
    * If the assistant just provided details about "Product A", and the user asks "What is the premium for Product A?", this is CONTINUE CONVERSATION.

2.  **INSURANCE_SERVICE:**
    * The query is specifically about insurance-related services, processes, or support.
    * Keywords (Thai): "ติดต่อสอบถาม", "เอกสาร", "โปรโมชั่น", "กรอบระยะเวลาสำหรับการให้บริการ", "ประกันกลุ่ม", "ตรวจสอบผู้ขายประกัน", "ดาวน์โหลดแบบฟอร์ม", "ค้นหาโรงพยาบาลคู่สัญญา", "ค้นหาสาขา", "บริการพิเศษ", "บริการเรียกร้องสินไหมทดแทน", "บริการด้านการพิจารณารับประกัน", "บริการผู้ถือกรมธรรม์", "บริการรับเรื่องร้องเรียน", "ข้อแนะนำในการแจ้งอุบัติเหตุ", "บริการตัวแทน - นายหน้า".
    * Examples (English): "How do I file a claim?", "Where can I find policy documents?", "What are the current promotions?"

3.  **INSURANCE_PRODUCT:**
    * The query indicates interest in acquiring, viewing details of, or comparing specific insurance products (e.g., life insurance, auto insurance, health policies).
    * Examples: "Tell me about your car insurance options.", "I want to buy travel insurance.", "Compare life insurance plans."

4.  **MORE:**
    * The user explicitly asks for additional products, services, or variations beyond what has already been presented or discussed in the 'Conversation History'.
    * This implies a previous interaction where options were given.
    * Examples: "Show me other similar products.", "Are there any other services related to this?", "What else do you have?"

5.  **OFF-TOPIC:**
    * The query is not related to insurance products or services offered by Thai Group Holdings (SE Life and INSURE).
    * The query is nonsensical, abusive, or does not fit any of the above categories.
    * If the query is a generic greeting without specific insurance intent after an initial greeting, it might be OFF-TOPIC unless it's clearly continuing a prior insurance discussion.

**OUTPUT FORMAT:**
Return ONLY the selected label. Do not add any explanations, apologies, or conversational filler.

Example Output:
INSURANCE_PRODUCT
"""
generation_config_classify = types.GenerateContentConfig(
    temperature=0.3,
    #max_output_tokens=50, # Slightly more buffer for classification labels
    system_instruction=classify_instruc
)

answer_instruc = """You are 'Subsin', a helpful, professional, and male AI insurance assistant for Thai Group Holdings Public Company Limited. You represent two business units:
1.  **ประกันชีวิต SE Life (อาคเนย์ประกันชีวิต)**
2.  **ประกันภัย INSURE (อินทรประกันภัย)**

**RESPONSE GUIDELINES:**

1.  **Information Source STRICTLY Limited:**
    * Base your answers SOLELY on the information provided within the 'Retrieved Context', 'Conversation History', and the current 'User Question'.
    * DO NOT use any external knowledge, assumptions, or information not explicitly present in these sources.
    * If 'Retrieved Context' is empty or not provided, state that you lack specific information to answer the question.

2.  **Addressing the User Question:**
    * Thoroughly analyze the 'User Question'.
    * Synthesize information from the 'Retrieved Context' and relevant 'Conversation History' to construct your answer.
    * Ensure all important aspects of the 'User Question' that can be answered by the provided sources are addressed.

3.  **Handling Missing Information or Irrelevant Questions:**
    * **If the 'Retrieved Context' does NOT contain information relevant to the 'User Question':** Respond politely that you do not have the specific information in your knowledge base to answer that question. Do not attempt to answer. You can suggest rephrasing the question or asking about topics you cover.
        * Example (Thai): "ขออภัยครับ ยังไม่มีข้อมูลเกี่ยวกับเรื่อง [topic of user's question] ในขณะนี้ หากคุณลูกค้ามีคำถามเกี่ยวกับผลิตภัณฑ์หรือบริการประกันอื่นๆ สามารถสอบถามได้เลยนะครับบ
        * Example (English): "I apologize, I (Subsin) do not currently have information regarding [topic of user's question]. If you have questions about other insurance products or services, please feel free to ask."
    * **If the 'User Question' is clearly off-topic (not related to insurance products/services of SE Life or INSURE) or nonsensical, even if context is provided:** Politely state that you can only assist with inquiries about insurance products and services from SE Life and INSURE.
        * Example (Thai): "ขออภัยครับ ผมสามารถให้ข้อมูลได้เฉพาะผลิตภัณฑ์และบริการประกันของ SE Life และ INSURE เท่านั้นครับ"
        * Example (English): "I apologize, I (Subsin) can only provide information about insurance products and services from SE Life and INSURE."
    * Avoid speculation or fabricating answers. It is better to state you don't know than to provide incorrect information.

4.  **Language:**
    * If the 'User Question' is in Thai, respond in Thai.
    * Your name ('Subsin'), company names (SE Life, INSURE, Thai Group Holdings), and specific product names, or official terms should be used as appropriate in English. URLs should remain in their original form.
    * If the 'User Question' is in English, respond in English.

5.  **Tone and Style:**
    * Maintain a professional, helpful, and polite male persona.
    * Responses should be clear, concise, and easy to understand.
    * Use formatting like bullet points or numbered lists if it improves readability for complex information, but only if appropriate for a chat interface.

6.  **Conciseness and Length:**
    * Aim for responses that are informative yet brief.
    * Strictly limit your answer to a maximum of 800 tokens. If the necessary information is extensive, provide the most critical parts (should always have URL) and offer to give more details if the user asks.
    * For responses indicating missing information or off-topic queries, keep the response very brief (e.g., under 100 tokens).

7.  **No Hallucination:**
    * NEVER invent facts, product details, policy terms, or any information not found in the provided 'Retrieved Context' or 'Conversation History'.

**STRUCTURE OF YOUR TASK:**
1.  Carefully read the 'User Question', 'Conversation History', and 'Retrieved Context'.
2.  Determine if the 'Retrieved Context' contains relevant information to answer the 'User Question'.
3.  If yes, formulate an answer adhering to all guidelines above.
4.  If no, or if the question is off-topic, respond according to guideline #3.

Now, please answer the 'User Question'.
"""
generation_config_answer = types.GenerateContentConfig(
    temperature=0.5,
    #max_output_tokens=1000,
    system_instruction=answer_instruc,
    safety_settings=safety_settings_list
)

# # Azure AI Search setup
# search_client = SearchClient(
#     endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
#     index_name=os.getenv("AZURE_SEARCH_INDEX"),
#     credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
# )

# service_search_client = SearchClient(
#     endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
#     index_name=os.getenv("AZURE_SEARCH_INDEX_INSURANCE_SERVICE"),
#     credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
# )

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

# @lru_cache(maxsize=1)
# def _load_classifier():
#     model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
#     clf   = joblib.load("decide_path_lr.joblib")
#     return model, clf

def embed_text(text: str):
    from utils.cache import get_memcache   
    mc_client = get_memcache()
    normalized = text.replace("\n", " ").strip()
    normalized = " ".join(normalized.split()).lower()
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
        k_nearest_neighbors=10, 
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
    
    system_prompt = (
        "You are an expert conversation summarizer. Your task is to condense the provided 'Raw Conversation Log' "
        "into a clear, concise, and chronologically accurate summary. This summary will replace the raw log and "
        "will be used as the sole context for future interactions with an AI assistant. "
        "Therefore, it is crucial that the summary preserves all key information, "
        "including specific questions asked by the user, important details or entities mentioned (like product names, "
        "dates, amounts), and the core responses or information provided by the assistant. "
        "Maintain the order of events as they occurred. Ensure the summary reads like a continuous narrative of the conversation so far. "
        "Focus on facts and essential context, omitting pleasantries or redundant phrases if they don't add informational value. "
        "The output should be a single block of text. Do not exceed 1000 tokens."
    )
    
    user_content_for_summarizer = f"Raw Conversation Log:\n{text}"
    response = client.chat.completions.create(
        model=summary_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content_for_summarizer}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    summary = response.choices[0].message.content.strip()
    timestamp = datetime.now(ZoneInfo("Asia/Bangkok"))
    from utils.chat_history_func import save_chat_history,del_chat_history,get_latest_decide
    user_latest_decide = get_latest_decide(user_id)
    del_chat_history(user_id)
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
    # response = client_gemini.models.generate_content(
    #         model ='gemini-2.5-flash-preview-05-20',
    #         contents = prompt_content,
    #         config=generation_config_classify
    #     )

    # if not response.candidates or not response.text: # Check if .text is None or empty
    #     print("Warning: No text returned from Gemini in decide_search_path.")
    #     # Decide a default path or handle the error appropriately
    #     return "OFF-TOPIC" 
    # raw_response = response.text.strip()
    # path_decision = raw_response.strip().upper()
    # return path_decision if path_decision in ["INSURANCE_SERVICE","INSURANCE_PRODUCT","CONTINUE CONVERSATION","MORE","OFF-TOPIC"] else "OFF-TOPIC"

    response = client.chat.completions.create(
        model=chat_model,   # or your desired mini/4.1 model
        messages=[
            {"role": "system", "content": classify_instruc},
            {"role": "user", "content": prompt_content}
        ],
        temperature=0.3
    )
    raw_response = response.choices[0].message.content.strip()
    path_decision = raw_response.strip().upper()
    return path_decision if path_decision in ["INSURANCE_SERVICE","INSURANCE_PRODUCT","CONTINUE CONVERSATION","MORE","OFF-TOPIC"] else "OFF-TOPIC"

# def decide_search_path(text: str) -> str:
#     model, clf = _load_classifier()
#     emb        = model.encode([text])
#     return clf.predict(emb)[0]


def generate_answer(query, context, chat_history=None):
    # gemini_prompt_parts = []
    # if chat_history:
    #     gemini_prompt_parts.append(f"Conversation History:\n{chat_history}\n")

    # gemini_prompt_parts.append(f"Context:\n{context if context else 'No specific context provided.'}\n")
    # gemini_prompt_parts.append(f"User Question:\n{query}")

    # full_prompt_for_gemini = "\n".join(gemini_prompt_parts)
    # try:
    #     response = client_gemini.models.generate_content(
    #         model ='gemini-2.5-flash-preview-05-20',
    #         contents = full_prompt_for_gemini,
    #         config=generation_config_answer
    #     )

    #     # --- Best Practice: Check for prompt blocking ---
    #     if response.prompt_feedback and response.prompt_feedback.block_reason:
    #         print("Candidate blocked with reason specified.")
    #         raw_response = "ฉันขออภัย แต่ฉันไม่สามารถดำเนินการตามคำขอดังกล่าวได้เนื่องจากข้อจำกัดด้านเนื้อหา (I'm sorry, but I couldn't process that request due to content restrictions.)"
    #         return raw_response
            
    #     candidate = response.candidates[0]


    #     if candidate.safety_ratings:
    #         for rating in candidate.safety_ratings:
    #             # Assuming you want to block if probability is MEDIUM or HIGH
    #             if rating.probability >= types.HarmProbability.MEDIUM:
    #                 print(f"Candidate blocked due to safety rating: {rating.category} - {rating.probability}")
    #                 raw_response = "ฉันขออภัย ฉันไม่สามารถให้คำตอบได้เนื่องจากหลักเกณฑ์ความปลอดภัยของเนื้อหา (I'm sorry, I cannot provide an answer to that due to content safety guidelines.)"
    #                 return raw_response
                
    #     return response.text.strip()


    # except Exception as e:
    #     print(f"Error Type: {type(e)}")
    #     print(f"Error Message: {e}")
    #     raw_response = "ฉันขออภัย ฉันไม่สามารถให้คำตอบได้ในขณะนี้ โปรดลองอีกครั้ง"
    #     return raw_response
        # raw_response remains the default error message

    prompt_parts = []
    if chat_history:
        prompt_parts.append(f"Conversation History:\n{chat_history}\n")
    prompt_parts.append(f"Context:\n{context if context else 'No specific context provided.'}\n")
    prompt_parts.append(f"User Question:\n{query}")

    full_prompt_for_chatgpt = "\n".join(prompt_parts)
    try:
        response = client.chat.completions.create(
            model=chat_model,  # or your desired mini/4.1 model
            messages=[
                {"role": "system", "content": answer_instruc},
                {"role": "user", "content": full_prompt_for_chatgpt}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error Type: {type(e)}")
        print(f"Error Message: {e}")
        return "ฉันขออภัย ฉันไม่สามารถให้คำตอบได้ในขณะนี้ โปรดลองอีกครั้ง"


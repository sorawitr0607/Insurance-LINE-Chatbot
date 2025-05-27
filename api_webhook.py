import os
import time
import threading
import pickle
import logging
from dotenv import load_dotenv
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi,
    ReplyMessageRequest, TextMessage,
    QuickReply, QuickReplyItem, MessageAction
)
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from datetime import datetime
from zoneinfo import ZoneInfo


# import sys
# sys.path.append(r"D:\RAG\AZURE\New Deploy (2)\LINE_RAG_API-main (2)\LINE_RAG_API-main")

from utils.chat_history_func import (
    get_conversation_state, del_chat_history, save_chat_history
)
from utils.rag_func import (
    decide_search_path, generate_answer, summarize_context, get_search_results
)

from utils.cache import get_memcache   

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_webhook")

# LINE API
app = Flask(__name__)
configuration = Configuration(access_token=os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))

    
mc_client = get_memcache()                 

MESSAGE_WINDOW = 2

FAQ_CACHED_ANSWERS = {
    "ศูนย์ดูแลลูกค้า": " Se Life : 02-255-5656 \n IN-SURE : 02-636-5656 \n เวลาทำการ : จันทร์ - ศุกร์ 08.30 - 17.00 น",
    "โปรโมชั่น SE Life" : "สิทธิพิเศษสำหรับลูกค้า SE Life วัคซีนป้องกันโรค จากโรงพยาบาลพญาไท 2 ราคาพิเศษ 690 บาท (จากปกติ 2,082 บาท) \n ดูข้อมูลเพิ่มเติมได้ที่ https://www.southeastlife.co.th/promotion/detail/vaccine-phyathai2",
    "โปรโมชั่น IN-SURE" : "สิทธิพิเศษสำหรับลูกค้า IN-SURE อินทรประกันภัย รับฟรีเครื่องดื่มพันธุ์ไทย 1 แก้ว เมื่อแจ้งเคลมประกันรถยนต์ (เคลมแห้ง) สะดวกทุกที่ ทุกเวลา ผ่านไลน์ THAI GROUP \n ดูข้อมูลเพิ่มเติมได้ที่ https://www.indara.co.th/promotion/panthai-insurance-claim-line",
    "Line Thai Group" : "ที่เดียวจบ ครบทุกบริการของอาคเนย์ประกันชีวิต เช่น ดูข้อมูลประกัน แก้ไขข้อมูลกรมธรรม์ หรือ แจ้งเคลมประกัน \n เป็นเพื่อนกับ Thai Group ได้เลยที่นี่ https://lin.ee/OGWXtpN "
}
   
FAQ_BUTTON_META = {
    "ศูนย์ดูแลลูกค้า": "https://raw.githubusercontent.com/sorawitr0607/LINE_RAG_API/main/icon_pic/customer_service.png",
    "โปรโมชั่น SE Life" : "https://raw.githubusercontent.com/sorawitr0607/LINE_RAG_API/main/icon_pic/selife_icon.png",
    "โปรโมชั่น IN-SURE" : "https://raw.githubusercontent.com/sorawitr0607/LINE_RAG_API/main/icon_pic/insure_icon.png",
    "Line Thai Group" : "https://raw.githubusercontent.com/sorawitr0607/LINE_RAG_API/main/icon_pic/line_icon.png"
}
    
FAQ_QUICK_REPLY = QuickReply(
    items = [
        QuickReplyItem(
            image_url=url,
            action=MessageAction(label=label[:20], text=label)  # label ≤ 20 chars for LINE UI
        )
        for label,url in FAQ_BUTTON_META.items()
    ])

def safe_reply(
    line_bot_api: MessagingApi,
    reply_token: str,
    messages: list[TextMessage],
    *,
    fallback_text: str = "ขออภัย ระบบขัดข้อง กรุณาลองใหม่อีกครั้งนะครับ 🙏"
):

    try:
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(reply_token=reply_token, messages=messages)
        )
    except Exception as exc:
        logger.exception("LINE reply failed", exc_info=exc)
        try:
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    reply_token=reply_token,
                    messages=[TextMessage(text=fallback_text)]
                )
            )
        except Exception:
            logger.error("Fallback LINE reply also failed.", exc_info=True)
            
def send_loading_indicator(user_id, duration=25):
    import requests
    headers = {
        'Authorization': f'Bearer {os.getenv("LINE_CHANNEL_ACCESS_TOKEN")}',
        'Content-Type': 'application/json'
    }
    data = {
        "chatId": user_id,
        "loadingSeconds": duration
    }
    requests.post('https://api.line.me/v2/bot/chat/loading/start', headers=headers, json=data)

def process_message_batch(user_id):

    time.sleep(MESSAGE_WINDOW)
    cache_key = f"message_buffer:{user_id}"
    cached_data = mc_client.get(cache_key)

    if not cached_data:
        return

    buffer_data = pickle.loads(cached_data)
    user_query = " ".join(buffer_data['messages'])
    reply_token = buffer_data['reply_token']
    
    if user_query in FAQ_CACHED_ANSWERS:
        response = FAQ_CACHED_ANSWERS[user_query]
        path_decision = 'OFF-TOPIC'
    else:
        chat_history,latest_decide, chat_user_latest = get_conversation_state(user_id)
        #print(len(chat_history))
        path_decision = decide_search_path(user_query,chat_history)
        
        # print(path_decision)
        
        
        if path_decision == "INSURANCE_SERVICE":
            context = get_search_results(query=user_query, top_k=3, skip_k=0, service = True)
        elif path_decision == "INSURANCE_PRODUCT":
            context = get_search_results(query=user_query, top_k=7, skip_k=0, service = False)
        elif path_decision == "CONTINUE CONVERSATION":
            summary_context_search = summarize_context(user_query,chat_user_latest)
            if latest_decide == "INSURANCE_SERVICE":
                context = get_search_results(query=summary_context_search, top_k=3, skip_k=0, service = True)
                path_decision = 'INSURANCE_SERVICE'
            else:
                context = get_search_results(query=summary_context_search, top_k=7, skip_k=0, service = False)
                path_decision = 'INSURANCE_PRODUCT'
        elif path_decision == "MORE":
            context = get_search_results(query=user_query, top_k=7, skip_k=7, service = False)
        else:
            chat_history = None
            context = ""
        
        response = generate_answer(user_query, context, chat_history)

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        safe_reply(                                         
            line_bot_api,
            reply_token,
            [TextMessage(text=response, quickReply=FAQ_QUICK_REPLY)]
        )

    timestamp = datetime.now(ZoneInfo("Asia/Bangkok"))
    save_chat_history(user_id, "user", user_query, timestamp,path_decision)
    timestamp = datetime.now(ZoneInfo("Asia/Bangkok"))
    save_chat_history(user_id, "assistant", response, timestamp,path_decision)
    
    # Clear buffer
    mc_client.delete(cache_key)

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_id = event.source.user_id
    message_text = event.message.text
    reply_token = event.reply_token
    
    if message_text == "CHAT RESET":
        del_chat_history(user_id)
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            safe_reply(                                     
                line_bot_api,
                reply_token,
                [TextMessage(text="แชทของคุณถูกรีเซ็ตเรียบร้อยแล้ว")]
            )
        return
    
    threading.Thread(target=send_loading_indicator, args=(user_id,25)).start()
    
    cache_key = f"message_buffer:{user_id}"
    cached_data = mc_client.get(cache_key)
    
    if cached_data:
       buffer_data = pickle.loads(cached_data)
       buffer_data['messages'].append(message_text)
       buffer_data['reply_token'] = reply_token  # latest reply token
    else:
       buffer_data = {
           'messages': [message_text],
           'reply_token': reply_token
       }
       threading.Thread(target=process_message_batch, args=(user_id,), daemon=True).start()
       
    mc_client.set(cache_key, pickle.dumps(buffer_data), MESSAGE_WINDOW+3)
    
    
@app.route("/webhook", methods=['POST'])
def webhook():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.info("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8000)
    # app.run(debug=True,use_reloader=False)
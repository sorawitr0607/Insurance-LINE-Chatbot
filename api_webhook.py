import os
from dotenv import load_dotenv
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import Configuration,ApiClient,MessagingApi,ReplyMessageRequest,TextMessage
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent,TextMessageContent
from datetime import datetime
from zoneinfo import ZoneInfo

# import sys
# sys.path.append(r"D:\RAG\AZURE\New Deploy (2)\LINE_RAG_API-main (2)\LINE_RAG_API-main")

from utils.chat_history_func import get_chat_history,del_chat_history,save_chat_history,get_latest_decide,get_latest_user_history
from utils.rag_func import decide_search_path,generate_answer,summarize_context,get_search_results #,retrieve_insurance_service_context,retrieve_context

load_dotenv()

# LINE API
app = Flask(__name__)
configuration = Configuration(access_token=os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))

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

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_query = event.message.text
    user_id = event.source.user_id
    chat_history = get_chat_history(user_id)
    #print(len(chat_history))
    path_decision = decide_search_path(user_query,chat_history)
    
    # print(path_decision)
    
    if path_decision == "RESET":
        del_chat_history(user_id)
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
        # context = retrieve_insurance_service_context(user_query)
        context = get_search_results(query=user_query, top_k=3, skip_k=0, service = True)
    elif path_decision == "INSURANCE_PRODUCT":
        # context = retrieve_context(user_query)
        context = get_search_results(query=user_query, top_k=7, skip_k=0, service = False)
    elif path_decision == "CONTINUE CONVERSATION":
        latest_decide = get_latest_decide(user_id)
        chat_user_latest = get_latest_user_history(user_id)
        summary_context_search = summarize_context(user_query,chat_user_latest)
        if latest_decide == "INSURANCE_SERVICE":
            # context = retrieve_insurance_service_context(summary_context_search)
            context = get_search_results(query=summary_context_search, top_k=3, skip_k=0, service = True)
            path_decision = 'INSURANCE_SERVICE'
        else:
            # context = retrieve_context(summary_context_search)
            context = get_search_results(query=summary_context_search, top_k=7, skip_k=0, service = False)
            path_decision = 'INSURANCE_PRODUCT'
    elif path_decision == "MORE":
        # context = retrieve_context(user_query,10)
        context = get_search_results(query=user_query, top_k=7, skip_k=7, service = False)
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

    timestamp = datetime.now(ZoneInfo("Asia/Bangkok"))
    save_chat_history(user_id, "user", user_query, timestamp,path_decision)
    timestamp = datetime.now(ZoneInfo("Asia/Bangkok"))
    save_chat_history(user_id, "assistant", response, timestamp,path_decision)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
    # app.run(debug=True,use_reloader=False)
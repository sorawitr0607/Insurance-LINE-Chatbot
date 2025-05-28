import sys
sys.path.append(r"D:\RAG\AZURE\New Deploy (3)\LINE_RAG_API")
import os, time, asyncio, pickle, logging
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Header, HTTPException, Body # Changed from Flask
# from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    QuickReply, QuickReplyItem, MessageAction, TextMessage, ReplyMessageRequest
)



from utils.clients import get_line_api                    # ← new
from utils.cache import get_memcache
from utils.chat_history_func import (
    get_conversation_state, del_chat_history, save_chat_history
)
from utils.rag_func import (
    decide_search_path, generate_answer, summarize_context, get_search_results
)


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_webhook")


# FastAPI App Initialization
app = FastAPI() # Changed from Flask
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))

_EXEC = ThreadPoolExecutor(max_workers=2)    
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


# Store the main event loop instance
main_event_loop = None

@app.on_event("startup")
async def startup_event():
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()
    logger.info("Main event loop captured on startup.")
    
    
async def _to_thread(fn, *args, **kwargs):
    """Run blocking function in the shared thread-pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_EXEC, partial(fn, *args, **kwargs))

async def _send_loading_indicator(user_id: str, seconds: int = 20) -> None:
    url = "https://api.line.me/v2/bot/chat/loading/start"
    headers = {
        "Authorization": f"Bearer {os.getenv('LINE_CHANNEL_ACCESS_TOKEN')}",
        "Content-Type": "application/json"
    }
    payload = {"chatId": user_id, "loadingSeconds": seconds}
    async with httpx.AsyncClient(http2=True, timeout=5) as client:
        await client.post(url, json=payload, headers=headers)


async def _run_rag_pipeline(user_id: str, buffer_data: dict[str, any]) -> tuple[str, str]:
    reply_token = None
    try:
        logger.info(f"[{user_id}] Starting RAG pipeline. Buffer: {buffer_data}")
        user_query = " ".join(buffer_data["messages"])
        reply_token = buffer_data["reply_token"] # Assign here
        logger.info(f"[{user_id}] User query: '{user_query}', Reply token: {reply_token}")
        
        if user_query in FAQ_CACHED_ANSWERS:
            answer = FAQ_CACHED_ANSWERS[user_query]
            path_decision = 'OFF-TOPIC'
            line_api = get_line_api()
            logger.info(f"[{user_id}] Attempting to send RAG answer via LINE API.")
            await _to_thread(
                line_api.reply_message_with_http_info,
                ReplyMessageRequest(
                    reply_token=reply_token,
                    messages=[TextMessage(text=answer, quickReply=FAQ_QUICK_REPLY)],
                ),
            )
            logger.info(f"[{user_id}] Successfully sent RAG answer.")

            timestamp = datetime.now(ZoneInfo("Asia/Bangkok"))
            await _to_thread(save_chat_history, user_id, "user", user_query, timestamp, path_decision)
            timestamp = datetime.now(ZoneInfo("Asia/Bangkok"))
            await _to_thread(save_chat_history, user_id, "assistant", answer, timestamp, path_decision) 
            return answer,path_decision

        chat_hist, latest_decision, latest_user = await _to_thread(
            get_conversation_state, user_id
        )
        logger.info(f"[{user_id}] Conversation state retrieved. Latest decision: {latest_decision}")

        path_decision = await _to_thread(decide_search_path, user_query)
        logger.info(f"[{user_id}] Path decision: {path_decision}")

        context = "" # Initialize context
        if path_decision == "INSURANCE_SERVICE":
            context = await _to_thread(get_search_results, user_query, 3, 0, True)
        elif path_decision == "INSURANCE_PRODUCT":
            context = await _to_thread(get_search_results, user_query, 7, 0, False)
        elif path_decision == "CONTINUE CONVERSATION":
            summary_ctx = await _to_thread(summarize_context, user_query, latest_user)
            service_flag = latest_decision == "INSURANCE_SERVICE"
            context = await _to_thread(
                get_search_results,
                summary_ctx,
                3 if service_flag else 7,
                0,
                service_flag,
            )
            path_decision = "INSURANCE_SERVICE" if service_flag else "INSURANCE_PRODUCT" # Update path_decision if it was "CONTINUE"
        elif path_decision == "MORE":
            context = await _to_thread(get_search_results, user_query, 7, 7, False)
        else:  # OFF-TOPIC
            logger.info(f"[{user_id}] Path decision is OFF-TOPIC or unrecognized. No context will be fetched for RAG.")
            context, chat_hist = "", None
        
        logger.info(f"[{user_id}] Context for RAG (length: {len(context)}): '{context[:200]}...'")


        answer = await _to_thread(generate_answer, user_query, context, chat_hist)
        

        line_api = get_line_api()
        logger.info(f"[{user_id}] Attempting to send RAG answer via LINE API.")
        await _to_thread(
            line_api.reply_message_with_http_info,
            ReplyMessageRequest(
                reply_token=reply_token,
                messages=[TextMessage(text=answer, quickReply=FAQ_QUICK_REPLY)],
            ),
        )
        logger.info(f"[{user_id}] Successfully sent RAG answer.")

        timestamp = datetime.now(ZoneInfo("Asia/Bangkok"))
        await _to_thread(save_chat_history, user_id, "user", user_query, timestamp, path_decision)
        timestamp = datetime.now(ZoneInfo("Asia/Bangkok"))
        await _to_thread(save_chat_history, user_id, "assistant", answer, timestamp, path_decision) 
        
        return answer, path_decision
    
    except Exception as e:
        logger.error(f"[{user_id}] Error in _run_rag_pipeline: {e}", exc_info=True)

async def process_message_batch(user_id: str) -> None:
    buffer_data = None # Initialize buffer_data
    try:
        logger.info(f"[{user_id}] process_message_batch: Waiting for MESSAGE_WINDOW ({MESSAGE_WINDOW}s).")
        await asyncio.sleep(MESSAGE_WINDOW)

        cache_key = f"message_buffer:{user_id}"
        cached_data = mc_client.get(cache_key)
        
        if not cached_data:
            logger.info(f"[{user_id}] process_message_batch: No cached data found after wait. Exiting.")
            return

        buffer_data = pickle.loads(cached_data)
        logger.info(f"[{user_id}] process_message_batch: Buffer data retrieved. Calling _run_rag_pipeline.")
        
        await _run_rag_pipeline(user_id, buffer_data) # Correctly awaiting the async function
        
        mc_client.delete(cache_key) # Delete cache only after successful processing
        logger.info(f"[{user_id}] process_message_batch: Successfully processed and deleted cache.")
    except Exception as e:
        logger.error(f"[{user_id}] Error in process_message_batch: {e}", exc_info=True)

# @handler.add(MessageEvent, message=TextMessageContent)
async def _async_handle_message_logic(event: MessageEvent):
    user_id = event.source.user_id
    message_text = event.message.text
    reply_token = event.reply_token
    
    logger.info(f"Received message: '{message_text}' from user: {user_id}") # Added logging
    
    if message_text == "CHAT RESET":
        await _to_thread(del_chat_history, user_id) #
        line_api = get_line_api()
        await _to_thread(
            line_api.reply_message_with_http_info,
            ReplyMessageRequest(
                reply_token=reply_token,
                messages=[TextMessage(text="แชทของคุณถูกรีเซ็ตเรียบร้อยแล้ว")],
            ),
        )
        return
    
    # 1) show “typing…” indicator (non-blocking)
    asyncio.create_task(_send_loading_indicator(user_id, 20))

    # 2) append to buffer in Memcached
    cache_key = f"message_buffer:{user_id}"
    cur       = mc_client.get(cache_key)
    now       = time.time()

    if cur:
        buf = pickle.loads(cur)
        buf["messages"].append(message_text)
        buf["reply_token"] = reply_token
        buf["timestamp"]   = now
    else:                                    # first message in a burst
        buf = {"messages": [message_text],
               "reply_token": reply_token,
               "timestamp":  now}
        # schedule the batch processor once
        asyncio.create_task(process_message_batch(user_id))
    try:
        pickled_buf = pickle.dumps(buf)
        # The expiry is MESSAGE_WINDOW + 3 = 5 seconds
        set_result = mc_client.set(cache_key, pickled_buf, time=MESSAGE_WINDOW + 3) 

        if set_result:
            logger.info(f"[{user_id}] Message buffer successfully SET in cache. Key: {cache_key}. Expires in: {MESSAGE_WINDOW + 3}s. Data size: {len(pickled_buf)} bytes.")
           
        else:
            # This else block might also be hit if set_result is None or other non-True falsy values
            logger.error(f"[{user_id}] FAILED to SET message buffer in cache. mc.set returned: {set_result}. Key: {cache_key}. Check Memcached server status and connection.")
    except Exception as e_set:
        logger.error(f"[{user_id}] EXCEPTION during mc.set for {cache_key}: {e_set}", exc_info=True)

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event: MessageEvent): # This is a synchronous function #
    logger.info(f"Sync Wrapper: Received event for user {event.source.user_id}, text: '{event.message.text}'. Scheduling async handler.") #
    if main_event_loop:
        # Submit the coroutine to be run on the main event loop
        future = asyncio.run_coroutine_threadsafe(_async_handle_message_logic(event), main_event_loop)
        logger.info(f"Sync Wrapper: Task for _async_handle_message_logic scheduled on main event loop for user {event.source.user_id}.")
        # Optionally, you can add a callback to the future if you need to handle results or exceptions
        # future.add_done_callback(lambda f: logger.info(f"Async task completed. Result: {f.result()} Exception: {f.exception()}"))
    else:
        logger.error("Sync Wrapper: Main event loop not available. Cannot schedule async task for user {event.source.user_id}.")
        # Handle this critical error, perhaps by replying with an error message to the user if possible,
        # though replying from here without the event loop is also problematic.
        # This indicates a setup issue.
    
    
@app.post("/webhook")
async def webhook(
    request: Request,
    x_line_signature: str = Header(..., alias="X-Line-Signature")):
    # body: str = Body(..., media_type="application/json"),
    body = await request.body()
    body_str = body.decode("utf-8")

    try:
        await _to_thread(handler.handle, body_str, x_line_signature)
    except InvalidSignatureError:
        logger.error("Invalid LINE signature.") # Added logging
        raise HTTPException(status_code=400, detail="Invalid LINE signature")
    except Exception as exc:
        logger.exception("Unhandled error in webhook")
        raise HTTPException(status_code=500, detail=str(exc))

    return "OK"

# if __name__ == "__main__":
#     #app.run(host="0.0.0.0", port=8000)
#     app.run(debug=True,use_reloader=False)

# uvicorn api_webhook:app --host 0.0.0.0 --port 8000 --reload


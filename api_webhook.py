## Import Library
import os, asyncio, logging
from typing import Any, DefaultDict
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

# API
import httpx
from functools import lru_cache,partial
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Header, HTTPException
from concurrent.futures import ThreadPoolExecutor

# Line Library
from linebot.v3 import WebhookHandler
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    QuickReply, QuickReplyItem, MessageAction, TextMessage, ReplyMessageRequest
)

# Loading Utils Script
from utils.clients import get_line_api                    
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

# Thread Worker
_EXEC = ThreadPoolExecutor(max_workers=int(os.getenv("MAX_WORKERS")))          

# Message Window Duration
MESSAGE_WINDOW = int(os.getenv("MESSAGE_WINDOW_EXP") or "2")

# FAQ Answer
FAQ_CACHED_ANSWERS = {
    "ศูนย์ดูแลลูกค้า": " Se Life : 02-255-5656 \n IN-SURE : 02-636-5656 \n เวลาทำการ : จันทร์ - ศุกร์ 08.30 - 17.00 น",
    "โปรโมชั่น SE Life" : os.getenv("PROMOTION_SELIFE") or "ยังไม่มีโปรโมชั่นสำหรับ SE Life ขณะนี้",
    "โปรโมชั่น IN-SURE" : os.getenv("PROMOTION_INSURE") or "ยังไม่มีโปรโมชั่นสำหรับ IN-SURE ขณะนี้",
    "Line Thai Group" : "ที่เดียวจบ ครบทุกบริการของอาคเนย์ประกันชีวิต เช่น ดูข้อมูลประกัน แก้ไขข้อมูลกรมธรรม์ หรือ แจ้งเคลมประกัน \n เป็นเพื่อนกับ Thai Group ได้เลยที่นี่ https://lin.ee/OGWXtpN "
}

# Icon Image
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
            action=MessageAction(label=label[:20], text=label)  # For LINE UI
        )
        for label,url in FAQ_BUTTON_META.items()
    ])


# Store the main event loop instance
main_event_loop = None
USER_BUFFERS: DefaultDict[str, dict[str, Any]] = defaultdict(lambda: {"messages": [], "reply_token": None, "task": None})

# Event Setup
@app.on_event("startup")
async def startup_event():
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()
    logger.info("Main event loop captured on startup.")
    
async def _to_thread(fn, *args, **kwargs):
    """Run blocking function in the shared thread-pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_EXEC, partial(fn, *args, **kwargs))

@lru_cache(maxsize=1)
def get_async_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        http2=True,
        timeout=httpx.Timeout(5.0),
        transport=httpx.AsyncHTTPTransport(retries=3)  # automatic 3-try back-off
    )

async def close_async_client() -> None:
    """Call this in FastAPI's shutdown event to close keep-alive connections."""
    await get_async_client().aclose()

# Main Function
## Line Loading Effect
async def _send_loading_indicator(user_id: str, seconds: int = 40) -> None:
    url = "https://api.line.me/v2/bot/chat/loading/start"
    headers = {
        "Authorization": f"Bearer {os.getenv('LINE_CHANNEL_ACCESS_TOKEN')}",
        "Content-Type": "application/json"
    }
    payload = {"chatId": user_id, "loadingSeconds": seconds}
    client = get_async_client()                     
    await client.post(url, json=payload, headers=headers)

## Run Main Event
@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event: MessageEvent):
    logger.info(f"Sync Wrapper: Received event for user {event.source.user_id}, text: '{event.message.text}'. Scheduling async handler.") #
    if main_event_loop:
        _ = asyncio.run_coroutine_threadsafe(_async_handle_message_logic(event), main_event_loop)
        logger.info(f"Sync Wrapper: Task for _async_handle_message_logic scheduled on main event loop for user {event.source.user_id}.")

    else:
        logger.error(f"Sync Wrapper: Main event loop not available. Cannot schedule async task for user {event.source.user_id}.")

## Handle Receiving Message
async def _async_handle_message_logic(event: MessageEvent):
    user_id = event.source.user_id
    message_text = event.message.text
    reply_token = event.reply_token
    logger.info(f"Received message: '{message_text}' from user: {user_id}") 
    ### Check 'CHAT_RESET'
    if message_text == "CHAT RESET":
        await _to_thread(del_chat_history, user_id) # chat_history_func
        line_api = get_line_api()                   # clients_func
        await _to_thread(
            line_api.reply_message_with_http_info,
            ReplyMessageRequest(
                reply_token=reply_token,
                messages=[TextMessage(text="แชทของคุณถูกรีเซ็ตเรียบร้อยแล้ว")],
            ),
        )
        USER_BUFFERS.pop(user_id, None)
        return
    buf = USER_BUFFERS[user_id]
    task = buf.get("task")
    if not task or task.done():
        asyncio.create_task(_send_loading_indicator(user_id, 30))

    buf["messages"].append(message_text)
    buf["reply_token"] = reply_token  # keep latest so we can reply

    # Debounce: cancel a pending batch task and reschedule
    if task := buf.get("task"):
        if not task.done():
            task.cancel()
    buf["task"] = asyncio.create_task(process_message_batch(user_id))

## Message Batch
async def process_message_batch(user_id: str) -> None:
    try:
        logger.info(f"[{user_id}] process_message_batch: Waiting for MESSAGE_WINDOW ({MESSAGE_WINDOW}s).")
        await asyncio.sleep(MESSAGE_WINDOW)
        buf = USER_BUFFERS[user_id]
        buffer_data = {"messages": buf["messages"][:], "reply_token": buf["reply_token"]}
        buf["messages"].clear()
        buf["reply_token"] = None
        buf["task"] = None
        await _run_rag_pipeline(user_id, buffer_data)
    except asyncio.CancelledError:
        return
    except Exception as e:
        logger.error(f"[{user_id}] Error in process_message_batch: {e}", exc_info=True)

## RAG Pipeline
async def _run_rag_pipeline(user_id: str, buffer_data: dict[str, Any]) -> tuple[str, str]| None:
    reply_token = None
    try:
        ### Check Empty
        if not buffer_data or not buffer_data.get("messages") or not buffer_data.get("reply_token"):
            logger.warning(f"[{user_id}] Empty buffer_data; aborting pipeline.")
            return None
        
        ### Start
        logger.info(f"[{user_id}] Starting RAG pipeline. Buffer: {buffer_data}")
        user_query = " ".join(buffer_data["messages"])
        reply_token = buffer_data["reply_token"] 
        logger.info(f"[{user_id}] User query: '{user_query}', Reply token: {reply_token}")

        ### Check FAQ
        if user_query in FAQ_CACHED_ANSWERS:
            answer = FAQ_CACHED_ANSWERS[user_query]
            path_decision = 'OFF-TOPIC'
            line_api = get_line_api() # clients_func
            logger.info(f"[{user_id}] Attempting to send RAG answer via LINE API.")
            await _to_thread(
                line_api.reply_message_with_http_info,
                ReplyMessageRequest(
                    reply_token=reply_token,
                    messages=[TextMessage(text=answer, quickReply=FAQ_QUICK_REPLY)],
                ),
            )
            logger.info(f"[{user_id}] Successfully sent RAG answer.")
            ### Save Chat
            timestamp = datetime.now(ZoneInfo("Asia/Bangkok"))
            await _to_thread(save_chat_history, user_id, "user", user_query, timestamp, path_decision) # chat_history_func
            timestamp = datetime.now(ZoneInfo("Asia/Bangkok"))
            await _to_thread(save_chat_history, user_id, "assistant", answer, timestamp, path_decision) # chat_history_func
            return answer,path_decision
        
        ### Chat History
        chat_hist, latest_decision, latest_user = await _to_thread(get_conversation_state, user_id) # chat_history_func
        logger.info(f"[{user_id}] Conversation state retrieved. Latest decision: {latest_decision}")

        ### Decide RAG Source
        classify_fut = _to_thread(decide_search_path, user_query, chat_hist)

        # launch both product+service searches in parallel (small top=3 each is cheap)
        prod_fut    = _to_thread(get_search_results, user_query, 7, 0, False)
        svc_fut     = _to_thread(get_search_results, user_query, 3, 0, True)
        more_fut    = _to_thread(get_search_results, user_query, 7, 7, False)
        path_decision = await classify_fut
        logger.info(f"[{user_id}] Path decision: {path_decision}")

        ### Retrieving Doc
        context = "" # Initialize context
        if path_decision == "INSURANCE_SERVICE":
            context = await svc_fut # rag_func
        elif path_decision == "INSURANCE_PRODUCT":
            context = await prod_fut # rag_func
        elif path_decision == "CONTINUE CONVERSATION":
            if latest_decision!='OFF-TOPIC':
                summary_ctx = await _to_thread(summarize_context, user_query, latest_user) # rag_func
                service_flag = latest_decision == "INSURANCE_SERVICE"
                context = await _to_thread(
                    get_search_results, # rag_func
                    summary_ctx,
                    3 if service_flag else 7,
                    0,
                    service_flag,
                )
                path_decision = "INSURANCE_SERVICE" if service_flag else "INSURANCE_PRODUCT"
            else:
                context = ""
                path_decision = 'OFF-TOPIC'
        elif path_decision == "MORE":
            context = await more_fut # rag_func
        else:
            logger.info(f"[{user_id}] Path decision is OFF-TOPIC or unrecognized. No context will be fetched for RAG.")
            context, chat_hist = "", None
        logger.info(f"[{user_id}] Context for RAG (length: {len(context)}): '{context[:200]}...'")

        ### Answer
        answer = await _to_thread(generate_answer, user_query, context, chat_hist) # rag_func
        
        ### Send Answer API
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

        ### Save Chat
        timestamp = datetime.now(ZoneInfo("Asia/Bangkok"))
        await _to_thread(save_chat_history, user_id, "user", user_query, timestamp, path_decision) 
        timestamp = datetime.now(ZoneInfo("Asia/Bangkok"))
        await _to_thread(save_chat_history, user_id, "assistant", answer, timestamp, path_decision) 
        
        return answer, path_decision
    
    except Exception as e:
        logger.error(f"[{user_id}] Error in _run_rag_pipeline: {e}", exc_info=True)



@app.post("/webhook")
async def webhook(
    request: Request,
    x_line_signature: str = Header(..., alias="X-Line-Signature")):
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

## 
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutdown event triggered. Closing async client and ThreadPoolExecutor.")
    await close_async_client()
    _EXEC.shutdown(wait=True) # Ensure threads complete
    logger.info("ThreadPoolExecutor shut down.")


# uvicorn api_webhook:app --host 0.0.0.0 --port 8000 --reload


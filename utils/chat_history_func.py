import os
from dotenv import load_dotenv
from pymongo import MongoClient


load_dotenv()

# MongoDB
mongo_uri = os.getenv("COSMOS_MONGO_URI")
mongo_db = os.getenv("COSMOS_MONGO_DB")
mongo_table = os.getenv("COSMOS_MONGO_TABLE")
mongo_client = MongoClient(mongo_uri)
db = mongo_client[mongo_db]
conversations = db[mongo_table]


def get_conversation_state(user_id, history_limit=20, summary_max_chars=3500):
    from utils.rag_func import summarize_text

    # 1) Single query for the most recent messages
    msgs = list(conversations.find(
        {"user_id": user_id},
        sort=[("timestamp", -1)],
        limit=history_limit
    ))
    if not msgs:
        return "", None, ""

    # Reverse back to chronological order
    msgs = list(reversed(msgs))

    # 2) Build raw history text for summarization
    history_text = "\n".join(f"{m['sender']}: {m['message']}" for m in msgs)
    summary = summarize_text(history_text, summary_max_chars, user_id)

    # 3) Extract the very last path_decision
    latest_decision = msgs[-1].get("path_decision")

    # 4) Extract the last 2 user messages
    user_msgs = [m["message"] for m in msgs if m["sender"] == "user"]
    # keep up to last 2
    latest_user_history = "\n".join(user_msgs[-2:]) if len(user_msgs) >= 2 else "\n".join(user_msgs)

    return summary, latest_decision, latest_user_history


def save_chat_history(user_id, sender, message, timestamp,path_decision):
    conversations.insert_one({
        "user_id": user_id,
        "sender": sender,
        "message": message,
        "timestamp": timestamp,
        "path_decision" : path_decision
    })
    
def del_chat_history(user_id):
    conversations.delete_many({"user_id": user_id})

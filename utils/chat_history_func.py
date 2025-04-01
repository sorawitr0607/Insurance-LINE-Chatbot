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

def get_chat_history(user_id, limit=20):
    from utils.rag_func import summarize_text
    messages = list(conversations.find(
        {"user_id": user_id},
        sort=[("timestamp", -1)],
        limit=limit
    ))
    messages.reverse()
    history_text = "\n".join([f"{m['sender']}: {m['message']}" for m in messages])
    latest_decide = get_latest_decide(user_id)
    summary = summarize_text(history_text,2500,user_id,latest_decide)
    return summary

def get_latest_decide(user_id, limit=1):
    messages = list(conversations.find(
        {"user_id": user_id},
        sort=[("timestamp", -1)],
        limit=limit
    ))
    messages.reverse()
    latest_decide = "\n".join([f"{m['path_decision']}" for m in messages])
    return latest_decide

def get_latest_user_history(user_id, limit=1):
    messages = list(conversations.find(
        {"user_id": user_id,"sender" : "user"},
        sort=[("timestamp", -1)],
        limit=limit
    ))
    messages.reverse()
    history_text = "\n".join([f"{m['sender']}: {m['message']}" for m in messages])
    return history_text

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

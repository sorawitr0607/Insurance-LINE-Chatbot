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
    summary = summarize_text(history_text,2500,user_id)
    return summary

def save_chat_history(user_id, sender, message, timestamp):
    conversations.insert_one({
        "user_id": user_id,
        "sender": sender,
        "message": message,
        "timestamp": timestamp
    })
    
def del_chat_history(user_id):
    conversations.delete_many({"user_id": user_id})
    
    
    



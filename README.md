# 🧠 LINE RAG API
A retrieval‑augmented chatbot for LINE that helps insurance customers get concise answers from proprietary knowledge bases.


📑 Table of Contents
📄 Description

✨ Key Features

🗂️ Repository Structure

🛠️ Installation

For Users

For Developers

🚀 Usage & Workflow

⚙️ Architecture

🧪 Data‑Science Highlights

🙌 Contributing

👤 Author

📄 Description
LINE_RAG_API is a production‑grade retrieval‑augmented generation (RAG) service that powers a LINE messaging chatbot for an insurance provider. It combines modern FastAPI web services, asynchronous concurrency, vector search over Azure Cognitive Search and large‑language models (OpenAI & Google Gemini) to deliver up‑to‑date answers from internal knowledge bases. Conversations are cached for seamless multi‑turn interactions and stored for analytics.

The pipeline works as follows:

Message buffering. Incoming LINE messages are stored in a memcached buffer for a configurable window. This prevents spamming the API when users type multiple messages quickly. The webhook initialises a thread pool and retrieves a memcache client from utils.cache.get_memcache()
raw.githubusercontent.com
.

Intent classification. When the buffer expires, the concatenated user messages are classified using a small logistic‑regression model or a prompt‑engineered Gemini classifier. The classifier labels queries as INSURANCE_SERVICE, INSURANCE_PRODUCT, CONTINUE CONVERSATION, MORE or OFF‑TOPIC
raw.githubusercontent.com
 with detailed guidelines and examples
raw.githubusercontent.com
.

Vector retrieval. Based on the predicted label, the API issues a vector search against either a service index (3 documents) or a product index (7 documents). The thread‑safe Azure Cognitive Search clients are created on demand in utils/clients.py
raw.githubusercontent.com
.

Answer generation. The retrieved context, conversation history and user question are passed to an LLM (OpenAI/Gemini) with a system prompt that forbids hallucination
raw.githubusercontent.com
. Predefined FAQs are served instantly from a cache
raw.githubusercontent.com
.

Reply and persistence. The answer is sent back through the LINE Messaging API with quick‑reply buttons
raw.githubusercontent.com
. Conversation turns and the chosen path are saved in MongoDB
raw.githubusercontent.com
 for future context and analytics.

✨ Key Features
Retrieval‑augmented generation: Combines document retrieval with LLM‑powered answer synthesis.

Intent classification: Uses both prompt‑engineered classification and a logistic‑regression model to decide the retrieval path.

Asynchronous FastAPI: High throughput via asyncio, a shared thread pool and an HTTPX async client with retry logic
raw.githubusercontent.com
.

Stateful conversations: Memcached buffers group messages; MongoDB stores chat history and path decisions
raw.githubusercontent.com
.

Pluggable clients: Modular functions create Azure Search, OpenAI, Gemini and LINE API clients
raw.githubusercontent.com
.

Multilingual support: Prompts include Thai and English examples for clear classification and responses
raw.githubusercontent.com
.

🗂️ Repository Structure
mermaid
Copy
Edit
graph TD
    A[Project Root] --> B(api_webhook.py)
    A --> C(utils/)
    C --> C1[clients.py]
    C --> C2[cache.py]
    C --> C3[chat_history_func.py]
    C --> C4[rag_func.py]
    A --> D[decide_path_lr.joblib]
    A --> E[requirements.txt]
    A --> F[icon_pic/]
Path	Description
api_webhook.py	Main FastAPI application. Handles the LINE webhook, buffers messages, runs the RAG pipeline and replies via the LINE API. Contains FAQ answers and quick‑reply button definitions
raw.githubusercontent.com
.
utils/clients.py	Factories for Azure Search, OpenAI/Gemini and LINE API clients
raw.githubusercontent.com
.
utils/cache.py	Memcached client using environment variables
raw.githubusercontent.com
.
utils/chat_history_func.py	Retrieves, summarises and persists chat history in MongoDB
raw.githubusercontent.com
.
utils/rag_func.py	Contains classification prompts, response prompts and functions to decide the retrieval path, summarise context, search and generate answers
raw.githubusercontent.com
.
decide_path_lr.joblib	Logistic‑regression model to classify user queries into retrieval paths.
icon_pic/	PNG icons used in quick‑reply buttons.
requirements.txt	Python dependencies.

🛠️ Installation
For Users
bash
Copy
Edit
pip install -r requirements.txt
Create a .env file and set your LINE credentials (LINE_CHANNEL_SECRET, LINE_CHANNEL_ACCESS_TOKEN), Azure Search keys (AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, AZURE_SEARCH_INDEX, AZURE_SEARCH_INDEX_INSURANCE_SERVICE), OpenAI/Gemini API keys, memcached connection parameters and MongoDB connection strings
raw.githubusercontent.com
raw.githubusercontent.com
. Then run:

bash
Copy
Edit
uvicorn api_webhook:app --host 0.0.0.0 --port 8000
Expose /callback to your LINE bot’s webhook URL.

For Developers
bash
Copy
Edit
# Clone the repository
git clone https://github.com/sorawitr0607/LINE_RAG_API.git
cd LINE_RAG_API

# Create & activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Install development tools (e.g., black, ruff) as needed. Adjust environment variables in .env to point to your own services and run tests or benchmarks.

🚀 Usage & Workflow
mermaid
Copy
Edit
graph LR
    A[User Message] --> B[Memcache Buffer]
    B --> C[Intent & Path Decision]
    C --> D[Vector Retrieval]
    D --> E[Answer Generation]
    E --> F[Reply via LINE]
Send a message via LINE. If the message matches a cached FAQ (e.g., “ศูนย์ดูแลลูกค้า” or “โปรโมชั่น SE Life”), an answer is returned immediately from FAQ_CACHED_ANSWERS
raw.githubusercontent.com
.

Intent & path decision – after the message window expires, the concatenated messages are classified using the logistic‑regression model or Gemini prompt
raw.githubusercontent.com
. The label determines whether to search the service index (3 docs), product index (7 docs) or follow‑up context
raw.githubusercontent.com
.

Vector retrieval – queries are vectorised and executed against Azure Cognitive Search using the appropriate client
raw.githubusercontent.com
.

Answer generation – an LLM synthesises an answer using only the retrieved context, conversation history and user question, avoiding hallucination
raw.githubusercontent.com
.

Reply – the answer and quick‑reply buttons are sent via the LINE API
raw.githubusercontent.com
. Messages and path decisions are saved in MongoDB for future context and analytics
raw.githubusercontent.com
.

⚙️ Architecture
Under the hood, LINE_RAG_API uses several cloud services and machine‑learning components:

Memcached for transient message buffers and concurrency control.

MongoDB (Cosmos DB) to persist conversations and path decisions
raw.githubusercontent.com
.

Azure Cognitive Search for vector search; each document is embedded using OpenAI embeddings and stored in indexes.

Google Gemini & OpenAI large‑language models for classification and answer generation. The classification prompt defines five possible labels with Thai/English examples
raw.githubusercontent.com
, while the answer prompt instructs the model to rely solely on retrieved context
raw.githubusercontent.com
.

Logistic regression model trained on labelled data to decide retrieval paths when prompt‑based classification is not used.

FastAPI & HTTPX for asynchronous request handling
raw.githubusercontent.com
.

LINE Messaging API for communication with users; quick replies are built from icons stored in icon_pic/
raw.githubusercontent.com
.

🧪 Data‑Science Highlights
This project demonstrates several skills valuable for data‑science roles:

Prompt engineering & LLM orchestration – carefully designed prompts guide both classification and answer synthesis
raw.githubusercontent.com
.

Vector search & embeddings – retrieval of relevant documents from a semantic index shows practical use of embeddings and vector databases.

Asynchronous programming – concurrency patterns (async/await, thread pools) ensure responsiveness and scalability
raw.githubusercontent.com
.

Machine‑learning integration – a logistic‑regression model complements the prompt‑based classifier, illustrating how classical ML can coexist with LLMs.

Data persistence & caching – persisting conversations and caching answers demonstrate awareness of state management and user experience
raw.githubusercontent.com
.

Modular code design – separation into utils modules promotes reusability and testability
raw.githubusercontent.com
raw.githubusercontent.com
.

🙌 Contributing
Contributions are welcome! Feel free to open issues or pull requests if you:

Want to support other messaging platforms.

Have ideas to improve classification or retrieval.

Find bugs or have suggestions for better prompts.

Please create a fork, make your changes in a feature branch and submit a pull request. See the existing code for examples of modular design and docstrings.

👤 Author
SorawitR – Data scientist & developer passionate about conversational AI and information retrieval. Reach out on GitHub if you have questions or feedback.

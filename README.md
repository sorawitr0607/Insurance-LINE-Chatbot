# 🧠 LINE RAG API

_A retrieval‑augmented chatbot for LINE that helps insurance customers get concise answers from proprietary knowledge bases._

![FastAPI](https://img.shields.io/badge/Made%20with-FastAPI-009688?logo=fastapi&logoColor=white) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python) ![Chatbot](https://img.shields.io/badge/RAG%20Chatbot-LLM%2BSearch-yellowgreen)

## 📑 Table of Contents

- [📄 Description](#-description)
  - [✨ Key Features](#-key-features)
- [🗂️ Repository Structure](#-repository-structure)
- [🛠️ Installation](#-installation)
  - [For Users](#for-users)
  - [For Developers](#for-developers)
- [🚀 Usage & Workflow](#-usage--workflow)
- [⚙️ Architecture](#-architecture)
- [🧪 Data‑Science Highlights](#-data-science-highlights)
- [🙌 Contributing](#-contributing)
- [👤 Author](#-author)

---

## 📄 Description

`LINE_RAG_API` is a production‑grade retrieval‑augmented generation (RAG) service that powers a LINE messaging chatbot for an insurance provider.  It combines modern **FastAPI** web services, **asynchronous concurrency**, **vector search** over Azure Cognitive Search and **large‑language models** (OpenAI & Google Gemini) to deliver up‑to‑date answers from internal knowledge bases.  Conversations are cached for seamless multi‑turn interactions and stored for analytics.

The pipeline works as follows:

1. **Message buffering.**  Incoming LINE messages are stored in a **memcached** buffer for a configurable window.  This prevents spamming the API when users type multiple messages quickly.  The webhook initialises a thread pool and retrieves a memcache client from `utils.cache.get_memcache()`:contentReference[oaicite:0]{index=0}.
2. **Intent classification.**  When the buffer expires, the concatenated user messages are classified using a small logistic‑regression model or a prompt‑engineered Gemini classifier.  The classifier labels queries as `INSURANCE_SERVICE`, `INSURANCE_PRODUCT`, `CONTINUE CONVERSATION`, `MORE` or `OFF‑TOPIC`:contentReference[oaicite:1]{index=1} with detailed guidelines and examples:contentReference[oaicite:2]{index=2}.
3. **Vector retrieval.**  Based on the predicted label, the API issues a **vector search** against either a service index (3 documents) or a product index (7 documents).  The thread‑safe Azure Cognitive Search clients are created on demand in `utils/clients.py`:contentReference[oaicite:3]{index=3}.
4. **Answer generation.**  The retrieved context, conversation history and user question are passed to an LLM (OpenAI/Gemini) with a system prompt that forbids hallucination:contentReference[oaicite:4]{index=4}.  Predefined FAQs are served instantly from a cache:contentReference[oaicite:5]{index=5}.
5. **Reply and persistence.**  The answer is sent back through the LINE Messaging API with quick‑reply buttons:contentReference[oaicite:6]{index=6}.  Conversation turns and the chosen path are saved in MongoDB:contentReference[oaicite:7]{index=7} for future context and analytics.

### ✨ Key Features

- **Retrieval‑augmented generation:** Combines document retrieval with LLM‑powered answer synthesis.
- **Intent classification:** Uses both prompt‑engineered classification and a logistic‑regression model to decide the retrieval path.
- **Asynchronous FastAPI:** High throughput via `asyncio`, a shared thread pool and an HTTPX async client with retry logic:contentReference[oaicite:8]{index=8}.
- **Stateful conversations:** Memcached buffers group messages; MongoDB stores chat history and path decisions:contentReference[oaicite:9]{index=9}.
- **Pluggable clients:** Modular functions create Azure Search, OpenAI, Gemini and LINE API clients:contentReference[oaicite:10]{index=10}.
- **Multilingual support:** Prompts include Thai and English examples for clear classification and responses:contentReference[oaicite:11]{index=11}.

---

## 🗂️ Repository Structure

```mermaid
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

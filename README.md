# MEAI: RL-RAG with Knowledge Graph for Drug QA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-MeAI-blue.svg)](https://github.com/tramit-work/MEAI-RL-RAG-with-Knowledge-Graph-for-Drug-QA)

---

## Description

**MeAI** là hệ thống hỏi đáp y khoa thông minh kết hợp:

- Knowledge Graph (KG)
- Retrieval-Augmented Generation (RAG)
- Large Language Model (Llama 3.2)
- Reinforcement Learning (RL)

Hệ thống giúp:
- Tăng độ chính xác câu trả lời y khoa  
- Khai thác tri thức dạng graph  
- Cải thiện chất lượng output bằng RL  

---

**Repository:**  
[https://github.com/tramit-work/MEAI-RL-RAG-with-Knowledge-Graph-for-Drug-QA](https://github.com/tramit-work/MEAI-RL-RAG-with-Knowledge-Graph-for-Drug-QA)

---

## Abstract

Mặc dù các hệ thống **Retrieval-Augmented Generation (RAG)** đã cải thiện đáng kể khả năng của mô hình ngôn ngữ lớn, chúng vẫn tồn tại một số hạn chế:

- Thiếu khả năng khai thác **tri thức có cấu trúc**
- Không có cơ chế **tối ưu câu trả lời sau khi sinh**

Để giải quyết vấn đề này, nghiên cứu đề xuất:

- Xây dựng **Knowledge Graph từ dữ liệu thuốc**
- Thực hiện **graph-based retrieval** để cung cấp context
- Sử dụng **LLM để sinh câu trả lời**
- Áp dụng **Reinforcement Learning** để chọn output tốt nhất

---

## Project Structure
```bash
MeAI-RL-RAG/
│
├── data-kg/                  # Knowledge Graph data (JSON)
│   ├── medicine-one.json
│   ├── ...
│   └── medicine-ten.json
│
├── static/                   # Frontend assets
│   ├── script.js
│   └── style.css
│
├── templates/                # HTML UI
│   └── index.html
│
├── main.py                   # FastAPI server
├── rl_rag_system.py          # Core system (KG + RAG + RL)
└── README.md
```

---

## Installation
### 1. Clone repository
```bash
git clone https://github.com/tramit-work/MEAI-RL-RAG-with-Knowledge-Graph-for-Drug-QA.git
cd MEAI-RL-RAG-with-Knowledge-Graph-for-Drug-QA
```
### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```
### 3. Install dependencies
```bash
pip install fastapi uvicorn transformers torch scikit-learn networkx jinja2
```
## Running the System
```bash
uvicorn main:app --reload
```
Truy cập hệ thống tại:
```bash
http://127.0.0.1:8000
```

---

## Methodology
### 1. Knowledge Graph
- Load từ data-kg/*.json
- Xây bằng networkx
Dạng:
```bash
Entity → Relation → Entity
```
- Sử dụng Neo4j để xây dựng graph
### 2. Retrieval
- Nhận câu hỏi từ người dùng
- Xác định thực thể liên quan
- Truy xuất các node lân cận trong graph
- Tạo context cho mô hình
### 3. Generation (RAG)
Prompt
```bash
Context: (KG data)
Question: (user input)
Answer:
```
Model sử :
```bash
meta-llama/Llama-3.2-3B-Instruct
```
### 4. Reinforcement Learning
- Sinh nhiều câu trả lời (episodes)
- Đánh giá bằng cosine similarity (embedding)
- Lựa chọn câu trả lời tốt nhất

---

## Technologies
| Component | Technology      |
| --------- | --------------- |
| Backend   | FastAPI         |
| LLM       | Llama 3.2       |
| Embedding | RoBERTa         |
| Graph     | Neo4j           |
| RL        | Custom RL loop  |
| Frontend  | HTML + CSS + JS |


## Notes
Model LLM khá nặng → nên dùng GPU

---

## Author
### Nguyễn Ngọc Bảo Trâm
AI Student – Graduation Thesis


















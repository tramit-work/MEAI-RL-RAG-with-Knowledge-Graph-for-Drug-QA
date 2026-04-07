import os
import json
import torch
import numpy as np
import networkx as nx
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

HF_TOKEN = ""   # nếu có token thì thêm

print("Loading LLM...")

generator_model_id = "meta-llama/Llama-3.2-3B-Instruct"

generator_tokenizer = AutoTokenizer.from_pretrained(
    generator_model_id,
    token=HF_TOKEN
)

generator_model = AutoModelForCausalLM.from_pretrained(
    generator_model_id,
    dtype=torch.float16,
    device_map="auto",
    token=HF_TOKEN
)

print("LLM Loaded")

print("Loading Embedding Model...")

embed_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
embed_model = AutoModel.from_pretrained("roberta-base")

print("Embedding Model Loaded")


def load_kg_from_folder(folder_path):
    G = nx.DiGraph()

    if not os.path.exists(folder_path):
        print("KG folder not found")
        return G

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            with open(os.path.join(folder_path, filename), "r") as f:
                data = json.load(f)

            for record in data:
                try:
                    path = record.get("p", {})
                    start = path.get("start", {}).get("properties", {}).get("name")
                    end = path.get("end", {}).get("properties", {}).get("name")
                    rel = path.get("segments", [])[0].get("relationship", {}).get("type", "RELATED_TO")

                    if start and end:
                        G.add_edge(start, end, relation=rel)

                except:
                    continue

    return G


def retrieve_kg_context(entity, G, depth=1):
    context = set()

    if entity not in G:
        return []

    nodes = [entity]

    for _ in range(depth):
        next_nodes = []

        for node in nodes:
            neighbors = list(G.successors(node))

            for nbr in neighbors:
                rel = G.get_edge_data(node, nbr).get("relation", "RELATED_TO")
                context.add(f"{node} -[{rel}]-> {nbr}")
                next_nodes.append(nbr)

        nodes = next_nodes

    return list(context)


def generate_response(query, context):
    if context:
        context_text = "\n".join(context)
        prompt = f"""
You are a medical AI assistant.

Use the medical knowledge graph if relevant.

Context:
{context_text}

Question: {query}
Answer:
"""
    else:
        prompt = f"""
You are a medical AI assistant.

Answer clearly and concisely.

Question: {query}
Answer:
"""

    inputs = generator_tokenizer(prompt, return_tensors="pt").to(generator_model.device)

    outputs = generator_model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.6
    )

    return generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_embedding(text):
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = embed_model(**inputs)

    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def calculate_reward(response, ground_truth):
    if ground_truth == "":
        return 0

    vec1 = get_embedding(response)
    vec2 = get_embedding(ground_truth)

    return cosine_similarity([vec1], [vec2])[0][0]


def rl_step(query, ground_truth, G):
    context = retrieve_kg_context(query, G)

    response = generate_response(query, context)

    reward = calculate_reward(response, ground_truth)

    return response, reward


def train_rl(query, ground_truth, G, episodes=2):
    best_reward = -1
    best_response = ""

    for _ in range(episodes):
        response, reward = rl_step(query, ground_truth, G)

        if reward > best_reward:
            best_reward = reward
            best_response = response

    return best_response
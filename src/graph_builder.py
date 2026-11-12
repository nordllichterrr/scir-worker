
import networkx as nx
from typing import List, Dict
from .utils import logger
import numpy as np

def build_graph_from_items(items: List[Dict]):
 
    G = nx.Graph()
    for it in items:
        iid = f"item:{it['item_id']}"
        G.add_node(iid, type="item", title=it.get("title"), description=it.get("description"),
                   format_tags=it.get("format_tags", []), complexity=it.get("complexity", 3),
                   raw=it.get("genres",""))
        for ent in it.get("entities", []):
            en = f"entity:{ent}"
            if not G.has_node(en):
                G.add_node(en, type="entity", name=ent)
            if not G.has_edge(iid, en):
                G.add_edge(iid, en, relation="has_entity", weight=1.0)
    logger.info("Graph built: nodes=%d, edges=%d", G.number_of_nodes(), G.number_of_edges())
    return G

def add_similarity_edges(G, items, embeddings, id2idx, top_k=5):

    idx2id = {v:k for k,v in id2idx.items()}

    from .vector_index import build_faiss_index, load_faiss_index, search_index
    import numpy as np
    import faiss
    xb = embeddings.astype('float32')
    norms = np.linalg.norm(xb, axis=1, keepdims=True)
    xb = xb / (norms + 1e-9)
    d = xb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(xb)
    D, I = index.search(xb, top_k+1) 
    for i, neighbors in enumerate(I):
        src_id = idx2id.get(i)
        if src_id is None:
            continue
        src_node = f"item:{src_id}"
        for j in neighbors[1:]:
            tgt_id = idx2id.get(int(j))
            if tgt_id is None:
                continue
            tgt_node = f"item:{tgt_id}"
            if not G.has_edge(src_node, tgt_node):
                G.add_edge(src_node, tgt_node, relation="similar_to", weight=1.0)
    logger.info("Similarity edges added to graph")
    return G

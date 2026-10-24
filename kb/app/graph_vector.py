"""Vector DB (FAISS) and Knowledge Graph (NetworkX) integration for Smart Knowledge Base."""
import re
import numpy as np
import networkx as nx
import faiss
from sklearn.decomposition import TruncatedSVD

def build_knowledge_graph(docs, chunks=None):
    """Constructs a directed Knowledge Graph connecting documents, entities, periods, and frameworks."""
    G = nx.DiGraph()
    for d in docs:
        did = d["doc_id"]
        d_node = f"doc:{did}"
        G.add_node(d_node, type="document", filename=d.get("filename", ""), title=d.get("title", ""))
        
        ent = d.get("entity", "").strip()
        if ent:
            e_node = f"entity:{ent.lower()}"
            G.add_node(e_node, type="entity", name=ent)
            G.add_edge(d_node, e_node, relation="HAS_ENTITY")
            G.add_edge(e_node, d_node, relation="CONTAINED_IN")
            
        period = d.get("period", "").strip()
        if period:
            for p in re.findall(r"\b20\d\d\b", period):
                p_node = f"period:{p}"
                G.add_node(p_node, type="period", year=p)
                G.add_edge(d_node, p_node, relation="COVERS_PERIOD")
                
        jurisdiction = d.get("jurisdiction", "").strip()
        if jurisdiction:
            j_node = f"jurisdiction:{jurisdiction.lower()}"
            G.add_node(j_node, type="jurisdiction", name=jurisdiction)
            G.add_edge(d_node, j_node, relation="UNDER_JURISDICTION")
            
        dtype = d.get("doc_type", "").strip()
        if dtype:
            dt_node = f"type:{dtype.lower()}"
            G.add_node(dt_node, type="doc_type", name=dtype)
            G.add_edge(d_node, dt_node, relation="IS_TYPE")

    # Subsidiary & Corporate Group Relationships
    known_relations = [
        ("entity:refex industries limited", "entity:refex green mobility limited", "HAS_SUBSIDIARY"),
        ("entity:refex industries limited", "entity:venwind refex power limited", "HAS_SUBSIDIARY"),
        ("entity:refex industries limited", "entity:refex mobility limited", "HAS_SUBSIDIARY"),
        ("entity:inox wind limited", "entity:inox renewable solutions limited", "DEMERGER_ENTITY"),
        ("entity:federal deposit insurance corporation (fdic)", "entity:office of the comptroller of the currency (occ)", "INTERAGENCY_PARTNER"),
        ("entity:federal deposit insurance corporation (fdic)", "entity:board of governors of the federal reserve system", "INTERAGENCY_PARTNER"),
    ]
    for src, dst, rel in known_relations:
        if G.has_node(src) and G.has_node(dst):
            G.add_edge(src, dst, relation=rel)
            G.add_edge(dst, src, relation=f"INVERSE_{rel}")
            
    return G

def expand_via_graph(G, doc_ids, query_tokens):
    """Traverses the Knowledge Graph to find linked subsidiaries, cross-references, or interagency documents."""
    expanded_ids = set(doc_ids)
    connected_relations = []
    
    for did in doc_ids:
        d_node = f"doc:{did}"
        if not G.has_node(d_node):
            continue
        # 1-hop: find entities of this doc
        for neighbor in G.neighbors(d_node):
            edge_data = G.get_edge_data(d_node, neighbor)
            rel = edge_data.get("relation", "")
            # 2-hop: see if that entity has subsidiaries or cross-references
            for second_neighbor in G.neighbors(neighbor):
                sec_edge = G.get_edge_data(neighbor, second_neighbor)
                sec_rel = sec_edge.get("relation", "")
                if second_neighbor.startswith("doc:"):
                    other_did = int(second_neighbor.split(":")[1])
                    if other_did not in expanded_ids:
                        expanded_ids.add(other_did)
                        connected_relations.append(f"{rel} -> {sec_rel} -> doc:{other_did}")
                        
    return list(expanded_ids), connected_relations

def build_faiss_index(X_sparse, n_components=64):
    """Builds a normalized FAISS vector index (IndexFlatIP) for fast dense vector similarity."""
    svd = TruncatedSVD(n_components=min(n_components, X_sparse.shape[1] - 1), random_state=42)
    X_dense = svd.fit_transform(X_sparse)
    norms = np.linalg.norm(X_dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_dense = (X_dense / norms).astype(np.float32)
    
    dim = X_dense.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X_dense)
    return index, svd

def faiss_search(index, svd, vec_model, query, top_k=20):
    """Executes a dense vector similarity search using FAISS."""
    q_sparse = vec_model.transform([query])
    q_dense = svd.transform(q_sparse).astype(np.float32)
    norm = np.linalg.norm(q_dense)
    if norm > 0:
        q_dense /= norm
    scores, indices = index.search(q_dense, top_k)
    return indices[0], scores[0]


# Stratum: Hybrid Tree-Graph Memory

Stratum is a hybrid tree-graph neural memory engine designed for long-context AI agents. It resolves the problem of **Stochastic Weight Interference** (where old states are shadowed by new weights instead of being cleanly updated) by routing all memory operations through a **Surprisal Gate** and a **Taxonomy Router**, followed by structured radix tree updates and emergent retrieval-time graph stitching.

---

## Ingestion Pipeline 

The ingestion pipeline determines whether a piece of information is novel enough to store, and how to structure it hierarchically.

```text
               ┌───────────────────────────┐
               │    1. INPUT MEMORY TEXT   │
               └─────────────┬─────────────┘
                             ▼
               ┌───────────────────────────┐
               │ 2. PERSPECTIVE NORMALIZER │
               └─────────────┬─────────────┘
                             ▼
               ┌───────────────────────────┐
               │    3. SURPRISAL GATING    │
               │   (Local Causal LM Loss)  │
               └──────┬─────────────┬──────┘
                      │             │
      Loss < 0.5      │             │  Loss >= 0.5
   (Topic is Known)   ▼             ▼  (Novel Topic)
               ┌───────────┐   ┌───────────────────────────┐
               │   SKIP    │   │  4. AGENTIC INTENT GATE   │
               │ (Discard) │   │    (Radix Path Router)    │
               └───────────┘   └────────────┬──────────────┘
                                            ▼
                               ┌───────────────────────────┐
                               │ 5. HIERARCHICAL EXPANSION │
                               └──────┬─────────────┬──────┘
                                      │             │
                     Vector & Payload ▼             ▼ Cache
                               ┌───────────┐   ┌───────────┐
                               │  QDRANT   │   │   REDIS   │
                               │ (Semantic)│   │  (Radix)  │
                               └───────────┘   └───────────┘
```

### Ingestion Phases

| Phase | Component | Description |
| :--- | :--- | :--- |
| **1** | **Perspective Normalizer** | Converts first-person sentences to third-person (e.g., *"I started"* $\rightarrow$ *"The user started"*) to align with future query formats. |
| **2** | **Surprisal Gate** | Calculates the **CrossEntropy Loss** of the text using a local `Qwen-1.5B` model. If the loss is $< 0.5$, it means the information is redundant and is skipped. |
| **3** | **Agentic Intent Gate** | Classifies the memory into a 3-level radix path: `users/{user_handle}/{subject}/{domain}/{detail}`. |
| **4** | **Hierarchical Expansion** | Breaks the path into all parent prefixes (e.g., `tech`, `tech/rust`, `tech/rust/basics`) to index the memory at multiple levels of abstraction. |
| **5** | **Dual-Store Write** | Upserts the vector and metadata into **Qdrant** (semantic layer) and stores the KV-cache in **Redis** (structural layer). |

> [!TIP]
> **Why CrossEntropy Loss for Surprisal?**
> CrossEntropy loss is a direct measure of negative log-likelihood ($-\log P(x)$). If the local language model can easily predict the next tokens in the sentence given the prior conversation context, the loss will be extremely low ($<0.5$). This indicates the information is already known, allowing Stratum to safely filter it out.

---

## Retrieval Pipeline 

Retrieval dynamically reconstructs the tree and graph relationships at query time to synthesize a chronologically accurate response.

```text
                             ┌───► [ 2a. SEMANTIC SEARCH ] ───┐
                             │       (Qdrant Cosine Match)    │
┌──────────────┐             │                                ▼
│  USER QUERY  │ ──►      [1. EMBED]                     [3. RADIX RERANK]
└──────────────┘             │                                ▲
                             │                                │
                             └───► [ 2b. PATH PREDICTION ] ───┘
                                     (Predict Likely Paths)
                                              │
                                              ▼
                                      [4. GRAPH STITCHING]
                                      (±10m Time Window)
                                              │
                                              ▼
                                    [5. RECURSIVE TREE WALK]
                                     (Up/Side Traversal)
                                              │
                                              ▼
                                     [6. CHRONO SORT & LLM]
```

### Retrieval Phases

1. **Query Embedding & Path Prediction**
   * The query is embedded into a vector.
   * The `TaxonomyLLM` predicts which radix paths are likely to contain the answer.
2. **Semantic Search & Radix Reranking**
   * The top $N$ memories are retrieved from Qdrant via cosine similarity.
   * If a retrieved memory's `radix_path` overlaps with the predicted paths, its score is boosted by up to **30%**:
     $$\text{Score}_{\text{boosted}} = \text{Score}_{\text{original}} \times (1.0 + 0.3 \times \text{overlap\_ratio})$$
3. **Temporal Graph Stitching**
   * Using the timestamp of the top retrieved memories, the engine queries Qdrant for any other memories that occurred within a **$\pm$10-minute window**. 
   * This forms implicit graph edges connecting events that happened together in time, regardless of their semantic category.
4. **Recursive Tree Walk**
   * The engine walks **sideways** (siblings) or **upwards** (parents) from the matched radix paths in Redis to gather wider hierarchical context.
5. **Chronological Sorting & Synthesis**
   * All gathered memories are sorted chronologically by parsing their date headers.
   * This ordered timeline is injected into the prompt of the generator LLM (`gpt-4o-mini`), which synthesizes the final natural language answer.

> [!IMPORTANT]
> **State Resolution**
> By sorting memories chronologically prior to generation, Stratum resolves contradictions (e.g., ensuring a *"fully recovered"* state chronologically succeeds a *"sick"* state) so the LLM always has the correct chronological context.

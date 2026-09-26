## RAG Pipeline Architecture

```text
[User Query]
     │
     ▼
┌─────────────────────────────────────────────┐
│ Step 1: Guardrails Gate (NeMo Guardrails) │
│                                             │
│ 📡 LLM Call 1:                              │
│ @llm2/meta-llama/llama-3.1-8b-instruct     │
│              via Portkey                    │
│                                             │
│ • Checks for jailbreaks                     │
│ • Checks for off-topic queries              │
│ • Handles greetings                         │
│                                             │
│ If blocked → Return refusal immediately     │
│ If clean   → Continue to Step 2             │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│ Step 2: Planner Node (LangGraph)           │
│                                             │
│ 📡 LLM Call 2:                              │
│ @llm1/openai/gpt-oss-20b                    │
│              via Portkey                    │
│                                             │
│ • Analyzes conversation history             │
│ • Determines query type                     │
│                                             │
│ ┌───────────────────┐   ┌─────────────────┐ │
│ │ CONVERSATIONAL    │   │ TECHNICAL QUERY │ │
│ └─────────┬─────────┘   └────────┬────────┘ │
│           │                      │          │
│           │                      ▼          │
│           │              Step 3: Retrieval │
│           │                      │          │
│           │                      ▼          │
│           │              ┌───────────────┐  │
│           │              │ Gemini        │  │
│           │              │ Embeddings    │  │
│           │              └───────┬───────┘  │
│           │                      │          │
│           │                      ▼          │
│           │              ┌───────────────┐  │
│           │              │ Qdrant        │  │
│           │              │ Vector DB     │  │
│           │              └───────┬───────┘  │
│           │                      │          │
│           │                      ▼          │
│           │              ┌───────────────┐  │
│           │              │ FlashRank     │  │
│           │              │ Reranker      │  │
│           │              └───────┬───────┘  │
│           │                      │          │
│           └──────────────┬──────┘           │
└──────────────────────────┼──────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────┐
│ Step 4: Responder Node                     │
│                                             │
│ 📡 LLM Call 3:                              │
│ Model defined in Portkey Config ID          │
│              via Portkey                    │
│                                             │
│ • API checks Redis response cache            │
│ • Exact/Semantic Hit → Skip the full graph  │
│ • Cache Miss          → Run planner/retrieval│
│                                             │
│ • Synthesizes final response using:         │
│   - Retrieved context                       │
│   - Conversation history                    │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
              [Final Answer]
                       │
                       ▼
                    [UI]
```

### Pipeline Flow

1. **Guardrails Gate — NeMo Guardrails**

   * Performs the first safety and intent check.
   * Detects jailbreak attempts, off-topic queries, and greetings.
   * Blocked requests receive an immediate refusal.
   * Valid requests continue to the Redis response-cache lookup.

2. **Redis Response Cache**

   * Runs after guardrails so cached responses cannot bypass safety checks.
   * The key contains the thread ID, normalized question, collection, responder model, and cache version.
   * Exact matches are checked first; paraphrases use embedding similarity with a conservative threshold.
   * A hit returns the complete answer and sources without running LangGraph.
   * A miss continues to the planner.

3. **Planner Node — LangGraph**

   * Analyzes the current query and conversation history.
   * Determines whether the request is:

     * `CONVERSATIONAL` — no retrieval required.
     * `TECHNICAL QUERY` — retrieval required.

4. **Retrieval Pipeline**

   * Checks the Redis retrieval cache first.
   * On a miss, generates embeddings and searches **Qdrant**.
   * Reranks retrieved results using the configured reranker.
   * Stores the Qdrant results in Redis for repeated searches.
   * The resulting context is passed to the responder.

5. **Responder Node**

   * Uses the model configured through the **Portkey Config ID**.
   * Generates a response using the retrieved context and conversation history.
   * The complete API response is stored in Redis after successful generation.

6. **Final Answer**

   * The generated response is returned to the UI.

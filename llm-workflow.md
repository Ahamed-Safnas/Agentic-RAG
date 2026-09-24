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
│ • Checks Portkey cache                      │
│ • Cache Hit  → Return cached response       │
│ • Cache Miss → Generate response             │
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
   * Valid requests continue to the planner.

2. **Planner Node — LangGraph**

   * Analyzes the current query and conversation history.
   * Determines whether the request is:

     * `CONVERSATIONAL` — no retrieval required.
     * `TECHNICAL QUERY` — retrieval required.

3. **Retrieval Pipeline**

   * Generates embeddings using **Gemini Embeddings**.
   * Searches the **Qdrant Vector Database**.
   * Reranks retrieved results using **FlashRank**.
   * The resulting context is passed to the responder.

4. **Responder Node**

   * Uses the model configured through the **Portkey Config ID**.
   * Checks Portkey cache before generating a response.
   * **Cache Hit:** Returns the cached response.
   * **Cache Miss:** Generates a response using the retrieved context and conversation history.

5. **Final Answer**

   * The generated response is returned to the UI.

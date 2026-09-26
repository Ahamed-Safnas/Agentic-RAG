"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

type Message = { role: "user" | "assistant"; content: string };
type Thread = {
  thread_id: string;
  title: string;
  message_count: number;
  updated_at?: string | null;
};
type QueryResponse = {
  answer?: string;
  thought_process?: string[];
  status?: string;
  sources?: string[];
  message?: string;
};

const newId = () => crypto.randomUUID();

export default function Home() {
  const [threadId, setThreadId] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [threads, setThreads] = useState<Thread[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [details, setDetails] = useState<QueryResponse | null>(null);
  const [error, setError] = useState("");

  const loadThreads = async () => {
    try {
      const response = await fetch("/api/memory", { cache: "no-store" });
      if (!response.ok) return;
      const data = (await response.json()) as { threads?: Thread[] };
      setThreads(data.threads || []);
    } catch {}
  };

  const loadThread = async (id: string) => {
    try {
      const response = await fetch(`/api/memory/${encodeURIComponent(id)}`, {
        cache: "no-store",
      });
      if (!response.ok) return;
      const data = (await response.json()) as { messages?: Message[] };
      setMessages(data.messages || []);
      setDetails(null);
      setError("");
    } catch {
      setError("Could not load conversation.");
    }
  };

  useEffect(() => {
    const stored = window.localStorage.getItem("agentic-rag-thread");
    const id = stored || newId();
    window.localStorage.setItem("agentic-rag-thread", id);
    setThreadId(id);
    void loadThread(id);
    void loadThreads();
  }, []);

  const startNewConversation = () => {
    const id = newId();
    window.localStorage.setItem("agentic-rag-thread", id);
    setThreadId(id);
    setMessages([]);
    setDetails(null);
    setError("");
  };

  const selectThread = (id: string) => {
    window.localStorage.setItem("agentic-rag-thread", id);
    setThreadId(id);
    void loadThread(id);
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const question = input.trim();
    if (!question || loading || !threadId) return;

    setInput("");
    setError("");
    setDetails(null);
    setMessages((current) => [...current, { role: "user", content: question }]);
    setLoading(true);

    try {
      const response = await fetch("/api/query", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ q: question, thread_id: threadId }),
      });
      const data = (await response.json()) as QueryResponse;
      if (!response.ok) {
        throw new Error(data.message || "Request failed.");
      }
      setMessages((current) => [
        ...current,
        { role: "assistant", content: data.answer || "No response." },
      ]);
      setDetails(data);
      void loadThreads();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed.");
    } finally {
      setLoading(false);
    }
  };

  const activeTitle = useMemo(() => {
    const current = threads.find((thread) => thread.thread_id === threadId);
    return current?.title || "New conversation";
  }, [threads, threadId]);

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">◆</div>
          <div><strong>Agent OS</strong><span>Enterprise RAG</span></div>
        </div>
        <button className="new-chat" onClick={startNewConversation}>＋ New conversation</button>
        <div className="sidebar-label">Conversations</div>
        <div className="thread-list">
          {threads.map((thread) => (
            <button
              key={thread.thread_id}
              className={thread.thread_id === threadId ? "thread active" : "thread"}
              onClick={() => selectThread(thread.thread_id)}
            >
              <span>{thread.title || "New conversation"}</span>
              <small>{thread.message_count}</small>
            </button>
          ))}
        </div>
        <div className="sidebar-footer">
          <span className="status-dot" /> RAG backend connected through secure proxy
        </div>
      </aside>

      <section className="chat-panel">
        <header className="chat-header">
          <div>
            <div className="eyebrow">ENTERPRISE AGENT</div>
            <h1>{activeTitle}</h1>
          </div>
          <div className="memory-badge">Memory · {threadId.slice(0, 8)}</div>
        </header>

        <div className="messages">
          {messages.length === 0 && (
            <div className="empty-state">
              <div className="hero-icon">✦</div>
              <h2>What can I help you find?</h2>
              <p>Ask questions about your connected enterprise documentation.</p>
            </div>
          )}

          {messages.map((message, index) => (
            <article key={`${message.role}-${index}`} className={`message ${message.role}`}>
              <div className="avatar">{message.role === "assistant" ? "✦" : "You"}</div>
              <div className="bubble">{message.content}</div>
            </article>
          ))}

          {loading && (
            <article className="message assistant">
              <div className="avatar">✦</div>
              <div className="bubble typing"><i /><i /><i /></div>
            </article>
          )}

          {error && <div className="error">{error}</div>}

          {details && (
            <details className="pipeline">
              <summary>Pipeline details</summary>
              {details.status && <p><b>Status:</b> {details.status}</p>}
              {details.thought_process?.length ? (
                <div><b>Reasoning:</b>{details.thought_process.map((step, i) => <div key={i}>• {step}</div>)}</div>
              ) : null}
              {details.sources?.length ? (
                <div className="sources"><b>Retrieved sources</b>{details.sources.map((source, i) => (
                  <details key={i}><summary>Chunk {i + 1}</summary><p>{source}</p></details>
                ))}</div>
              ) : null}
            </details>
          )}
        </div>

        <form className="composer" onSubmit={submit}>
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void submit(event);
              }
            }}
            placeholder="Ask about your documentation..."
            rows={1}
            disabled={loading}
          />
          <button type="submit" disabled={loading || !input.trim()} aria-label="Send">↑</button>
        </form>
        <div className="composer-note">Enter to send · Shift + Enter for a new line</div>
      </section>
    </main>
  );
}


import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "../styles.css";

type Role = "user" | "assistant";
type Message = {
  id: string;
  role: Role;
  content: string;
  time: string;
  used?: Set<string>;
  sources?: SourceChip[];
};
type SourceChip = { id: string; path: string; chunk_id: number; preview: string; score?: number; };

const backendURL = "http://localhost:8000"; // FastAPI backend

function nowHHMM() {
  const d = new Date();
  return `${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`;
}

export default function ChatBubbles() {
  const [q, setQ] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [activeSource, setActiveSource] = useState<SourceChip | null>(null);
  const esRef = useRef<EventSource | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, activeSource]);

  const ask = () => {
    const text = q.trim();
    if (!text) return;

    esRef.current?.close();

    const userMsg: Message = { id: crypto.randomUUID(), role: "user", content: text, time: nowHHMM() };
    const assistantMsg: Message = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "",
      time: nowHHMM(),
      used: new Set(),
      sources: []
    };

    setMessages(prev => [...prev, userMsg, assistantMsg]);
    setQ("");

    const es = new EventSource(`${backendURL}/chat/stream?q=${encodeURIComponent(text)}&top_k=5`);

    // Receive sources first (named SSE event)
    es.addEventListener("sources", (ev: MessageEvent) => {
      try {
        const payload = JSON.parse(ev.data);
        const srcs: SourceChip[] = payload?.sources || [];
        setMessages(prev => prev.map(m => m.id === assistantMsg.id ? { ...m, sources: srcs } : m));
      } catch (e) { console.error("sources parse error:", e); }
    });

    // Stream tokens on default message

    
// es.onmessage = (ev) => {
//   try {
//     const { token, error } = JSON.parse(ev.data);

//     // Surface backend errors to the assistant bubble so it doesn't hang
//     if (error) {
//       setMessages((prev) =>
//         prev.map((m) =>
//           m.id === assistantMsg.id
//             ? { ...m, content: `⚠️ Error: ${error}` } // <-- show error in the bubble
//             : m
//         )
//       );
//       es.close();
//       return;
//     }

//     // Normal token streaming path
//     if (token) {
//       setMessages((prev) =>
//         prev.map((m) => {
//           if (m.id !== assistantMsg.id) return m;
//           const next = m.content + token;
//           const matches = [...next.matchAll(/\[(S\d+)\]/g)].map((mm) => mm[1]);
//           const newer = new Set(m.used);
//           matches.forEach((id) => newer.add(id));
//           return { ...m, content: next, used: newer };
//         })
//       );
//     }
//   } catch (e) {
//     console.error("message parse error", e);
//   }
// };

    es.onmessage = (ev) => {
      try {
        const { token, error } = JSON.parse(ev.data);
        if (error) { console.error(error); es.close(); return; }
        if (token) {
          setMessages(prev => prev.map(m => {
            if (m.id !== assistantMsg.id) return m;
            const next = m.content + token;
            const matches = [...next.matchAll(/\[(S\d+)\]/g)].map(m => m[1]);
            const newer = new Set(m.used);
            matches.forEach(id => newer.add(id));
            return { ...m, content: next, used: newer };
          }));
        }
      } catch (e) { console.error("message parse error", e); }
    };

   es.onerror = (err) => { console.error("SSE error", err); es.close(); };
  esRef.current = es;
  };

  const copyLastAssistant = () => {
    const last = [...messages].reverse().find(m => m.role === "assistant" && m.content);
    if (last?.content) navigator.clipboard.writeText(last.content);
  };

  return (
    <div className="chatWrap">
      <div className="inputRow">
        <input
          className="input"
          value={q}
          onChange={e => setQ(e.target.value)}
          placeholder="Ask a question…"
        />
        <button className="btn" onClick={ask}>Ask</button>
        <button
          className="btn ghost"
          onClick={copyLastAssistant}
          disabled={!messages.some(m => m.role === "assistant" && m.content)}
        >
          Copy last answer
        </button>
      </div>

      <div className="bubbles">
        {messages.map(m => (
          <div key={m.id} className={`row ${m.role}`}>
            {m.role === "assistant" && <div className="avatar ai" title="Assistant">AI</div>}
            <div className={`bubble ${m.role}`}>
              <div className="meta">
                <span className="role">{m.role === "user" ? "You" : "Assistant"}</span>
                <span className="time">{m.time}</span>
              </div>

              {m.role === "assistant" ? (
                <div className="content markdown" aria-live="polite">
                  {m.content ? (
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                  ) : (
                    <em>Thinking…</em>
                  )}
                </div>
              ) : (
                <div className="content">{m.content}</div>
              )}

              {m.role === "assistant" && m.sources && m.sources.length > 0 && (
                <div className="badgeRow">
                  {m.sources.map(s => {
                    const cited = m.used?.has(s.id);
                    return (
                      <button
                        key={s.id}
                        className={`badge ${cited ? "used" : ""}`}
                        title={`${s.path}#${s.chunk_id}`}
                        onClick={() => setActiveSource(s)}
                      >
                        {s.id}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
            {m.role === "user" && <div className="avatar you" title="You">Sakthi</div>}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {activeSource && (
        <div className="preview">
          <div className="previewHead">
            <strong>{activeSource.id}</strong>
            <span className="muted">{activeSource.path}#{activeSource.chunk_id}</span>
            {typeof activeSource.score === "number" && <span className="score">{activeSource.score.toFixed(3)}</span>}
            <button className="close" onClick={() => setActiveSource(null)} aria-label="Close preview">×</button>
          </div>
          <div className="previewBody">{activeSource.preview}</div>
        </div>
      )}
    </div>
  );
}

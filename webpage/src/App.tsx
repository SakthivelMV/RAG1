
import React from "react";
import ChatBubbles from "./components/ChatBubbles";
import "./styles.css";

export default function App() {
  return (
    <div style={{ maxWidth: 920, margin: "40px auto", padding: "0 16px" }}>
      <h1 className="title">System Design FAQ Chatbot</h1>
      <p className="subtitle">Ask about System Design Topics.</p>
      <ChatBubbles />
    </div>
  );
}

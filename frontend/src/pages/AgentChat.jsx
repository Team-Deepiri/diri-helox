import React, { useState } from 'react';
import { externalApi } from '../api/externalApi';

export default function AgentChat() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [stream, setStream] = useState(false);

  const send = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    const userMsg = { role: 'user', content: input };
    setMessages((m) => [...m, userMsg]);
    setInput('');
    setLoading(true);
    try {
      if (stream) {
        const base = import.meta.env.VITE_CYREX_URL || 'http://localhost:8000';
        const res = await fetch(`${base}/agent/message/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content: userMsg.content })
        });
        const reader = res.body.getReader();
        let decoder = new TextDecoder();
        let acc = '';
        setMessages((m) => [...m, { role: 'assistant', content: '' }]);
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          acc += decoder.decode(value, { stream: true });
          setMessages((m) => {
            const copy = [...m];
            copy[copy.length - 1] = { role: 'assistant', content: acc };
            return copy;
          });
        }
      } else {
        const res = await externalApi.cyrexMessage(userMsg.content);
        const assistant = { role: 'assistant', content: res?.data?.message || 'No response' };
        setMessages((m) => [...m, assistant]);
      }
    } catch (err) {
      setMessages((m) => [...m, { role: 'assistant', content: 'Error contacting agent' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto p-4">
      <h1 className="text-2xl font-semibold mb-4">Agent Chat (Python)</h1>
      <div className="border rounded p-3 h-96 overflow-y-auto bg-white mb-3">
        {messages.length === 0 && (
          <div className="text-gray-500">Ask the agent for an adventure plan...</div>
        )}
        {messages.map((m, idx) => (
          <div key={idx} className={`mb-2 ${m.role === 'user' ? 'text-right' : 'text-left'}`}>
            <span className={`inline-block px-3 py-2 rounded ${m.role === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-800'}`}>
              {m.content}
            </span>
          </div>
        ))}
        {loading && <div className="text-sm text-gray-500">Thinking...</div>}
      </div>
      <form onSubmit={send} className="flex gap-2 items-center">
        <input
          className="flex-1 border rounded px-3 py-2"
          placeholder="Type a message"
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <label className="flex items-center gap-2 text-sm text-gray-600">
          <input type="checkbox" checked={stream} onChange={(e) => setStream(e.target.checked)} />
          Stream
        </label>
        <button className="bg-blue-600 text-white px-4 py-2 rounded" type="submit" disabled={loading}>
          Send
        </button>
      </form>
    </div>
  );
}



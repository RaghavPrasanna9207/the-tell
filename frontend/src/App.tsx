import { useEffect, useState } from "react";
import "./App.css";

const API_BASE = "http://localhost:8000";
const TRUSTED_CONTACT_KEY = "scamshield_trusted_contact";

interface TechniqueCard {
  technique: string;
  plain_name: string;
  span: string;
  confidence: number;
  explanation: string;
  source: string;
}

interface AnalyzeResponse {
  is_clean: boolean;
  cards: TechniqueCard[];
}

function highlightSpan(text: string, span: string) {
  const idx = text.indexOf(span);
  if (idx === -1) return text;
  return (
    <>
      {text.slice(0, idx)}
      <mark>{text.slice(idx, idx + span.length)}</mark>
      {text.slice(idx + span.length)}
    </>
  );
}

export default function App() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [trustedContact, setTrustedContact] = useState("");

  useEffect(() => {
    const saved = localStorage.getItem(TRUSTED_CONTACT_KEY);
    if (saved) setTrustedContact(saved);
  }, []);

  function saveTrustedContact(value: string) {
    setTrustedContact(value);
    localStorage.setItem(TRUSTED_CONTACT_KEY, value);
  }

  async function handleAnalyze() {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `Request failed (${res.status})`);
      }
      setResult(await res.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  function shareText() {
    if (!result) return text;
    const names = result.cards.map((c) => c.plain_name).join(", ");
    return `I got this message and it looks like it's using: ${names}.\n\nOriginal message:\n${text}`;
  }

  return (
    <div className="app">
      <div className="brand">
        <svg
          className="brand-mark"
          width="30"
          height="30"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M12 2.5 L19.5 5.3 V11 C19.5 16 16 19.8 12 21.5 C8 19.8 4.5 16 4.5 11 V5.3 Z" />
          <circle cx="10.3" cy="10.3" r="2.6" />
          <line x1="12.2" y1="12.2" x2="14.6" y2="14.6" />
        </svg>
        <h1>ScamShield Explainer</h1>
      </div>
      <p className="subtitle">
        Paste a suspicious message. This tool names the manipulation technique being used on
        you, not just whether it looks like a scam.
      </p>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Paste the message here..."
        rows={6}
      />
      <button onClick={handleAnalyze} disabled={loading || !text.trim()}>
        {loading ? "Analyzing..." : "Analyze"}
      </button>

      {error && <div className="banner error">{error}</div>}

      {result && result.is_clean && (
        <div className="banner clean">
          No manipulation techniques detected. This does not mean the message is safe —
          stay cautious with anything asking for money or personal details.
        </div>
      )}

      {result && !result.is_clean && (
        <div className="cards">
          {result.cards.map((card, i) => (
            <div className="card" key={i}>
              <h3>{card.plain_name}</h3>
              <p className="quoted">"{highlightSpan(text, card.span)}"</p>
              <p>{card.explanation}</p>
              <p className="source">Source: {card.source}</p>
            </div>
          ))}
        </div>
      )}

      <section className="layer3">
        <h2>Break the isolation</h2>
        <p>The scam depends on you not talking to anyone. Do it anyway.</p>

        <div className="trusted-contact">
          <label htmlFor="contact">Trusted contact's phone number</label>
          <input
            id="contact"
            type="tel"
            value={trustedContact}
            onChange={(e) => saveTrustedContact(e.target.value)}
            placeholder="e.g. +91 98765 43210"
          />
          {trustedContact && <a href={`tel:${trustedContact}`}>Call them now</a>}
        </div>

        <a href="tel:1930" className="resource-primary">
          Call 1930 (National Cyber Crime Helpline)
        </a>
        <a href="https://cybercrime.gov.in" target="_blank" rel="noreferrer">
          Report at cybercrime.gov.in
        </a>

        {result && !result.is_clean && (
          <a
            href={`https://wa.me/?text=${encodeURIComponent(shareText())}`}
            target="_blank"
            rel="noreferrer"
          >
            Send this to them on WhatsApp
          </a>
        )}
      </section>

      <p className="disclaimer">
        Not legal advice. This is not a diagnosis of any specific message and does not replace
        contacting the police or 1930 for anything real.
      </p>
    </div>
  );
}

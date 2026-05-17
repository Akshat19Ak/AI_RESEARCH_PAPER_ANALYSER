import { useState, useEffect, useCallback, useRef } from 'react'
import * as api from './api'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

let mermaidPromise = null

async function loadMermaid() {
  if (!mermaidPromise) {
    mermaidPromise = import('mermaid').then(({ default: mermaid }) => {
      mermaid.initialize({ startOnLoad: false, theme: 'dark', themeVariables: {
        primaryColor: '#6366f1', primaryTextColor: '#f1f5f9', primaryBorderColor: '#4f46e5',
        lineColor: '#64748b', secondaryColor: '#1e1b4b', tertiaryColor: '#0c0c14',
      }})
      return mermaid
    })
  }
  return mermaidPromise
}

/* ── Hooks ────────────────────────────────────── */
function useLoadingMessage(loading, messages, interval = 2500) {
  const [idx, setIdx] = useState(0)
  useEffect(() => {
    if (!loading) { setIdx(0); return }
    const t = setInterval(() => setIdx(i => Math.min(i + 1, messages.length - 1)), interval)
    return () => clearInterval(t)
  }, [loading, messages, interval])
  return messages[idx]
}

/* ── Shared: Sources Panel ────────────────────── */
function Sources({ sources }) {
  const [open, setOpen] = useState(false)
  if (!sources?.length) return null
  return (
    <div style={{ marginTop: '1rem' }}>
      <button className="sources-toggle" onClick={() => setOpen(!open)}>
        {open ? '▼' : '▶'} 📎 {sources.length} source chunks retrieved
      </button>
      {open && sources.map((s, i) => (
        <div key={i} className="source-item">
          <strong>Chunk {s.chunk_index}</strong>
          {s.reranker_score != null && <span> · Score: {s.reranker_score.toFixed(3)}</span>}
          <p style={{ margin: '0.3rem 0 0' }}>{s.content?.slice(0, 250)}...</p>
        </div>
      ))}
    </div>
  )
}

function Metrics({ m }) {
  if (!m) return null
  return (
    <div className="metrics-row">
      {[['Answer Relevance', m.answer_relevance], ['Faithfulness', m.faithfulness],
        ['Context Precision', m.context_precision], ['Overall', m.overall_score]].map(([l, v]) => (
        <div key={l} className="metric"><div className="metric-val">{v?.toFixed(0)}%</div><div className="metric-lbl">{l}</div></div>
      ))}
    </div>
  )
}

/* ── Page: Landing ────────────────────────────── */
function Landing({ onStart }) {
  return (
    <>
      <section className="hero">
        <div className="hero-badges">
          <span className="badge badge-purple">Hybrid RAG Pipeline</span>
          <span className="badge badge-green">Free LLM (Groq)</span>
          <span className="badge badge-cyan">Scalable with Pinecone</span>
        </div>
        <h1>Understand Any<br /><span className="gradient">Research Paper in Minutes</span></h1>
        <p>Upload a PDF, paste a URL, or snap an image — get instant summaries, methodology flowcharts, and AI-powered Q&A. Built with hybrid retrieval, cross-encoder reranking, and real-time evaluation metrics.</p>
        <button className="btn btn-primary btn-lg" onClick={onStart}>Get Started →</button>
      </section>
      <section className="container-wide" style={{ paddingTop: '1rem' }}>
        <div className="features">
          {[
            ['⚡', 'Quick Summary', 'Get a structured summary — problem, method, results, limitations — in one click.'],
            ['🔀', 'Methodology Flowchart', 'Auto-generate a visual Mermaid flowchart of the paper\'s architecture.'],
            ['💬', 'Chat Q&A', 'Ask anything about the paper with cited answers and evaluation metrics.'],
            ['🔍', 'Section Deep-Dive', 'Drill into specific sections like methodology, experiments, or related work.'],
            ['📊', 'Evaluation Metrics', 'Real-time faithfulness, relevance, and precision scores on every answer.'],
            ['🎯', 'Multi-Format Input', 'Supports PDF, images (OCR), ArXiv URLs, and web pages.'],
          ].map(([icon, title, desc]) => (
            <div key={title} className="feature-card">
              <div className="feature-icon">{icon}</div>
              <div className="feature-title">{title}</div>
              <div className="feature-desc">{desc}</div>
            </div>
          ))}
        </div>
      </section>
    </>
  )
}

/* ── Page: Setup (API Keys) ───────────────────── */
function Setup({ onNext, health }) {
  const [groq, setGroq] = useState('')
  const [pinecone, setPinecone] = useState('')
  const [saving, setSaving] = useState(false)
  const [err, setErr] = useState('')

  const save = async () => {
    if (!groq.trim() && !health?.groq_configured) { setErr('Groq API key is required.'); return }
    setSaving(true); setErr('')
    try {
      await api.configure({ groq_api_key: groq, pinecone_api_key: pinecone, vector_store_type: pinecone ? 'pinecone' : 'faiss' })
      onNext()
    } catch (e) { setErr(e.message) }
    setSaving(false)
  }

  return (
    <div className="container">
      <div className="steps">
        <div className="step active"><div className="step-num">1</div> Configure</div>
        <div className="step-line" />
        <div className="step"><div className="step-num">2</div> Upload</div>
        <div className="step-line" />
        <div className="step"><div className="step-num">3</div> Analyze</div>
      </div>
      <div className="setup-card">
        <h2>🔑 Configure API Keys</h2>
        <p>Enter your API keys to get started. Keys are stored in memory only — never saved to disk.</p>
        <div className="setup-fields">
          <div className="field">
            <label>Groq API Key {health?.groq_configured && <span style={{ color: 'var(--green)' }}>✓ configured in backend</span>}</label>
            <input className="input" type="password" value={groq} onChange={e => setGroq(e.target.value)}
              placeholder={health?.groq_configured ? '••• already set (override optional)' : 'gsk_...'} />
            <span style={{ fontSize: '0.75rem', color: 'var(--text-3)' }}>Free at <a href="https://console.groq.com/keys" target="_blank" rel="noreferrer" style={{ color: 'var(--cyan)' }}>console.groq.com</a></span>
          </div>
          <div className="field">
            <label>Pinecone API Key <span style={{ color: 'var(--text-3)' }}>(optional — for cloud storage)</span></label>
            <input className="input" type="password" value={pinecone} onChange={e => setPinecone(e.target.value)}
              placeholder={health?.pinecone_configured ? '••• already set' : 'pc-... (or leave blank for local FAISS)'} />
          </div>
          {err && <p style={{ color: 'var(--red)', fontSize: '0.85rem' }}>{err}</p>}
          <button className="btn btn-primary btn-full" onClick={save} disabled={saving}>
            {saving ? <><span className="spinner" /> Saving...</> : 'Continue →'}
          </button>
          {(health?.groq_configured) && (
            <button className="btn btn-ghost btn-full" onClick={onNext}>Skip — use backend keys</button>
          )}
        </div>
      </div>
    </div>
  )
}

/* ── Page: Upload ─────────────────────────────── */
function Upload({ onUploaded, groqKey }) {
  const [source, setSource] = useState('pdf')
  const [file, setFile] = useState(null)
  const [url, setUrl] = useState('')
  const [uploading, setUploading] = useState(false)
  const [err, setErr] = useState('')
  
  const loadingMsg = useLoadingMessage(uploading, [
    'Extracting text from document...',
    'Cleaning and formatting text...',
    'Chunking document into logical sections...',
    'Generating dense embeddings via FAISS/Pinecone...',
    'Generating sparse embeddings via BM25...',
    'Finalizing index for hybrid retrieval...'
  ])

  const upload = async () => {
    setUploading(true); setErr('')
    try {
      let r
      if (source === 'url') { r = await api.uploadUrl(url.trim(), groqKey) }
      else { r = await api.uploadFile(file, groqKey) }
      onUploaded(r)
    } catch (e) { setErr(e.message) }
    setUploading(false)
  }

  const canSubmit = source === 'url' ? url.trim() : file
  return (
    <div className="container">
      <div className="steps">
        <div className="step done"><div className="step-num">✓</div> Configure</div>
        <div className="step-line" />
        <div className="step active"><div className="step-num">2</div> Upload</div>
        <div className="step-line" />
        <div className="step"><div className="step-num">3</div> Analyze</div>
      </div>
      <div className="upload-card">
        <h2 style={{ marginBottom: '0.3rem' }}>📄 Upload Your Document</h2>
        <p style={{ color: 'var(--text-2)', fontSize: '0.88rem', marginBottom: '1rem' }}>Upload a PDF or paste a URL to begin analysis.</p>
        
        {uploading ? (
          <div style={{ textAlign: 'center', padding: '3rem 1rem' }}>
            <div className="spinner" style={{ width: '30px', height: '30px', borderWidth: '3px', marginBottom: '1rem' }} />
            <p style={{ color: 'var(--text)', fontWeight: 600 }}>{loadingMsg}</p>
          </div>
        ) : (
          <>
            <div className="source-tabs">
              {[['pdf', '📄 PDF'], ['url', '🌐 URL']].map(([id, label]) => (
                <button key={id} className={`source-tab ${source === id ? 'active' : ''}`} onClick={() => { setSource(id); setFile(null); setUrl('') }}>{label}</button>
              ))}
            </div>
            {source === 'url' ? (
              <input className="input" value={url} onChange={e => setUrl(e.target.value)} placeholder="https://arxiv.org/html/2401.xxxxx" />
            ) : (
              <div className="upload-zone" onClick={() => document.getElementById('f-input').click()}>
                <input id="f-input" type="file" accept=".pdf" onChange={e => setFile(e.target.files?.[0])} />
                {file ? (
                  <><div className="icon">✅</div><p className="filename">{file.name}</p><p>Click to change</p></>
                ) : (
                  <><div className="icon">📁</div><p>Click to select a PDF</p></>
                )}
              </div>
            )}
            {err && <p style={{ color: 'var(--red)', fontSize: '0.85rem', marginTop: '0.5rem' }}>{err}</p>}
            <button className="btn btn-primary btn-full" style={{ marginTop: '1rem' }} onClick={upload} disabled={!canSubmit}>
              🚀 Analyze Document
            </button>
          </>
        )}
      </div>
    </div>
  )
}

/* ── Tab: Quick Summary ───────────────────────── */
function SummaryTab({ sid, apiKey }) {
  const [result, setResult] = useState(''); const [sources, setSources] = useState([]); const [loading, setLoading] = useState(false)
  const msg = useLoadingMessage(loading, ['Retrieving relevant chunks...', 'Analyzing context...', 'Structuring summary...', 'Finalizing output...'], 1500)
  
  const gen = async () => {
    setLoading(true)
    try { const r = await api.generateSummary(sid, apiKey); setResult(r.result); setSources(r.sources || []) }
    catch (e) { alert(e.message) } setLoading(false)
  }
  return (
    <div>
      <h2 style={{ marginBottom: '0.3rem' }}>⚡ Quick Summary</h2>
      <p style={{ color: 'var(--text-2)', marginBottom: '1.25rem' }}>One-click structured analysis: Problem → Method → Results → Limitations → Future Work</p>
      <button className="btn btn-primary" onClick={gen} disabled={loading}>
        {loading ? <><span className="spinner" /> {msg}</> : '⚡ Generate Summary'}
      </button>
      {result && <div className="result-card"><div className="md"><ReactMarkdown remarkPlugins={[remarkGfm]}>{result}</ReactMarkdown></div></div>}
      <Sources sources={sources} />
    </div>
  )
}

/* ── Tab: Flowchart ───────────────────────────── */
function FlowchartTab({ sid, apiKey }) {
  const [mermaidCode, setMermaidCode] = useState(''); const [sources, setSources] = useState([]); const [loading, setLoading] = useState(false)
  const msg = useLoadingMessage(loading, ['Retrieving methodology sections...', 'Identifying key components...', 'Generating Mermaid syntax...', 'Rendering diagram...'], 2000)
  const ref = useRef(null)

  const gen = async () => {
    setLoading(true)
    try {
      const r = await api.generateFlowchart(sid, apiKey)
      setMermaidCode(r.mermaid); setSources(r.sources || [])
    } catch (e) { alert(e.message) }
    setLoading(false)
  }

  useEffect(() => {
    let cancelled = false
    const renderMermaid = async () => {
      if (!mermaidCode || !ref.current) return
      ref.current.innerHTML = ''
      try {
        const mermaid = await loadMermaid()
        const { svg } = await mermaid.render('mermaid-chart', mermaidCode)
        if (!cancelled && ref.current) ref.current.innerHTML = svg
      } catch {
        if (!cancelled && ref.current) ref.current.innerHTML = `<pre style="color:var(--text-2);font-size:0.85rem;">${mermaidCode}</pre>`
      }
    }
    renderMermaid()
    return () => { cancelled = true }
  }, [mermaidCode])

  return (
    <div>
      <h2 style={{ marginBottom: '0.3rem' }}>🔀 Methodology Flowchart</h2>
      <p style={{ color: 'var(--text-2)', marginBottom: '1.25rem' }}>Auto-generate a visual diagram of the paper's pipeline and architecture</p>
      <button className="btn btn-primary" onClick={gen} disabled={loading}>
        {loading ? <><span className="spinner" /> {msg}</> : '🔀 Generate Flowchart'}
      </button>
      {mermaidCode && <div className="mermaid-container" ref={ref} />}
      <Sources sources={sources} />
    </div>
  )
}

/* ── Tab: Chat Q&A ────────────────────────────── */
function ChatTab({ sid, apiKey }) {
  const [history, setHistory] = useState([]); const [q, setQ] = useState(''); const [diff, setDiff] = useState('Expert')
  const [loading, setLoading] = useState(false); const [lastM, setLastM] = useState(null); const [lastS, setLastS] = useState([])
  const sugs = ['What problem does this paper solve?', 'Explain the main method simply', 'What datasets or experiments were used?', 'What are the key limitations?']
  const msg = useLoadingMessage(loading, ['Running hybrid search...', 'Cross-encoder reranking...', 'Generating cited response...', 'Calculating evaluation metrics...'], 1000)

  const ask = async (text) => {
    const question = typeof text === 'string' ? text : q; if (!question.trim()) return
    setLoading(true)
    try {
      const r = await api.chatAsk(sid, question.trim(), diff, apiKey)
      setHistory(h => [...h, { role: 'user', content: question.trim() }, { role: 'ai', content: r.answer }])
      setLastM(r.metrics); setLastS(r.sources || []); setQ('')
    } catch (e) { alert(e.message) }
    setLoading(false)
  }

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
        <div><h2 style={{ marginBottom: '0.15rem' }}>💬 Chat Q&A</h2><p style={{ color: 'var(--text-2)', fontSize: '0.88rem' }}>Ask anything — answers are cited and scored</p></div>
        <div className="toggle-group">
          {['Beginner', 'Expert'].map(d => <button key={d} className={`toggle-opt ${diff === d ? 'active' : ''}`} onClick={() => setDiff(d)}>{d}</button>)}
        </div>
      </div>
      {history.length > 0 ? (
        <div className="chat-box">{history.map((m, i) => (
          <div key={i} className={`msg ${m.role === 'user' ? 'msg-user' : 'msg-ai'}`}>
            <div className="msg-role">{m.role === 'user' ? '👤 You' : '🤖 Analyst'}</div>
            <div className="md"><ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown></div>
          </div>
        ))}</div>
      ) : (
        <div className="suggestions">{sugs.map(s => <button key={s} className="suggestion" onClick={() => { setQ(s); ask(s); }}>{s}</button>)}</div>
      )}
      <Metrics m={lastM} /><Sources sources={lastS} />
      <div className="chat-input-row">
        <textarea className="textarea" value={q} onChange={e => setQ(e.target.value)} placeholder="Type your question..."
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask() } }} />
        <button className="btn btn-primary" onClick={() => ask()} disabled={loading || !q.trim()} style={{ alignSelf: 'flex-end' }}>
          {loading ? <span className="spinner" /> : '→'}
        </button>
      </div>
      {loading && <p style={{ fontSize: '0.8rem', color: 'var(--text-3)', textAlign: 'right', marginTop: '0.5rem' }}>{msg}</p>}
    </div>
  )
}

/* ── Tab: Deep Dive ───────────────────────────── */
function DeepDiveTab({ sid, apiKey }) {
  const [result, setResult] = useState(''); const [sources, setSources] = useState([]); const [loading, setLoading] = useState(false)
  const msg = useLoadingMessage(loading, ['Retrieving full paper context...', 'Analyzing methodology deeply...', 'Structuring complete explanation...'], 2000)

  const dive = async () => {
    setLoading(true)
    try { const r = await api.deepDive(sid, apiKey); setResult(r.result); setSources(r.sources || []) }
    catch (e) { alert(e.message) } setLoading(false)
  }

  return (
    <div>
      <h2 style={{ marginBottom: '0.3rem' }}>🔍 Full Paper Deep-Dive</h2>
      <p style={{ color: 'var(--text-2)', marginBottom: '1rem' }}>Generate an extremely comprehensive, expert-level masterclass explaining the entire research paper from start to finish.</p>
      <button className="btn btn-primary" onClick={dive} disabled={loading}>
        {loading ? <><span className="spinner" /> {msg}</> : '🔍 Generate Full Deep Dive'}
      </button>
      {result && <div className="result-card"><div className="md"><ReactMarkdown remarkPlugins={[remarkGfm]}>{result}</ReactMarkdown></div></div>}
      <Sources sources={sources} />
    </div>
  )
}

/* ── Tab: Insights ────────────────────────────── */
function InsightsTab({ sid, apiKey }) {
  const [result, setResult] = useState(''); const [sources, setSources] = useState([]); const [loading, setLoading] = useState(false)
  const msg = useLoadingMessage(loading, ['Scanning entire document...', 'Extracting 5 key takeaways...', 'Formatting insights...'], 1500)
  
  const gen = async () => {
    setLoading(true)
    try { const r = await api.generateInsights(sid, apiKey); setResult(r.result); setSources(r.sources || []) }
    catch (e) { alert(e.message) } setLoading(false)
  }
  return (
    <div>
      <h2 style={{ marginBottom: '0.3rem' }}>💡 Quick Insights</h2>
      <p style={{ color: 'var(--text-2)', marginBottom: '1.25rem' }}>Get the top 5 key takeaways and concepts from the paper instantly.</p>
      <button className="btn btn-primary" onClick={gen} disabled={loading}>
        {loading ? <><span className="spinner" /> {msg}</> : '💡 Extract Insights'}
      </button>
      {result && <div className="result-card"><div className="md"><ReactMarkdown remarkPlugins={[remarkGfm]}>{result}</ReactMarkdown></div></div>}
      <Sources sources={sources} />
    </div>
  )
}

/* ── Tab: Interview Prep ──────────────────────── */
function InterviewTab({ sid, apiKey }) {
  const [qaList, setQaList] = useState([]); const [sources, setSources] = useState([]); const [loading, setLoading] = useState(false)
  const [openIdx, setOpenIdx] = useState(null)
  const msg = useLoadingMessage(loading, ['Analyzing core contributions...', 'Generating 5 challenging questions...', 'Preparing ideal answers...'], 2000)
  
  const gen = async () => {
    setLoading(true)
    try { 
      const r = await api.generateInterview(sid, apiKey)
      // Parse the Q: A: format
      const blocks = r.result.split('---').map(b => b.trim()).filter(b => b)
      const parsed = blocks.map(b => {
        const qMatch = b.match(/Q:\s*(.*)/)
        const aMatch = b.match(/A:\s*([\s\S]*)/)
        return { 
          q: qMatch ? qMatch[1].trim() : 'Question not parsed properly', 
          a: aMatch ? aMatch[1].trim() : b 
        }
      })
      setQaList(parsed); setSources(r.sources || []); setOpenIdx(null)
    }
    catch (e) { alert(e.message) } setLoading(false)
  }
  return (
    <div>
      <h2 style={{ marginBottom: '0.3rem' }}>🎯 Interview & Study Prep</h2>
      <p style={{ color: 'var(--text-2)', marginBottom: '1.25rem' }}>Generate exactly 5 technical questions and ideal answers to test your understanding of the paper.</p>
      <button className="btn btn-primary" onClick={gen} disabled={loading}>
        {loading ? <><span className="spinner" /> {msg}</> : '🎯 Generate Quiz'}
      </button>
      {qaList.length > 0 && (
        <div className="result-card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {qaList.map((item, i) => (
            <div key={i} style={{ padding: '1rem', background: 'var(--surface)', borderRadius: '8px', border: '1px solid var(--border)' }}>
              <div style={{ fontWeight: 600, fontSize: '1.05rem', color: 'var(--text)', marginBottom: '0.75rem' }}>{i + 1}. {item.q}</div>
              {openIdx === i ? (
                <>
                  <div className="md"><ReactMarkdown remarkPlugins={[remarkGfm]}>{item.a}</ReactMarkdown></div>
                  <button className="btn btn-ghost" style={{ marginTop: '0.5rem', fontSize: '0.8rem' }} onClick={() => setOpenIdx(null)}>Hide Answer</button>
                </>
              ) : (
                <button className="btn btn-ghost" onClick={() => setOpenIdx(i)}>Show Answer</button>
              )}
            </div>
          ))}
        </div>
      )}
      <Sources sources={sources} />
    </div>
  )
}

/* ── Main App ─────────────────────────────────── */
export default function App() {
  const [page, setPage] = useState('landing') // landing | setup | upload | dashboard
  const [health, setHealth] = useState(null)
  const [groqKey, setGroqKey] = useState('')
  const [sessionId, setSessionId] = useState(null)
  const [meta, setMeta] = useState({})
  const [chunks, setChunks] = useState(0)
  const [tab, setTab] = useState('summary')

  const check = useCallback(async () => {
    try { setHealth(await api.healthCheck()) } catch { setHealth(null) }
  }, [])

  useEffect(() => { check(); const t = setInterval(check, 15000); return () => clearInterval(t) }, [check])

  const handleUploaded = (r) => {
    setSessionId(r.session_id); setMeta(r.metadata || {}); setChunks(r.chunk_count || 0); setTab('summary'); setPage('dashboard')
  }

  const reset = async () => {
    if (sessionId) try { await api.deleteSession(sessionId) } catch {}
    setSessionId(null); setMeta({}); setChunks(0); setPage('upload')
  }

  const tabs = [
    ['summary', '⚡ Summary'], 
    ['insights', '💡 Insights'],
    ['flowchart', '🔀 Flowchart'], 
    ['chat', '💬 Chat Q&A'], 
    ['deepdive', '🔍 Deep Dive'],
    ['interview', '🎯 Study Prep']
  ]

  return (
    <>
      {/* ── Navbar ── */}
      <nav className="navbar">
        <div className="nav-brand" style={{ cursor: 'pointer' }} onClick={() => { if (!sessionId) setPage('landing') }}>
          🔬 <span>ResearchLens</span>
        </div>
        <div className="nav-actions">
          <div className="nav-status">
            <span className={`dot ${health ? 'on' : 'off'}`} />
            {health ? 'API Online' : 'Backend offline'}
          </div>
          {sessionId && <button className="btn btn-ghost" style={{ fontSize: '0.8rem', padding: '0.35rem 0.8rem' }} onClick={reset}>New Paper</button>}
        </div>
      </nav>

      {/* ── Pages ── */}
      {page === 'landing' && <Landing onStart={() => setPage('setup')} />}

      {page === 'setup' && (
        <Setup health={health} onNext={() => setPage('upload')} />
      )}

      {page === 'upload' && <Upload groqKey={groqKey} onUploaded={handleUploaded} />}

      {page === 'dashboard' && (
        <>
          <div className="dash-header">
            <div className="dash-meta">
              <span className="badge badge-purple">{(meta.type || 'doc').toUpperCase()}</span>
              <span className="stat"><strong>{chunks}</strong> chunks</span>
              <span className="stat"><strong>{meta.pages || '—'}</strong> pages</span>
              <span className="stat"><strong>{(meta.char_count || 0).toLocaleString()}</strong> chars</span>
              <span className="badge badge-green">HYBRID</span>
              <span className="badge badge-green">RERANKED</span>
            </div>
          </div>
          <div className="dash-tabs" style={{ flexWrap: 'wrap' }}>
            {tabs.map(([id, label]) => (
              <button key={id} className={`dash-tab ${tab === id ? 'active' : ''}`} onClick={() => setTab(id)}>{label}</button>
            ))}
          </div>
          <div className="dash-body">
            {/* Render all tabs but hide inactive ones to cache their state */}
            <div style={{ display: tab === 'summary' ? 'block' : 'none' }}>
              <SummaryTab sid={sessionId} apiKey={groqKey} />
            </div>
            <div style={{ display: tab === 'insights' ? 'block' : 'none' }}>
              <InsightsTab sid={sessionId} apiKey={groqKey} />
            </div>
            <div style={{ display: tab === 'flowchart' ? 'block' : 'none' }}>
              <FlowchartTab sid={sessionId} apiKey={groqKey} />
            </div>
            <div style={{ display: tab === 'chat' ? 'block' : 'none' }}>
              <ChatTab sid={sessionId} apiKey={groqKey} />
            </div>
            <div style={{ display: tab === 'deepdive' ? 'block' : 'none' }}>
              <DeepDiveTab sid={sessionId} apiKey={groqKey} />
            </div>
            <div style={{ display: tab === 'interview' ? 'block' : 'none' }}>
              <InterviewTab sid={sessionId} apiKey={groqKey} />
            </div>
          </div>
        </>
      )}
    </>
  )
}

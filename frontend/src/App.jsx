import React, { useState, useEffect } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

function App() {
  const [backendStatus, setBackendStatus] = useState('Connecting to backend...');
  const [isConnected, setIsConnected] = useState(false);

  // Repository & Indexing State
  const [repoUrl, setRepoUrl] = useState('');
  const [indexStep, setIndexStep] = useState('');
  const [isIndexing, setIsIndexing] = useState(false);
  const [indexedRepoInfo, setIndexedRepoInfo] = useState(null);
  const [indexError, setIndexError] = useState(null);

  // RAG Chat Stream History State
  const [chatHistory, setChatHistory] = useState([
    {
      id: 1,
      sender: 'assistant',
      text: 'Hello! Paste a public GitHub repository URL above and click "Index Repository". Once indexed, ask me questions or use the Developer Tools Console below!',
      sources: [],
      snippets: []
    }
  ]);
  const [userQuery, setUserQuery] = useState('');
  const [isReplying, setIsReplying] = useState(false);
  const [chatError, setChatError] = useState(null);

  // Phase 12 Tools State
  const [activeTool, setActiveTool] = useState('search'); // 'search', 'file', 'structure'
  const [searchKeyword, setSearchKeyword] = useState('');
  const [targetFilePath, setTargetFilePath] = useState('');
  const [toolLoading, setToolLoading] = useState(false);
  const [toolResult, setToolResult] = useState(null);
  const [toolError, setToolError] = useState(null);

  // Check backend health on mount
  useEffect(() => {
    fetch(`${API_BASE_URL}/api/health`)
      .then((res) => res.json())
      .then((data) => {
        setBackendStatus(data.message);
        setIsConnected(true);
      })
      .catch((err) => {
        console.error('Error connecting to backend:', err);
        setBackendStatus('Connection failed');
        setIsConnected(false);
      });
  }, []);

  // 1-Click Pipeline: Ingest -> Load Docs -> Chunk -> Build FAISS Index
  const handleIndexRepository = async (e) => {
    e.preventDefault();
    if (!repoUrl.trim() || isIndexing) return;

    setIsIndexing(true);
    setIndexError(null);
    setIndexStep('Cloning & Analyzing Repository...');

    try {
      // Step 1: Ingest Repo
      const ingestRes = await fetch(`${API_BASE_URL}/api/ingest-repo`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repo_url: repoUrl.trim() }),
      });
      const ingestData = await ingestRes.json();
      if (!ingestRes.ok) throw new Error(ingestData.detail || 'Failed to ingest repository');

      // Step 2: Load Docs & Chunk
      setIndexStep('Converting files & chunking code...');
      const chunkRes = await fetch(`${API_BASE_URL}/api/chunk-documents`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repo_url: repoUrl.trim(), chunk_size: 1000, chunk_overlap: 200 }),
      });
      const chunkData = await chunkRes.json();
      if (!chunkRes.ok) throw new Error(chunkData.detail || 'Failed to chunk repository files');

      // Step 3: Build FAISS Vector Index
      setIndexStep('Building FAISS Vector Database...');
      const faissRes = await fetch(`${API_BASE_URL}/api/build-faiss-index`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repo_url: repoUrl.trim(), chunk_size: 1000, chunk_overlap: 200 }),
      });
      const faissData = await faissRes.json();
      if (!faissRes.ok) throw new Error(faissData.detail || 'Failed to build vector index');

      // Store Indexed Summary
      setIndexedRepoInfo({
        owner: ingestData.owner,
        repo_name: ingestData.repo_name,
        repo_url: ingestData.repo_url,
        total_files: ingestData.total_files,
        total_chunks: chunkData.total_chunks
      });

      setIndexStep('done');

      setChatHistory([
        {
          id: Date.now(),
          sender: 'assistant',
          text: `Repository "${ingestData.owner}/${ingestData.repo_name}" indexed successfully (${ingestData.total_files} files, ${chunkData.total_chunks} code vectors ready). Ask questions or use tools!`,
          sources: [],
          snippets: []
        }
      ]);
    } catch (err) {
      console.error('Indexing Error:', err);
      setIndexError(err.message);
    } finally {
      setIsIndexing(false);
    }
  };

  // Submit RAG Question with History
  const handleSendRagMessage = async (e) => {
    e.preventDefault();
    if (!userQuery.trim() || !indexedRepoInfo || isReplying) return;

    const queryText = userQuery.trim();
    setUserQuery('');
    setChatError(null);

    const userMsgObj = {
      id: Date.now(),
      sender: 'user',
      text: queryText
    };
    const updatedHistory = [...chatHistory, userMsgObj];
    setChatHistory(updatedHistory);
    setIsReplying(true);

    const historyPayload = chatHistory
      .filter((msg) => msg.id !== 1)
      .map((msg) => ({
        role: msg.sender === 'user' ? 'user' : 'assistant',
        content: msg.text
      }));

    try {
      const response = await fetch(`${API_BASE_URL}/api/rag-query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          repo_url: indexedRepoInfo.repo_url,
          user_question: queryText,
          history: historyPayload,
          top_k: 4
        }),
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Failed to generate answer');

      const aiMsgObj = {
        id: Date.now() + 1,
        sender: 'assistant',
        text: data.answer,
        sources: data.sources || [],
        snippets: data.retrieved_chunks || [],
        model: data.model_used
      };

      setChatHistory([...updatedHistory, aiMsgObj]);
    } catch (err) {
      console.error('RAG Error:', err);
      setChatError(err.message);
    } finally {
      setIsReplying(false);
    }
  };

  // Phase 12 Tool Handlers
  const handleRunToolSearchCode = async (e) => {
    e.preventDefault();
    if (!searchKeyword.trim() || !indexedRepoInfo || toolLoading) return;

    setToolLoading(true);
    setToolError(null);
    setToolResult(null);

    try {
      const res = await fetch(`${API_BASE_URL}/api/tools/search-code`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          repo_url: indexedRepoInfo.repo_url,
          keyword: searchKeyword.trim()
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Tool execution failed');
      setToolResult(data);
    } catch (err) {
      setToolError(err.message);
    } finally {
      setToolLoading(false);
    }
  };

  const handleRunToolGetFileContent = async (e) => {
    e.preventDefault();
    if (!targetFilePath.trim() || !indexedRepoInfo || toolLoading) return;

    setToolLoading(true);
    setToolError(null);
    setToolResult(null);

    try {
      const res = await fetch(`${API_BASE_URL}/api/tools/get-file-content`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          repo_url: indexedRepoInfo.repo_url,
          file_path: targetFilePath.trim()
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Tool execution failed');
      setToolResult(data);
    } catch (err) {
      setToolError(err.message);
    } finally {
      setToolLoading(false);
    }
  };

  const handleRunToolRepoStructure = async () => {
    if (!indexedRepoInfo || toolLoading) return;

    setToolLoading(true);
    setToolError(null);
    setToolResult(null);

    try {
      const res = await fetch(`${API_BASE_URL}/api/tools/repo-structure`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repo_url: indexedRepoInfo.repo_url }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Tool execution failed');
      setToolResult(data);
    } catch (err) {
      setToolError(err.message);
    } finally {
      setToolLoading(false);
    }
  };

  return (
    <div className="app-shell">
      {/* Navbar Header */}
      <header className="navbar">
        <div className="nav-brand">
          <h1>AI GitHub Repository Coding Assistant</h1>
        </div>
        <div className={`connection-badge ${isConnected ? 'online' : 'offline'}`}>
          <span className="dot"></span>
          <span>{backendStatus}</span>
        </div>
      </header>

      <main className="main-layout">
        {/* TOP PANEL: Repository URL Entry & Indexing Controls */}
        <section className="repo-control-panel">
          <h2>1. Target GitHub Repository</h2>
          <form onSubmit={handleIndexRepository} className="url-form">
            <input
              type="url"
              placeholder="Enter GitHub URL (e.g. https://github.com/pallets/flask)"
              value={repoUrl}
              onChange={(e) => setRepoUrl(e.target.value)}
              disabled={isIndexing || !isConnected}
              required
            />
            <button type="submit" className="index-btn" disabled={isIndexing || !repoUrl.trim() || !isConnected}>
              {isIndexing ? 'Indexing Repo...' : 'Index Repository'}
            </button>
          </form>

          {isIndexing && (
            <div className="progress-banner">
              <span className="spinner"></span>
              <span>{indexStep}</span>
            </div>
          )}

          {indexError && (
            <div className="error-alert">
              <strong>Indexing Error:</strong> {indexError}
            </div>
          )}

          {indexedRepoInfo && (
            <div className="active-repo-banner">
              <div className="repo-title-row">
                <span className="active-badge">ACTIVE INDEX</span>
                <h3>{indexedRepoInfo.owner} / <span>{indexedRepoInfo.repo_name}</span></h3>
              </div>
              <div className="repo-meta-row">
                <span>📁 {indexedRepoInfo.total_files} Files</span>
                <span>⚡ {indexedRepoInfo.total_chunks} Code Vectors</span>
                <span>🛠️ Tools Active</span>
                <a href={indexedRepoInfo.repo_url} target="_blank" rel="noopener noreferrer">View on GitHub ↗</a>
              </div>
            </div>
          )}
        </section>

        {/* PHASE 12: Developer Tools Panel */}
        {indexedRepoInfo && (
          <section className="card tools-panel">
            <h2>🛠️ Developer Tool Execution Console (Phase 12)</h2>
            <div className="tools-tabs">
              <button
                className={`tab-btn ${activeTool === 'search' ? 'active' : ''}`}
                onClick={() => { setActiveTool('search'); setToolResult(null); }}
              >
                1. search_code()
              </button>
              <button
                className={`tab-btn ${activeTool === 'file' ? 'active' : ''}`}
                onClick={() => { setActiveTool('file'); setToolResult(null); }}
              >
                2. get_file_content()
              </button>
              <button
                className={`tab-btn ${activeTool === 'structure' ? 'active' : ''}`}
                onClick={() => { setActiveTool('structure'); setToolResult(null); handleRunToolRepoStructure(); }}
              >
                3. get_repository_structure()
              </button>
            </div>

            <div className="tool-body">
              {activeTool === 'search' && (
                <form onSubmit={handleRunToolSearchCode} className="tool-form">
                  <input
                    type="text"
                    placeholder="Enter search keyword (e.g. createUser or Flask or route)"
                    value={searchKeyword}
                    onChange={(e) => setSearchKeyword(e.target.value)}
                    required
                  />
                  <button type="submit" disabled={toolLoading}>
                    {toolLoading ? 'Executing Tool...' : 'Execute search_code()'}
                  </button>
                </form>
              )}

              {activeTool === 'file' && (
                <form onSubmit={handleRunToolGetFileContent} className="tool-form">
                  <input
                    type="text"
                    placeholder="Enter relative file path (e.g. src/flask/app.py)"
                    value={targetFilePath}
                    onChange={(e) => setTargetFilePath(e.target.value)}
                    required
                  />
                  <button type="submit" disabled={toolLoading}>
                    {toolLoading ? 'Fetching File...' : 'Execute get_file_content()'}
                  </button>
                </form>
              )}

              {activeTool === 'structure' && toolLoading && (
                <div className="progress-banner">
                  <span className="spinner"></span>
                  <span>Generating Repository Tree Structure...</span>
                </div>
              )}

              {toolError && (
                <div className="error-alert">
                  <strong>Tool Error:</strong> {toolError}
                </div>
              )}

              {/* Tool Output Display */}
              {toolResult && (
                <div className="tool-output-box">
                  <div className="tool-output-header">
                    <strong>Tool Result: <code>{toolResult.tool}</code></strong>
                  </div>

                  {toolResult.tool === 'search_code' && (
                    <div className="matches-list">
                      <p>Found <strong>{toolResult.total_matches}</strong> matches for keyword <code>"{toolResult.keyword}"</code>:</p>
                      {toolResult.matches.map((m, idx) => (
                        <div key={idx} className="tool-match-item">
                          <span><code>{m.file_path}</code> (Line {m.line_number})</span>
                          <pre><code>{m.line_content}</code></pre>
                        </div>
                      ))}
                    </div>
                  )}

                  {toolResult.tool === 'get_file_content' && (
                    <div className="file-content-view">
                      <p>File: <code>{toolResult.file_path}</code> ({toolResult.total_lines} lines, {toolResult.total_characters} chars)</p>
                      <pre><code>{toolResult.content}</code></pre>
                    </div>
                  )}

                  {toolResult.tool === 'get_repository_structure' && (
                    <div className="tree-view">
                      <p>Repository File Tree Preview ({toolResult.total_files} files):</p>
                      <ul>
                        {toolResult.file_tree.map((filePath, idx) => (
                          <li key={idx}><code>📄 {filePath}</code></li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
          </section>
        )}

        {/* BOTTOM PANEL: Chat Console */}
        <section className="chat-console-panel">
          <div className="console-header">
            <h2>2. Conversational RAG Console</h2>
            {indexedRepoInfo ? (
              <span className="status-pill ready">RAG Active</span>
            ) : (
              <span className="status-pill waiting">Waiting for Repo Index</span>
            )}
          </div>

          <div className="chat-stream">
            {chatHistory.map((msg) => (
              <div key={msg.id} className={`chat-message ${msg.sender}`}>
                <div className="avatar">
                  {msg.sender === 'user' ? '🧑‍💻' : '🤖'}
                </div>

                <div className="message-body">
                  <div className="msg-header">
                    <strong>{msg.sender === 'user' ? 'You' : 'AI Coding Assistant'}</strong>
                    {msg.model && <span className="model-tag">{msg.model}</span>}
                  </div>

                  <div className="msg-text">{msg.text}</div>

                  {msg.sources && msg.sources.length > 0 && (
                    <div className="sources-container">
                      <span className="sources-label">📌 Source Citations:</span>
                      <div className="sources-chips">
                        {msg.sources.map((src, idx) => (
                          <span key={idx} className="chip">📄 <code>{src}</code></span>
                        ))}
                      </div>
                    </div>
                  )}

                  {msg.snippets && msg.snippets.length > 0 && (
                    <details className="context-snippets-accordion">
                      <summary>Inspect {msg.snippets.length} Retrieved Context Snippets</summary>
                      <div className="snippets-drawer">
                        {msg.snippets.map((snip, sIdx) => (
                          <div key={sIdx} className="snip-card">
                            <div className="snip-header">
                              <code>{snip.file_path}</code>
                              <span className="score">Score: {snip.similarity_score}</span>
                            </div>
                            <pre><code>{snip.content}</code></pre>
                          </div>
                        ))}
                      </div>
                    </details>
                  )}
                </div>
              </div>
            ))}

            {isReplying && (
              <div className="chat-message assistant thinking">
                <div className="avatar">🤖</div>
                <div className="message-body">
                  <span className="typing">Retrieving code vectors & generating grounded answer...</span>
                </div>
              </div>
            )}
          </div>

          {chatError && (
            <div className="error-alert chat-err">
              <strong>Chat Error:</strong> {chatError}
            </div>
          )}

          <form onSubmit={handleSendRagMessage} className="chat-input-bar">
            <input
              type="text"
              placeholder={
                indexedRepoInfo
                  ? 'Ask a question (e.g. "Where is routing handled?" or "Explain UserService")'
                  : 'Index a repository above first to start asking questions...'
              }
              value={userQuery}
              onChange={(e) => setUserQuery(e.target.value)}
              disabled={isReplying || !indexedRepoInfo}
            />
            <button type="submit" disabled={isReplying || !userQuery.trim() || !indexedRepoInfo}>
              {isReplying ? 'Thinking...' : 'Send'}
            </button>
          </form>
        </section>
      </main>
    </div>
  );
}

export default App;

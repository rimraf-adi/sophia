/**
 * Sophia Frontend Engine
 * Multi-turn Live Agentic RAG Client with SSE Streaming & Citation Linking
 */

document.addEventListener('DOMContentLoaded', () => {
  // --- State ---
  let sessionId = localStorage.getItem('sophia_session_id') || `sess_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
  localStorage.setItem('sophia_session_id', sessionId);
  let useCache = true;
  let activeEventSource = null;

  // --- Elements ---
  const chatViewport = document.getElementById('chat-viewport');
  const heroSection = document.getElementById('hero-section');
  const threadContainer = document.getElementById('thread-container');
  const searchForm = document.getElementById('search-form');
  const queryInput = document.getElementById('query-input');
  const sendBtn = document.getElementById('send-btn');
  const newThreadBtn = document.getElementById('new-thread-btn');
  const toggleCacheBtn = document.getElementById('toggle-cache-btn');
  const sessionDisplay = document.getElementById('session-display');
  const poolStatusText = document.getElementById('pool-status-text');

  // Modal elements
  const modelsModal = document.getElementById('models-modal');
  const navModelsBtn = document.getElementById('nav-models-modal');
  const closeModalBtn = document.getElementById('close-modal-btn');
  const modalBody = document.getElementById('modal-body');

  sessionDisplay.textContent = `Session: ${sessionId.substring(0, 14)}...`;

  // --- Auto-resize textarea ---
  queryInput.addEventListener('input', () => {
    queryInput.style.height = 'auto';
    queryInput.style.height = `${Math.min(queryInput.scrollHeight, 140)}px`;
  });

  function startSearch() {
    const query = queryInput.value.trim();
    if (!query) return;

    queryInput.value = '';
    queryInput.style.height = 'auto';
    executeSearchStream(query);
  }

  // Handle Enter keypress
  queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      e.stopPropagation();
      startSearch();
    }
  });

  // Handle Send Button click
  sendBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    startSearch();
  });

  // Handle Form submit prevention
  searchForm.addEventListener('submit', (e) => {
    e.preventDefault();
    e.stopPropagation();
    startSearch();
    return false;
  });

  // --- Toggle Cache ---
  toggleCacheBtn.addEventListener('click', () => {
    useCache = !useCache;
    toggleCacheBtn.classList.toggle('active', useCache);
    toggleCacheBtn.innerHTML = useCache 
      ? '<i class="fa-solid fa-bolt"></i> Cache: ON'
      : '<i class="fa-solid fa-bolt-slash"></i> Cache: OFF';
  });

  // --- New Thread ---
  newThreadBtn.addEventListener('click', () => {
    sessionId = `sess_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
    localStorage.setItem('sophia_session_id', sessionId);
    sessionDisplay.textContent = `Session: ${sessionId.substring(0, 14)}...`;
    threadContainer.innerHTML = '';
    heroSection.style.display = 'flex';
    queryInput.focus();
  });

  // --- Suggestion Chips ---
  document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const q = chip.getAttribute('data-query');
      if (q) {
        queryInput.value = q;
        startSearch();
      }
    });
  });

  // --- Fetch Key Pool Diagnostics ---
  async function updatePoolStatus() {
    try {
      const res = await fetch('/api/models');
      const data = await res.json();
      if (data.pool && data.pool.groq) {
        poolStatusText.textContent = `${data.pool.groq.total_keys} Groq Keys Active`;
      }
    } catch (e) {
      console.warn('Failed to fetch pool status', e);
    }
  }
  updatePoolStatus();

  // --- Modal Events ---
  navModelsBtn.addEventListener('click', async () => {
    modelsModal.classList.add('active');
    modalBody.innerHTML = '<div class="loading-spinner"><i class="fa-solid fa-spinner fa-spin"></i> Fetching active pool stats...</div>';
    try {
      const res = await fetch('/api/models');
      const data = await res.json();
      let keysHtml = '';
      if (data.pool && data.pool.groq && data.pool.groq.keys) {
        keysHtml = data.pool.groq.keys.map((k, i) => `
          <div class="key-card">
            <strong>Key [${i+1}]:</strong> <code>${k.masked_key}</code><br>
            <span style="color: ${k.is_available ? '#10B981' : '#EF4444'}">
              ${k.is_available ? '● Healthy' : '● Cooling down'}
            </span> | Requests: ${k.total_requests}
          </div>
        `).join('');
      }

      let modelsHtml = (data.rotating_models || []).map(m => `
        <li style="margin-bottom: 4px; font-family: monospace; font-size: 0.88rem;">• ${m}</li>
      `).join('');

      modalBody.innerHTML = `
        <h3 style="margin-bottom: 8px; font-size: 1rem; color: var(--accent-cyan);">Active Rotating Models:</h3>
        <ul style="margin-left: 20px; margin-bottom: 20px;">${modelsHtml}</ul>
        
        <h3 style="margin-bottom: 8px; font-size: 1rem; color: var(--accent-emerald);">Groq Key Pool:</h3>
        <div class="keys-grid">${keysHtml}</div>
      `;
    } catch (e) {
      modalBody.innerHTML = `<div style="color: #EF4444;">Failed to load pool diagnostics: ${e.message}</div>`;
    }
  });

  closeModalBtn.addEventListener('click', () => modelsModal.classList.remove('active'));
  modelsModal.addEventListener('click', (e) => {
    if (e.target === modelsModal) modelsModal.classList.remove('active');
  });

  // --- Execution Engine: SSE Streaming ---
  function executeSearchStream(query) {
    if (activeEventSource) {
      activeEventSource.close();
      activeEventSource = null;
    }

    heroSection.style.display = 'none';
    sendBtn.disabled = true;

    // 1. Add User Block
    const userBlock = document.createElement('div');
    userBlock.className = 'user-query-block';
    userBlock.innerHTML = `
      <div class="user-avatar"><i class="fa-solid fa-user"></i></div>
      <div class="user-query-text">${escapeHtml(query)}</div>
    `;
    threadContainer.appendChild(userBlock);

    // 2. Add Assistant Response Block
    const responseBlock = document.createElement('div');
    responseBlock.className = 'response-block';

    const stepper = document.createElement('div');
    stepper.className = 'pipeline-progress';
    stepper.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> <span>Initializing search pipeline...</span>';
    responseBlock.appendChild(stepper);

    const sourcesContainer = document.createElement('div');
    sourcesContainer.className = 'sources-section';
    sourcesContainer.style.display = 'none';
    sourcesContainer.innerHTML = '<div class="section-label"><i class="fa-solid fa-globe"></i> Sources</div><div class="sources-grid"></div>';
    responseBlock.appendChild(sourcesContainer);

    const answerContent = document.createElement('div');
    answerContent.className = 'answer-content';
    responseBlock.appendChild(answerContent);

    const followupsContainer = document.createElement('div');
    followupsContainer.className = 'followups-section';
    followupsContainer.style.display = 'none';
    followupsContainer.innerHTML = '<div class="section-label"><i class="fa-solid fa-lightbulb"></i> Related Questions</div><div class="followups-list"></div>';
    responseBlock.appendChild(followupsContainer);

    threadContainer.appendChild(responseBlock);
    chatViewport.scrollTop = chatViewport.scrollHeight;

    // SSE Stream Setup
    let accumulatedText = '';
    let sourcesList = [];
    const streamUrl = `/api/query/stream?q=${encodeURIComponent(query)}&session_id=${encodeURIComponent(sessionId)}&use_cache=${useCache}`;
    activeEventSource = new EventSource(streamUrl);

    activeEventSource.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        const { event_type, data } = payload;

        if (event_type === 'status') {
          stepper.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> <span>${escapeHtml(data)}</span>`;
        } else if (event_type === 'query_rewritten') {
          stepper.innerHTML = `<i class="fa-solid fa-magnifying-glass"></i> <span>Searching for: <strong>"${escapeHtml(data)}"</strong></span>`;
        } else if (event_type === 'sources') {
          sourcesList = data || [];
          if (sourcesList.length > 0) {
            sourcesContainer.style.display = 'block';
            const grid = sourcesContainer.querySelector('.sources-grid');
            grid.innerHTML = sourcesList.map(s => `
              <a href="${s.url}" target="_blank" rel="noopener" class="source-card" id="source-${s.index || s.source_id}">
                <div class="source-domain">
                  <i class="fa-solid fa-link"></i> ${s.domain || 'web'}
                </div>
                <div class="source-title">[${s.index || s.source_id}] ${escapeHtml(s.title || 'Untitled')}</div>
              </a>
            `).join('');
          }
        } else if (event_type === 'token') {
          accumulatedText += data;
          renderAnswerWithCitations(answerContent, accumulatedText);
          chatViewport.scrollTop = chatViewport.scrollHeight;
        } else if (event_type === 'follow_ups') {
          const questions = data || [];
          if (questions.length > 0) {
            followupsContainer.style.display = 'block';
            const list = followupsContainer.querySelector('.followups-list');
            list.innerHTML = questions.map(q => `
              <button class="followup-btn" data-query="${escapeHtml(q)}">
                <span>${escapeHtml(q)}</span>
                <i class="fa-solid fa-plus"></i>
              </button>
            `).join('');

            list.querySelectorAll('.followup-btn').forEach(btn => {
              btn.addEventListener('click', () => {
                const followQuery = btn.getAttribute('data-query');
                if (followQuery) {
                  queryInput.value = followQuery;
                  startSearch();
                }
              });
            });
          }
        } else if (event_type === 'done') {
          stepper.style.display = 'none';
          if (activeEventSource) {
            activeEventSource.close();
            activeEventSource = null;
          }
          sendBtn.disabled = false;
          updatePoolStatus();
        } else if (event_type === 'error') {
          stepper.innerHTML = `<i class="fa-solid fa-triangle-exclamation" style="color:#EF4444;"></i> <span style="color:#EF4444;">${escapeHtml(data)}</span>`;
          if (activeEventSource) {
            activeEventSource.close();
            activeEventSource = null;
          }
          sendBtn.disabled = false;
        }
      } catch (err) {
        console.error('Error parsing SSE event', err);
      }
    };

    activeEventSource.onerror = (err) => {
      console.warn('SSE stream closed/errored', err);
      if (activeEventSource) {
        activeEventSource.close();
        activeEventSource = null;
      }
      stepper.style.display = 'none';
      sendBtn.disabled = false;
    };
  }

  // --- Render Answer with Interactive Citations ---
  function renderAnswerWithCitations(container, rawMarkdown) {
    if (!rawMarkdown) return;
    
    // Parse standard markdown
    let html = marked.parse(rawMarkdown);

    // Replace [1], [2], [1][2] markers with interactive styled chips
    html = html.replace(/\[(\d+)\]/g, (match, id) => {
      return `<a href="#source-${id}" class="citation-chip" data-id="${id}">[${id}]</a>`;
    });

    container.innerHTML = html;

    // Apply syntax highlighting
    container.querySelectorAll('pre code').forEach((block) => {
      hljs.highlightElement(block);
    });

    // Add hover and scroll click listener to citation chips
    container.querySelectorAll('.citation-chip').forEach((chip) => {
      chip.addEventListener('click', (e) => {
        e.preventDefault();
        const srcId = chip.getAttribute('data-id');
        const targetCard = document.getElementById(`source-${srcId}`);
        if (targetCard) {
          targetCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
          targetCard.style.outline = '2px solid var(--accent-cyan)';
          setTimeout(() => targetCard.style.outline = 'none', 1500);
        }
      });
    });
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
  }
});

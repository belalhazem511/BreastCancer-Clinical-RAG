const STORAGE_QUESTION = 'bcai_current_question';
const STORAGE_SOURCE = 'bcai_source_filter';
const STORAGE_HISTORY = 'bcai_citation_history';
const STORAGE_RECENT = 'bcai_recent_chats';

const conversation = document.getElementById('conversation');
const chatInput = document.getElementById('chatInput');
const sendButton = document.getElementById('chatSendButton');
const evidenceDrawer = document.getElementById('evidenceDrawer');
const drawerClose = document.getElementById('drawerClose');
const copyCitationButton = document.getElementById('copyCitationButton');
const drawerOpenPdfBtn = document.getElementById('drawerOpenPdfBtn');
const drawerPrevBtn = document.getElementById('drawerPrevBtn');
const drawerNextBtn = document.getElementById('drawerNextBtn');
const drawerSnippet = document.getElementById('drawerSnippet');
const recentChatsContainer = document.getElementById('recentChatsContainer');
const exportChatButton = document.getElementById('exportChatButton');

let latestQuestion = '';
let activeCitation = null;
let currentRetrievedCitations = [];
let currentCitationIndex = 0;
let responseInProgress = false;
let chatMessages = [];

// Determine API base URL (works seamlessly on any port or host)
const API_BASE = (window.location.protocol === 'http:' || window.location.protocol === 'https:')
  ? ''
  : (localStorage.getItem('bcai_api_base') || 'http://127.0.0.1:8080');

// Format date relative or concise
function formatChatTime(date = new Date()) {
  const hours = date.getHours().toString().padStart(2, '0');
  const minutes = date.getMinutes().toString().padStart(2, '0');
  return `${hours}:${minutes}`;
}

// Render User Message
function createUserMessage(question) {
  latestQuestion = question;
  chatMessages.push({ role: 'user', content: question, time: new Date().toLocaleTimeString() });
  
  const group = document.createElement('div');
  group.className = 'message-group';

  const message = document.createElement('div');
  message.className = 'user-message';
  message.textContent = question;

  group.appendChild(message);
  conversation.appendChild(group);
  scrollToBottom();
  saveRecentChat(question);
}

// Animated AI Process Indicator
function createProcess() {
  const group = document.createElement('div');
  group.className = 'message-group';
  group.id = 'activeProcessGroup';
  group.innerHTML = `
    <div class="ai-process">
      <div class="process-orb"></div>
      <div class="process-copy">
        <strong class="process-title">Searching clinical guidelines...</strong>
        <span class="process-subtitle">Scanning NICE NG101, CG81, and CG164 guideline vectors</span>
      </div>
    </div>
  `;
  conversation.appendChild(group);
  scrollToBottom();
  return group;
}

function updateProcess(group, title, subtitle) {
  if (!group || !group.querySelector('.ai-process')) return;
  const box = group.querySelector('.ai-process');
  const titleEl = group.querySelector('.process-title');
  const subtitleEl = group.querySelector('.process-subtitle');
  if (titleEl) titleEl.textContent = title;
  if (subtitleEl) subtitleEl.textContent = subtitle;
  box.animate([
    { opacity: 0.7, transform: 'translateY(2px)' },
    { opacity: 1, transform: 'translateY(0)' }
  ], { duration: 200, easing: 'ease' });
}

// Render Structured Answer Card
function createAnswer(data) {
  const citations = data.citations || [];
  currentRetrievedCitations = citations;
  currentCitationIndex = 0;
  activeCitation = citations.length > 0 ? citations[0] : null;

  chatMessages.push({
    role: 'assistant',
    data: data,
    time: new Date().toLocaleTimeString()
  });

  const group = document.createElement('div');
  group.className = 'message-group';

  const summary = data.summary || 'Evidence-grounded clinical response from NICE guidelines.';
  const recs = (data.recommendations || []).map(r => `<li>${escapeHtml(r)}</li>`).join('');
  const evidence = (data.supporting_evidence || []).map(e => `<li>${escapeHtml(e)}</li>`).join('');
  const confidence = data.confidence || 'High';
  const confidenceReason = data.confidence_reason || 'Evidence grounded in NICE guidelines';
  const sourceMatch = data.source_match || '94%';

  // Dynamic confidence dot color
  let dotColor = '#4bbe7d'; // green
  if (confidence.toLowerCase() === 'medium') dotColor = '#f5a623';
  if (confidence.toLowerCase() === 'low') dotColor = '#e74c3c';

  // Build source chips
  let chipsHtml = '';
  if (citations.length > 0) {
    citations.forEach((c, idx) => {
      chipsHtml += `
        <button class="source-chip evidence-trigger" type="button" data-citation-index="${idx}" title="${escapeHtml(c.source_name)}">
          ${c.source} ${c.section}
        </button>
        <button class="source-chip evidence-trigger" type="button" data-citation-index="${idx}">
          ${c.pages}
        </button>
      `;
    });
  } else {
    chipsHtml = `<span style="font-size:12px;color:#888;">No applicable NICE source citations</span>`;
  }

  group.innerHTML = `
    <article class="answer-card">
      <div class="answer-header">
        <div class="mini-ai"></div>
        <h2>Evidence-based answer</h2>
      </div>
      
      <p><strong>${escapeHtml(summary)}</strong></p>

      ${recs ? `
        <h3>Recommendations</h3>
        <ul>${recs}</ul>
      ` : ''}

      ${evidence ? `
        <h3>Supporting evidence</h3>
        <ul>${evidence}</ul>
      ` : ''}

      <div class="source-area">
        <div class="source-label">Sources used</div>
        <div class="source-chips">
          ${chipsHtml}
        </div>
      </div>

      <div class="confidence-card">
        <span>Evidence confidence</span>
        <strong class="high-confidence">
          <i class="confidence-dot" style="background:${dotColor};box-shadow:0 0 0 3px ${dotColor}22;"></i>
          ${confidence}
        </strong>
      </div>

      ${citations.length > 0 ? `
        <button class="view-evidence-button evidence-trigger" type="button" data-citation-index="0">
          View supporting evidence & PDF source →
        </button>
      ` : ''}
    </article>
  `;

  conversation.appendChild(group);

  // Attach event listeners to chips and buttons
  group.querySelectorAll('.evidence-trigger').forEach((button) => {
    button.addEventListener('click', () => {
      const idx = parseInt(button.dataset.citationIndex || '0', 10);
      if (currentRetrievedCitations[idx]) {
        openEvidence(currentRetrievedCitations[idx], idx);
      } else if (activeCitation) {
        openEvidence(activeCitation, 0);
      }
    });
  });

  // Save each citation to history
  citations.forEach(c => {
    saveCitationHistory(latestQuestion, c);
  });

  scrollToBottom();
}

// Call Real Backend API
async function askRag(question, sourceFilter = '') {
  if (responseInProgress) return;
  responseInProgress = true;

  const process = createProcess();

  // Progress animation timers
  const t1 = setTimeout(() => {
    updateProcess(process, 'Retrieving relevant guideline sections...', 'Matching semantic and BM25 keyword evidence');
  }, 450);

  const t2 = setTimeout(() => {
    updateProcess(process, 'Evaluating clinical criteria...', 'Checking NICE guideline sections and recommendations');
  }, 1100);

  const t3 = setTimeout(() => {
    updateProcess(process, 'Synthesizing evidence-based response...', 'Grounding answer strictly in retrieved sources');
  }, 1800);

  try {
    const payload = {
      question: question,
      source_filter: sourceFilter || null
    };

    const response = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    clearTimeout(t1);
    clearTimeout(t2);
    clearTimeout(t3);

    if (!response.ok) {
      throw new Error(`Server returned status ${response.status}`);
    }

    const data = await response.json();

    // Small delay to ensure smooth transition
    setTimeout(() => {
      process.remove();
      createAnswer(data);
      responseInProgress = false;
    }, 200);

  } catch (error) {
    clearTimeout(t1);
    clearTimeout(t2);
    clearTimeout(t3);
    console.error('RAG Query Error:', error);

    process.remove();
    createErrorMessage(error.message);
    responseInProgress = false;
  }
}

function createErrorMessage(details) {
  const group = document.createElement('div');
  group.className = 'message-group';
  group.innerHTML = `
    <article class="answer-card" style="border-color: rgba(231,76,60,0.3); background: #fff8f8;">
      <div class="answer-header">
        <div class="mini-ai" style="background:#e74c3c;"></div>
        <h2 style="color:#c0392b;">Service Notice</h2>
      </div>
      <p>The clinical AI service could not complete the request. Please make sure the backend server is running.</p>
      <p style="font-size:12px;color:#888;">Error details: ${escapeHtml(details)}</p>
      <button class="view-evidence-button" type="button" style="border-color:#e74c3c;color:#c0392b;" onclick="retryLastQuestion()">
        Retry Question
      </button>
    </article>
  `;
  conversation.appendChild(group);
  scrollToBottom();
}

function retryLastQuestion() {
  if (latestQuestion) {
    const filter = localStorage.getItem(STORAGE_SOURCE) || '';
    askRag(latestQuestion, filter);
  }
}

// Send Message Handler
function sendMessage() {
  const question = chatInput.value.trim();
  if (!question || responseInProgress) {
    if (!question) chatInput.focus();
    return;
  }

  const sourceFilter = localStorage.getItem(STORAGE_SOURCE) || '';
  createUserMessage(question);
  chatInput.value = '';
  chatInput.style.height = 'auto';
  askRag(question, sourceFilter);
}

// Save Citation History
function saveCitationHistory(question, citation) {
  if (!question || !citation) return;

  let history = [];
  try {
    history = JSON.parse(localStorage.getItem(STORAGE_HISTORY) || '[]');
    if (!Array.isArray(history)) history = [];
  } catch {
    history = [];
  }

  // Avoid exact duplicates
  const isDuplicate = history.some(item => 
    item.question === question && item.source === citation.source && item.section === citation.section
  );

  if (!isDuplicate) {
    history.unshift({
      id: Date.now() + Math.floor(Math.random() * 1000),
      question: question,
      source: citation.source,
      section: citation.section,
      pages: citation.pages,
      description: citation.description || citation.section_name || '',
      date: new Date().toLocaleString()
    });
    localStorage.setItem(STORAGE_HISTORY, JSON.stringify(history.slice(0, 60)));
  }
}

// Save Recent Chats
function saveRecentChat(question) {
  if (!question) return;
  let recents = [];
  try {
    recents = JSON.parse(localStorage.getItem(STORAGE_RECENT) || '[]');
    if (!Array.isArray(recents)) recents = [];
  } catch {
    recents = [];
  }

  // Filter out identical existing question
  recents = recents.filter(r => r.title.toLowerCase() !== question.toLowerCase());
  recents.unshift({
    id: Date.now(),
    title: question,
    date: 'Today'
  });

  localStorage.setItem(STORAGE_RECENT, JSON.stringify(recents.slice(0, 10)));
  renderRecentChats();
}

function renderRecentChats() {
  if (!recentChatsContainer) return;
  let recents = [];
  try {
    recents = JSON.parse(localStorage.getItem(STORAGE_RECENT) || '[]');
    if (!Array.isArray(recents)) recents = [];
  } catch {
    recents = [];
  }

  if (recents.length === 0) {
    recents = [
      { id: 1, title: 'Treatment options for breast cancer', date: 'Today' },
      { id: 2, title: 'Endocrine therapy overview', date: 'Yesterday' },
      { id: 3, title: 'NICE NG101 summary', date: '3 days ago' }
    ];
  }

  recentChatsContainer.innerHTML = '';
  recents.forEach((item, index) => {
    const btn = document.createElement('button');
    btn.className = `recent-chat ${index === 0 ? 'active-recent' : ''}`;
    btn.type = 'button';
    btn.innerHTML = `
      <span class="recent-dot"></span>
      <span>
        <strong>${escapeHtml(item.title)}</strong>
        <small>${item.date || 'Recent'}</small>
      </span>
    `;
    btn.addEventListener('click', () => {
      document.querySelectorAll('.recent-chat').forEach(b => b.classList.remove('active-recent'));
      btn.classList.add('active-recent');
      chatInput.value = item.title;
      sendMessage();
    });
    recentChatsContainer.appendChild(btn);
  });
}

// Open Evidence Drawer
function openEvidence(citation, index = 0) {
  if (!citation) return;
  activeCitation = citation;
  currentCitationIndex = index;

  document.getElementById('drawerSource').textContent = citation.shortSource || 'NICE';
  document.getElementById('drawerSection').textContent = citation.sectionNumber || citation.section || '';
  document.getElementById('drawerPages').textContent = citation.pageRange || citation.pages || '';
  document.getElementById('drawerDescription').textContent = citation.description || citation.source_name || '';
  document.getElementById('drawerFilename').textContent = citation.filename || `${citation.shortSource}.pdf`;
  document.getElementById('drawerPageLabel').textContent = `Page ${citation.firstPage || citation.start_page || 1}`;
  document.getElementById('drawerPageCount').textContent = citation.pageCount || `${citation.start_page || 1} / 108`;
  document.getElementById('drawerPreviewTitle').textContent = citation.previewTitle || citation.section_name || '';
  
  if (drawerSnippet) {
    drawerSnippet.textContent = citation.text || 'Relevant recommendation retrieved from this NICE guideline.';
  }

  if (drawerOpenPdfBtn) {
    drawerOpenPdfBtn.onclick = () => {
      const pageNum = citation.start_page || citation.firstPage || 1;
      const pdfUrl = `${API_BASE}${citation.pdf_url || `/api/pdf/${citation.filename}`}#page=${pageNum}`;
      window.open(pdfUrl, '_blank');
    };
  }

  evidenceDrawer.classList.remove('closed');
}

function closeEvidence() {
  evidenceDrawer.classList.add('closed');
}

// Drawer previous/next navigation for multiple retrieved chunks
if (drawerPrevBtn) {
  drawerPrevBtn.addEventListener('click', () => {
    if (currentRetrievedCitations.length > 1) {
      currentCitationIndex = (currentCitationIndex - 1 + currentRetrievedCitations.length) % currentRetrievedCitations.length;
      openEvidence(currentRetrievedCitations[currentCitationIndex], currentCitationIndex);
    }
  });
}

if (drawerNextBtn) {
  drawerNextBtn.addEventListener('click', () => {
    if (currentRetrievedCitations.length > 1) {
      currentCitationIndex = (currentCitationIndex + 1) % currentRetrievedCitations.length;
      openEvidence(currentRetrievedCitations[currentCitationIndex], currentCitationIndex);
    }
  });
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    conversation.scrollTop = conversation.scrollHeight;
  });
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// Export Chat Functionality
if (exportChatButton) {
  exportChatButton.addEventListener('click', () => {
    if (chatMessages.length === 0) {
      alert('No conversation messages to export.');
      return;
    }

    let exportContent = `# BreastCancer.ai - Clinical Consultation Transcript\n`;
    exportContent += `Export Date: ${new Date().toLocaleString()}\n`;
    exportContent += `Grounded Guideline Sources: NICE NG101, NICE CG81, NICE CG164\n\n`;
    exportContent += `============================================================\n\n`;

    chatMessages.forEach(msg => {
      if (msg.role === 'user') {
        exportContent += `[${msg.time}] CLINICIAN QUESTION:\n${msg.content}\n\n`;
      } else if (msg.role === 'assistant' && msg.data) {
        const d = msg.data;
        exportContent += `[${msg.time}] AI EVIDENCE-BASED ANSWER:\n`;
        exportContent += `Summary: ${d.summary}\n\n`;
        if (d.recommendations && d.recommendations.length) {
          exportContent += `Recommendations:\n`;
          d.recommendations.forEach(r => exportContent += `• ${r}\n`);
          exportContent += `\n`;
        }
        if (d.supporting_evidence && d.supporting_evidence.length) {
          exportContent += `Supporting Evidence:\n`;
          d.supporting_evidence.forEach(e => exportContent += `• ${e}\n`);
          exportContent += `\n`;
        }
        if (d.citations && d.citations.length) {
          exportContent += `Sources & Citations:\n`;
          d.citations.forEach(c => exportContent += `• ${c.source} ${c.section} (${c.pages})\n`);
          exportContent += `\n`;
        }
        exportContent += `Evidence Confidence: ${d.confidence}\n`;
        exportContent += `------------------------------------------------------------\n\n`;
      }
    });

    exportContent += `\nDisclaimer: This clinical summary is generated strictly based on NICE guideline evidence and does not replace professional medical judgement.\n`;

    const blob = new Blob([exportContent], { type: 'text/markdown;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `BreastCancer_AI_Transcript_${Date.now()}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });
}

// Copy Citation Handler
if (copyCitationButton) {
  copyCitationButton.addEventListener('click', async () => {
    if (!activeCitation && currentRetrievedCitations.length > 0) {
      activeCitation = currentRetrievedCitations[0];
    }
    if (!activeCitation) return;

    const text = `${activeCitation.source} - ${activeCitation.section}, ${activeCitation.pages} (${activeCitation.source_name || ''})`;
    try {
      await navigator.clipboard.writeText(text);
      copyCitationButton.textContent = 'Copied ✓';
      setTimeout(() => { copyCitationButton.textContent = 'Copy Citation'; }, 1300);
    } catch {
      copyCitationButton.textContent = 'Copy unavailable';
      setTimeout(() => { copyCitationButton.textContent = 'Copy Citation'; }, 1300);
    }
  });
}

// Event Listeners
sendButton.addEventListener('click', sendMessage);
chatInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});
chatInput.addEventListener('input', () => {
  chatInput.style.height = 'auto';
  chatInput.style.height = `${Math.min(chatInput.scrollHeight, 120)}px`;
});
drawerClose.addEventListener('click', closeEvidence);

// Voice Recognition (Speech-to-Text) in Chat
const chatVoiceMicButton = document.getElementById('chatVoiceMicButton');
if (chatVoiceMicButton) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SpeechRecognition) {
    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    let isListening = false;
    const originalPlaceholder = chatInput.placeholder;

    chatVoiceMicButton.addEventListener('click', () => {
      if (isListening) {
        recognition.stop();
      } else {
        try {
          recognition.start();
        } catch (err) {
          console.warn('Chat speech recognition start error:', err);
        }
      }
    });

    recognition.onstart = () => {
      isListening = true;
      chatVoiceMicButton.classList.add('is-recording');
      chatVoiceMicButton.setAttribute('title', 'Listening... Click to stop');
      chatInput.placeholder = 'Listening... Speak your question now';
    };

    recognition.onresult = (event) => {
      let transcript = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
      }
      if (transcript.trim()) {
        chatInput.value = transcript;
        chatInput.style.height = 'auto';
        chatInput.style.height = `${Math.min(chatInput.scrollHeight, 120)}px`;
      }
    };

    recognition.onerror = (event) => {
      console.warn('Chat speech recognition error:', event.error);
      stopListening();
      if (event.error === 'not-allowed') {
        alert('Microphone access was blocked. Please enable microphone permission in your browser to use voice input.');
      }
    };

    recognition.onend = () => {
      stopListening();
      const question = chatInput.value.trim();
      if (question) {
        setTimeout(() => {
          sendMessage();
        }, 350);
      } else {
        chatInput.focus();
      }
    };

    function stopListening() {
      isListening = false;
      chatVoiceMicButton.classList.remove('is-recording');
      chatVoiceMicButton.setAttribute('title', 'Click to speak');
      chatInput.placeholder = originalPlaceholder;
    }
  } else {
    chatVoiceMicButton.addEventListener('click', () => {
      alert('Speech recognition is not supported in this browser. Please use Google Chrome, Microsoft Edge, or Safari.');
    });
  }
}

// Initialize Page
renderRecentChats();

const initialQuestion = localStorage.getItem(STORAGE_QUESTION);
const initialSource = localStorage.getItem(STORAGE_SOURCE) || '';

if (initialQuestion) {
  // Clear stored question so subsequent reloads don't re-trigger unnecessarily
  localStorage.removeItem(STORAGE_QUESTION);
  createUserMessage(initialQuestion);
  askRag(initialQuestion, initialSource);
}

requestAnimationFrame(() => {
  requestAnimationFrame(() => document.body.classList.remove('chat-preload'));
});

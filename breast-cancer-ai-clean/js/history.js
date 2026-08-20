const STORAGE_HISTORY = 'bcai_citation_history';
const STORAGE_QUESTION = 'bcai_current_question';
const STORAGE_SOURCE = 'bcai_source_filter';

const historyList = document.getElementById('historyList');
const historyEmpty = document.getElementById('historyEmpty');
const historyCount = document.getElementById('historyCount');
const historySearch = document.getElementById('historySearch');
const clearHistoryButton = document.getElementById('clearHistoryButton');

function getHistory() {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_HISTORY) || '[]');
    return Array.isArray(value) ? value : [];
  } catch {
    return [];
  }
}

function renderHistory(search = '') {
  const needle = search.trim().toLowerCase();
  const history = getHistory().filter((item) => {
    const haystack = `${item.question || ''} ${item.source || ''} ${item.section || ''} ${item.pages || ''} ${item.description || ''}`.toLowerCase();
    return haystack.includes(needle);
  });

  historyList.innerHTML = '';
  historyCount.textContent = `${history.length} ${history.length === 1 ? 'citation' : 'citations'}`;
  historyEmpty.hidden = history.length !== 0;
  historyList.style.display = history.length ? 'flex' : 'none';

  history.forEach((item) => {
    const card = document.createElement('article');
    card.className = 'history-card';

    const left = document.createElement('div');
    const time = document.createElement('div');
    time.className = 'history-time';
    time.textContent = item.date || '';

    const question = document.createElement('div');
    question.className = 'history-question';
    question.textContent = item.question || '';

    const chips = document.createElement('div');
    chips.className = 'history-chips';
    [item.source, item.section, item.pages].forEach((label) => {
      const chip = document.createElement('span');
      chip.className = 'history-chip';
      chip.textContent = label || '';
      chips.appendChild(chip);
    });

    left.append(time, question, chips);

    const actions = document.createElement('div');
    actions.className = 'history-actions';
    const openPdf = document.createElement('a');
    openPdf.className = 'history-open-pdf';
    openPdf.textContent = 'View PDF ↗';
    const shortCode = String(item.source || '').includes('CG164') ? 'CG164' : (String(item.source || '').includes('CG81') ? 'CG81' : 'NG101');
    const startPgMatch = String(item.pages || '').match(/\d+/);
    const startPg = startPgMatch ? startPgMatch[0] : '1';
    openPdf.href = `/api/pdf/${shortCode}.pdf#page=${startPg}`;
    openPdf.target = '_blank';
    openPdf.rel = 'noopener';
    openPdf.style.cssText = 'font-size:12px;color:#2e7d32;background:rgba(75,190,125,0.12);padding:4px 8px;border-radius:6px;text-decoration:none;font-weight:600;display:inline-flex;align-items:center;';

    const copy = document.createElement('button');
    copy.className = 'copy-history';
    copy.type = 'button';
    copy.textContent = 'Copy';
    const askAgain = document.createElement('button');
    askAgain.className = 'ask-again';
    askAgain.type = 'button';
    askAgain.textContent = 'Ask again';
    actions.append(openPdf, copy, askAgain);

    copy.addEventListener('click', async () => {
      const citation = `${item.source} - ${item.section}, ${item.pages}`;
      try {
        await navigator.clipboard.writeText(citation);
        copy.textContent = 'Copied ✓';
        setTimeout(() => { copy.textContent = 'Copy'; }, 1200);
      } catch {
        copy.textContent = 'Unavailable';
        setTimeout(() => { copy.textContent = 'Copy'; }, 1200);
      }
    });

    askAgain.addEventListener('click', () => {
      localStorage.setItem(STORAGE_QUESTION, item.question || '');
      const source = String(item.source || '').includes('CG164') ? 'CG164' : (String(item.source || '').includes('CG81') ? 'CG81' : 'NG101');
      localStorage.setItem(STORAGE_SOURCE, source);
      window.location.href = 'chat.html';
    });

    card.append(left, actions);
    historyList.appendChild(card);
  });
}

historySearch.addEventListener('input', () => renderHistory(historySearch.value));
clearHistoryButton.addEventListener('click', () => {
  localStorage.removeItem(STORAGE_HISTORY);
  renderHistory(historySearch.value);
});
renderHistory();

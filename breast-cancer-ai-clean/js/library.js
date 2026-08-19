const STORAGE_QUESTION = 'bcai_current_question';
const STORAGE_SOURCE = 'bcai_source_filter';

const sourceSearch = document.getElementById('sourceSearch');
const sourceCards = Array.from(document.querySelectorAll('.source-card'));
const emptySourceSearch = document.getElementById('emptySourceSearch');
const modal = document.getElementById('detailsModal');
const modalClose = document.getElementById('modalClose');
const modalTitle = document.getElementById('modalTitle');
const modalText = document.getElementById('modalText');
const modalOpenPdfBtn = document.getElementById('modalOpenPdfBtn');

const API_BASE = (window.location.protocol === 'http:' || window.location.protocol === 'https:')
  ? ''
  : (localStorage.getItem('bcai_api_base') || 'http://127.0.0.1:8080');

const sourceDetails = {
  NG101: {
    title: 'NICE NG101',
    text: 'Early and locally advanced breast cancer: diagnosis and management. This guideline is built into the system knowledge base and is available automatically to the clinical RAG assistant.',
    filename: 'NG101.pdf'
  },
  CG81: {
    title: 'NICE CG81',
    text: 'Advanced breast cancer: diagnosis and treatment. This guideline is built into the system knowledge base and is available automatically to the clinical RAG assistant.',
    filename: 'CG81.pdf'
  },
  CG164: {
    title: 'NICE CG164',
    text: 'Familial breast cancer: classification, care and managing breast cancer and related risks in people with a family history of breast cancer. This guideline is built into the system knowledge base and is available automatically to the clinical RAG assistant.',
    filename: 'CG164.pdf'
  }
};

let currentModalSource = 'NG101';

sourceSearch.addEventListener('input', () => {
  const value = sourceSearch.value.trim().toLowerCase();
  let visible = 0;
  sourceCards.forEach((card) => {
    const matches = (card.dataset.search || '').includes(value);
    card.style.display = matches ? 'flex' : 'none';
    if (matches) visible += 1;
  });
  emptySourceSearch.style.display = visible === 0 ? 'block' : 'none';
});

document.querySelectorAll('.ask-source-button').forEach((button) => {
  button.addEventListener('click', () => {
    const source = button.dataset.source;
    const question = source === 'CG164'
      ? 'What does NICE CG164 recommend for surveillance and genetic testing in familial breast cancer?'
      : (source === 'CG81'
        ? 'What does NICE CG81 recommend for advanced breast cancer diagnosis and systemic treatment?'
        : 'What does NICE NG101 recommend for early and locally advanced breast cancer?');

    localStorage.setItem(STORAGE_QUESTION, question);
    localStorage.setItem(STORAGE_SOURCE, source);
    window.location.href = 'chat.html';
  });
});

document.querySelectorAll('.details-button').forEach((button) => {
  button.addEventListener('click', () => {
    const source = button.dataset.source;
    currentModalSource = source;
    const details = sourceDetails[source] || sourceDetails.NG101;
    modalTitle.textContent = details.title;
    modalText.textContent = details.text;
    modal.hidden = false;
  });
});

if (modalOpenPdfBtn) {
  modalOpenPdfBtn.addEventListener('click', () => {
    const details = sourceDetails[currentModalSource] || sourceDetails.NG101;
    const pdfUrl = `${API_BASE}/api/pdf/${details.filename}`;
    window.open(pdfUrl, '_blank');
  });
}

function closeModal() {
  modal.hidden = true;
}

modalClose.addEventListener('click', closeModal);
modal.addEventListener('click', (event) => {
  if (event.target === modal) closeModal();
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && !modal.hidden) closeModal();
});

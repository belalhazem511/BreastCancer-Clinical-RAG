const STORAGE_QUESTION = 'bcai_current_question';
const STORAGE_SOURCE = 'bcai_source_filter';

const questionInput = document.getElementById('questionInput');
const sendButton = document.getElementById('homeSendButton');
const questionBox = document.getElementById('questionBox');
const homeMain = document.getElementById('homeMain');
const suggestionCards = document.querySelectorAll('.suggestion-card');
let navigating = false;

function openChatWithQuestion(question, source = '') {
  const clean = question.trim();
  if (!clean || navigating) return;

  navigating = true;
  localStorage.setItem(STORAGE_QUESTION, clean);
  if (source) localStorage.setItem(STORAGE_SOURCE, source);
  else localStorage.removeItem(STORAGE_SOURCE);

  questionBox.classList.add('sending');
  sendButton.animate([
    { transform: 'scale(1)' },
    { transform: 'scale(.88)' },
    { transform: 'scale(1)' }
  ], { duration: 220, easing: 'ease' });

  setTimeout(() => homeMain.classList.add('is-leaving'), 70);
  setTimeout(() => { window.location.href = 'chat.html'; }, 300);
}

function sendCurrentQuestion() {
  const question = questionInput.value.trim();
  if (!question) {
    questionInput.focus();
    return;
  }
  openChatWithQuestion(question);
}

sendButton.addEventListener('click', sendCurrentQuestion);
questionInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') {
    event.preventDefault();
    sendCurrentQuestion();
  }
});

suggestionCards.forEach((card) => {
  card.addEventListener('click', () => {
    questionInput.value = card.dataset.question || '';
    questionInput.focus();
    questionBox.animate([
      { transform: 'scale(.996)' },
      { transform: 'scale(1)' }
    ], { duration: 170, easing: 'ease' });
  });
});

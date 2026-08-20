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

// Voice Recognition (Speech-to-Text)
const homeVoiceMicButton = document.getElementById('homeVoiceMicButton');
if (homeVoiceMicButton) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SpeechRecognition) {
    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    let isListening = false;
    const originalPlaceholder = questionInput.placeholder;

    homeVoiceMicButton.addEventListener('click', () => {
      if (isListening) {
        recognition.stop();
      } else {
        try {
          recognition.start();
        } catch (err) {
          console.warn('Speech recognition start error:', err);
        }
      }
    });

    recognition.onstart = () => {
      isListening = true;
      homeVoiceMicButton.classList.add('is-recording');
      homeVoiceMicButton.setAttribute('title', 'Listening... Click to stop');
      questionInput.placeholder = 'Listening... Speak your clinical question now';
    };

    recognition.onresult = (event) => {
      let transcript = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
      }
      if (transcript.trim()) {
        questionInput.value = transcript;
      }
    };

    recognition.onerror = (event) => {
      console.warn('Speech recognition error:', event.error);
      stopListening();
      if (event.error === 'not-allowed') {
        alert('Microphone access was blocked. Please enable microphone permission in your browser to use voice input.');
      }
    };

    recognition.onend = () => {
      stopListening();
      questionInput.focus();
    };

    function stopListening() {
      isListening = false;
      homeVoiceMicButton.classList.remove('is-recording');
      homeVoiceMicButton.setAttribute('title', 'Click to speak');
      questionInput.placeholder = originalPlaceholder;
    }
  } else {
    homeVoiceMicButton.addEventListener('click', () => {
      alert('Speech recognition is not supported in this browser. Please use Google Chrome, Microsoft Edge, or Safari.');
    });
  }
}

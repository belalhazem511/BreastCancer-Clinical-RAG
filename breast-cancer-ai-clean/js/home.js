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

// Voice Recognition & Interactive Orb Controller
const homeVoiceMicButton = document.getElementById('homeVoiceMicButton');
const heroOrbWrap = document.getElementById('heroOrbWrap');
const assistantStatusBadge = document.getElementById('assistantStatusBadge');
const assistantStatusText = document.getElementById('assistantStatusText');

let currentRecognition = null;
let isRecordingActive = false;
let silenceTimer = null;
const originalPlaceholder = questionInput.placeholder;

function updateOrbVoiceState(state, customText = '') {
  if (!heroOrbWrap || !assistantStatusBadge) return;

  heroOrbWrap.classList.remove('is-listening', 'is-processing');
  assistantStatusBadge.classList.remove('is-listening', 'is-processing');

  if (state === 'listening') {
    heroOrbWrap.classList.add('is-listening');
    assistantStatusBadge.classList.add('is-listening');
    if (homeVoiceMicButton) homeVoiceMicButton.classList.add('is-recording');
    if (assistantStatusText) assistantStatusText.textContent = customText || 'Listening to your voice...';
    questionInput.placeholder = 'Listening... Speak your clinical question now';
  } else if (state === 'processing') {
    heroOrbWrap.classList.add('is-processing');
    assistantStatusBadge.classList.add('is-processing');
    if (homeVoiceMicButton) homeVoiceMicButton.classList.remove('is-recording');
    if (assistantStatusText) assistantStatusText.textContent = customText || 'Analyzing NICE guidelines...';
  } else {
    if (homeVoiceMicButton) homeVoiceMicButton.classList.remove('is-recording');
    if (assistantStatusText) assistantStatusText.textContent = 'AI assistant ready';
    questionInput.placeholder = originalPlaceholder;
  }
}

function startHomeVoiceRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    alert('Speech recognition is not supported in this browser. Please use Google Chrome, Microsoft Edge, or Safari.');
    return;
  }

  if (isRecordingActive) {
    stopHomeVoiceRecognition(true);
    return;
  }

  try {
    if (currentRecognition) {
      try { currentRecognition.abort(); } catch (e) {}
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    isRecordingActive = true;
    updateOrbVoiceState('listening');

    recognition.onstart = () => {
      isRecordingActive = true;
      updateOrbVoiceState('listening');
    };

    recognition.onresult = (event) => {
      let finalTranscript = '';
      let interimTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        } else {
          interimTranscript += event.results[i][0].transcript;
        }
      }

      const spokenText = (finalTranscript || interimTranscript).trim();
      if (spokenText) {
        questionInput.value = spokenText;
        
        // Reset silence timer on new speech
        clearTimeout(silenceTimer);
        silenceTimer = setTimeout(() => {
          stopHomeVoiceRecognition(true);
        }, 1600);
      }
    };

    recognition.onerror = (event) => {
      console.warn('Speech recognition error event:', event.error);
      if (event.error === 'not-allowed') {
        alert('Microphone access was blocked. Please allow microphone permissions in your browser.');
      }
      stopHomeVoiceRecognition(false);
    };

    recognition.onend = () => {
      if (isRecordingActive) {
        stopHomeVoiceRecognition(true);
      }
    };

    currentRecognition = recognition;
    recognition.start();
  } catch (err) {
    console.error('Failed to start speech recognition:', err);
    stopHomeVoiceRecognition(false);
  }
}

function stopHomeVoiceRecognition(shouldSubmit = true) {
  isRecordingActive = false;
  clearTimeout(silenceTimer);

  if (currentRecognition) {
    try {
      currentRecognition.stop();
    } catch (e) {}
    currentRecognition = null;
  }

  const question = questionInput.value.trim();
  if (shouldSubmit && question) {
    localStorage.setItem('bcai_voice_mode', '1');
    updateOrbVoiceState('processing');
    setTimeout(() => {
      sendCurrentQuestion();
    }, 380);
  } else {
    updateOrbVoiceState('idle');
    questionInput.focus();
  }
}

if (homeVoiceMicButton) {
  homeVoiceMicButton.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    startHomeVoiceRecognition();
  });
}

if (heroOrbWrap) {
  heroOrbWrap.addEventListener('click', () => {
    startHomeVoiceRecognition();
  });
}

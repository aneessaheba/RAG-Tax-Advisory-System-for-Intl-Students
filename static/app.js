let studentProfile = null;

// ── Profile form ──────────────────────────────────────

document.getElementById('profile-form').addEventListener('submit', function (e) {
  e.preventDefault();
  const form = e.target;

  const incomeChecked = form.querySelectorAll('input[name="income_types"]:checked');
  const incomeTypes = Array.from(incomeChecked).map(cb => cb.value);

  studentProfile = {
    visa_type:        form.visa_type.value,
    home_country:     form.home_country.value.trim() || 'Unknown',
    first_entry_year: form.first_entry_year.value.trim() || '2023',
    tax_year:         form.tax_year.value.trim() || '2024',
    income_types:     incomeTypes.length ? incomeTypes : ['None'],
    state:            form.state.value.trim() || 'CA',
    has_ssn_or_itin:  form.has_ssn.checked,
  };

  document.getElementById('profile-status').textContent = 'Profile saved.';
  document.getElementById('question-input').disabled = false;
  document.getElementById('send-btn').disabled = false;
  document.getElementById('question-input').focus();
});

// ── Send message ──────────────────────────────────────

document.getElementById('send-btn').addEventListener('click', sendMessage);

document.getElementById('question-input').addEventListener('keydown', function (e) {
  if (e.key === 'Enter') sendMessage();
});

function sendMessage() {
  if (!studentProfile) return;

  const input = document.getElementById('question-input');
  const question = input.value.trim();
  if (!question) return;

  input.value = '';
  input.disabled = true;
  document.getElementById('send-btn').disabled = true;

  appendUserMessage(question);
  const typingEl = appendTyping();

  fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, student_info: studentProfile }),
  })
    .then(r => r.json())
    .then(data => {
      typingEl.remove();
      appendBotMessage(question, data);
    })
    .catch(() => {
      typingEl.remove();
      appendBotMessage(question, {
        answer: 'Something went wrong. Please try again.',
        refused: true,
      });
    })
    .finally(() => {
      input.disabled = false;
      document.getElementById('send-btn').disabled = false;
      input.focus();
    });
}

// ── Message rendering ─────────────────────────────────

function appendUserMessage(text) {
  const el = document.createElement('div');
  el.className = 'user-msg';
  el.textContent = text;
  getMessages().appendChild(el);
  scrollToBottom();
}

function appendTyping() {
  const el = document.createElement('div');
  el.className = 'bot-msg typing-indicator';
  el.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
  getMessages().appendChild(el);
  scrollToBottom();
  return el;
}

function appendBotMessage(question, data) {
  const el = document.createElement('div');
  el.className = 'bot-msg';

  // Answer text
  const answerEl = document.createElement('div');
  answerEl.className = 'answer-text';
  answerEl.innerHTML = formatText(data.answer);
  el.appendChild(answerEl);

  // Meta row + feedback (only for successful non-refused answers)
  if (!data.refused) {
    const meta = document.createElement('div');
    meta.className = 'meta-row';

    const stats = document.createElement('span');
    stats.className = 'meta-stats';
    stats.textContent = data.used_fallback
      ? `Retrieval: ${data.retrieval_latency}s | Gemini unavailable (fallback) | Confidence: ${data.confidence}`
      : `${data.total_latency}s · Confidence: ${data.confidence} · ~${data.input_tokens}/${data.output_tokens} tok`;
    meta.appendChild(stats);

    // 👍 / 👎 feedback buttons
    const feedbackBtns = document.createElement('div');
    feedbackBtns.className = 'feedback-btns';

    const upBtn   = makeBtn('👍');
    const downBtn = makeBtn('👎');
    let feedbackGiven = false;

    upBtn.addEventListener('click', () => {
      if (feedbackGiven) return;
      feedbackGiven = true;
      upBtn.classList.add('selected');
      downBtn.disabled = true;
      postFeedback(question, data.answer, 1);
    });

    downBtn.addEventListener('click', () => {
      if (feedbackGiven) return;
      feedbackGiven = true;
      downBtn.classList.add('selected');
      upBtn.disabled = true;
      postFeedback(question, data.answer, 0);
    });

    feedbackBtns.appendChild(upBtn);
    feedbackBtns.appendChild(downBtn);
    meta.appendChild(feedbackBtns);
    el.appendChild(meta);
  }

  getMessages().appendChild(el);
  scrollToBottom();
}

// ── Helpers ───────────────────────────────────────────

function makeBtn(label) {
  const btn = document.createElement('button');
  btn.className = 'feedback-btn';
  btn.textContent = label;
  return btn;
}

function postFeedback(question, answer, rating) {
  fetch('/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, answer, rating }),
  });
}

function formatText(text) {
  // Convert **bold** markers and split into paragraphs
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .split('\n')
    .map(line => line.trim())
    .filter(line => line.length > 0)
    .map(line => `<p>${line}</p>`)
    .join('');
}

function getMessages() {
  return document.getElementById('messages');
}

function scrollToBottom() {
  const msgs = getMessages();
  msgs.scrollTop = msgs.scrollHeight;
}

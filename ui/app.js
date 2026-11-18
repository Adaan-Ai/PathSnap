const form = document.querySelector('#predict-form');
const statusNode = document.querySelector('#status');
const resultNode = document.querySelector('#result');
const errorNode = document.querySelector('#error');
const button = document.querySelector('#submit');

async function checkHealth() {
  try {
    const response = await fetch('/health');
    if (!response.ok) throw new Error('API did not return ready');
    statusNode.textContent = 'API status: ready';
    statusNode.classList.add('ready');
  } catch {
    statusNode.textContent = 'API status: unavailable';
    statusNode.classList.add('failed');
  }
}

function showResult(data) {
  document.querySelector('#observations').textContent = data.observations.toLocaleString();
  document.querySelector('#matched').textContent = data.matched_observations.toLocaleString();
  document.querySelector('#coverage').textContent = `${(data.candidate_coverage * 100).toFixed(1)}%`;
  document.querySelector('#continuity').textContent = `${(data.route_continuity * 100).toFixed(1)}%`;
  const body = document.querySelector('#predictions');
  body.replaceChildren();
  for (const item of data.predictions) {
    const row = document.createElement('tr');
    const cells = [item.observation_index + 1, item.timestamp_s, item.arc_id ?? 'No match', item.candidate_arc_ids.join(', ') || '—'];
    for (const value of cells) {
      const cell = document.createElement('td');
      cell.textContent = String(value);
      row.append(cell);
    }
    body.append(row);
  }
  resultNode.hidden = false;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  errorNode.hidden = true;
  resultNode.hidden = true;
  button.disabled = true;
  button.textContent = 'Matching…';
  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ trip_id: new FormData(form).get('trip_id').trim() }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Prediction request failed');
    showResult(data);
  } catch (error) {
    errorNode.textContent = error.message;
    errorNode.hidden = false;
  } finally {
    button.disabled = false;
    button.textContent = 'Match route';
  }
});

checkHealth();

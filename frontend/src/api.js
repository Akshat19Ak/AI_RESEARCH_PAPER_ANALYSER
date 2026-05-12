/**
 * api.js — HTTP client for calling the FastAPI backend.
 * All API calls go through /api/* which Vite proxies to localhost:8000.
 * In production, set VITE_API_URL to the actual backend URL.
 */

const BASE = import.meta.env.VITE_API_URL || '/api';

async function request(endpoint, options = {}) {
  const url = `${BASE}${endpoint}`;
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || `API error ${res.status}`);
  return data;
}

export async function healthCheck() {
  return request('/health');
}

export async function configure(config) {
  return request('/configure', { method: 'POST', body: JSON.stringify(config) });
}

export async function uploadFile(file, apiKey = '') {
  const form = new FormData();
  form.append('file', file);
  form.append('api_key', apiKey);
  const url = `${BASE}/upload`;
  const res = await fetch(url, { method: 'POST', body: form });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'Upload failed');
  return data;
}

export async function uploadUrl(url, apiKey = '') {
  return request('/upload-url', {
    method: 'POST',
    body: JSON.stringify({ url, api_key: apiKey }),
  });
}

export async function generateSummary(sessionId, apiKey = '') {
  return request('/summary', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId, api_key: apiKey }),
  });
}

export async function chatAsk(sessionId, question, difficulty = 'Expert', apiKey = '') {
  return request('/chat', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId, question, difficulty, api_key: apiKey }),
  });
}

export async function getInsights(sessionId, apiKey = '') {
  return request('/insights', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId, api_key: apiKey }),
  });
}

export async function deepDive(sessionId, apiKey = '') {
  return request('/deepdive', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId, api_key: apiKey }),
  });
}

export async function interviewPrep(sessionId, apiKey = '') {
  return request('/interview', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId, api_key: apiKey }),
  });
}

export async function comparePapers(sessionId, paper1Summary, apiKey = '') {
  return request('/compare', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId, paper1_summary: paper1Summary, api_key: apiKey }),
  });
}

export async function generateFlowchart(sessionId, apiKey = '') {
  return request('/flowchart', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId, api_key: apiKey }),
  });
}

export async function generateInsights(sessionId, apiKey = '') {
  return request('/insights', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId, api_key: apiKey }),
  });
}

export async function generateInterview(sessionId, apiKey = '') {
  return request('/interview', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId, api_key: apiKey }),
  });
}

export async function getMetrics(sessionId) {
  return request(`/metrics/${sessionId}`);
}

export async function deleteSession(sessionId) {
  return request(`/session/${sessionId}`, { method: 'DELETE' });
}

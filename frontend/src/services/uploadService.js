

import api from './authService';

const baseUrl = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000/api';

export const uploadService = {
  async uploadThermalImages(files, metadata = {}) {
    const formData = new FormData();
    for (const file of files) {
      formData.append('files', file);
    }
    formData.append('ambient_temperature', metadata.ambientTemperature || '34.0');
    formData.append('notes', metadata.notes || 'React frontend upload');
    const response = await api.post('/upload/thermal-images', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: metadata.onProgress || undefined,
    });
    return response.data;
  },

  async presign(file, batchId) {
    const resp = await api.post('/upload/thermal-images/presign', {
      batch_id: batchId,
      filename: file.name,
      content_type: file.type || 'application/octet-stream',
    });
    return resp.data; // { provider, url, headers, key }
  },

  async putPresigned(url, headers, file, onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('PUT', url);
      for (const [k, v] of Object.entries(headers || {})) {
        xhr.setRequestHeader(k, v);
      }
      xhr.upload.onprogress = (e) => {
        if (onProgress && e.lengthComputable) {
          onProgress(Math.round((e.loaded * 100) / e.total));
        }
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) resolve(true);
        else reject(new Error(`Upload failed: ${xhr.status}`));
      };
      xhr.onerror = () => reject(new Error('Network error during upload'));
      xhr.send(file);
    });
  },

  async confirm(batchId, files, ambientTemperature = '34.0', notes = 'React presigned upload') {
    const payload = {
      batch_id: batchId,
      files: files.map((f) => ({
        filename: f.filename,
        key: f.key,
        content_type: f.contentType || 'application/octet-stream',
        size_bytes: f.sizeBytes || undefined,
      })),
      ambient_temperature: parseFloat(String(ambientTemperature)),
      notes,
    };
    const resp = await api.post('/upload/thermal-images/confirm', payload);
    return resp.data;
  },

  // Get batch status
  async getBatchStatus(batchId) {
    const response = await api.get(`/upload/batch/${batchId}/status`);
    return response.data;
  },

  // Get batch files
  async getBatchFiles(batchId) {
    const response = await api.get(`/upload/batch/${batchId}/files`);
    return response.data;
  },

  // Subscribe to batch progress via SSE
  subscribeBatchProgress(batchId, onEvent) {
    const token = localStorage.getItem('token');
    const url = `${baseUrl}/upload/batch/${batchId}/stream${token ? `?token=${encodeURIComponent(token)}` : ''}`;
    const es = new EventSource(url, { withCredentials: false });
    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (onEvent) onEvent(data);
      } catch (_) {}
    };
    es.onerror = () => {
      es.close();
    };
    return () => es.close();
  },

  // Download PDF by report id
  async downloadPdf(reportId) {
    const token = localStorage.getItem('token');
    const url = `${baseUrl}/reports/download/${reportId}.pdf`;
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const resp = await fetch(url, { headers });
    if (!resp.ok) {
      throw new Error(`Failed to download PDF: ${resp.status}`);
    }
    const blob = await resp.blob();
    const link = document.createElement('a');
    link.href = window.URL.createObjectURL(blob);
    link.download = `report_${reportId}.pdf`;
    document.body.appendChild(link);
    link.click();
    link.remove();
  },

  // Delete batch (admin only)
  async deleteBatch(batchId) {
    const response = await api.delete(`/upload/batch/${batchId}`);
    return response.data;
  },
}; 
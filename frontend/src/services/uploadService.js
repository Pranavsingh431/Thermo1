import api from './authService';

export const uploadService = {
  // Upload thermal images
  async uploadThermalImages(files, metadata = {}) {
    const formData = new FormData();
    
    // Add files
    for (const file of files) {
      formData.append('files', file);
    }
    
    // Add metadata
    formData.append('ambient_temperature', metadata.ambientTemperature || '34.0');
    formData.append('notes', metadata.notes || 'React frontend upload');
    
    const response = await api.post('/upload/thermal-images', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: metadata.onProgress || undefined,
    });
    
    return response.data;
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
    const baseUrl = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000/api';
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
    const baseUrl = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000/api';
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
  }
}; 
import api from './authService';

export const reportsService = {
  async generateReport(analysisId, { includePdf = true, includeLlm = true, format = 'comprehensive' } = {}) {
    const resp = await api.post(`/reports/generate/${analysisId}?format=${format}&include_pdf=${includePdf}&include_llm=${includeLlm}`);
    return resp.data;
  },

  async sendEmail(analysisId) {
    const resp = await api.post(`/reports/email/${analysisId}`);
    return resp.data;
  },

  async exportData(analysisId, fmt = 'json') {
    const resp = await api.get(`/reports/export/${analysisId}?format=${fmt}`);
    return resp.data;
  }
};

export default reportsService;


import api from './authService';

export const dashboardService = {
  // Get dashboard statistics
  async getStats() {
    const response = await api.get('/dashboard/stats');
    return response.data;
  },

  // Get recent analyses
  async getRecentAnalyses(limit = 10) {
    const response = await api.get(`/dashboard/recent-analyses?limit=${limit}`);
    return response.data;
  },

  // Get substations summary
  async getSubstations() {
    const response = await api.get('/dashboard/substations');
    return response.data;
  },

  // Get thermal scans
  async getThermalScans(params = {}) {
    const queryParams = new URLSearchParams(params).toString();
    const response = await api.get(`/dashboard/thermal-scans?${queryParams}`);
    return response.data;
  },

  // Task runs for observability
  async getTaskRuns(limit = 50, status, taskName) {
    const params = new URLSearchParams();
    params.set('limit', String(limit));
    if (status) params.set('status', status);
    if (taskName) params.set('task_name', taskName);
    const response = await api.get(`/tasks/task-runs?${params.toString()}`);
    return response.data;
  },

  // DLQ listing and requeue
  async getDlq(limit = 50) {
    const response = await api.get(`/tasks/dlq?limit=${limit}`);
    return response.data;
  },
  async requeueDlqById(taskId) {
    const response = await api.post(`/tasks/dlq/requeue`, { task_id: taskId });
    return response.data;
  },
  async requeueDlqByIndex(index) {
    const response = await api.post(`/tasks/dlq/requeue`, { index });
    return response.data;
  },

  // Get analysis detections
  async getAnalysisDetections(analysisId) {
    const response = await api.get(`/dashboard/analysis/${analysisId}/detections`);
    return response.data;
  }
}; 
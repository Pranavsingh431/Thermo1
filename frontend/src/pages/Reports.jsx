import React, { useEffect, useState } from 'react';
import { Typography, Card, Table, Tag, Button, Space, message } from 'antd';
import { dashboardService } from '../services/dashboardService';

const { Title } = Typography;

const Reports = () => {
  const [taskRuns, setTaskRuns] = useState([]);
  const [dlq, setDlq] = useState([]);
  const [loading, setLoading] = useState(false);

  const loadData = async () => {
    try {
      setLoading(true);
      const [runs, dlqItems] = await Promise.all([
        dashboardService.getTaskRuns(50),
        dashboardService.getDlq(50),
      ]);
      setTaskRuns(runs);
      setDlq(dlqItems);
    } catch (e) {
      message.error('Failed to load task runs / DLQ');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
    // auto-refresh every 10s
    const t = setInterval(loadData, 10000);
    return () => clearInterval(t);
  }, []);

  const columnsRuns = [
    { title: 'Task', dataIndex: 'task_name', key: 'task_name' },
    { title: 'Task ID', dataIndex: 'task_id', key: 'task_id' },
    { title: 'Status', dataIndex: 'status', key: 'status', render: (s) => <Tag>{s}</Tag> },
    { title: 'Started', dataIndex: 'started_at', key: 'started_at' },
    { title: 'Finished', dataIndex: 'finished_at', key: 'finished_at' },
    { title: 'Duration(s)', dataIndex: 'duration_seconds', key: 'duration_seconds' },
    { title: 'Worker', dataIndex: 'worker_hostname', key: 'worker_hostname' },
  ];

  const columnsDlq = [
    { title: 'Task', dataIndex: 'task', key: 'task' },
    { title: 'Task ID', dataIndex: 'task_id', key: 'task_id' },
    { title: 'Error', dataIndex: 'error', key: 'error', ellipsis: true },
    { title: 'When', dataIndex: 'when', key: 'when' },
    { title: 'Actions', key: 'actions', render: (_, rec, idx) => (
      <Space>
        <Button size="small" onClick={async () => {
          try {
            await dashboardService.requeueDlqById(rec.task_id);
            message.success('Requeued');
            loadData();
          } catch (e) {
            message.error('Requeue failed');
          }
        }}>Requeue</Button>
      </Space>
    ) },
  ];

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>📊 Reports & Operations</Title>
      <Card style={{ marginBottom: 16 }} title="Background Task Runs">
        <Table rowKey={(r) => `${r.task_id}-${r.started_at}`} loading={loading} columns={columnsRuns} dataSource={taskRuns} pagination={{ pageSize: 10 }} />
      </Card>
      <Card title="Dead Letter Queue (Failures)">
        <Table rowKey={(r, i) => r.task_id || String(i)} loading={loading} columns={columnsDlq} dataSource={dlq} pagination={{ pageSize: 10 }} />
      </Card>
    </div>
  );
};

export default Reports; 
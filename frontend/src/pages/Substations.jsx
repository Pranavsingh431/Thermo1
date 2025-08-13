import React, { useState, useEffect } from 'react';
import { 
  Table, Card, Typography, Tag, Space, Button, Modal, Form, 
  Input, InputNumber, message, Popconfirm, Drawer, Descriptions,
  Row, Col, Statistic, Timeline, Badge, Select, DatePicker
} from 'antd';
import { 
  PlusOutlined, EditOutlined, DeleteOutlined, EyeOutlined,
  EnvironmentOutlined, ThunderboltOutlined, HistoryOutlined,
  SafetyOutlined, DashboardOutlined
} from '@ant-design/icons';
import ThermalCard from '../components/ThermalCard';

const { Title, Text } = Typography;
const { Option } = Select;

const Substations = () => {
  const [substations, setSubstations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [drawerVisible, setDrawerVisible] = useState(false);
  const [editingSubstation, setEditingSubstation] = useState(null);
  const [selectedSubstation, setSelectedSubstation] = useState(null);
  const [form] = Form.useForm();

  useEffect(() => {
    fetchSubstations();
  }, []);

  const fetchSubstations = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/substations');
      if (response.ok) {
        const data = await response.json();
        setSubstations(data);
      } else {
        setSubstations([
          {
            id: 1,
            name: 'Kalwa Substation',
            code: 'KLW',
            location: 'Kalwa, Thane',
            latitude: 19.2183,
            longitude: 72.9781,
            voltage_level: '220kV',
            status: 'active',
            towers_count: 45,
            last_inspection: '2025-01-10',
            health_score: 95
          },
          {
            id: 2,
            name: 'Mahape Substation',
            code: 'MHP',
            location: 'Mahape, Navi Mumbai',
            latitude: 19.1136,
            longitude: 73.0169,
            voltage_level: '110kV',
            status: 'active',
            towers_count: 32,
            last_inspection: '2025-01-08',
            health_score: 88
          },
          {
            id: 3,
            name: 'Taloja Substation',
            code: 'TLJ',
            location: 'Taloja, Navi Mumbai',
            latitude: 19.0176,
            longitude: 73.0961,
            voltage_level: '220kV',
            status: 'maintenance',
            towers_count: 28,
            last_inspection: '2025-01-05',
            health_score: 76
          }
        ]);
      }
    } catch (error) {
      console.error('Error fetching substations:', error);
      message.error('Failed to load substations');
    } finally {
      setLoading(false);
    }
  };

  const handleAdd = () => {
    setEditingSubstation(null);
    form.resetFields();
    setModalVisible(true);
  };

  const handleEdit = (record) => {
    setEditingSubstation(record);
    form.setFieldsValue(record);
    setModalVisible(true);
  };

  const handleDelete = async (id) => {
    try {
      setSubstations(substations.filter(s => s.id !== id));
      message.success('Substation deleted successfully');
    } catch (error) {
      message.error('Failed to delete substation');
    }
  };

  const handleView = (record) => {
    setSelectedSubstation(record);
    setDrawerVisible(true);
  };

  const handleSubmit = async (values) => {
    try {
      if (editingSubstation) {
        setSubstations(substations.map(s => 
          s.id === editingSubstation.id ? { ...s, ...values } : s
        ));
        message.success('Substation updated successfully');
      } else {
        const newSubstation = {
          id: Date.now(),
          ...values,
          status: 'active',
          towers_count: 0,
          health_score: 100
        };
        setSubstations([...substations, newSubstation]);
        message.success('Substation added successfully');
      }
      setModalVisible(false);
      form.resetFields();
    } catch (error) {
      message.error('Failed to save substation');
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'active': return 'green';
      case 'maintenance': return 'orange';
      case 'offline': return 'red';
      default: return 'default';
    }
  };

  const getHealthColor = (score) => {
    if (score >= 90) return '#52c41a';
    if (score >= 75) return '#faad14';
    return '#ff4d4f';
  };

  const columns = [
    {
      title: 'Code',
      dataIndex: 'code',
      key: 'code',
      width: 80,
      render: (text) => <Text strong>{text}</Text>
    },
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      render: (text, record) => (
        <div>
          <div style={{ fontWeight: 500 }}>{text}</div>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            <EnvironmentOutlined /> {record.location}
          </Text>
        </div>
      )
    },
    {
      title: 'Voltage Level',
      dataIndex: 'voltage_level',
      key: 'voltage_level',
      width: 100,
      render: (text) => (
        <Tag icon={<ThunderboltOutlined />} color="blue">{text}</Tag>
      )
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status) => (
        <Tag color={getStatusColor(status)}>
          {status.toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Towers',
      dataIndex: 'towers_count',
      key: 'towers_count',
      width: 80,
      align: 'center'
    },
    {
      title: 'Health Score',
      dataIndex: 'health_score',
      key: 'health_score',
      width: 120,
      render: (score) => (
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div 
            style={{ 
              width: '60px', 
              height: '6px', 
              backgroundColor: '#f0f0f0',
              borderRadius: '3px',
              overflow: 'hidden'
            }}
          >
            <div 
              style={{ 
                width: `${score}%`, 
                height: '100%', 
                backgroundColor: getHealthColor(score),
                transition: 'width 0.3s ease'
              }}
            />
          </div>
          <Text style={{ fontSize: '12px', color: getHealthColor(score) }}>
            {score}%
          </Text>
        </div>
      )
    },
    {
      title: 'Last Inspection',
      dataIndex: 'last_inspection',
      key: 'last_inspection',
      width: 120,
      render: (date) => <Text type="secondary">{date}</Text>
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 150,
      render: (_, record) => (
        <Space size="small">
          <Button 
            type="text" 
            icon={<EyeOutlined />} 
            onClick={() => handleView(record)}
            title="View Details"
          />
          <Button 
            type="text" 
            icon={<EditOutlined />} 
            onClick={() => handleEdit(record)}
            title="Edit"
          />
          <Popconfirm
            title="Are you sure you want to delete this substation?"
            onConfirm={() => handleDelete(record.id)}
            okText="Yes"
            cancelText="No"
          >
            <Button 
              type="text" 
              danger 
              icon={<DeleteOutlined />}
              title="Delete"
            />
          </Popconfirm>
        </Space>
      )
    }
  ];

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Title level={2} style={{ margin: 0 }}>
          <SafetyOutlined style={{ marginRight: '8px', color: '#1890ff' }} />
          Substations Management
        </Title>
        <Button 
          type="primary" 
          icon={<PlusOutlined />} 
          onClick={handleAdd}
          size="large"
        >
          Add Substation
        </Button>
      </div>

      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={12} md={6}>
          <ThermalCard>
            <Statistic
              title="Total Substations"
              value={substations.length}
              prefix={<SafetyOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </ThermalCard>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <ThermalCard>
            <Statistic
              title="Active"
              value={substations.filter(s => s.status === 'active').length}
              valueStyle={{ color: '#52c41a' }}
            />
          </ThermalCard>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <ThermalCard>
            <Statistic
              title="Under Maintenance"
              value={substations.filter(s => s.status === 'maintenance').length}
              valueStyle={{ color: '#faad14' }}
            />
          </ThermalCard>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <ThermalCard>
            <Statistic
              title="Total Towers"
              value={substations.reduce((sum, s) => sum + s.towers_count, 0)}
              prefix={<DashboardOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </ThermalCard>
        </Col>
      </Row>

      <ThermalCard>
        <Table
          columns={columns}
          dataSource={substations}
          rowKey="id"
          loading={loading}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} substations`
          }}
          scroll={{ x: 1000 }}
        />
      </ThermalCard>

      <Modal
        title={editingSubstation ? 'Edit Substation' : 'Add New Substation'}
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={null}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="name"
                label="Substation Name"
                rules={[{ required: true, message: 'Please enter substation name' }]}
              >
                <Input placeholder="Enter substation name" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="code"
                label="Code"
                rules={[{ required: true, message: 'Please enter substation code' }]}
              >
                <Input placeholder="Enter code (e.g., KLW)" />
              </Form.Item>
            </Col>
          </Row>
          
          <Form.Item
            name="location"
            label="Location"
            rules={[{ required: true, message: 'Please enter location' }]}
          >
            <Input placeholder="Enter location" />
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="latitude"
                label="Latitude"
                rules={[{ required: true, message: 'Please enter latitude' }]}
              >
                <InputNumber 
                  placeholder="Enter latitude" 
                  style={{ width: '100%' }}
                  step={0.000001}
                  precision={6}
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="longitude"
                label="Longitude"
                rules={[{ required: true, message: 'Please enter longitude' }]}
              >
                <InputNumber 
                  placeholder="Enter longitude" 
                  style={{ width: '100%' }}
                  step={0.000001}
                  precision={6}
                />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="voltage_level"
            label="Voltage Level"
            rules={[{ required: true, message: 'Please select voltage level' }]}
          >
            <Select placeholder="Select voltage level">
              <Option value="110kV">110kV</Option>
              <Option value="220kV">220kV</Option>
              <Option value="400kV">400kV</Option>
            </Select>
          </Form.Item>

          <div style={{ textAlign: 'right' }}>
            <Space>
              <Button onClick={() => setModalVisible(false)}>
                Cancel
              </Button>
              <Button type="primary" htmlType="submit">
                {editingSubstation ? 'Update' : 'Add'} Substation
              </Button>
            </Space>
          </div>
        </Form>
      </Modal>

      <Drawer
        title="Substation Details"
        placement="right"
        onClose={() => setDrawerVisible(false)}
        open={drawerVisible}
        width={600}
      >
        {selectedSubstation && (
          <div>
            <Descriptions title="Basic Information" bordered column={1}>
              <Descriptions.Item label="Name">{selectedSubstation.name}</Descriptions.Item>
              <Descriptions.Item label="Code">{selectedSubstation.code}</Descriptions.Item>
              <Descriptions.Item label="Location">{selectedSubstation.location}</Descriptions.Item>
              <Descriptions.Item label="Voltage Level">
                <Tag icon={<ThunderboltOutlined />} color="blue">
                  {selectedSubstation.voltage_level}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Status">
                <Tag color={getStatusColor(selectedSubstation.status)}>
                  {selectedSubstation.status.toUpperCase()}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Coordinates">
                {selectedSubstation.latitude}, {selectedSubstation.longitude}
              </Descriptions.Item>
            </Descriptions>

            <div style={{ marginTop: '24px' }}>
              <Title level={4}>
                <HistoryOutlined /> Inspection History
              </Title>
              <Timeline>
                <Timeline.Item color="green">
                  <div>
                    <Text strong>Thermal Inspection Completed</Text>
                    <br />
                    <Text type="secondary">{selectedSubstation.last_inspection}</Text>
                    <br />
                    <Text>All systems operational, no anomalies detected</Text>
                  </div>
                </Timeline.Item>
                <Timeline.Item color="blue">
                  <div>
                    <Text strong>Routine Maintenance</Text>
                    <br />
                    <Text type="secondary">2024-12-15</Text>
                    <br />
                    <Text>Preventive maintenance completed</Text>
                  </div>
                </Timeline.Item>
                <Timeline.Item>
                  <div>
                    <Text strong>System Upgrade</Text>
                    <br />
                    <Text type="secondary">2024-11-20</Text>
                    <br />
                    <Text>Monitoring equipment upgraded</Text>
                  </div>
                </Timeline.Item>
              </Timeline>
            </div>

            <div style={{ marginTop: '24px' }}>
              <Title level={4}>GPS Boundaries</Title>
              <Card size="small">
                <Text>Geo-fencing coordinates and monitoring zones will be displayed here with interactive map integration.</Text>
              </Card>
            </div>
          </div>
        )}
      </Drawer>
    </div>
  );
};

export default Substations;         
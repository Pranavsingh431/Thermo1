import React from 'react';
import { Card } from 'antd';

const ThermalCard = ({ children, ...props }) => {
  return (
    <Card 
      className="thermal-card"
      {...props}
    >
      {children}
    </Card>
  );
};

export default ThermalCard;

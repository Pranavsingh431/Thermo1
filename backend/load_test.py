from locust import HttpUser, task, between
import random
import os
import json
import base64
from io import BytesIO
from PIL import Image

class ThermalInspectionUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Initialize user session and login"""
        self.login()
        self.auth_headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
    
    def login(self):
        """Authenticate user and get access token"""
        login_data = {
            "username": random.choice(["admin", "engineer", "operator"]),
            "password": random.choice(["admin123", "engineer123", "operator123"])
        }
        
        response = self.client.post("/api/auth/login", json=login_data)
        if response.status_code == 200:
            self.access_token = response.json().get("access_token")
        else:
            self.access_token = "dummy_token"
    
    @task(3)
    def upload_thermal_image(self):
        """Test thermal image upload and processing"""
        try:
            thermal_image = self._generate_mock_thermal_image()
            
            files = {
                'file': ('thermal_test.jpg', thermal_image, 'image/jpeg')
            }
            
            metadata = {
                'substation_id': random.randint(1, 6),
                'tower_id': f'T{random.randint(1, 50):03d}',
                'ambient_temperature': round(random.uniform(25.0, 40.0), 1),
                'weather_condition': random.choice(['clear', 'cloudy', 'humid']),
                'line_selection': random.choice(['incoming', 'outgoing']),
                'phase_selection': random.choice(['R', 'Y', 'B'])
            }
            
            response = self.client.post(
                "/api/upload/thermal-image",
                files=files,
                data=metadata,
                headers={'Authorization': f'Bearer {self.access_token}'}
            )
            
            if response.status_code == 200:
                upload_result = response.json()
                image_id = upload_result.get('image_id')
                if image_id:
                    self._check_processing_status(image_id)
            
        except Exception as e:
            print(f"Upload error: {e}")
    
    @task(2)
    def view_dashboard(self):
        """Test dashboard data loading"""
        try:
            response = self.client.get("/api/dashboard/stats", headers=self.auth_headers)
            if response.status_code == 200:
                stats = response.json()
                
                response = self.client.get("/api/dashboard/recent-scans", headers=self.auth_headers)
                
        except Exception as e:
            print(f"Dashboard error: {e}")
    
    @task(2)
    def view_thermal_scans(self):
        """Test thermal scans listing and filtering"""
        try:
            params = {
                'page': random.randint(1, 3),
                'limit': random.choice([10, 20, 50]),
                'substation_id': random.choice([None, 1, 2, 3]),
                'status': random.choice([None, 'completed', 'processing'])
            }
            
            clean_params = {k: v for k, v in params.items() if v is not None}
            
            response = self.client.get("/api/thermal-scans", params=clean_params, headers=self.auth_headers)
            
        except Exception as e:
            print(f"Thermal scans error: {e}")
    
    @task(1)
    def generate_report(self):
        """Test report generation"""
        try:
            report_data = {
                'substation_id': random.randint(1, 6),
                'date_range': {
                    'start': '2025-01-01',
                    'end': '2025-01-13'
                },
                'report_type': random.choice(['thermal_analysis', 'monthly_inspection', 'performance'])
            }
            
            response = self.client.post("/api/reports/generate", json=report_data, headers=self.auth_headers)
            
            if response.status_code == 200:
                report_result = response.json()
                report_id = report_result.get('report_id')
                if report_id:
                    self._check_report_status(report_id)
            
        except Exception as e:
            print(f"Report generation error: {e}")
    
    @task(1)
    def view_ai_results(self):
        """Test AI analysis results viewing"""
        try:
            params = {
                'page': random.randint(1, 3),
                'limit': 20,
                'status': random.choice([None, 'normal', 'warning', 'critical'])
            }
            
            clean_params = {k: v for k, v in params.items() if v is not None}
            
            response = self.client.get("/api/ai-results", params=clean_params, headers=self.auth_headers)
            
        except Exception as e:
            print(f"AI results error: {e}")
    
    @task(1)
    def view_substations(self):
        """Test substations management"""
        try:
            response = self.client.get("/api/substations", headers=self.auth_headers)
            
            if response.status_code == 200:
                substations = response.json()
                if substations:
                    substation_id = random.choice(substations)['id']
                    response = self.client.get(f"/api/substations/{substation_id}", headers=self.auth_headers)
            
        except Exception as e:
            print(f"Substations error: {e}")
    
    @task(1)
    def health_check(self):
        """Test system health endpoints"""
        try:
            response = self.client.get("/api/health")
            
            response = self.client.get("/api/health/detailed", headers=self.auth_headers)
            
        except Exception as e:
            print(f"Health check error: {e}")
    
    def _generate_mock_thermal_image(self):
        """Generate a mock thermal image for testing"""
        try:
            img = Image.new('RGB', (640, 480), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            
            img_buffer = BytesIO()
            img.save(img_buffer, format='JPEG', quality=85)
            img_buffer.seek(0)
            
            return img_buffer
            
        except Exception as e:
            print(f"Mock image generation error: {e}")
            return BytesIO(b"mock_image_data")
    
    def _check_processing_status(self, image_id):
        """Check image processing status"""
        try:
            response = self.client.get(f"/api/upload/status/{image_id}", headers=self.auth_headers)
            
        except Exception as e:
            print(f"Status check error: {e}")
    
    def _check_report_status(self, report_id):
        """Check report generation status"""
        try:
            response = self.client.get(f"/api/reports/status/{report_id}", headers=self.auth_headers)
            
        except Exception as e:
            print(f"Report status error: {e}")

class AdminUser(ThermalInspectionUser):
    """Admin user with additional privileges"""
    weight = 1
    
    def on_start(self):
        self.login_as_admin()
        self.auth_headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
    
    def login_as_admin(self):
        login_data = {"username": "admin", "password": "admin123"}
        response = self.client.post("/api/auth/login", json=login_data)
        if response.status_code == 200:
            self.access_token = response.json().get("access_token")
        else:
            self.access_token = "dummy_admin_token"
    
    @task(1)
    def manage_users(self):
        """Test user management endpoints"""
        try:
            response = self.client.get("/api/admin/users", headers=self.auth_headers)
            
        except Exception as e:
            print(f"User management error: {e}")
    
    @task(1)
    def system_monitoring(self):
        """Test system monitoring endpoints"""
        try:
            response = self.client.get("/api/admin/system-stats", headers=self.auth_headers)
            
            response = self.client.get("/api/admin/performance-metrics", headers=self.auth_headers)
            
        except Exception as e:
            print(f"System monitoring error: {e}")

class EngineerUser(ThermalInspectionUser):
    """Engineer user focused on analysis tasks"""
    weight = 3
    
    def on_start(self):
        self.login_as_engineer()
        self.auth_headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
    
    def login_as_engineer(self):
        login_data = {"username": "engineer", "password": "engineer123"}
        response = self.client.post("/api/auth/login", json=login_data)
        if response.status_code == 200:
            self.access_token = response.json().get("access_token")
        else:
            self.access_token = "dummy_engineer_token"
    
    @task(4)
    def detailed_analysis(self):
        """Perform detailed thermal analysis"""
        try:
            response = self.client.get("/api/analysis/detailed", headers=self.auth_headers)
            
        except Exception as e:
            print(f"Detailed analysis error: {e}")

class OperatorUser(ThermalInspectionUser):
    """Operator user focused on routine operations"""
    weight = 6
    
    def on_start(self):
        self.login_as_operator()
        self.auth_headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
    
    def login_as_operator(self):
        login_data = {"username": "operator", "password": "operator123"}
        response = self.client.post("/api/auth/login", json=login_data)
        if response.status_code == 200:
            self.access_token = response.json().get("access_token")
        else:
            self.access_token = "dummy_operator_token"

if __name__ == "__main__":
    print("Load testing configuration:")
    print("- Admin users: 10% (system management)")
    print("- Engineer users: 30% (detailed analysis)")
    print("- Operator users: 60% (routine operations)")
    print("- Target: 1000+ concurrent users")
    print("- Duration: 10+ minutes")
    print("- Focus: Upload, analysis, reporting workflows")

import os
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_thermal_inspection.db")

from app.database import Base, SessionLocal, engine  # noqa: E402
from app.models.ai_analysis import AIAnalysis  # noqa: E402
from app.models.thermal_scan import ThermalScan  # noqa: E402
from app.models.user import User  # noqa: E402


@pytest.fixture(scope="function", autouse=True)
def setup_database():
    """Create tables before each test and clean up after."""
    from sqlalchemy import event

    def enable_foreign_keys(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    event.listen(engine, "connect", enable_foreign_keys)

    # Create all tables
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_unique_file_hash_enforced():
    session = SessionLocal()
    try:
        # Create a user to satisfy FK
        suffix = uuid.uuid4().hex[:8]
        u = User(
            username=f"tmp_u1_{suffix}",
            email=f"tmp_u1_{suffix}@example.com",
            full_name="Tmp U1",
            role="engineer"
        )
        u.set_password("pass")
        session.add(u)
        session.commit()
        session.refresh(u)

        common_hash = (uuid.uuid4().hex + uuid.uuid4().hex)[:64]
        s1 = ThermalScan(
            original_filename="a.jpg",
            file_path="/tmp/a.jpg",
            file_size_bytes=1,
            file_hash=common_hash,
            capture_timestamp=text("datetime('now')"),
            uploaded_by=u.id,
        )
        session.add(s1)
        session.commit()

        s2 = ThermalScan(
            original_filename="b.jpg",
            file_path="/tmp/b.jpg",
            file_size_bytes=1,
            file_hash=common_hash,
            capture_timestamp=text("datetime('now')"),
            uploaded_by=u.id,
        )
        session.add(s2)
        with pytest.raises(IntegrityError):
            session.commit()
    finally:
        session.rollback()
        session.close()


def test_uploaded_by_fk_enforced():
    session = SessionLocal()
    try:
        s = ThermalScan(
            original_filename="c.jpg",
            file_path="/tmp/c.jpg",
            file_size_bytes=1,
            file_hash="cafebabe" * 8,
            capture_timestamp=text("datetime('now')"),
            uploaded_by=999999,  # non-existent
        )
        session.add(s)
        with pytest.raises(IntegrityError):
            session.commit()
    finally:
        session.rollback()
        session.close()


def test_ai_analysis_fk_cascade_delete():
    session = SessionLocal()
    try:
        # Create user and scan
        suffix = uuid.uuid4().hex[:8]
        u = User(
            username=f"tmp_u2_{suffix}",
            email=f"tmp_u2_{suffix}@example.com",
            full_name="Tmp U2",
            role="engineer"
        )
        u.set_password("pass")
        session.add(u)
        session.commit()
        session.refresh(u)

        s = ThermalScan(
            original_filename="d.jpg",
            file_path="/tmp/d.jpg",
            file_size_bytes=1,
            file_hash=(uuid.uuid4().hex + uuid.uuid4().hex)[:64],
            capture_timestamp=text("datetime('now')"),
            uploaded_by=u.id,
        )
        session.add(s)
        session.commit()
        session.refresh(s)

        a = AIAnalysis(
            thermal_scan_id=s.id,
            model_version="test",
            analysis_status="completed",
            processing_duration_seconds=0.1,
            is_good_quality=True,
            quality_score=1.0,
            ambient_temperature=34.0,
            max_temperature_detected=40.0,
            min_temperature_detected=30.0,
            avg_temperature=35.0,
            temperature_variance=1.0,
            total_hotspots=0,
            critical_hotspots=0,
            potential_hotspots=0,
            normal_zones=10,
            total_components_detected=0,
            nuts_bolts_count=0,
            mid_span_joints_count=0,
            polymer_insulators_count=0,
            overall_risk_level="low",
            risk_score=0.0,
            requires_immediate_attention=False,
            summary_text="ok",
        )
        session.add(a)
        session.commit()

        analysis_id = a.id

        # Delete scan; cascade should remove analysis
        session.delete(s)
        session.commit()

        remaining = session.query(AIAnalysis).filter(
            AIAnalysis.id == analysis_id
        ).first()
        assert remaining is None
    finally:
        session.rollback()
        session.close()

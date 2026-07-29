-- Database initialization script for PostgreSQL

-- Create Table: telemetry_logs
CREATE TABLE IF NOT EXISTS telemetry_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT (now() at time zone 'utc') NOT NULL,
    unit_number INTEGER NOT NULL,
    time_in_cycles INTEGER NOT NULL,
    op_setting_1 DOUBLE PRECISION,
    op_setting_2 DOUBLE PRECISION,
    op_setting_3 DOUBLE PRECISION,
    sensor_1 DOUBLE PRECISION,
    sensor_2 DOUBLE PRECISION,
    sensor_3 DOUBLE PRECISION,
    sensor_4 DOUBLE PRECISION,
    sensor_5 DOUBLE PRECISION,
    sensor_6 DOUBLE PRECISION,
    sensor_7 DOUBLE PRECISION,
    sensor_8 DOUBLE PRECISION,
    sensor_9 DOUBLE PRECISION,
    sensor_10 DOUBLE PRECISION,
    sensor_11 DOUBLE PRECISION,
    sensor_12 DOUBLE PRECISION,
    sensor_13 DOUBLE PRECISION,
    sensor_14 DOUBLE PRECISION,
    sensor_15 DOUBLE PRECISION,
    sensor_16 DOUBLE PRECISION,
    sensor_17 DOUBLE PRECISION,
    sensor_18 DOUBLE PRECISION,
    sensor_19 DOUBLE PRECISION,
    sensor_20 DOUBLE PRECISION,
    sensor_21 DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_telemetry_unit ON telemetry_logs(unit_number);
CREATE INDEX IF NOT EXISTS idx_telemetry_cycles ON telemetry_logs(time_in_cycles);
CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry_logs(timestamp);

-- Create Table: prediction_logs
CREATE TABLE IF NOT EXISTS prediction_logs (
    id SERIAL PRIMARY KEY,
    telemetry_id INTEGER REFERENCES telemetry_logs(id) ON DELETE CASCADE,
    unit_number INTEGER NOT NULL,
    failure_probability DOUBLE PRECISION NOT NULL,
    predicted_label INTEGER NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT (now() at time zone 'utc') NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_prediction_unit ON prediction_logs(unit_number);
CREATE INDEX IF NOT EXISTS idx_prediction_prob ON prediction_logs(failure_probability);
CREATE INDEX IF NOT EXISTS idx_prediction_timestamp ON prediction_logs(timestamp);

-- Create Table: alerts
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT (now() at time zone 'utc') NOT NULL,
    unit_number INTEGER NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    resolved BOOLEAN DEFAULT FALSE NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_alerts_unit ON alerts(unit_number);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON alerts(resolved);
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);

-- Create Views for Power BI Integration
CREATE OR REPLACE VIEW view_machine_health_summary AS
SELECT 
    t.unit_number,
    MAX(t.time_in_cycles) as total_cycles,
    AVG(p.failure_probability) as avg_failure_probability,
    (
        SELECT p2.failure_probability 
        FROM prediction_logs p2 
        WHERE p2.unit_number = t.unit_number 
        ORDER BY p2.timestamp DESC LIMIT 1
    ) as current_failure_probability,
    COUNT(CASE WHEN a.resolved = FALSE THEN 1 END) as open_alerts_count
FROM telemetry_logs t
LEFT JOIN prediction_logs p ON t.id = p.telemetry_id
LEFT JOIN alerts a ON t.unit_number = a.unit_number
GROUP BY t.unit_number;

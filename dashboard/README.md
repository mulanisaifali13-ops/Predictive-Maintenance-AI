# Module 6: Power BI Integration Guide

This directory documents how to connect **Power BI** to the PostgreSQL database of the AI Predictive Maintenance Platform, and provides the SQL analytical views and DAX formulas used to generate the report.

---

## 1. Database Connection Settings

Power BI Desktop connects natively to **PostgreSQL**. Use the following credentials when running locally via Docker Compose:

- **Data Source Type**: PostgreSQL database
- **Server**: `localhost:5432` (or the IP of your AWS EC2 instance / Render URL)
- **Database**: `predictive_maintenance`
- **Username**: `postgres`
- **Password**: `postgres_password`
- **SSL Mode**: Prefer / Disable (depending on certificates; local docker works with Disable/Prefer)

---

## 2. Recommended Database Views

Create the following views in PostgreSQL to optimize reporting performance and keep Power BI data loading simple.

### View 1: Machine Health Summary (`view_machine_health_summary`)
Provides a single row per machine with current failure risk and cycle totals:
```sql
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
```

---

## 3. Key DAX Calculations

Create these measures in Power BI under your tables to calculate key performance indicators (KPIs):

### Measure 1: System Health Index (%)
Calculates the percentage of equipment operating below the critical failure threshold:
```dax
System Health Index = 
VAR TotalMachines = DISTINCTCOUNT(view_machine_health_summary[unit_number])
VAR CriticalMachines = CALCULATE(
    DISTINCTCOUNT(view_machine_health_summary[unit_number]),
    view_machine_health_summary[current_failure_probability] >= 0.70
)
RETURN
IF(TotalMachines > 0, (TotalMachines - CriticalMachines) / TotalMachines * 100, 100)
```

### Measure 2: Mean Time Between Failures (MTBF)
Calculates the average cycles completed by machines between failure events:
```dax
MTBF (Cycles) = 
VAR TotalCycles = SUM(view_machine_health_summary[total_cycles])
VAR TotalFailures = COUNTROWS(FILTER(view_machine_health_summary, view_machine_health_summary[current_failure_probability] >= 0.70))
RETURN
IF(TotalFailures > 0, TotalCycles / TotalFailures, TotalCycles)
```

### Measure 3: Overall Equipment Effectiveness (OEE) - Simplified
OEE combines Availability, Performance, and Quality:
```dax
Availability Rate = 
VAR ScheduledCycles = 300
VAR TotalCycles = AVERAGE(view_machine_health_summary[total_cycles])
RETURN 
DIVIDE(TotalCycles, ScheduledCycles, 0)

OEE (%) = 
[Availability Rate] * 0.95 * 0.99 * 100
```

---

## 4. Visual Dashboard Mockup Recommendations
We recommend setting up a 3-tab report in Power BI:
1. **Overview Dashboard**: High-level KPIs (System Health Index, Open Alerts, MTBF) and a list of assets colored by status (Red: Critical, Orange: Warning, Green: Normal).
2. **Telemetry Deep-Dive**: Line charts showing sensor trends (Sensor 2, 4, 11, and 12) grouped by `unit_number` over `time_in_cycles`, allowing engineers to trace actual degradations.
3. **Model Monitoring**: Histograms of failure probability distribution and confusion metrics tracked via MLflow API integration (Power BI Python scripts can fetch MLflow parameters directly).

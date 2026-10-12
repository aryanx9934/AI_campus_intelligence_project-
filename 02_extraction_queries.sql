-- ============================================================
-- 02_extraction_queries.sql
-- SQL layer of the AI-Powered Campus Intelligence Dashboard.
-- Run against campus_intelligence.db (SQLite).
-- These are the queries you'd point to in an interview when asked
-- "walk me through your SQL extraction step."
-- ============================================================

-- 1. Basic sanity check: row count and duplicate detection
SELECT COUNT(*) AS total_rows FROM students_raw;

SELECT student_id, COUNT(*) AS occurrences
FROM students_raw
GROUP BY student_id
HAVING COUNT(*) > 1;

-- 2. Extract a clean base table: valid records only
--    (drops rows with impossible values, e.g. negative internship_count)
SELECT *
FROM students_raw
WHERE internship_count >= 0
  AND cgpa BETWEEN 0 AND 10;

-- 3. Department-level performance summary (feeds the dashboard KPI cards)
SELECT
    department,
    COUNT(*) AS num_students,
    ROUND(AVG(cgpa), 2) AS avg_cgpa,
    ROUND(AVG(attendance_pct), 1) AS avg_attendance,
    SUM(at_risk) AS at_risk_count,
    ROUND(100.0 * SUM(placed) / COUNT(*), 1) AS placement_rate_pct
FROM students_raw
WHERE internship_count >= 0
GROUP BY department
ORDER BY placement_rate_pct DESC;

-- 4. At-risk students requiring intervention (feeds the "watchlist" view)
SELECT
    student_id,
    department,
    semester,
    cgpa,
    attendance_pct,
    backlog_count
FROM students_raw
WHERE at_risk = 1
  AND internship_count >= 0
ORDER BY cgpa ASC, attendance_pct ASC;

-- 5. Placement funnel by academic band (feeds the placement analytics view)
SELECT
    CASE
        WHEN cgpa >= 8.5 THEN 'A: 8.5-10'
        WHEN cgpa >= 7.0 THEN 'B: 7.0-8.49'
        WHEN cgpa >= 5.5 THEN 'C: 5.5-6.99'
        ELSE 'D: <5.5'
    END AS cgpa_band,
    COUNT(*) AS num_students,
    SUM(placed) AS placed_count,
    ROUND(100.0 * SUM(placed) / COUNT(*), 1) AS placement_rate_pct
FROM students_raw
WHERE internship_count >= 0
GROUP BY cgpa_band
ORDER BY cgpa_band;

-- 6. First-generation learner outcomes (equity-lens reporting)
SELECT
    first_gen_learner,
    COUNT(*) AS num_students,
    ROUND(AVG(cgpa), 2) AS avg_cgpa,
    ROUND(100.0 * SUM(placed) / COUNT(*), 1) AS placement_rate_pct,
    ROUND(100.0 * SUM(at_risk) / COUNT(*), 1) AS at_risk_rate_pct
FROM students_raw
WHERE internship_count >= 0
GROUP BY first_gen_learner;

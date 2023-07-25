-- The view need_meeting should return all students name when:
-- The scores are under (strict) to 80
-- AND no last_meeting date OR more than a month

DROP VIEW IF EXISTS need_meeting;
CREATE VIEW need_meeting AS
       SELECT name
       FROM students
       WHERE score < 80 AND (last_meeting IS NULL OR DATEDIFF(CURDATE(), last_meeting) > 30);

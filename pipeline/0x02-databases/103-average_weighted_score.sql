-- Procedure ComputeAverageScoreForUser is taking 1 input:
-- user_id, a users.id value
DELIMITER //

CREATE PROCEDURE ComputeAverageWeightedScoreForUser (IN user_id_new INTEGER)
BEGIN
	UPDATE users SET average_score=(
	SELECT SUM(score * weight) / SUM(weight) FROM corrections
	JOIN projects
	ON corrections.project_id=projects.id
	WHERE user_id=user_id_new);
END; //
DELIMITER ;

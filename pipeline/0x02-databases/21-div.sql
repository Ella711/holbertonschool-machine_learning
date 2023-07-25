-- creates a function SafeDiv
-- takes 2 arguments:
-- a, INT
-- b, INT
-- And returns a / b or 0 if b == 0
CREATE FUNCTION SafeDiv (a INT, b INT)
RETURNS FLOAT
BEGIN
    IF b = 0 THEN
        RETURN 0;
    ELSE
        RETURN (a / b);
    END IF;
END

SELECT
    CONCAT(MonthName, ' ', [Year]) AS MonthYear,
    CPI_Value
FROM [dbo].[historical-cpi-u]
UNPIVOT
(
    CPI_Value FOR MonthName IN
    ([Jan], [Feb], [Mar], [Apr], [May], [Jun],
     [Jul], [Aug], [Sep], [Oct], [Nov], [Dec])
) AS u
WHERE
    [Year] BETWEEN 2018 AND 2026
    AND NOT (
        [Year] = 2026
        AND MonthName NOT IN ('Jan', 'Feb', 'Mar')
    );
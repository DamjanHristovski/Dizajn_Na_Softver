SHOW DATABASES;
USE berza_data;

CREATE TABLE company(
	company_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255)
);
	
CREATE TABLE daily_data(
	company_id INT,
    date DATE,
    last_transaction DECIMAL(10,2),
    max_price DECIMAL(10,2),
    min_price DECIMAL(10,2),
    average_price DECIMAL(10,2),
    volume DECIMAL (10,2),
    BEST_profit DECIMAL (20,2),
    total_profit DECIMAL (20,2),
    FOREIGN KEY (company_id) REFERENCES company(company_id)
);

ALTER TABLE company ADD UNIQUE (name)
SHOW TABLES;
SET SQL_SAFE_UPDATES = 0;
DELETE FROM daily_data;
DROP TABLE daily_data;
DELETE FROM company;
DROP TABLE company;
SET SQL_SAFE_UPDATES = 1;
select * from company;
select * from daily_data;

SELECT company_id, date,
       FORMAT(last_transaction, 2) AS last_transaction,
       FORMAT(max_price, 2) AS max_price,
       FORMAT(min_price, 2) AS min_price,
       FORMAT(average_price, 2) AS average_price,
       FORMAT(volume, 2) AS volume,
       FORMAT(BEST_profit, 2) AS BEST_profit,
       FORMAT(total_profit, 2) AS total_profit
FROM daily_data;

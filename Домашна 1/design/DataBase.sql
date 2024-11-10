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
    quantity INT,
    BEST_profit DECIMAL (10,2),
    total_profit DECIMAL (10,2),
    FOREIGN KEY (company_id) REFERENCES company(company_id)
);
ALTER TABLE company ADD UNIQUE (name)
SHOW TABLES; //Komanda za prikaz na tabeli
SET SQL_SAFE_UPDATES = 0; //Komandi za reset na tabeli
DELETE FROM daily_data;
DROP TABLE daily_data;
DELETE FROM company;
DROP TABLE company;
SET SQL_SAFE_UPDATES = 1;
select * from company; //Komanda za prikaz na podatoci
select * from daily_data; //Komanda za prikaz na podatoci


	
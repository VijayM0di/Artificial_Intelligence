    
    CREATE TABLE supplier (
    supplier_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each supplier, auto-incremented
    supplier_name VARCHAR(100) NOT NULL, -- Name of the supplier company or entity
    contact_info VARCHAR(100), -- Contact details (e.g., phone, email) of the supplier
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- Date and time when the supplier record was created
);

CREATE TABLE drug_category (
    category_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each drug category, auto-incremented
    category_name VARCHAR(50) NOT NULL -- Name of the drug category (e.g., Antibiotics, Painkillers)
);

CREATE TABLE drug (
    drug_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each drug, auto-incremented
    category_id INT, -- ID of the category this drug belongs to
    drug_name VARCHAR(100) NOT NULL, -- Name of the drug (e.g., Amoxicillin)
    manufacturer VARCHAR(100), -- Name of the company that produces the drug
    price DECIMAL(10,2), -- Price per unit of the drug
    stock INT -- Current stock quantity of the drug in inventory
);

CREATE TABLE pharmacy (
    pharmacy_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each pharmacy, auto-incremented
    pharmacy_name VARCHAR(100) NOT NULL, -- Name of the pharmacy
    location VARCHAR(100), -- Physical location or address of the pharmacy
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- Date and time when the pharmacy record was created
);

CREATE TABLE employee (
    employee_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each employee, auto-incremented
    pharmacy_id INT, -- ID of the pharmacy where the employee works
    first_name VARCHAR(50), -- Employee's first name
    last_name VARCHAR(50), -- Employee's last name
    role VARCHAR(50) -- Employee's job role (e.g., Pharmacist, Cashier)
);

CREATE TABLE sales (
    sale_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each sale, auto-incremented
    pharmacy_id INT, -- ID of the pharmacy where the sale occurred
    employee_id INT, -- ID of the employee who made the sale
    sale_date DATE, -- Date the sale took place
    total_amount DECIMAL(10,2) -- Total monetary amount of the sale
);

CREATE TABLE sale_details (
    sale_detail_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each sale detail entry, auto-incremented
    sale_id INT, -- ID of the sale this detail belongs to
    drug_id INT, -- ID of the drug sold
    quantity INT, -- Number of units of the drug sold
    subtotal DECIMAL(10,2) -- Total cost for this drug in the sale (quantity * price)
);

CREATE TABLE inventory (
    inventory_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each inventory record, auto-incremented
    drug_id INT, -- ID of the drug being tracked
    supplier_id INT, -- ID of the supplier providing the drug
    stock_level INT, -- Current stock level of the drug from this supplier
    last_updated DATE -- Date when the stock level was last updated
);

CREATE TABLE prescription (
    prescription_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each prescription, auto-incremented
    sale_id INT, -- ID of the sale associated with this prescription
    doctor_name VARCHAR(100), -- Name of the doctor who issued the prescription
    prescription_date DATE -- Date the prescription was issued
);

CREATE TABLE feedback (
    feedback_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each feedback entry, auto-incremented
    pharmacy_id INT, -- ID of the pharmacy receiving the feedback
    rating INT CHECK (rating >= 1 AND rating <= 5), -- Customer rating (1 to 5 stars)
    comments TEXT, -- Customer comments or notes
    feedback_date DATE -- Date the feedback was submitted
);
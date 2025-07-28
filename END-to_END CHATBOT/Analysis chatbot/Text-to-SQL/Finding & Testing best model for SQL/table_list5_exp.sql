CREATE TABLE supplier (
    supplier_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each supplier, automatically incremented
    supplier_name VARCHAR(100), -- Name of the supplier company or entity
    contact VARCHAR(100) -- Contact information for the supplier (e.g., phone, email)
);

CREATE TABLE raw_material (
    material_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each raw material, automatically incremented
    material_name VARCHAR(100), -- Name of the raw material (e.g., Steel, Cotton)
    unit_cost DECIMAL(10,2) -- Cost per unit of the raw material
);

CREATE TABLE product (
    product_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each product, automatically incremented
    product_name VARCHAR(100), -- Name of the product (e.g., "Steel Table")
    sale_price DECIMAL(10,2) -- Price at which the product is sold per unit
);

CREATE TABLE production (
    production_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each production batch, automatically incremented
    product_id INT, -- ID of the product being produced
    production_date DATE, -- Date when the production occurred
    quantity INT, -- Number of units produced in this batch
    FOREIGN KEY (product_id) REFERENCES product(product_id) -- Links to product table
);

CREATE TABLE employee (
    employee_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each employee, automatically incremented
    first_name VARCHAR(50), -- Employee's first name
    role VARCHAR(50) -- Employee's job role (e.g., Salesperson, Production Worker)
);

CREATE TABLE sales (
    sale_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each sale, automatically incremented
    product_id INT, -- ID of the product sold
    employee_id INT, -- ID of the employee who made the sale
    sale_date DATE, -- Date when the sale took place
    quantity_sold INT, -- Number of units sold
    total_amount DECIMAL(10,2), -- Total monetary amount of the sale
    FOREIGN KEY (product_id) REFERENCES product(product_id), -- Links to product table
    FOREIGN KEY (employee_id) REFERENCES employee(employee_id) -- Links to employee table
);

CREATE TABLE inventory (
    inventory_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each inventory record, automatically incremented
    material_id INT, -- ID of the raw material being tracked
    stock_level INT, -- Current stock quantity of the raw material
    FOREIGN KEY (material_id) REFERENCES raw_material(material_id) -- Links to raw_material table
);

CREATE TABLE customer (
    customer_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each customer, automatically incremented
    company_name VARCHAR(100), -- Name of the customer company or entity
    contact VARCHAR(100) -- Contact information for the customer (e.g., phone, email)
);

CREATE TABLE order_ (
    order_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each order, automatically incremented
    customer_id INT, -- ID of the customer who placed the order
    sale_id INT, -- ID of the sale associated with this order
    order_date DATE, -- Date when the order was placed
    FOREIGN KEY (customer_id) REFERENCES customer(customer_id), -- Links to customer table
    FOREIGN KEY (sale_id) REFERENCES sales(sale_id) -- Links to sales table
);


-- production.product_id can be joined with product.product_id -- Connects production batches to the product being produced.
-- sales.product_id can be joined with product.product_id -- Ties sales to the product sold.
-- sales.employee_id can be joined with employee.employee_id -- Links sales to the employee who made them.
-- inventory.material_id can be joined with raw_material.material_id -- Associates inventory records with the raw materials they track.
-- order_.customer_id can be joined with customer.customer_id -- Connects orders to the customer who placed them.
-- order_.sale_id can be joined with sales.sale_id -- Links orders to their corresponding sale.
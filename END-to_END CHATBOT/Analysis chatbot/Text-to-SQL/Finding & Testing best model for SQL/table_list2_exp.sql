CREATE TABLE store (
    store_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each store, automatically incremented
    store_name VARCHAR(50) NOT NULL, -- Name of the store
    location VARCHAR(100) -- Physical location or address of the store
);

CREATE TABLE department (
    department_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each department, automatically incremented
    department_name VARCHAR(50) NOT NULL -- Name of the department (e.g., Electronics, Clothing)
);

CREATE TABLE category (
    category_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each product category, automatically incremented
    category_name VARCHAR(50) NOT NULL -- Name of the product category (e.g., TVs, Shirts)
);

CREATE TABLE product (
    product_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each product, automatically incremented
    category_id INT, -- ID of the category this product belongs to
    product_name VARCHAR(100), -- Name of the product (e.g., "LED TV 55-inch")
    price DECIMAL(10,2), -- Price per unit of the product
    FOREIGN KEY (category_id) REFERENCES category(category_id) -- Links to category table
);

CREATE TABLE employee (
    employee_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each employee, automatically incremented
    store_id INT, -- ID of the store where the employee works
    department_id INT, -- ID of the department the employee is assigned to
    first_name VARCHAR(50), -- Employee's first name
    last_name VARCHAR(50), -- Employee's last name
    FOREIGN KEY (store_id) REFERENCES store(store_id), -- Links to store table
    FOREIGN KEY (department_id) REFERENCES department(department_id) -- Links to department table
);

CREATE TABLE customer (
    customer_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each customer, automatically incremented
    first_name VARCHAR(50), -- Customer's first name
    email VARCHAR(100) -- Customer's email address
);

CREATE TABLE sales (
    sale_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each sale, automatically incremented
    store_id INT, -- ID of the store where the sale occurred
    employee_id INT, -- ID of the employee who processed the sale
    customer_id INT, -- ID of the customer who made the purchase
    sale_date DATETIME, -- Date and time when the sale took place
    total_amount DECIMAL(10,2), -- Total monetary amount of the sale
    FOREIGN KEY (store_id) REFERENCES store(store_id), -- Links to store table
    FOREIGN KEY (employee_id) REFERENCES employee(employee_id), -- Links to employee table
    FOREIGN KEY (customer_id) REFERENCES customer(customer_id) -- Links to customer table
);

CREATE TABLE sale_items (
    sale_item_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each sale item, automatically incremented
    sale_id INT, -- ID of the sale this item belongs to
    product_id INT, -- ID of the product sold
    quantity INT, -- Number of units of the product sold
    subtotal DECIMAL(10,2), -- Subtotal amount for this specific product in the sale
    FOREIGN KEY (sale_id) REFERENCES sales(sale_id), -- Links to sales table
    FOREIGN KEY (product_id) REFERENCES product(product_id) -- Links to product table
);

CREATE TABLE stock (
    stock_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each stock record, automatically incremented
    product_id INT, -- ID of the product being tracked
    store_id INT, -- ID of the store holding this stock
    quantity INT, -- Current stock quantity of the product at the store
    FOREIGN KEY (product_id) REFERENCES product(product_id), -- Links to product table
    FOREIGN KEY (store_id) REFERENCES store(store_id) -- Links to store table
);






-- product.category_id can be joined with category.category_id -- Connects each product to its category.
-- employee.store_id can be joined with store.store_id -- Associates employees with the store they work at.
-- employee.department_id can be joined with department.department_id -- Links employees to their assigned department.
-- sales.store_id can be joined with store.store_id -- Ties sales to the store where they occurred.
-- sales.employee_id can be joined with employee.employee_id -- Connects sales to the employee who processed them.
-- sales.customer_id can be joined with customer.customer_id -- Links sales to the customer who made the purchase.
-- sale_items.sale_id can be joined with sales.sale_id -- Associates sale items with their corresponding sale.
-- sale_items.product_id can be joined with product.product_id -- Connects sale items to the specific product sold.
-- stock.product_id can be joined with product.product_id -- Ties stock records to the products they track.
-- stock.store_id can be joined with store.store_id -- Links stock records to the store holding the inventory.
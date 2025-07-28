CREATE TABLE warehouse (
    warehouse_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each warehouse, automatically incremented
    location VARCHAR(100), -- Physical location or address of the warehouse
    capacity INT -- Maximum storage capacity of the warehouse (e.g., in units or volume)
);

CREATE TABLE product_category (
    category_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each product category, automatically incremented
    category_name VARCHAR(50) -- Name of the product category (e.g., Electronics, Furniture)
);

CREATE TABLE product (
    product_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each product, automatically incremented
    category_id INT, -- ID of the category this product belongs to
    product_name VARCHAR(100), -- Name of the product (e.g., "LED TV")
    wholesale_price DECIMAL(10,2), -- Wholesale price per unit of the product
    FOREIGN KEY (category_id) REFERENCES product_category(category_id) -- Links to product_category table
);

CREATE TABLE supplier (
    supplier_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each supplier, automatically incremented
    supplier_name VARCHAR(100), -- Name of the supplier company or entity
    contact VARCHAR(100) -- Contact information for the supplier (e.g., phone, email)
);

CREATE TABLE customer (
    customer_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each customer, automatically incremented
    company_name VARCHAR(100), -- Name of the customer company or entity
    contact VARCHAR(100) -- Contact information for the customer (e.g., phone, email)
);

CREATE TABLE employee (
    employee_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each employee, automatically incremented
    first_name VARCHAR(50), -- Employee's first name
    warehouse_id INT, -- ID of the warehouse where the employee works
    role VARCHAR(50), -- Employee's job role (e.g., Warehouse Manager, Order Picker)
    FOREIGN KEY (warehouse_id) REFERENCES warehouse(warehouse_id) -- Links to warehouse table
);

CREATE TABLE order_ (
    order_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each order, automatically incremented
    customer_id INT, -- ID of the customer who placed the order
    warehouse_id INT, -- ID of the warehouse fulfilling the order
    employee_id INT, -- ID of the employee who processed the order
    order_date DATE, -- Date when the order was placed
    total_amount DECIMAL(10,2), -- Total monetary amount of the order
    FOREIGN KEY (customer_id) REFERENCES customer(customer_id), -- Links to customer table
    FOREIGN KEY (warehouse_id) REFERENCES warehouse(warehouse_id), -- Links to warehouse table
    FOREIGN KEY (employee_id) REFERENCES employee(employee_id) -- Links to employee table
);

CREATE TABLE order_detail (
    order_detail_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each order detail, automatically incremented
    order_id INT, -- ID of the order this detail belongs to
    product_id INT, -- ID of the product ordered
    quantity INT, -- Number of units of the product ordered
    subtotal DECIMAL(10,2), -- Subtotal amount for this specific product in the order
    FOREIGN KEY (order_id) REFERENCES order_(order_id), -- Links to order_ table
    FOREIGN KEY (product_id) REFERENCES product(product_id) -- Links to product table
);

CREATE TABLE inventory (
    inventory_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each inventory record, automatically incremented
    product_id INT, -- ID of the product being tracked
    warehouse_id INT, -- ID of the warehouse storing the product
    supplier_id INT, -- ID of the supplier providing the product
    stock_level INT, -- Current stock quantity of the product in the warehouse
    FOREIGN KEY (product_id) REFERENCES product(product_id), -- Links to product table
    FOREIGN KEY (warehouse_id) REFERENCES warehouse(warehouse_id), -- Links to warehouse table
    FOREIGN KEY (supplier_id) REFERENCES supplier(supplier_id) -- Links to supplier table
);

CREATE TABLE shipment (
    shipment_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each shipment, automatically incremented
    order_id INT, -- ID of the order being shipped
    shipment_date DATE, -- Date when the shipment was sent
    tracking_number VARCHAR(50), -- Tracking number for the shipment
    FOREIGN KEY (order_id) REFERENCES order_(order_id) -- Links to order_ table
);




-- product.category_id can be joined with product_category.category_id -- Connects each product to its category.
-- employee.warehouse_id can be joined with warehouse.warehouse_id -- Ties employees to the warehouse they work at.
-- order_.customer_id can be joined with customer.customer_id -- Links orders to the customer who placed them.
-- order_.warehouse_id can be joined with warehouse.warehouse_id -- Associates orders with the warehouse fulfilling them.
-- order_.employee_id can be joined with employee.employee_id -- Connects orders to the employee who processed them.
-- order_detail.order_id can be joined with order_.order_id -- Ties order details to their corresponding order.
-- order_detail.product_id can be joined with product.product_id -- Links order details to the specific product ordered.
-- inventory.product_id can be joined with product.product_id -- Associates inventory records with the product being tracked.
-- inventory.warehouse_id can be joined with warehouse.warehouse_id -- Connects inventory to the warehouse storing it.
-- inventory.supplier_id can be joined with supplier.supplier_id -- Links inventory to the supplier providing the product.
-- shipment.order_id can be joined with order_.order_id -- Ties shipments to the order being shipped.-- 
CREATE TABLE vehicle_type (
    type_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each vehicle type, automatically incremented
    type_name VARCHAR(50) -- Name of the vehicle type (e.g., Sedan, Truck)
);

CREATE TABLE vehicle (
    vehicle_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each vehicle, automatically incremented
    type_id INT, -- ID of the vehicle type this vehicle belongs to
    model VARCHAR(100), -- Model name of the vehicle (e.g., "Camry")
    manufacturer VARCHAR(50), -- Name of the vehicle manufacturer (e.g., "Toyota")
    price DECIMAL(12,2), -- Price of the vehicle
    stock INT, -- Current stock quantity of the vehicle
    FOREIGN KEY (type_id) REFERENCES vehicle_type(type_id) -- Links to vehicle_type table
);

CREATE TABLE customer (
    customer_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each customer, automatically incremented
    first_name VARCHAR(50), -- Customer's first name
    last_name VARCHAR(50), -- Customer's last name
    email VARCHAR(100) -- Customer's email address
);

CREATE TABLE employee (
    employee_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each employee, automatically incremented
    first_name VARCHAR(50), -- Employee's first name
    last_name VARCHAR(50), -- Employee's last name
    role VARCHAR(50) -- Employee's job role (e.g., Salesperson, Mechanic)
);

CREATE TABLE sale (
    sale_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each sale, automatically incremented
    vehicle_id INT, -- ID of the vehicle sold
    customer_id INT, -- ID of the customer who purchased the vehicle
    employee_id INT, -- ID of the employee who facilitated the sale
    sale_date DATE, -- Date when the sale occurred
    sale_price DECIMAL(12,2), -- Final sale price of the vehicle
    FOREIGN KEY (vehicle_id) REFERENCES vehicle(vehicle_id), -- Links to vehicle table
    FOREIGN KEY (customer_id) REFERENCES customer(customer_id), -- Links to customer table
    FOREIGN KEY (employee_id) REFERENCES employee(employee_id) -- Links to employee table
);

CREATE TABLE service (
    service_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each service record, automatically incremented
    vehicle_id INT, -- ID of the vehicle being serviced
    customer_id INT, -- ID of the customer requesting the service
    service_date DATE, -- Date when the service was performed
    cost DECIMAL(10,2), -- Cost of the service
    FOREIGN KEY (vehicle_id) REFERENCES vehicle(vehicle_id), -- Links to vehicle table
    FOREIGN KEY (customer_id) REFERENCES customer(customer_id) -- Links to customer table
);

CREATE TABLE supplier (
    supplier_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each supplier, automatically incremented
    supplier_name VARCHAR(100), -- Name of the supplier company or entity
    contact VARCHAR(100) -- Contact information for the supplier (e.g., phone, email)
);

CREATE TABLE inventory (
    inventory_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each inventory record, automatically incremented
    vehicle_id INT, -- ID of the vehicle being tracked
    supplier_id INT, -- ID of the supplier providing the vehicle
    stock_level INT, -- Current stock level of the vehicle
    FOREIGN KEY (vehicle_id) REFERENCES vehicle(vehicle_id), -- Links to vehicle table
    FOREIGN KEY (supplier_id) REFERENCES supplier(supplier_id) -- Links to supplier table
);

CREATE TABLE feedback (
    feedback_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each feedback entry, automatically incremented
    customer_id INT, -- ID of the customer providing the feedback
    rating INT, -- Rating given by the customer (e.g., 1 to 5)
    comments TEXT, -- Additional comments or notes from the customer
    FOREIGN KEY (customer_id) REFERENCES customer(customer_id) -- Links to customer table
);




-- vehicle.type_id can be joined with vehicle_type.type_id -- Connects each vehicle to its type.
-- sale.vehicle_id can be joined with vehicle.vehicle_id -- Ties sales to the vehicle sold.
-- sale.customer_id can be joined with customer.customer_id -- Links sales to the customer who purchased the vehicle.
-- sale.employee_id can be joined with employee.employee_id -- Associates sales with the employee who facilitated them.
-- service.vehicle_id can be joined with vehicle.vehicle_id -- Connects service records to the vehicle being serviced.
-- service.customer_id can be joined with customer.customer_id -- Links service records to the customer requesting them.
-- inventory.vehicle_id can be joined with vehicle.vehicle_id -- Ties inventory records to the vehicle being tracked.
-- inventory.supplier_id can be joined with supplier.supplier_id -- Associates inventory with the supplier providing the vehicle.
-- feedback.customer_id can be joined with customer.customer_id -- Connects feedback to the customer providing it.-- 
CREATE TABLE restaurant (
    restaurant_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each restaurant, automatically incremented
    name VARCHAR(100), -- Name of the restaurant
    location VARCHAR(100) -- Physical location or address of the restaurant
);

CREATE TABLE menu_category (
    category_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each menu category, automatically incremented
    category_name VARCHAR(50) -- Name of the menu category (e.g., Appetizers, Desserts)
);

CREATE TABLE menu_item (
    item_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each menu item, automatically incremented
    category_id INT, -- ID of the category this menu item belongs to
    item_name VARCHAR(100), -- Name of the menu item (e.g., "Chicken Wings")
    price DECIMAL(10,2), -- Price of the menu item
    FOREIGN KEY (category_id) REFERENCES menu_category(category_id) -- Links to menu_category table
);

CREATE TABLE employee (
    employee_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each employee, automatically incremented
    restaurant_id INT, -- ID of the restaurant where the employee works
    first_name VARCHAR(50), -- Employee's first name
    role VARCHAR(50) -- Employee's job role (e.g., Waiter, Chef)
    FOREIGN KEY (restaurant_id) REFERENCES restaurant(restaurant_id) -- Links to restaurant table
);

CREATE TABLE customer (
    customer_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each customer, automatically incremented
    first_name VARCHAR(50), -- Customer's first name
    email VARCHAR(100) -- Customer's email address
);

CREATE TABLE order_ (
    order_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each order, automatically incremented
    restaurant_id INT, -- ID of the restaurant where the order was placed
    employee_id INT, -- ID of the employee who processed the order
    customer_id INT, -- ID of the customer who placed the order
    order_date DATETIME, -- Date and time when the order was placed
    total_amount DECIMAL(10,2), -- Total monetary amount of the order
    FOREIGN KEY (restaurant_id) REFERENCES restaurant(restaurant_id), -- Links to restaurant table
    FOREIGN KEY (employee_id) REFERENCES employee(employee_id), -- Links to employee table
    FOREIGN KEY (customer_id) REFERENCES customer(customer_id) -- Links to customer table
);

CREATE TABLE order_detail (
    order_detail_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each order detail, automatically incremented
    order_id INT, -- ID of the order this detail belongs to
    item_id INT, -- ID of the menu item ordered
    quantity INT, -- Number of units of the menu item ordered
    subtotal DECIMAL(10,2), -- Subtotal amount for this specific menu item in the order
    FOREIGN KEY (order_id) REFERENCES order_(order_id), -- Links to order_ table
    FOREIGN KEY (item_id) REFERENCES menu_item(item_id) -- Links to menu_item table
);

CREATE TABLE feedback (
    feedback_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each feedback entry, automatically incremented
    customer_id INT, -- ID of the customer providing the feedback
    restaurant_id INT, -- ID of the restaurant being reviewed
    rating INT, -- Rating given by the customer (e.g., 1 to 5)
    comments TEXT, -- Additional comments or notes from the customer
    FOREIGN KEY (customer_id) REFERENCES customer(customer_id), -- Links to customer table
    FOREIGN KEY (restaurant_id) REFERENCES restaurant(restaurant_id) -- Links to restaurant table
);


-- menu_item.category_id can be joined with menu_category.category_id -- Connects each menu item to its category.
-- employee.restaurant_id can be joined with restaurant.restaurant_id -- Ties employees to the restaurant they work at.
-- order_.restaurant_id can be joined with restaurant.restaurant_id -- Links orders to the restaurant where they were placed.
-- order_.employee_id can be joined with employee.employee_id -- Connects orders to the employee who processed them.
-- order_.customer_id can be joined with customer.customer_id -- Associates orders with the customer who placed them.
-- order_detail.order_id can be joined with order_.order_id -- Ties order details to their corresponding order.
-- order_detail.item_id can be joined with menu_item.item_id -- Links order details to the specific menu item ordered.
-- feedback.customer_id can be joined with customer.customer_id -- Connects feedback to the customer providing it.
-- feedback.restaurant_id can be joined with restaurant.restaurant_id -- Associates feedback with the restaurant being reviewed.
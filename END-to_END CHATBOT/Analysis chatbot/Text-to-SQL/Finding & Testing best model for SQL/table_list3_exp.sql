CREATE TABLE user (
    user_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each user, automatically incremented
    username VARCHAR(50) UNIQUE, -- Unique username chosen by the user
    email VARCHAR(100), -- User's email address
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- Date and time when the user account was created
);

CREATE TABLE category (
    category_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each product category, automatically incremented
    category_name VARCHAR(50) -- Name of the product category (e.g., Electronics, Clothing)
);

CREATE TABLE product (
    product_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each product, automatically incremented
    category_id INT, -- ID of the category this product belongs to
    product_name VARCHAR(100), -- Name of the product (e.g., "Wireless Mouse")
    price DECIMAL(10,2), -- Price per unit of the product
    stock INT -- Current stock quantity of the product
    FOREIGN KEY (category_id) REFERENCES category(category_id) -- Links to category table
);

CREATE TABLE order_ (
    order_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each order, automatically incremented
    user_id INT, -- ID of the user who placed the order
    order_date DATETIME, -- Date and time when the order was placed
    status ENUM('PENDING', 'SHIPPED', 'DELIVERED'), -- Current status of the order
    total_amount DECIMAL(10,2), -- Total monetary amount of the order
    FOREIGN KEY (user_id) REFERENCES user(user_id) -- Links to user table
);

CREATE TABLE order_item (
    order_item_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each order item, automatically incremented
    order_id INT, -- ID of the order this item belongs to
    product_id INT, -- ID of the product ordered
    quantity INT, -- Number of units of the product ordered
    subtotal DECIMAL(10,2), -- Subtotal amount for this specific product in the order
    FOREIGN KEY (order_id) REFERENCES order_(order_id), -- Links to order_ table
    FOREIGN KEY (product_id) REFERENCES product(product_id) -- Links to product table
);

CREATE TABLE payment (
    payment_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each payment, automatically incremented
    order_id INT, -- ID of the order associated with this payment
    amount DECIMAL(10,2), -- Amount paid for the order
    payment_date DATETIME, -- Date and time when the payment was made
    method ENUM('CARD', 'PAYPAL', 'COD'), -- Payment method used (Credit Card, PayPal, Cash on Delivery)
    FOREIGN KEY (order_id) REFERENCES order_(order_id) -- Links to order_ table
);

CREATE TABLE shipping (
    shipping_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each shipping record, automatically incremented
    order_id INT, -- ID of the order being shipped
    shipping_date DATE, -- Date when the order was shipped
    tracking_number VARCHAR(50), -- Tracking number for the shipment
    FOREIGN KEY (order_id) REFERENCES order_(order_id) -- Links to order_ table
);

CREATE TABLE review (
    review_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each review, automatically incremented
    product_id INT, -- ID of the product being reviewed
    user_id INT, -- ID of the user who wrote the review
    rating INT, -- Rating given by the user (e.g., 1 to 5)
    comment TEXT, -- Text comment or feedback about the product
    FOREIGN KEY (product_id) REFERENCES product(product_id), -- Links to product table
    FOREIGN KEY (user_id) REFERENCES user(user_id) -- Links to user table
);



-- product.category_id can be joined with category.category_id -- Connects each product to its category.
-- order_.user_id can be joined with user.user_id -- Ties orders to the user who placed them.
-- order_item.order_id can be joined with order_.order_id -- Links order items to their corresponding order.
-- order_item.product_id can be joined with product.product_id -- Connects order items to the specific product ordered.
-- payment.order_id can be joined with order_.order_id -- Associates payments with the order they cover.
-- shipping.order_id can be joined with order_.order_id -- Links shipping records to the order being shipped.
-- review.product_id can be joined with product.product_id -- Ties reviews to the product being reviewed.
-- review.user_id can be joined with user.user_id -- Connects reviews to the user who wrote them.
CREATE TABLE customer (
    customer_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each customer, automatically incremented
    first_name VARCHAR(50), -- Customer's first name
    email VARCHAR(100) UNIQUE -- Customer's unique email address
);

CREATE TABLE plan (
    plan_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each subscription plan, automatically incremented
    plan_name VARCHAR(50), -- Name of the plan (e.g., "Basic", "Premium")
    monthly_price DECIMAL(10,2) -- Monthly cost of the plan
);

CREATE TABLE subscription (
    subscription_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each subscription, automatically incremented
    customer_id INT, -- ID of the customer who subscribed
    plan_id INT, -- ID of the plan subscribed to
    start_date DATE, -- Date when the subscription began
    end_date DATE, -- Date when the subscription ends (if applicable)
    status ENUM('ACTIVE', 'CANCELLED'), -- Current status of the subscription
    FOREIGN KEY (customer_id) REFERENCES customer(customer_id), -- Links to customer table
    FOREIGN KEY (plan_id) REFERENCES plan(plan_id) -- Links to plan table
);

CREATE TABLE payment (
    payment_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each payment, automatically incremented
    subscription_id INT, -- ID of the subscription this payment is for
    amount DECIMAL(10,2), -- Amount paid
    payment_date DATE, -- Date when the payment was made
    method ENUM('CARD', 'PAYPAL', 'BANK'), -- Payment method used (Credit Card, PayPal, Bank Transfer)
    FOREIGN KEY (subscription_id) REFERENCES subscription(subscription_id) -- Links to subscription table
);

CREATE TABLE usage (
    usage_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each usage record, automatically incremented
    subscription_id INT, -- ID of the subscription this usage pertains to
    usage_date DATE, -- Date when the usage was recorded
    metric_value INT, -- Measured value of usage (e.g., data in MB, minutes used)
    FOREIGN KEY (subscription_id) REFERENCES subscription(subscription_id) -- Links to subscription table
);

CREATE TABLE employee (
    employee_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each employee, automatically incremented
    first_name VARCHAR(50), -- Employee's first name
    role VARCHAR(50) -- Employee's job role (e.g., Support Agent, Manager)
);

CREATE TABLE support_ticket (
    ticket_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each support ticket, automatically incremented
    customer_id INT, -- ID of the customer who raised the ticket
    employee_id INT, -- ID of the employee handling the ticket
    issue TEXT, -- Description of the customer's issue
    status ENUM('OPEN', 'CLOSED'), -- Current status of the ticket
    FOREIGN KEY (customer_id) REFERENCES customer(customer_id), -- Links to customer table
    FOREIGN KEY (employee_id) REFERENCES employee(employee_id) -- Links to employee table
);

CREATE TABLE feedback (
    feedback_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each feedback entry, automatically incremented
    customer_id INT, -- ID of the customer providing the feedback
    rating INT, -- Rating given by the customer (e.g., 1 to 5)
    comments TEXT, -- Additional comments or notes from the customer
    FOREIGN KEY (customer_id) REFERENCES customer(customer_id) -- Links to customer table
);



-- subscription.customer_id can be joined with customer.customer_id -- Connects subscriptions to the customer who subscribed.
-- subscription.plan_id can be joined with plan.plan_id -- Links subscriptions to the specific plan subscribed to.
-- payment.subscription_id can be joined with subscription.subscription_id -- Ties payments to their corresponding subscription.
-- usage.subscription_id can be joined with subscription.subscription_id -- Associates usage records with the subscription they pertain to.
-- support_ticket.customer_id can be joined with customer.customer_id -- Links support tickets to the customer who raised them.
-- support_ticket.employee_id can be joined with employee.employee_id -- Connects support tickets to the employee handling them.
-- feedback.customer_id can be joined with customer.customer_id -- Ties feedback to the customer providing it.-- 
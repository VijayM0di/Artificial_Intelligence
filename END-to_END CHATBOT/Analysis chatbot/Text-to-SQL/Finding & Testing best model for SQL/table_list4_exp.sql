CREATE TABLE agent (
    agent_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each agent, automatically incremented
    first_name VARCHAR(50), -- Agent's first name
    last_name VARCHAR(50), -- Agent's last name
    phone VARCHAR(20) -- Agent's phone number
);

CREATE TABLE property_type (
    type_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each property type, automatically incremented
    type_name VARCHAR(50) NOT NULL -- Name of the property type (e.g., House, Apartment)
);

CREATE TABLE property (
    property_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each property, automatically incremented
    type_id INT, -- ID of the property type this property belongs to
    address TEXT, -- Full address of the property
    price DECIMAL(12,2), -- Listed price of the property
    status ENUM('AVAILABLE', 'SOLD', 'PENDING'), -- Current status of the property
    FOREIGN KEY (type_id) REFERENCES property_type(type_id) -- Links to property_type table
);

CREATE TABLE client (
    client_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each client, automatically incremented
    first_name VARCHAR(50), -- Client's first name
    last_name VARCHAR(50), -- Client's last name
    email VARCHAR(100) -- Client's email address
);

CREATE TABLE sale (
    sale_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each sale, automatically incremented
    property_id INT, -- ID of the property sold
    agent_id INT, -- ID of the agent who facilitated the sale
    client_id INT, -- ID of the client who purchased the property
    sale_date DATE, -- Date when the sale was completed
    sale_price DECIMAL(12,2), -- Final sale price of the property
    FOREIGN KEY (property_id) REFERENCES property(property_id), -- Links to property table
    FOREIGN KEY (agent_id) REFERENCES agent(agent_id), -- Links to agent table
    FOREIGN KEY (client_id) REFERENCES client(client_id) -- Links to client table
);

CREATE TABLE commission (
    commission_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each commission record, automatically incremented
    sale_id INT, -- ID of the sale this commission is tied to
    agent_id INT, -- ID of the agent receiving the commission
    amount DECIMAL(10,2), -- Commission amount earned by the agent
    FOREIGN KEY (sale_id) REFERENCES sale(sale_id), -- Links to sale table
    FOREIGN KEY (agent_id) REFERENCES agent(agent_id) -- Links to agent table
);

CREATE TABLE appointment (
    appointment_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each appointment, automatically incremented
    agent_id INT, -- ID of the agent conducting the appointment
    client_id INT, -- ID of the client attending the appointment
    property_id INT, -- ID of the property being shown or discussed
    appointment_date DATETIME, -- Date and time of the appointment
    FOREIGN KEY (agent_id) REFERENCES agent(agent_id), -- Links to agent table
    FOREIGN KEY (client_id) REFERENCES client(client_id), -- Links to client table
    FOREIGN KEY (property_id) REFERENCES property(property_id) -- Links to property table
);

CREATE TABLE expense (
    expense_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each expense record, automatically incremented
    property_id INT, -- ID of the property associated with the expense
    amount DECIMAL(10,2), -- Amount of the expense
    description TEXT, -- Details or reason for the expense (e.g., repairs, marketing)
    expense_date DATE, -- Date when the expense was incurred
    FOREIGN KEY (property_id) REFERENCES property(property_id) -- Links to property table
);

CREATE TABLE feedback (
    feedback_id INT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each feedback entry, automatically incremented
    client_id INT, -- ID of the client providing the feedback
    rating INT, -- Rating given by the client (e.g., 1 to 5)
    comments TEXT, -- Additional comments or notes from the client
    FOREIGN KEY (client_id) REFERENCES client(client_id) -- Links to client table
);


-- property.type_id can be joined with property_type.type_id -- Connects each property to its type.
-- sale.property_id can be joined with property.property_id -- Ties sales to the property sold.
-- sale.agent_id can be joined with agent.agent_id -- Links sales to the agent who facilitated them.
-- sale.client_id can be joined with client.client_id -- Connects sales to the client who purchased the property.
-- commission.sale_id can be joined with sale.sale_id -- Associates commissions with their corresponding sale.
-- commission.agent_id can be joined with agent.agent_id -- Links commissions to the agent receiving them.
-- appointment.agent_id can be joined with agent.agent_id -- Ties appointments to the agent conducting them.
-- appointment.client_id can be joined with client.client_id -- Connects appointments to the client attending them.
-- appointment.property_id can be joined with property.property_id -- Links appointments to the property being shown.
-- expense.property_id can be joined with property.property_id -- Associates expenses with the property they relate to.
-- feedback.client_id can be joined with client.client_id -- Connects feedback to the client providing it.
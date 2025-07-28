# Healthcare Bot using LangGraph with Medical Reasoning & Customer Care
# Supports MSSQL Server database integration

import getpass
import os
import json
import re
from datetime import datetime, date, timedelta
from typing import Annotated, Optional, Union, TypedDict, List, Dict, Any
import pyodbc
import pandas as pd
import requests
from enum import Enum

# Gemini imports for AI capabilities
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage

# Configuration
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

# API Keys
_set_env("GEMINI_API_KEY")
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Database Configuration - MSSQL Server
class DatabaseConfig:
    def __init__(self):
        self.server = os.getenv("MSSQL_SERVER", "localhost")
        self.database = os.getenv("MSSQL_DATABASE", "RX_AI_ML")
        self.username = os.getenv("MSSQL_USERNAME", "master")
        self.password = os.getenv("MSSQL_PASSWORD", "YourStrong@Passw0rd")
        self.driver = "{ODBC Driver 17 for SQL Server}"
    
    def get_connection_string(self):
        if self.username and self.password:
            return f"DRIVER={self.driver};SERVER={self.server};DATABASE={self.database};UID={self.username};PWD={self.password}"
        else:
            # Windows Authentication
            return f"DRIVER={self.driver};SERVER={self.server};DATABASE={self.database};Trusted_Connection=yes"

# Healthcare Bot State with enhanced medical context
class HealthcareState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    patient_id: Optional[str]
    conversation_type: Optional[str]  # "medical_inquiry", "appointment", "billing", "general"
    medical_context: Optional[Dict[str, Any]]  # Store medical history, symptoms, etc.
    urgency_level: Optional[str]  # "low", "medium", "high", "emergency"
    escalation_needed: Optional[bool]
    session_metadata: Optional[Dict[str, Any]]

# Conversation Types
class ConversationType(Enum):
    MEDICAL_INQUIRY = "medical_inquiry"
    APPOINTMENT = "appointment"
    BILLING = "billing"
    PRESCRIPTION = "prescription"
    TEST_RESULTS = "test_results"
    GENERAL = "general"
    EMERGENCY = "emergency"

# Medical Database Tools
class HealthcareDatabaseTools:
    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config
    
    def get_connection(self):
        return pyodbc.connect(self.db_config.get_connection_string())
    
    def get_patient_info(self, patient_id: str) -> Dict[str, Any]:
        """Fetch patient information and medical history"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Patient basic info
            patient_query = """
            SELECT p.patient_id, p.first_name, p.last_name, p.date_of_birth, 
                   p.gender, p.phone, p.email, p.emergency_contact,
                   p.insurance_provider, p.policy_number
            FROM patients p
            WHERE p.patient_id = ?
            """
            cursor.execute(patient_query, (patient_id,))
            patient_data = cursor.fetchone()
            
            if not patient_data:
                return {"error": "Patient not found"}
            
            # Medical history
            history_query = """
            SELECT TOP 10 mh.condition_name, mh.diagnosis_date, mh.status, 
                   mh.treatment_notes, d.doctor_name
            FROM medical_history mh
            JOIN doctors d ON mh.doctor_id = d.doctor_id
            WHERE mh.patient_id = ?
            ORDER BY mh.diagnosis_date DESC
            """
            cursor.execute(history_query, (patient_id,))
            medical_history = cursor.fetchall()
            
            # Current medications
            medication_query = """
            SELECT m.medication_name, pm.dosage, pm.frequency, 
                   pm.start_date, pm.end_date, pm.status
            FROM patient_medications pm
            JOIN medications m ON pm.medication_id = m.medication_id
            WHERE pm.patient_id = ? AND pm.status = 'active'
            """
            cursor.execute(medication_query, (patient_id,))
            medications = cursor.fetchall()
            
            # Recent appointments
            appointment_query = """
            SELECT TOP 5 a.appointment_id, a.appointment_date, a.appointment_time,
                   a.status, a.reason, d.doctor_name, d.specialization
            FROM appointments a
            JOIN doctors d ON a.doctor_id = d.doctor_id
            WHERE a.patient_id = ?
            ORDER BY a.appointment_date DESC
            """
            cursor.execute(appointment_query, (patient_id,))
            appointments = cursor.fetchall()
            
            conn.close()
            
            return {
                "patient_info": dict(zip([col[0] for col in cursor.description], patient_data)) if patient_data else None,
                "medical_history": [dict(zip([col[0] for col in cursor.description], row)) for row in medical_history],
                "medications": [dict(zip([col[0] for col in cursor.description], row)) for row in medications],
                "recent_appointments": [dict(zip([col[0] for col in cursor.description], row)) for row in appointments]
            }
            
        except Exception as e:
            return {"error": f"Database error: {str(e)}"}
    
    def search_doctors(self, specialization: str = None, availability_date: str = None) -> List[Dict]:
        """Search for available doctors"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            query = """
            SELECT d.doctor_id, d.doctor_name, d.specialization, d.phone, 
                   d.email, d.years_experience, d.rating
            FROM doctors d
            WHERE d.status = 'active'
            """
            params = []
            
            if specialization:
                query += " AND d.specialization LIKE ?"
                params.append(f"%{specialization}%")
            
            cursor.execute(query, params)
            doctors = cursor.fetchall()
            conn.close()
            
            return [dict(zip([col[0] for col in cursor.description], row)) for row in doctors]
            
        except Exception as e:
            return [{"error": f"Database error: {str(e)}"}]
    
    def book_appointment(self, patient_id: str, doctor_id: str, preferred_date: str, reason: str) -> Dict:
        """Book an appointment"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Check availability (simplified)
            check_query = """
            SELECT COUNT(*) FROM appointments 
            WHERE doctor_id = ? AND appointment_date = ? AND status != 'cancelled'
            """
            cursor.execute(check_query, (doctor_id, preferred_date))
            existing_count = cursor.fetchone()[0]
            
            if existing_count >= 8:  # Assuming 8 slots per day
                return {"success": False, "message": "No available slots for this date"}
            
            # Book appointment
            insert_query = """
            INSERT INTO appointments (patient_id, doctor_id, appointment_date, 
                                    appointment_time, status, reason, created_date)
            VALUES (?, ?, ?, '09:00', 'scheduled', ?, GETDATE())
            """
            cursor.execute(insert_query, (patient_id, doctor_id, preferred_date, reason))
            conn.commit()
            
            # Get appointment ID
            cursor.execute("SELECT @@IDENTITY")
            appointment_id = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "success": True, 
                "appointment_id": appointment_id,
                "message": "Appointment booked successfully"
            }
            
        except Exception as e:
            return {"success": False, "message": f"Booking failed: {str(e)}"}
    
    def get_test_results(self, patient_id: str, test_type: str = None) -> List[Dict]:
        """Get patient test results"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            query = """
            SELECT tr.test_id, tr.test_name, tr.test_date, tr.result_value, 
                   tr.normal_range, tr.status, tr.notes, d.doctor_name
            FROM test_results tr
            JOIN doctors d ON tr.ordered_by = d.doctor_id
            WHERE tr.patient_id = ?
            """
            params = [patient_id]
            
            if test_type:
                query += " AND tr.test_name LIKE ?"
                params.append(f"%{test_type}%")
            
            query += " ORDER BY tr.test_date DESC"
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            conn.close()
            
            return [dict(zip([col[0] for col in cursor.description], row)) for row in results]
            
        except Exception as e:
            return [{"error": f"Database error: {str(e)}"}]

# Medical Knowledge Base using Gemini
class MedicalKnowledgeBase:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-2.0-flash")
    
    def analyze_symptoms(self, symptoms: List[str], patient_context: Dict = None) -> Dict:
        """Analyze symptoms and provide medical insights"""
        
        context = ""
        if patient_context:
            age = self._calculate_age(patient_context.get('date_of_birth'))
            context = f"Patient context: Age {age}, Gender: {patient_context.get('gender', 'Unknown')}"
            
            if patient_context.get('medical_history'):
                conditions = [h.get('condition_name') for h in patient_context['medical_history'][:3]]
                context += f", Medical history: {', '.join(conditions)}"
            
            if patient_context.get('medications'):
                meds = [m.get('medication_name') for m in patient_context['medications'][:3]]
                context += f", Current medications: {', '.join(meds)}"
        
        prompt = f"""
        As a medical AI assistant, analyze the following symptoms and provide insights:
        
        Symptoms: {', '.join(symptoms)}
        {context}
        
        Please provide:
        1. Possible conditions (with likelihood assessment)
        2. Recommended immediate actions
        3. Whether emergency care is needed
        4. Suggested next steps
        5. Questions to ask for better diagnosis
        
        IMPORTANT: Always include disclaimers about seeking professional medical advice.
        Be empathetic and avoid causing unnecessary alarm.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return {
                "analysis": response.text,
                "urgency_level": self._assess_urgency(symptoms),
                "emergency_keywords": self._check_emergency_keywords(symptoms)
            }
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _calculate_age(self, birth_date):
        if isinstance(birth_date, str):
            birth_date = datetime.strptime(birth_date, '%Y-%m-%d').date()
        return (date.today() - birth_date).days // 365
    
    def _assess_urgency(self, symptoms: List[str]) -> str:
        emergency_symptoms = [
            'chest pain', 'difficulty breathing', 'severe bleeding', 'unconscious',
            'heart attack', 'stroke', 'severe abdominal pain', 'high fever'
        ]
        
        high_urgency = [
            'persistent pain', 'vomiting', 'dizziness', 'swelling', 'rash'
        ]
        
        symptoms_text = ' '.join(symptoms).lower()
        
        for symptom in emergency_symptoms:
            if symptom in symptoms_text:
                return "emergency"
        
        for symptom in high_urgency:
            if symptom in symptoms_text:
                return "high"
        
        return "medium"
    
    def _check_emergency_keywords(self, symptoms: List[str]) -> List[str]:
        emergency_keywords = [
            'chest pain', 'difficulty breathing', 'unconscious', 'severe bleeding',
            'heart attack', 'stroke', 'seizure', 'poisoning'
        ]
        
        symptoms_text = ' '.join(symptoms).lower()
        found_keywords = [kw for kw in emergency_keywords if kw in symptoms_text]
        return found_keywords

# Healthcare Assistant with Advanced Reasoning
class HealthcareAssistant:
    def __init__(self, db_tools: HealthcareDatabaseTools):
        self.model = genai.GenerativeModel(
            "gemini-2.0-flash",
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            }
        )
        self.db_tools = db_tools
        self.knowledge_base = MedicalKnowledgeBase()
        
        # Available tools
        self.tools = {
            "get_patient_info": self.db_tools.get_patient_info,
            "search_doctors": self.db_tools.search_doctors,
            "book_appointment": self.db_tools.book_appointment,
            "get_test_results": self.db_tools.get_test_results,
            "analyze_symptoms": self.knowledge_base.analyze_symptoms,
        }
    
    def __call__(self, state: HealthcareState) -> Dict[str, Any]:
        """Main healthcare assistant processing"""
        messages = state.get("messages", [])
        patient_id = state.get("patient_id")
        conversation_type = state.get("conversation_type", "general")
        medical_context = state.get("medical_context", {})
        
        # Build conversation context
        conversation_text = self._build_conversation_context(messages)
        
        # Determine conversation type if not set
        if not conversation_type or conversation_type == "general":
            conversation_type = self._classify_conversation(conversation_text)
        
        # System prompt based on conversation type
        system_prompt = self._get_system_prompt(conversation_type, patient_id, medical_context)
        
        try:
            # Generate response
            full_prompt = f"{system_prompt}\n\nConversation History:\n{conversation_text}\n\nHealthcare Assistant:"
            response = self.model.generate_content(full_prompt)
            response_text = response.text
            
            # Check for tool usage
            tool_calls = self._extract_tool_calls(response_text)
            
            if tool_calls:
                # Execute tools
                tool_results = []
                for tool_call in tool_calls:
                    result = self._execute_tool(tool_call, state)
                    tool_results.append(result)
                
                # Generate enhanced response with tool results
                tool_context = "\n".join([f"Tool {t['name']}: {t['result']}" for t in tool_results])
                enhanced_prompt = f"{full_prompt}\n\nTool Results:\n{tool_context}\n\nFinal Response (provide helpful, empathetic response):"
                final_response = self.model.generate_content(enhanced_prompt)
                response_text = final_response.text
            
            # Assess urgency and escalation need
            urgency_level = self._assess_response_urgency(response_text, conversation_text)
            escalation_needed = urgency_level in ["high", "emergency"]
            
            # Update medical context
            updated_context = self._update_medical_context(medical_context, conversation_text, response_text)
            
            return {
                "messages": [AIMessage(content=response_text)],
                "conversation_type": conversation_type,
                "medical_context": updated_context,
                "urgency_level": urgency_level,
                "escalation_needed": escalation_needed
            }
            
        except Exception as e:
            error_response = AIMessage(
                content=f"I apologize, but I'm experiencing technical difficulties. For urgent medical concerns, please contact your healthcare provider directly or call emergency services. Error: {str(e)}"
            )
            return {
                "messages": [error_response],
                "escalation_needed": True
            }
    
    def _get_system_prompt(self, conversation_type: str, patient_id: str = None, medical_context: Dict = None) -> str:
        """Get specialized system prompt based on conversation type"""
        
        base_prompt = """You are an advanced healthcare AI assistant with expertise in both medical knowledge and customer service. You provide compassionate, accurate, and helpful responses while maintaining professional medical standards.

CRITICAL GUIDELINES:
- Always emphasize that you're an AI assistant, not a replacement for professional medical advice
- For serious symptoms or emergencies, immediately recommend contacting healthcare providers or emergency services
- Be empathetic and reassuring while being factually accurate
- Maintain patient privacy and confidentiality
- Use clear, understandable language avoiding excessive medical jargon
"""
        
        if conversation_type == "medical_inquiry":
            return base_prompt + """
MEDICAL INQUIRY MODE:
- Analyze symptoms carefully and provide educational information
- Always recommend professional medical evaluation for diagnosis
- Assess urgency levels and guide appropriate care-seeking behavior
- Ask relevant follow-up questions to better understand the situation
- Provide general health information and preventive care guidance

Available tools:
- get_patient_info(patient_id): Get patient medical history and context
- analyze_symptoms(symptoms_list, patient_context): Analyze symptoms with medical knowledge
"""
        
        elif conversation_type == "appointment":
            return base_prompt + """
APPOINTMENT MANAGEMENT MODE:
- Help schedule, reschedule, or cancel appointments efficiently
- Provide information about preparation for appointments
- Suggest appropriate specialists based on medical needs
- Explain appointment procedures and what to expect

Available tools:
- search_doctors(specialization, availability_date): Find suitable doctors
- book_appointment(patient_id, doctor_id, date, reason): Schedule appointments
- get_patient_info(patient_id): Access patient information for context
"""
        
        elif conversation_type == "test_results":
            return base_prompt + """
TEST RESULTS MODE:
- Help interpret test results in understandable terms
- Explain normal ranges and what results mean
- Emphasize the importance of discussing results with healthcare providers
- Schedule follow-up appointments if needed

Available tools:
- get_test_results(patient_id, test_type): Retrieve patient test results
- get_patient_info(patient_id): Get patient context for result interpretation
"""
        
        else:
            return base_prompt + """
GENERAL HEALTHCARE SUPPORT MODE:
- Provide comprehensive healthcare assistance
- Identify the specific type of help needed
- Route to appropriate specialists or departments
- Offer general health information and support

Available tools: All tools available based on identified needs
"""
    
    def _classify_conversation(self, conversation_text: str) -> str:
        """Classify the type of conversation based on content"""
        text_lower = conversation_text.lower()
        
        # Emergency keywords
        if any(word in text_lower for word in ['emergency', 'urgent', 'severe pain', 'chest pain', 'breathing']):
            return "emergency"
        
        # Medical inquiry keywords
        if any(word in text_lower for word in ['symptoms', 'pain', 'feeling', 'hurts', 'diagnosis', 'condition']):
            return "medical_inquiry"
        
        # Appointment keywords
        if any(word in text_lower for word in ['appointment', 'schedule', 'book', 'reschedule', 'cancel', 'doctor']):
            return "appointment"
        
        # Test results keywords
        if any(word in text_lower for word in ['test results', 'lab results', 'blood work', 'x-ray', 'scan']):
            return "test_results"
        
        # Prescription keywords
        if any(word in text_lower for word in ['prescription', 'medication', 'refill', 'dosage', 'side effects']):
            return "prescription"
        
        # Billing keywords
        if any(word in text_lower for word in ['bill', 'insurance', 'payment', 'cost', 'coverage']):
            return "billing"
        
        return "general"
    
    def _build_conversation_context(self, messages: List[BaseMessage]) -> str:
        """Build conversation context from messages"""
        context = []
        for msg in messages[-10:]:  # Last 10 messages
            if isinstance(msg, HumanMessage):
                role = "Patient"
            elif isinstance(msg, AIMessage):
                role = "Assistant"
            else:
                role = "System"
            
            content = msg.content if hasattr(msg, 'content') else str(msg)
            context.append(f"{role}: {content}")
        return "\n".join(context)
    
    def _extract_tool_calls(self, text: str) -> List[Dict]:
        """Extract tool calls from response text"""
        tool_calls = []
        
        patterns = {
            "get_patient_info": r'get_patient_info\(["\']([^"\']+)["\']\)',
            "search_doctors": r'search_doctors\(([^)]*)\)',
            "book_appointment": r'book_appointment\(([^)]+)\)',
            "get_test_results": r'get_test_results\(([^)]*)\)',
            "analyze_symptoms": r'analyze_symptoms\(([^)]+)\)',
        }
        
        for tool_name, pattern in patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                tool_calls.append({
                    "name": tool_name,
                    "args": match
                })
        
        return tool_calls
    
    def _execute_tool(self, tool_call: Dict, state: HealthcareState) -> Dict:
        """Execute a tool call with state context"""
        tool_name = tool_call["name"]
        args = tool_call["args"]
        
        try:
            if tool_name == "get_patient_info":
                result = self.tools[tool_name](args)
            elif tool_name == "analyze_symptoms":
                # Parse symptoms from args
                symptoms = [s.strip().strip('"\'') for s in args.split(',')]
                patient_context = state.get("medical_context", {}).get("patient_info")
                result = self.tools[tool_name](symptoms, patient_context)
            elif tool_name in ["search_doctors", "get_test_results"]:
                # Handle optional parameters
                result = self.tools[tool_name]()
            else:
                result = f"Tool {tool_name} execution not implemented"
            
            return {"name": tool_name, "result": str(result)[:2000]}  # Limit result length
        except Exception as e:
            return {"name": tool_name, "result": f"Error: {str(e)}"}
    
    def _assess_response_urgency(self, response_text: str, conversation_text: str) -> str:
        """Assess urgency level based on response and conversation"""
        emergency_indicators = [
            'emergency', 'immediately', 'call 911', 'urgent care', 'severe'
        ]
        
        high_indicators = [
            'see a doctor', 'medical attention', 'concerning', 'monitor closely'
        ]
        
        combined_text = (response_text + " " + conversation_text).lower()
        
        if any(indicator in combined_text for indicator in emergency_indicators):
            return "emergency"
        elif any(indicator in combined_text for indicator in high_indicators):
            return "high"
        else:
            return "medium"
    
    def _update_medical_context(self, current_context: Dict, conversation: str, response: str) -> Dict:
        """Update medical context based on conversation"""
        updated_context = current_context.copy()
        
        # Extract symptoms mentioned
        symptom_keywords = [
            'pain', 'fever', 'headache', 'nausea', 'fatigue', 'dizziness',
            'rash', 'cough', 'shortness of breath', 'chest pain'
        ]
        
        mentioned_symptoms = []
        text_lower = conversation.lower()
        for symptom in symptom_keywords:
            if symptom in text_lower:
                mentioned_symptoms.append(symptom)
        
        if mentioned_symptoms:
            updated_context['current_symptoms'] = mentioned_symptoms
        
        # Update timestamp
        updated_context['last_updated'] = datetime.now().isoformat()
        
        return updated_context

# Graph Creation and Workflow
def create_healthcare_bot():
    """Create the healthcare bot graph"""
    
    # Initialize components
    db_config = DatabaseConfig()
    db_tools = HealthcareDatabaseTools(db_config)
    assistant = HealthcareAssistant(db_tools)
    
    # Create the graph
    workflow = StateGraph(HealthcareState)
    
    # Add nodes
    workflow.add_node("healthcare_assistant", assistant)
    
    # Simple flow for now - can be extended with routing logic
    workflow.add_edge(START, "healthcare_assistant")
    workflow.add_edge("healthcare_assistant", END)
    
    # Add memory for conversation persistence
    memory = MemorySaver()
    
    # Compile the graph
    app = workflow.compile(checkpointer=memory)
    return app

# Interactive Healthcare Bot
def run_healthcare_bot():
    """Run the interactive healthcare bot"""
    
    app = create_healthcare_bot()
    
    print("🏥 Advanced Healthcare Assistant Bot")
    print("===================================")
    print("I can help with medical inquiries, appointments, test results, and general healthcare support.")
    print("⚠️  Important: This is an AI assistant. For emergencies, call your local emergency number.")
    print("Type 'quit' to exit, 'help' for commands\n")
    
    # Session configuration
    config = {
        "configurable": {
            "thread_id": f"healthcare_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        }
    }
    
    # Initialize state
    state = {
        "messages": [],
        "patient_id": None,
        "conversation_type": "general",
        "medical_context": {},
        "urgency_level": "low",
        "escalation_needed": False,
        "session_metadata": {"start_time": datetime.now().isoformat()}
    }
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Thank you for using the Healthcare Assistant. Stay healthy!")
                break
            
            if user_input.lower() == 'help':
                print("""
Available commands:
- Ask about symptoms: "I have a headache and fever"
- Book appointments: "I need to schedule an appointment with a cardiologist"
- Check test results: "Can you show me my recent blood test results?"
- General questions: "What should I do for a minor cut?"
- Set patient ID: "My patient ID is 12345"
""")
                continue
            
            if user_input.lower().startswith('patient id'):
                patient_id = user_input.split()[-1]
                state["patient_id"] = patient_id
                print(f"✅ Patient ID set to: {patient_id}")
                continue
            
            if not user_input:
                continue
            
            # Add user message
            state["messages"].append(HumanMessage(content=user_input))
            
            # Process with healthcare bot
            result = app.invoke(state, config)
            
            # Get assistant response
            assistant_response = result["messages"][-1].content
            print(f"🤖 Healthcare Assistant: {assistant_response}\n")
            
            # Check for escalation
            if result.get("escalation_needed"):
                print("⚠️  NOTICE: This conversation may require immediate medical attention.")
                print("   Please consider contacting your healthcare provider or emergency services.\n")
            
            # Update state
            state = result
            
        except KeyboardInterrupt:
            print("\n👋 Session interrupted. Take care!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again or contact technical support.\n")

# SQL Schema for Healthcare Database (MSSQL Server)
HEALTHCARE_DB_SCHEMA = """
-- Healthcare Database Schema for MSSQL Server

-- Patients table
CREATE TABLE patients (
    patient_id VARCHAR(20) PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    date_of_birth DATE NOT NULL,
    gender VARCHAR(10),
    phone VARCHAR(15),
    email VARCHAR(100),
    address TEXT,
    emergency_contact VARCHAR(100),
    insurance_provider VARCHAR(100),
    policy_number VARCHAR(50),
    created_date DATETIME DEFAULT GETDATE(),
    updated_date DATETIME DEFAULT GETDATE()
);

-- Doctors table
CREATE TABLE doctors (
    doctor_id VARCHAR(20) PRIMARY KEY,
    doctor_name VARCHAR(100) NOT NULL,
    specialization VARCHAR(100),
    phone VARCHAR(15),
    email VARCHAR(100),
    years_experience INT,
    rating DECIMAL(3,2),
    status VARCHAR(20) DEFAULT 'active',
    created_date DATETIME DEFAULT GETDATE()
);

-- Appointments table
CREATE TABLE appointments (
    appointment_id INT IDENTITY(1,1) PRIMARY KEY,
    patient_id VARCHAR(20) FOREIGN KEY REFERENCES patients(patient_id),
    doctor_id VARCHAR(20) FOREIGN KEY REFERENCES doctors(doctor_id),
    appointment_date DATE NOT NULL,
    appointment_time TIME NOT NULL,
    status VARCHAR(20) DEFAULT 'scheduled',
    reason TEXT,
    notes TEXT,
    created_date DATETIME DEFAULT GETDATE(),
    updated_date DATETIME DEFAULT GETDATE()
);

-- Medical History table
CREATE TABLE medical_history (
    history_id INT IDENTITY(1,1) PRIMARY KEY,
    patient_id VARCHAR(20) FOREIGN KEY REFERENCES patients(patient_id),
    condition_name VARCHAR(200) NOT NULL,
    diagnosis_date DATE,
    status VARCHAR(50),
    treatment_notes TEXT,
    doctor_id VARCHAR(20) FOREIGN KEY REFERENCES doctors(doctor_id),
    created_date DATETIME DEFAULT GETDATE()
);

-- Medications table
CREATE TABLE medications (
    medication_id INT IDENTITY(1,1) PRIMARY KEY,
    medication_name VARCHAR(200) NOT NULL,
    generic_name VARCHAR(200),
    medication_type VARCHAR(100),
    manufacturer VARCHAR(100),
    description TEXT
);

-- Patient Medications table
CREATE TABLE patient_medications (
    prescription_id INT IDENTITY(1,1) PRIMARY KEY,
    patient_id VARCHAR(20) FOREIGN KEY REFERENCES patients(patient_id),
    medication_id INT FOREIGN KEY REFERENCES medications(medication_id),
    doctor_id VARCHAR(20) FOREIGN KEY REFERENCES doctors(doctor_id),
    dosage VARCHAR(100),
    frequency VARCHAR(100),
    start_date DATE,
    end_date DATE,
    status VARCHAR(20) DEFAULT 'active',
    instructions TEXT,
    created_date DATETIME DEFAULT GETDATE()
);

-- Test Results table
CREATE TABLE test_results (
    test_id INT IDENTITY(1,1) PRIMARY KEY,
    patient_id VARCHAR(20) FOREIGN KEY REFERENCES patients(patient_id),
    test_name VARCHAR(200) NOT NULL,
    test_date DATE NOT NULL,
    result_value VARCHAR(500),
    normal_range VARCHAR(200),
    unit VARCHAR(50),
    status VARCHAR(50),
    notes TEXT,
    ordered_by VARCHAR(20) FOREIGN KEY REFERENCES doctors(doctor_id),
    created_date DATETIME DEFAULT GETDATE()
);

-- Sample data insertion scripts
INSERT INTO doctors (doctor_id, doctor_name, specialization, phone, email, years_experience, rating) VALUES
('DOC001', 'Dr. Sarah Johnson', 'Cardiology', '555-0101', 'sarah.johnson@hospital.com', 15, 4.8),
('DOC002', 'Dr. Michael Chen', 'Internal Medicine', '555-0102', 'michael.chen@hospital.com', 12, 4.7),
('DOC003', 'Dr. Emily Rodriguez', 'Pediatrics', '555-0103', 'emily.rodriguez@hospital.com', 8, 4.9),
('DOC004', 'Dr. James Wilson', 'Orthopedics', '555-0104', 'james.wilson@hospital.com', 20, 4.6),
('DOC005', 'Dr. Lisa Thompson', 'Dermatology', '555-0105', 'lisa.thompson@hospital.com', 10, 4.8);

INSERT INTO medications (medication_name, generic_name, medication_type, manufacturer) VALUES
('Lisinopril', 'Lisinopril', 'ACE Inhibitor', 'Generic Pharma'),
('Metformin', 'Metformin HCl', 'Diabetes Medication', 'Generic Pharma'),
('Ibuprofen', 'Ibuprofen', 'NSAID', 'Generic Pharma'),
('Amoxicillin', 'Amoxicillin', 'Antibiotic', 'Generic Pharma'),
('Simvastatin', 'Simvastatin', 'Statin', 'Generic Pharma');
"""

if __name__ == "__main__":
    print("🏥 Healthcare Bot with LangGraph")
    print("===============================")
    
    print("\nDatabase Setup Instructions:")
    print("1. Set up MSSQL Server database")
    print("2. Run the provided schema script")
    print("3. Set environment variables:")
    print("   - MSSQL_SERVER")
    print("   - MSSQL_DATABASE") 
    print("   - MSSQL_USERNAME")
    print("   - MSSQL_PASSWORD")
    print("   - GEMINI_API_KEY")
    
    print("\nLangGraph Architecture:")
    print("- State: Healthcare-specific state with medical context")
    print("- Nodes: Healthcare assistant with medical reasoning")
    print("- Tools: Database queries and medical analysis")
    print("- Memory: Conversation persistence with medical context")
    
    print("\nStarting Healthcare Bot...")
    run_healthcare_bot()
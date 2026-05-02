import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from twilio.rest import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Admin Details provided by user
ADMIN_EMAIL = "himanshu_kumar_ds_23@jagannathuniversityncr.ac.in"
ADMIN_PHONE = "+919268631084" # Ensure E.164 format for Twilio

def send_email_notification(new_username, new_email):
    sender_email = os.getenv("SMTP_EMAIL")
    sender_password = os.getenv("SMTP_PASSWORD")
    
    if not sender_email or not sender_password:
        return False, "SMTP credentials missing"

    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = ADMIN_EMAIL
        msg['Subject'] = f"🚨 New User Registration: {new_username}"
        
        body = f"""
        Hello Admin,
        
        A new user has just registered on the Road Accident Prediction platform!
        
        User Details:
        - Username: {new_username}
        - Email: {new_email}
        
        Best regards,
        Road Accident Prediction System
        """
        msg.attach(MIMEText(body, 'plain'))
        
        # Using Gmail's SMTP server as default
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True, "Email sent"
    except Exception as e:
        return False, str(e)

def send_sms_notification(new_username, new_email):
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_number = os.getenv("TWILIO_PHONE_NUMBER")
    
    if not account_sid or not auth_token or not twilio_number:
        return False, "Twilio credentials missing"
        
    try:
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=f"🚨 New User Alert! Username: {new_username}, Email: {new_email} has registered on the platform.",
            from_=twilio_number,
            to=ADMIN_PHONE
        )
        return True, "SMS sent"
    except Exception as e:
        return False, str(e)

def notify_admin_of_new_user(new_username, new_email):
    print(f"\n--- [SYSTEM LOG] NEW USER NOTIFICATION ---")
    print(f"Triggering notification for Admin ({ADMIN_EMAIL} | {ADMIN_PHONE})")
    print(f"New User: {new_username} ({new_email})")
    
    email_success, email_msg = send_email_notification(new_username, new_email)
    sms_success, sms_msg = send_sms_notification(new_username, new_email)
    
    print(f"Email Status: {'SUCCESS' if email_success else 'SKIPPED/FAILED'} - {email_msg}")
    print(f"SMS Status: {'SUCCESS' if sms_success else 'SKIPPED/FAILED'} - {sms_msg}")
    print("------------------------------------------\n")

def send_user_welcome_email(user_email, first_name, login_id):
    sender_email = os.getenv("SMTP_EMAIL")
    sender_password = os.getenv("SMTP_PASSWORD")
    
    if not sender_email or not sender_password or not user_email:
        return False, "SMTP credentials missing or no user email provided"

    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = user_email
        msg['Subject'] = "🎉 Welcome to Road Accident Prediction Platform!"
        
        body = f"""
        Hello {first_name},
        
        Your account has been successfully created!
        
        Please save your login details:
        - Login ID: {login_id}
        
        You can use this Login ID and the password you created to sign in to the platform.
        
        Best regards,
        Road Accident Prediction System
        """
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True, "User Email sent"
    except Exception as e:
        return False, str(e)

def send_user_welcome_sms(user_contact, first_name, login_id):
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_number = os.getenv("TWILIO_PHONE_NUMBER")
    
    # Twilio requires E.164 format (e.g. +919268631084)
    # Since users might not enter the '+', we can attempt a naive clean or just pass it directly.
    if not user_contact.startswith("+"):
        user_contact = "+" + user_contact.lstrip("0")
        
    if not account_sid or not auth_token or not twilio_number or not user_contact:
        return False, "Twilio credentials missing or no contact provided"
        
    try:
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=f"Hi {first_name}, your account is successfully created! Your Login ID is: {login_id}. Welcome to the Road Accident Prediction platform.",
            from_=twilio_number,
            to=user_contact
        )
        return True, "User SMS sent"
    except Exception as e:
        return False, str(e)

def notify_user_of_registration(user_email, user_contact, first_name, login_id):
    print(f"\n--- [SYSTEM LOG] NEW USER WELCOME NOTIFICATION ---")
    print(f"Triggering notification for User ({user_email} | {user_contact})")
    
    email_success, email_msg = send_user_welcome_email(user_email, first_name, login_id)
    sms_success, sms_msg = send_user_welcome_sms(user_contact, first_name, login_id)
    
    print(f"User Email Status: {'SUCCESS' if email_success else 'SKIPPED/FAILED'} - {email_msg}")
    print(f"User SMS Status: {'SUCCESS' if sms_success else 'SKIPPED/FAILED'} - {sms_msg}")
    print("--------------------------------------------------\n")

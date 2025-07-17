import logging
import azure.functions as func
import pandas as pd
import numpy as np
import re
import time
import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv(override=True)

# Define your system and user prompt
system_prompt = '''You are an email classifier for ARM ONE, a digital investment platform in Nigeria. Classify customer emails into one of the 20 categories below.
Context

ARM ONE: Digital platform for Asset Management and Investment Management
Investment Products include: Mutual Funds e,.g. Money Market Funds, Fixed Income Funds, Equity Funds, Real Estate Funds, etc.
Securities Products include: Stocks, bonds, fixed incomes, et.c.
Payment platforms: Monnify, Lemfi, Paystack, Flutterwave, Mono
Subscription: Customer deposits/investments
Redemption: Customer withdrawals
CSCS: Central Securities Clearing System, a Nigerian clearing house for securities transactions
PFA: Pension Fund Administrators
RSA: Retirement Savings Account, a Nigerian pension account
ARM Contact Addresses: Ikoyi, Lagos, Port-Harcourt, Abuja, Onitsha, Ibadan
Requests: Means customer wanting an action to be taken
Enquiries: Means customer asking questions or wanting to know more about a process
Complaints: Means customer is dissatisfied or angry about a service or product
Status Update: Means customer wants to know the status of a request or enquiry
Acknowledgement: Means customer wants confirmation of a request or action taken (Like Subscription or Redemption Acknowledgement)

Account Access & Authentication
•	Activate Account - First-time access requests for ARM ONE app/website
•	Login Details Request - Username/password credential requests
•	Password Reset - Password change requests for account recovery
Account Information & Management
•	Unstamped Statement Request - Transaction breakdown for specified date ranges
•	Stamped Statement Request - Stamped Transaction breakdown for specified date ranges around Embassy Letter Requests
•	Investment Details - Requests for specific information regarding investment accounts, including membership number, portfolio composition, and related financial inquiries
•	Membership ID - Unique account identifier requests
•	Relationship Manager Request - Assigned sales team member inquiries
•	Account Linking – Securities and Investment Account under one login (For Example, linking a Mutual Fund Account with a Securities Account, or linking a Securities Account with a Mutual Fund Account)
•	Account Upgrade - Tier 1 (Basic) to Tier 3 (Premium) upgrade requests
•	Account Cancellation Request - Requests to close or cancel an account or stop investments subscription
•	Account Switching - Multiple account consolidation under single login (For example Parent and Children Account, Husband and Wife Account, etc.)
•	Investment Subscription Enquiry - Customer questions regarding the fund payment or subscription process (of investment products)
•	Investment Subscription Acknowledgement - Acknowledgement OR Confirmation of subscription request from client after making payment(A customer had made a subscription and wants an acknowledgement or confirmation)
•	Investment Subscription Request - Formal instructions from a customer to subscribe to a mutual fund
•	Purchase Mutual Fund - Requests or information related to initiating mutual fund investments
•	Purchase Securities - Instructions or queries about buying equities or tradable securities
•	Sell Securities - Requests or actions related to selling owned securities or equities
•	Investment Redemption Status Update - Updates on the completion, delay, or progress of a fund withdrawal (on Investment Products)
•	Investment Subscription Status Update - Updates on the completion, delay, or progress of a fund payment (on Investment Products)
•	Investment Redemption Request - Formal instructions from a customer to withdraw from investments products
•	Investment Redemption Acknowledgement – Acknowledgement of redemption request from client after making withdrawals (A customer had made a redemption and wants an acknowledgement)
•	Investment Redemption Enquiry - Customer questions regarding the withdrawal or redemption process of Investment Products
•	Unit Transfer Enquiry- Customer wants to know the process for Unit transfer 
•	Unit Transfer Request- Requests to move investment units between accounts or across mutual funds
•	Direct Debit Setup - Requests to authorize automatic bank deductions via the ArmOne platform
•	Securities Redemption Enquiry - Customer questions regarding the withdrawal or redemption process of securities products
•	Securities Redemption Request - Formal instructions from a customer to withdraw from securities products
•	Securities Subscription Enquiry -  Customer questions regarding the payment process of securities products
•	Securities Subscription Status Update - Updates on the completion, delay, or progress of a fund payment (on Securities Products)
•	Securities Redemption Status Update - Updates on the completion, delay, or progress of a fund withdrawal  (on Securities Products)
•	Securities Subscription Acknowledgement – Acknowledgement or confirmation of subscription request from client after making payment (A customer had made a subscription and wants an acknowledgement)
•	Securities Redemption Acknowledgement – Acknowledgement or confirmation of redemption request from client after a client makes a withdrawal (A customer had made a redemption and wants an acknowledgement)
•	
Investment Details & Support
•	Interest Rate Inquiry - Annual interest rate affecting fund returns
•	Dividend Mandate Enquiry – Customer wants to know the process for dividend mandate or asks questions regarding dividend payments or recovery
•	Dividend Mandate Request - Update dividend reinvestment/payment instructions
•	Embassy Letter Request - Investment documentation for visa applications
•	Embassy Letter Enquiry – Customer wants to know the process to get Embassy Letter
•	Record Update Request - Requests to update personal information or account records / Client has requested to change an information in his Biodata with us
General Inquiries
•	Investment Account Opening Enquiries - Customer Requests for New account setup process for Mutual Funds (MMF)
•	Account Opened Inquiry SEC - Customer Requests for New account setup process for Securities
•	Contact Address General - Customer Requests for ARM office locations and addresses
•	Contact Address Lagos - Customer Requests for ARM office locations and addresses in Lagos
•	Contact Address Ikoyi - Customer Requests for Ikoyi ARM office location and address
•	Contact Address Port-Harcourt - Customer Requests for PH ARM office location and address
•	Contact Address Abuja - Customer Requests for Abuja ARM office location and address
•	Contact Address Onitsha - Customer Requests for Onitsha ARM office location and address
•	Contact Address Ibadan - Customer Requests for Ibadan ARM office location and address
•	Product Information Enquiry - Customer Requests for information about ARM products and services




Instructions
Classify each email into exactly one category. Consider the primary intent and most specific applicable category.
'''

user_prompt_template = '''You are a customer service Email assistant for an Asset Management and Investment company. Your task is to read incoming emails from customers and classify them into one or more of the following categories. Customers may have multiple requests in a single email, so identify up to 3 different categories if applicable:
these are the categories:

Activate Account
Login Details Request
Password Reset
Unstamped Statement Request
Stamped Statement Request
Investment Details
Membership ID
Relationship Manager Request
Account Linking
Account Upgrade
Account Cancellation Request
Account Switching
Investment Subscription Enquiry
Investment Subscription Acknowledgement
Investment Subscription Request
Purchase Mutual Fund
Purchase Securities
Sell Securities
Investment Redemption Status Update
Investment Subscription Status Update
Investment Redemption Request
Investment Redemption Acknowledgement
Investment Redemption Enquiry
Unit Transfer Enquiry
Unit Transfer Request
Direct Debit Setup
Securities Redemption Enquiry
Securities Redemption Request
Securities Subscription Enquiry
Securities Subscription Status Update
Securities Redemption Status Update
Securities Subscription Acknowledgement
Securities Redemption Acknowledgement
Interest Rate Inquiry
Dividend Mandate Enquiry
Dividend Mandate Request
Embassy Letter Request
Embassy Letter Enquiry
Record Update Request
Investment Account Opening Enquiries
Account Opened Inquiry SEC
Contact Address General
Contact Address Lagos
Contact Address Ikoyi
Contact Address Port-Harcourt
Contact Address Abuja
Contact Address Onitcha
Contact Address Ibadan
Product Information Enquiry


Instructions:
- Analyze the email content carefully for multiple requests or topics
- Identify up to 3 different categories that apply to the email
- If only one category applies, only provide Category1
- If two categories apply, provide Category1 and Category2
- If three categories apply, provide Category1, Category2, and Category3
- If the email does not fit any categories, leave the category as blank(dont fill in anything)
- List categories in order of importance/prominence in the email
- Embassy Letter Request has Statement Request as a subcategory, so if both are present, classify as Embassy Letter Request
- Do not classify emails that are not related to the categories above
- Usually, Statement Requests have investment details, so if confused, prioritize Statement Request over Investment Details
- For each category assigned, provide a confidence level (0-100%) indicating how certain you are about the classification
- Attach confidence levels to only categories that are assigned
- There is a difference between Subscription Request and Subscription Enquiry so as Redemption Request and Redemption Enquiry, so ensure you classify them correctly
- Login issues on app or website should be classified under ARM ONE App Issues
- If there seems to be an attachment in the mail, links or documents, dont classify the email, just return blank for all categories and confidence levels
- If the email is a reply to a previous email, do not classify it, just return blank for all categories and confidence levels
- Treasury Bills/ T Bills, Commercial Papers, Bonds are securities products
- Dont Classify Anything that looks like a spam or phishing email, just return blank for all categories and confidence levels
- Dont Classify anything that looks like a complaint, or looks like a customer is angry or showing dissatisfaction, just return blank for all categories and confidence levels
- Take note of Investment/Securities Subscription Status Update and Investment/Securities Redemption Status Update, they are quite tricky, so ensure you classify them correctly This is an example of how the Email looks like: "   [Hello,    I made a payment of N250, 000 into my virtual account since over a month ago. I wanted to use the money to subscribe to FG bond but the money has not been processed and made available to me.    Kindly help or advise on what to do.    Thank you.    Regards.],[   Good morning, I made a transaction to my money market account and it has not been credited.   Thanks, Sola] "
- If client does not specify invesment products or you are not sure for subsciption request, subscription enquiry, redemption request or redemption enquiry, just classify as Investment Subscription Request, Investment Subscription Enquiry, Investment Redemption Request or Investment Redemption Enquiry respectively.
- If clients complain about payment charges from paystack while using ARM one, dont classify the email, just return blank for all categories and confidence levels
- If clients complain about payment charges from Monnify while using ARM one, dont classify the email, just return blank for all categories and confidence levels
- If clients complain about payment charges from Lemfi while using ARM one, dont classify the email, just return blank for all categories and confidence levels
- If clients complain about payment charges from Flutterwave while using ARM one, dont classify the email, just return blank for all categories and confidence levels
- If clients complain about payment charges from Mono while using ARM one, dont classify the email, just return blank for all categories and confidence levels
- If clients complain about payment charges from any other payment platform while using ARM one, dont classify the email, just return blank for all categories and confidence levels
- Dont Classify anything that looks like a complaint, or looks like a customer is angry or showing dissatisfaction, just return blank for all categories and confidence levels
- Dont Classify anything thats related to pensions, retirement, or pension funds, just return blank for all categories and confidence levels.
- This is an example of Account Switching:    Hello   I need help with connecting my daughter’s account to mine so i can view together in the App Thank you
- Dont Classify any mail that looks like has an attachment, links or documents, just return blank for all categories and confidence levels
- If the given CONTACT ADDRESS is not one of the ARM Contact Addresses, return General Contact Address
- Any Issue related to App Issues, should also be classified under ARM ONE App Issues
- Cliets asking for Portfolio Analysis also means they want Statement Request, so classify as Stamped Statement Request or Unstamped Statement Request depending on the request
- If Cliets are having issues accessing their account, it should be classified under ARM ONE App Issues
Email Sender:
"""
{email_sender}
"""
 
Email Subject:
"""
{email_subject}
"""

Email Body:
"""
{email_body}
"""

Output format:
Category1: [Primary Category]
Confidence1: [Confidence percentage for Category1 (0-100%)]
Category2: [Secondary Category if applicable]
Confidence2: [Confidence percentage for Category2 (0-100%)]
Category3: [Tertiary Category if applicable]
Confidence3: [Confidence percentage for Category3 (0-100%)]
'''

def clean_email(df):
    # Define the sentence to be removed
    sentence_to_remove = "EXTERNAL: This is an external email and may be malicious. Please take care when clicking links or opening attachments."
    sentence_to_remove2 = "Sent from my iPhone"
    sentence_to_remove3 = "Sent from my Samsung Galaxy smartphone"
    
    # Define regex patterns for phone numbers and membership IDs
    phone_NIN_pattern = re.compile(r'\b\d{11}\b')
    membership_id_pattern = re.compile(r'\b\d{7}\b')
    account_nos_pattern = re.compile(r'\b\d{10}\b')
    email_pattern = re.compile(r'\b\w*@\w*\b')
    
    # Function to clean text by removing unwanted characters
    def clean_text(text):
        if isinstance(text, str):
            text = text.replace('\n', ' ')
            text = text.replace('\xa0', ' ')
            text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-printable characters
        return text
    
    # Function to apply regex and handle non-string values
    def mask_text(text):
        if isinstance(text, str):
            text = text.replace(sentence_to_remove, '')
            text = text.replace(sentence_to_remove2, '')
            text = text.replace(sentence_to_remove3, '')
            text = phone_NIN_pattern.sub('***********', text)
            text = membership_id_pattern.sub('*******',text)
            text = account_nos_pattern.sub('**********',text)
            text = email_pattern.sub('***@email.com', text)
        return text
    
    # Clean the EmailBody column
    df['EmailBody'] = df['EmailBody'].apply(clean_text)
    
    # Apply the mask_text function to the EmailBody column
    df['EmailBody'] = df['EmailBody'].apply(mask_text)
    
    # Note: Sender column is NOT cleaned - keeping original sender information intact
    
    return df

# Function to call the Azure OpenAI API
def classify_email(email_sender, email_subject, email_body, azure_api_key, azure_endpoint, azure_deployment_name, api_version):
    try:
        # Ensure inputs are strings
        email_sender = str(email_sender) if email_sender is not None else ""
        email_subject = str(email_subject) if email_subject is not None else ""
        email_body = str(email_body) if email_body is not None else ""
        
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            api_key=azure_api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )
        
        # Call Azure OpenAI API
        response = client.chat.completions.create(
            model=azure_deployment_name,  # In Azure OpenAI, you use the deployment name instead of model name
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_template.format(email_sender=email_sender, email_subject=email_subject, email_body=email_body)}
            ],
            max_tokens=4096,
            temperature=1.0,
            top_p=1.0,
        )
        
        # Parse response
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error with Azure OpenAI API: {e}")
        # Add a small delay before retrying to avoid rate limits
        time.sleep(5)
        return "Error: API call failed"

# Function to extract multiple categories and their confidence levels from the response
def extract_categories_with_confidence(response):
    # Ensure response is a string
    if not isinstance(response, str):
        response = str(response)
    
    # Initialize categories and confidence levels
    category1 = ""
    confidence1 = ""
    category2 = ""
    confidence2 = ""
    category3 = ""
    confidence3 = ""
    
    # Look for Category and Confidence patterns
    lines = response.split('\n')
    for line in lines:
        line = line.strip()
        
        if line.startswith("Category1:"):
            category1 = line.split("Category1:")[1].strip()
            category1 = category1.replace('[', '').replace(']', '')
        elif line.startswith("Confidence1:"):
            confidence1 = line.split("Confidence1:")[1].strip()
            confidence1 = extract_confidence_percentage(confidence1)
        elif line.startswith("Category2:"):
            category2 = line.split("Category2:")[1].strip()
            category2 = category2.replace('[', '').replace(']', '')
        elif line.startswith("Confidence2:"):
            confidence2 = line.split("Confidence2:")[1].strip()
            confidence2 = extract_confidence_percentage(confidence2)
        elif line.startswith("Category3:"):
            category3 = line.split("Category3:")[1].strip()
            category3 = category3.replace('[', '').replace(']', '')
        elif line.startswith("Confidence3:"):
            confidence3 = line.split("Confidence3:")[1].strip()
            confidence3 = extract_confidence_percentage(confidence3)
    
    # If no structured format found, try to parse the old format for backward compatibility
    if not category1 and "Category:" in response:
        category1 = response.split("Category:")[1].strip()
        category1 = category1.replace('[', '').replace(']', '')
        confidence1 = "80"  # Default confidence for backward compatibility
    
    # Clean up empty or "if applicable" responses
    def clean_category(cat):
        if not cat or cat.lower() in ['', 'n/a', 'none', 'if applicable', 'not applicable']:
            return ""
        return cat
    
    def clean_confidence(conf):
        if not conf or conf == "":
            return ""
        return conf
    
    return (clean_category(category1), clean_confidence(confidence1),
            clean_category(category2), clean_confidence(confidence2),
            clean_category(category3), clean_confidence(confidence3))

# Function to extract confidence percentage from text
def extract_confidence_percentage(confidence_text):
    if not confidence_text:
        return ""
    
    # Remove common words and symbols
    confidence_text = confidence_text.replace('%', '').replace('percent', '').strip()
    confidence_text = re.sub(r'[^\d.]', '', confidence_text)  # Keep only digits and decimal points
    
    try:
        # Convert to float and ensure it's between 0-100
        conf_value = float(confidence_text)
        if conf_value > 100:
            conf_value = 100
        elif conf_value < 0:
            conf_value = 0
        return str(int(conf_value))  # Return as integer string
    except (ValueError, TypeError):
        return "80"  # Default confidence if parsing fails

# Process the dataframe
def process_dataframe(df, azure_api_key, azure_endpoint, azure_deployment_name, api_version, batch_size=10):
    results_cat1 = []
    results_conf1 = []
    results_cat2 = []
    results_conf2 = []
    results_cat3 = []
    results_conf3 = []
    
    # Process in batches
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        
        batch_results_cat1 = []
        batch_results_conf1 = []
        batch_results_cat2 = []
        batch_results_conf2 = []
        batch_results_cat3 = []
        batch_results_conf3 = []
        
        for _, row in batch.iterrows():
            # Add error handling for missing columns
            required_fields = ['Subject', 'EmailBody']
            missing_fields = [field for field in required_fields if field not in row]
            
            if missing_fields:
                error_msg = f"Missing required fields: {', '.join(missing_fields)}"
                batch_results_cat1.append(error_msg)
                batch_results_conf1.append("")
                batch_results_cat2.append("")
                batch_results_conf2.append("")
                batch_results_cat3.append("")
                batch_results_conf3.append("")
                continue
            
            # Get sender (optional field)
            email_sender = row.get('Sender', '')
            email_subject = row['Subject']
            email_body = row['EmailBody']
            
            if (pd.isna(email_subject) or email_subject == "") and (pd.isna(email_body) or email_body == ""):
                batch_results_cat1.append("No email content")
                batch_results_conf1.append("")
                batch_results_cat2.append("")
                batch_results_conf2.append("")
                batch_results_cat3.append("")
                batch_results_conf3.append("")
                continue
                
            response = classify_email(email_sender, email_subject, email_body, azure_api_key, azure_endpoint, azure_deployment_name, api_version)
            category1, confidence1, category2, confidence2, category3, confidence3 = extract_categories_with_confidence(response)
            
            batch_results_cat1.append(category1)
            batch_results_conf1.append(confidence1)
            batch_results_cat2.append(category2)
            batch_results_conf2.append(confidence2)
            batch_results_cat3.append(category3)
            batch_results_conf3.append(confidence3)
            
            # Add a small delay to avoid rate limits
            time.sleep(0.5)
        
        results_cat1.extend(batch_results_cat1)
        results_conf1.extend(batch_results_conf1)
        results_cat2.extend(batch_results_cat2)
        results_conf2.extend(batch_results_conf2)
        results_cat3.extend(batch_results_cat3)
        results_conf3.extend(batch_results_conf3)
        
    return results_cat1, results_conf1, results_cat2, results_conf2, results_cat3, results_conf3

# Create the FunctionApp
app = func.FunctionApp()

@app.function_name(name="classify_emails")
@app.route(route="classify_emails", auth_level=func.AuthLevel.FUNCTION)
def classify_emails_http(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        # Try to get Azure OpenAI configuration from environment variables
        azure_api_key = os.environ.get('AZURE_OPENAI_API_KEY') or os.getenv('azure_api_key2')
        azure_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT') or os.getenv('azure_endpoint2')
        azure_deployment_name = os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME') or os.getenv('azure_deployment_name2')
        # api_version = os.environ.get('AZURE_OPENAI_API_VERSION') or os.getenv('api_version2', '2025-01-01')
        api_version = os.environ.get('AZURE_OPENAI_API_VERSION') or os.getenv('api_version2')

        # Ensure all environment variables are strings
        if azure_api_key is not None: azure_api_key = str(azure_api_key)
        if azure_endpoint is not None: azure_endpoint = str(azure_endpoint)
        if azure_deployment_name is not None: azure_deployment_name = str(azure_deployment_name)
        if api_version is not None: api_version = str(api_version)

        # Validate Azure OpenAI configuration
        if not azure_api_key or not azure_endpoint or not azure_deployment_name:
            return func.HttpResponse(
                "Please configure your Azure OpenAI settings in the application settings. Required: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME",
                status_code=400
            )

        # Get request body
        try:
            req_body = req.get_json()
        except ValueError:
            return func.HttpResponse(
                "Invalid request body. Please provide a valid JSON payload with email data.",
                status_code=400
            )

        # Check if the payload contains email data
        if not req_body or not isinstance(req_body, dict):
            return func.HttpResponse(
                "Please provide a valid JSON payload with email data.",
                status_code=400
            )

        # Handle different possible payload formats
        emails_data = None
        
        # Option 1: Payload has a direct list of emails
        if 'emails' in req_body and isinstance(req_body['emails'], list):
            emails_data = req_body['emails']
        # Option 2: The entire payload is the list of emails
        elif isinstance(req_body, list):
            emails_data = req_body
        # Option 3: Single email in the payload
        elif 'Subject' in req_body and 'EmailBody' in req_body:
            emails_data = [req_body]
        
        if not emails_data:
            return func.HttpResponse(
                "Invalid payload format. Expected a list of email objects with 'Subject' and 'EmailBody' fields (and optionally 'Sender'), or a single email object.",
                status_code=400
            )

        # Convert to DataFrame
        df = pd.DataFrame(emails_data)
        logging.info(f"Processed payload with {len(df)} emails")

        # Check if required columns exist
        required_columns = ['Subject', 'EmailBody']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return func.HttpResponse(
                f"Error: Missing required columns: {', '.join(missing_columns)}. Each email object must have 'Subject' and 'EmailBody' fields. 'Sender' field is optional.",
                status_code=400
            )

        # Log available columns for debugging
        logging.info(f"Available columns: {list(df.columns)}")
        
        # Clean the email data
        df = clean_email(df)

        # Process the dataframe
        logging.info(f"Processing {len(df)} emails using Azure OpenAI deployment: {azure_deployment_name}")
        categories1, confidences1, categories2, confidences2, categories3, confidences3 = process_dataframe(df, azure_api_key, azure_endpoint, azure_deployment_name, api_version)

        # Add results as new columns
        df['EmailCategory1'] = categories1
        df['EmailConfidence1'] = confidences1
        df['EmailCategory2'] = categories2
        df['EmailConfidence2'] = confidences2
        df['EmailCategory3'] = categories3
        df['EmailConfidence3'] = confidences3

        # Prepare the response
        response_data = df.to_dict(orient='records')
        
        # Return JSON response
        return func.HttpResponse(
            body=json.dumps({"results": response_data}),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return func.HttpResponse(
            f"An error occurred: {str(e)}",
            status_code=500
        )
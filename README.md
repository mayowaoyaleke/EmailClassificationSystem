# Email Classification System

This project is an **AI-powered email classification system** designed for ---- , a digital investment platform in Nigeria. The system classifies customer emails into predefined categories to streamline customer service operations.

## Features

- **Email Classification**: Classifies emails into up to 3 categories based on their content.
- **Azure OpenAI Integration**: Uses Azure OpenAI for natural language processing.
- **Data Cleaning**: Cleans email content to remove sensitive information like phone numbers, membership IDs, and email addresses.
- **Confidence Levels**: Provides confidence levels (0-100%) for each classification.
- **Error Handling**: Handles missing fields, invalid payloads, and API errors gracefully.

## Categories

The system supports classification into the following categories:

- Account Access & Authentication (e.g., Activate Account, Password Reset)
- Account Information & Management (e.g., Stamped Statement Request, Account Linking)
- Investment Details & Support (e.g., Investment Subscription Request, Unit Transfer Request)
- General Inquiries (e.g., Contact Address Lagos, Product Information Enquiry)

For a full list of categories, refer to the `system_prompt` in [function_app.py](ARMEMAILCLASS2/function_app.py).


## Setup Instructions

### Prerequisites

- Python 3.8 or later
- Azure Functions Core Tools
- Azure OpenAI Service
- Visual Studio Code with the following extensions:
  - [Azure Functions](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-azurefunctions)
  - [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/Email_Classification_System.git
   cd Email_Classification_System/ARMEMAILCLASS2

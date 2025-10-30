""" Python script that scraps Gmail emails, and uses OpenAI's API 
to provide a summary of trainee feedback on mini-mock interviews over time."""

import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from dotenv import load_dotenv
import base64
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pathlib import Path
from openai import OpenAI

# Load environment variables
load_dotenv()

# Gmail API scope - read-only access
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']


def authenticate_gmail() -> Any:
    """Authenticate and return Gmail API service."""
    creds: Optional[Credentials] = None

    # Token.json stores the user's access and refresh tokens
    # It's created automatically when the authorization flow completes for the first time
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    # If there are no (valid) credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)


def get_feedback_emails(service: Any) -> List[Dict[str, Any]]:
    """Fetch all emails with 'Mini-Mock Interview Feedback' in the subject."""
    try:
        # Search for emails with the specific subject
        results = service.users().messages().list(
            userId='me',
            q='subject:"Mini-Mock Interview Feedback"',
            maxResults=10
        ).execute()

        messages = results.get('messages', [])

        if not messages:
            print("No feedback emails found.")
            return []

        print(f"Found {len(messages)} feedback email(s)")

        # Fetch full message details for each email
        emails: List[Dict[str, Any]] = []
        for msg in messages[:10]:
            message = service.users().messages().get(
                userId='me',
                id=msg['id'],
                format='full'
            ).execute()
            emails.append(message)

        return emails

    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def extract_email_body(message: Dict[str, Any]) -> str:
    """Extract plain text body from email message."""
    payload = message.get('payload', {})
    body: str = ""

    # Check if body is directly in payload
    if 'body' in payload and 'data' in payload['body']:
        body = base64.urlsafe_b64decode(
            payload['body']['data']).decode('utf-8')
    # Check if body is in parts (multipart message)
    elif 'parts' in payload:
        for part in payload['parts']:
            if part['mimeType'] == 'text/plain':
                if 'data' in part['body']:
                    body = base64.urlsafe_b64decode(
                        part['body']['data']).decode('utf-8')
                    break

    return body


def extract_email_date(message: Dict[str, Any]) -> str:
    """Return the email timestamp in local time if available."""
    internal_date = message.get('internalDate')
    if internal_date:
        try:
            timestamp = datetime.fromtimestamp(
                int(internal_date) / 1000, tz=timezone.utc
            ).astimezone()
            return timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')
        except (OSError, OverflowError, ValueError):
            pass

    headers = message.get('payload', {}).get('headers', [])
    for header in headers:
        if header.get('name') == 'Date' and header.get('value'):
            return header['value']

    return 'Unknown date'


def summarize_feedback_with_openai(email_bodies: List[str]) -> Optional[str]:
    """Send email bodies to OpenAI and return the summary text."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("OpenAI API key not found. Skipping summary generation.")
        return None

    client = OpenAI(api_key=api_key)

    prompt = (
        "Here are a list of emails of interview feedback that trainees have received on their "
        "behavioural interviews. The format of all the emails are the same. It starts off with a "
        "RAG scoring system on 5 different topics. Followed by more detailed feedback about any "
        "questions that were asked in the interview. Provide a summary of what the trainee has "
        "improved over time, and what they still need to work on. Make the summary concise and clear."
        "No more than 200    words."
    )

    combined_emails = "\n\n".join(
        f"Email {index + 1}:\n{body}" for index, body in enumerate(email_bodies)
    )

    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are an expert interview coach summarizing performance trends."},
                {"role": "user", "content": f"{prompt}\n\n{combined_emails}"}
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"Failed to generate summary with OpenAI: {exc}")
        return None


def save_summary_markdown(summary: Optional[str]) -> Path:
    """Write a Markdown report containing the GPT analysis."""
    output_path = Path("feedback_summary.md")
    lines = [
        "# Trainee Interview Feedback Summary",
        "",
        "## GPT Analysis",
        "",
    ]

    if summary:
        lines.append(summary)
    else:
        lines.append("_No summary generated._")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    # Authenticate with Gmail
    print("Authenticating with Gmail...")
    service = authenticate_gmail()
    print("Authentication successful!\n")

    # Fetch feedback emails
    print("Fetching feedback emails...")
    emails = get_feedback_emails(service)

    # Collect email bodies for summarization
    email_bodies: List[str] = []
    for email in emails:
        body = extract_email_body(email)
        email_bodies.append(body)

    summary = summarize_feedback_with_openai(email_bodies)

    output_file = save_summary_markdown(summary)
    print(f"\nSaved Markdown summary to {output_file.resolve()}")


if __name__ == '__main__':
    main()

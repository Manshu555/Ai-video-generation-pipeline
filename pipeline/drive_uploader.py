"""
Optional Google Drive upload for final submission.
Requires credentials.json from Google Cloud Console (Drive API enabled).
"""
from pathlib import Path


def upload_to_drive(file_path: Path, folder_name: str = "Viral Reel Videos") -> str | None:
    """
    Upload a file to Google Drive inside a named folder.
    Returns the shareable link or None on failure.
    """
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        import os

        SCOPES = ["https://www.googleapis.com/auth/drive.file"]
        creds = None

        token_path = Path("token.json")
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

        if not creds or not creds.valid:
            from google.auth.transport.requests import Request
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                from config import GOOGLE_DRIVE_CREDENTIALS_FILE
                flow = InstalledAppFlow.from_client_secrets_file(
                    GOOGLE_DRIVE_CREDENTIALS_FILE, SCOPES
                )
                creds = flow.run_local_server(port=0)
            token_path.write_text(creds.to_json())

        service = build("drive", "v3", credentials=creds)

        # Find or create folder
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = service.files().list(q=query, fields="files(id)").execute()
        folders = results.get("files", [])

        if folders:
            folder_id = folders[0]["id"]
        else:
            folder_meta = {
                "name": folder_name,
                "mimeType": "application/vnd.google-apps.folder",
            }
            folder = service.files().create(body=folder_meta, fields="id").execute()
            folder_id = folder["id"]

        # Upload file
        file_meta = {"name": file_path.name, "parents": [folder_id]}
        media = MediaFileUpload(str(file_path), resumable=True)
        uploaded = service.files().create(body=file_meta, media_body=media, fields="id").execute()
        file_id = uploaded["id"]

        # Make shareable
        service.permissions().create(
            fileId=file_id,
            body={"type": "anyone", "role": "reader"},
        ).execute()

        link = f"https://drive.google.com/file/d/{file_id}/view"
        print(f"[Drive] Uploaded: {link}")
        return link

    except Exception as e:
        print(f"[Drive] Upload failed: {e}")
        return None

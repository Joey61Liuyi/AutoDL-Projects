from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Rename the downloaded JSON file to client_secrets.json
# The client_secrets.json file needs to be in the same directory as the script.
gauth = GoogleAuth()
drive = GoogleDrive(gauth)

# # List files in Google Drive
# fileList = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
# for file1 in fileList:
#   print('title: %s, id: %s' % (file1['title'], file1['id']))

# Upload files to your Google Drive
upload_file_list = ['cifar100_non_iid_setting.npy', 'cifar10_non_iid_setting.npy']
for upload_file in upload_file_list:
    gfile = drive.CreateFile({'parents': [{'id': '1B7btb5VBlWNppuNCjo6b4F4-byZqDJV5'}]})
    # Read file and set it as a content of this instance.
    gfile.SetContentFile(upload_file)
    gfile.Upload() # Upload the file.
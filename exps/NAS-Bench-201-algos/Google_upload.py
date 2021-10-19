# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
#
# # Rename the downloaded JSON file to client_secrets.json
# # The client_secrets.json file needs to be in the same directory as the script.
# gauth = GoogleAuth()
# drive = GoogleDrive(gauth)
#
# # # List files in Google Drive
# # fileList = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
# # for file1 in fileList:
# #   print('title: %s, id: %s' % (file1['title'], file1['id']))
#
# # Upload files to your Google Drive
# upload_file_list = ['cifar100_non_iid_setting.npy', 'cifar10_non_iid_setting.npy']
# for upload_file in upload_file_list:
#     gfile = drive.CreateFile({'parents': [{'id': '1B7btb5VBlWNppuNCjo6b4F4-byZqDJV5'}]})
#     # Read file and set it as a content of this instance.
#     gfile.SetContentFile(upload_file)
#     gfile.Upload() # Upload the file.


import ast
import re

file_proposal_ours = './Ours_Search_darts.log'
file_proposal_fednas = './FedNAS_128.log'

genotype_list = {}
user_list = {}
user = 0
for line in open(file_proposal_ours):
    if "<<<--->>>" in line:
        tep_dict = ast.literal_eval(re.search('({.+})', line).group(0))
        count = 0
        for j in tep_dict['normal']:
            for k in j:
                if 'skip_connect' in k[0]:
                    count += 1
        if count == 2:
            genotype_list[user % 5] = tep_dict['normal']
            user_list[user % 5] = user / 5
        user += 1

print(genotype_list)
print(user_list)
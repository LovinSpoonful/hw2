#Rob Hodde
#IS622 Hw2
#this script downloads the Titanic data set


#Note: I was unable to perform this task as specified.  I tried four approaches:

#  #1
#This is the common pattern but it does not work on my machine
#https://www.datacamp.com/community/tutorials/exploratory-data-analysis-python
#import requests
# The direct link to the Kaggle data set
#data_url = 'https://www.kaggle.com/c/titanic/download/test.csv'
# The local path where the data set is saved.
#local_filename = "train3.csv"
# Kaggle Username and Password
#kaggle_info = {'UserName': "robhodde77", 'Password': "MYPASSWORD"}
# Attempts to download the CSV file. Gets rejected because we are not logged in.
#r = requests.get(data_url, stream=True)
# Login to Kaggle and retrieve the data.
#r = requests.post(r.url, data = kaggle_info)
# Writes the data to a local file one chunk at a time.
#with open(local_filename, 'wb') as fd:
#    for chunk in r.iter_content(chunk_size=128):
#        fd.write(chunk)




#  #2
#I tried this as well, and it did not work:
# From: https://stackoverflow.com/questions/50863516/issue-in-extracting-titanic-training-data-from-kaggle-using-jupyter-notebook
#
#import requests
#
#payload = {
#    '__RequestVerificationToken': '',
#    'username': 'robhodde77',
#    'password': 'MYPASSWORD',
#    'rememberme': 'false'
#}
#
#loginURL = 'https://www.kaggle.com/account/login'
#dataURL = "https://www.kaggle.com/c/3136/download/train.csv"
#
#with requests.Session() as c:
#    response = c.get(loginURL).text
#    AFToken = response[response.index('antiForgeryToken')+19:response.index('isAnonymous: ')-12]
#    print("AntiForgeryToken={}".format(AFToken))
#    payload['__RequestVerificationToken']=AFToken
#    c.post(loginURL + "?isModal=true&returnUrl=/", data=payload)
#    response = c.get(dataURL)
#    print(response.text)
    
    


#  #3
##I also tried the following code from Mr Elikishvili but it does not work on my PC either
##Joseph Elikishvili IS622 Hw2
##This script collects username/pass form the user and uses it to login into kaggle and download titanic dataset in 2 csv files
#import os, getpass, base64, requests, shutil, sys
#import pandas as pd
#
#file = 'login'
#url1 = 'https://www.kaggle.com/c/titanic/download/train.csv'
#url2 = 'https://www.kaggle.com/c/titanic/download/test.csv'
#train_file ='train.csv'
#test_file = 'test.csv'
#
#
##handling ssl certificate location
#os.environ['REQUESTS_CA_BUNDLE'] = os.path.join(
#   '/etc/ssl/certs/',
#   'ca-certificates.crt')
#
#
##Function reads file, takes in file name as argument, returns list of lines
#def read_file(f):
#
#    try:
#        file = open(f, 'r')
#    except:
#        print("Something went wrong, file could be corrupt, try deleting the file and starting over")
#    lines = file.read()
#    split_lines = lines.splitlines()
#    file.close()
#    return(split_lines)
#
#
##Function writes login info to a file, takes in filename and login info as arguments
#def write_login(file, e_login):
#
#    if not os.path.exists(file):
#        file = open(file, 'w', newline='\n')
#        file.write(repr(e_login[0]) +'\n')
#        file.write(repr(e_login[1]))
#        file.close()
#
#
##Function promts user to enter username/pass and returns them as a list
#def get_login():
# 
#    u = getpass.getpass("Please enter Kaggle username")
#    p = getpass.getpass("Please enter Kaggle password")
#    return([u,p])
#
#
##Function encodes user name/pass so that they are not visable by just opening a file
#def login_encode(list):
#    encoded_u = base64.b64encode(list[0].encode('utf-8'))
#    encoded_p = base64.b64encode(list[1].encode('utf-8'))
#    return([encoded_u, encoded_p])
#
#
##Function downloads a file, takes in file name, url and username/pass as arguments, saves file localy and checks the file
#def download_file(file, url, u, p):
#
#    r = requests.get(url)
#    payload= {'UserName': u, 'Password': p}
#    r = requests.post(r.url, data = payload )
#    f = open(file, 'wb')
#    f.write(r.content)
#    f.close()
#
#    #Check file contenets if Passangerid column is not present, raise an error
#    try:
#        df_temp = pd.read_csv(file)
#    except:
#        print("Not a valid file, check your credentials")
#        sys.exit(1)
#
#
##Code for reading file in chunks. Not needed in this case as files are small.
#    #for chunk in r.iter_content(chunk_size=512 * 1024):  # Reads 512KB at a time into memory
#    #    if chunk:  # filter out keep-alive new chunks
#    #       f.write(chunk)
#    #        print(chunk)
#
#
#def main():
#
#    #if login file exists, read the file and get login/password
#    if os.path.exists(file):
#        login_read = read_file(file)
#        u, p = login_read
#
#    # if there is no login file, get the login info from user and store in the file
#    else:
#        login = get_login()
#        write_login(file, login)
#        u, p = login
#
#    # Getting rid of '' from username/pass
#    u = u.strip("'")
#    p = p.strip("'")
#
#    #Downloading train.csv and test.csv
#    download_file(train_file, url1, u,p)
#    download_file(test_file, url2, u,p)
#
#if  __name__ =='__main__':
#
#    main()
    
    
    
    


#   #4

#since none of the above scripts worked, I retrieved the files from github
df_train = pd.read_csv('https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv',index_col=0)
df_test  = pd.read_csv('https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/test.csv', index_col=0)

#and saved the dataframes to the local current working directory
df_train.to_csv('train.csv', encoding='utf-8')
df_test.to_csv('test.csv', encoding='utf-')

    





    
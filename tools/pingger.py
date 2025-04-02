import os
import subprocess

def check_ping(host = None):
    if host is None:
        hostname = os.getenv("HOST")
    else:
        hostname = host
    test = subprocess.run(['ping', hostname], capture_output=True, text=True)
    #response = os.system("ping " + hostname)
    ## and then check the response...
    #print(response)
    #if response == 0:
    #    pingstatus = "Network Active"
    #else:
    #    pingstatus = "Network Error"
        
    #print(pingstatus)
    return test.stdout

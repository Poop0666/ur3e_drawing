import os

def check_ping():
    hostname = os.getenv("HOST")
    response = os.system("ping " + hostname)
    # and then check the response...
    if response == 0:
        pingstatus = "Network Active"
    else:
        pingstatus = "Network Error"
        
    print(pingstatus)

import os

def check_ping():
    hostname = "169.254.123.187"
    response = os.system("ping " + hostname)
    # and then check the response...
    if response == 0:
        pingstatus = "Network Active"
    else:
        pingstatus = "Network Error"
        
    print(pingstatus)

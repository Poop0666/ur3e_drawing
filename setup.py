import os


def create_shortcut(target, shortcut_name, ico):
    """ create a shortcut"""
    
    desktop = os.path.join(os.path.join(os.environ["USERPROFILE"]), "Desktop")
    
    shell = win32com.client.Dispatch('WScript.Shell')
    
    # create the shortcut
    shortcut = shell.CreateShortcut(os.path.join(desktop, f"{shortcut_name}.lnk"))
    
    # define the shortcut's property
    shortcut.TargetPath = target  # shortcut's target
    shortcut.WorkingDirectory = os.path.dirname(target)  
    shortcut.IconLocation = ico
    shortcut.save()  


if __name__ == "__main__":
    os.system("python -m pip install -r requirements.txt")
    import win32com.client
    target = os.path.join(os.path.dirname(__file__), "app.pyw")
    ico = os.path.join(os.path.dirname(__file__), "tools/robotic-arm.ico")
    create_shortcut(target, "UR3E", ico)
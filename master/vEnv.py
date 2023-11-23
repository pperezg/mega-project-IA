if __name__ == "__main__":

    #Basic imports for creating the virtual environment
    from subprocess import run

    createvenv = input('Please indicate your OS: Linux (L), MacOS (M), Windows (W) \n')

    while createvenv not in ['L', 'M', 'W']:
        createvenv = input('Please indicate your OS: Linux (L), MacOS (M), Windows (W) \n')

    if createvenv in ['M', 'L']:
        #Creation of the virtual environment
        run(["python", "-m", "venv", "venvIAFinal"])
        #Installation of all the necessary packages
        run(["venvIAFinal/bin/python", "-m", "pip", "install", "--upgrade", "pip"])
        run(["venvIAFinal/bin/pip", "install", "-r", "./master/requirements.txt"])

        #Runing the project files using the virtual environment
        run(["venvIAFinal/bin/python", "master/main.py"])

    if createvenv=='W':
        #Creation of the virtual environment
        run(["python", "-m", "venv", "venvIAFinal"])
        #Installation of all the necessary packages
        run(["venvIAFinal/Script/python.exe", "-m", "pip", "install", "--upgrade", "pip"])
        run(["venvIAFinal/Script/pip.exe", "install", "-r", "./master/requirements.txt"])

        #Runing the project files using the virtual environment
        run(["venvIAFinal/Script/python.exe", "master/main.py"])
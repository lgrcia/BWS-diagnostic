activate Conda35
pyinstaller -y --onefile --paths "F:\py_ProtoBWS_1.0" --add-data="F:\py_ProtoBWS_1.0/data/parameters.cfg;data" --add-data="F:\py_ProtoBWS_1.0/images/cern_logo.jpg;images" main.py


@echo off
python -m PyInstaller --additional-hooks-dir=hooks --add-data="C:\Users\ZM\Desktop\ssmc-infrastructure\ADC drive\ADCS\venv\Lib\site-packages\tensorflow;." --hidden-import tensorflow --noconsole --onefile --clean --icon="C:\Users\ZM\Desktop\ssmc-infrastructure\ADC drive\ADCS\assets\icon.ico" --name ADCS start_adcs.py

python -m PyInstaller --additional-hooks-dir=hooks --add-data="C:\Users\ZM\AppData\Roaming\Python\Python38\site-packages\tensorflow\python\util;." --hidden-import=tensorflow --noconsole --onefile --clean --icon="C:\Users\ZM\Desktop\ssmc-infrastructure\ADC drive\ADCS\assets\icon.ico" --name ADCS start_adcs.py

python -m PyInstaller --paths venv\Lib\site-packages\tensorflow\tensorflow --hidden-import=tensorflow._api.v2.compat --hidden-import=jinja2 --hidden-import=pkg_resources --noconsole --onefile --clean --icon="C:\Users\ZM\Desktop\ssmc-infrastructure\ADC drive\ADCS\assets\icon.ico" --name ADCS ADCS.spec

python -m PyInstaller --additional-hooks-dir=hooks --onefile --clean ADCS.spec

pause
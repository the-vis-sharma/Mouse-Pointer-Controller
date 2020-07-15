:: create virtual env
virtualenv myenv

:: activate virtual env
call myenv\Scripts\activate

:: setup env vars for intel OpenVino toolkit
call "C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"

:: install project dependancies
call pip install -r requirements.txt

:: download face detection model from intel OpenVino model zoo
call python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\tools\model_downloader\downloader.py" --name "face-detection-adas-binary-0001" -o model

:: download facial landmarks detection model from OpenVino model zoo
call python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\tools\model_downloader\downloader.py" --name "landmarks-regression-retail-0009" -o model 

:: download head pose estimation model from OpenVino model zoo
call python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\tools\model_downloader\downloader.py" --name "head-pose-estimation-adas-0001" -o model

:: download gaze estimation model from OpenVino model zoo
call python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\tools\model_downloader\downloader.py" --name "gaze-estimation-adas-0002" -o model 


:: python app.py -f ..\model\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001.xml -l ..\model\intel\landmarks-regression-retail-0009\FP16\landmarks-regression-retail-0009.xml -hp ..\model\intel\head-pose-estimation-adas-0001\FP16\head-pose-estimation-adas-0001.xml -g ..\model\intel\gaze-estimation-adas-0002\FP16\gaze-estimation-adas-0002.xml -i ..\bin\demo.mp4 -v f l hp g
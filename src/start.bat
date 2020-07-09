call "C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"

:: virtualenv myenv

call ..\myenv\Scripts\activate

call python app.py -f ..\model\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001.xml -l ..\model\intel\landmarks-regression-retail-0009\FP16\landmarks-regression-retail-0009.xml -hp ..\model\intel\head-pose-estimation-adas-0001\FP16\head-pose-estimation-adas-0001.xml -i ..\bin\test.png
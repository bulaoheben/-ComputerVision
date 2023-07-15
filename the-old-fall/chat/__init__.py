import pythoncom
from aip import AipSpeech
from pyaudio import PyAudio, paInt16
import requests, json, wave, time, pyttsx3
from win32com import client

import sys



# 百度AI，配置信息
from pyttsx3.drivers import sapi5

APP_ID = '36099346'  # 百度ai应用id
API_KEY = 'XVQLMdyynkaeLX14AniSXyT1'  # 百度ai应用的键值
SECRET_KEY = '8fncxHf8EVhddevGnmkdf51RwkSdq3Af'
client1 = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
# 图灵机器，配置信息
TulingUrl = 'http://openapi.tuling123.com/openapi/api/v2'
turing_api_key = "8e10ed55bbbc453696d3ac52519d457e"  # 图灵机器人api_key


# 读取文件
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


# 用户语音输入
def SaveVoice():
    pa = PyAudio()
    wf = wave.open(r'T.wav', 'wb')  # 打开wav文件
    wf.setnchannels(1)  # 配置声道数
    wf.setsampwidth(2)  # 采样宽度2bytes
    wf.setframerate(16000)  # 采样率
    stream = pa.open(format=paInt16, channels=wf.getnchannels(), rate=wf.getframerate(), input=True,
                     frames_per_buffer=1024)  # 打开一个stream
    buff = []  # 存储声音信息
    start = time.time()  # 开始运行时间戳
    print('用户说：')
    while time.time() < start + 6:  # 录制6秒
        buff.append(stream.read(wf.getframerate()))
    stream.close()  # 关闭stream
    pa.terminate()
    wf.writeframes(b''.join(buff))
    wf.close()  # 关闭wave


# 接受机器人响应
def RobotSpeakText(usersay="你好"):
    robot = {
        "perception": {
            "inputText": {
                "text": usersay
            }
        },
        "userInfo": {
            "apiKey": turing_api_key,
            "userId": 'Dudu'
        }
    }
    response = json.loads(requests.post(TulingUrl, None, robot, headers={'Content-Type': 'charset=UTF-8'}).text)
    return str(response['results'][0]['values']['text'])


# 语音发声
def RobotVoice(robotsay):
    # engine = pyttsx3.init('sapi5')
    # engine = pyttsx3.init()
    # engine.setProperty('rate', 100)
    # engine.setProperty('volume', 0.6)  # 音量
    # engine.say(robotsay)
    # print(111)
    # engine.runAndWait()

    # tts = pyttsx3.Engine()
    # tts = pyttsx3.init()
    # voices = tts.getProperty('voices')
    # for voice in voices:
    #     print('id = {} \n name = {} \n'.format(voice.id, voice.name))
    pythoncom.CoInitialize()
    engine = client.Dispatch("SAPI.SpVoice")
    engine.Speak(robotsay)


# 主函数
def main():
    while True:
        # try:
        #     SaveVoice()
        #     result = client.asr(get_file_content('T.wav'), 'wav', 16000, {'dev_pid': 1536})  # 识别本地文件
        #     usersay = result['result'][0]
        #     print(usersay)
        #     robotsay = RobotSpeakText(usersay)
        #     print('小嘟嘟说：')
        #     print(robotsay)
        #     if robotsay is None:
            robotsay = "我没听清，你在说一遍"
            RobotVoice(robotsay)
        # except Exception as e:  # 异常处理
        #     print("出现异常", e)
        #     break


if __name__ == '__main__':
    main()

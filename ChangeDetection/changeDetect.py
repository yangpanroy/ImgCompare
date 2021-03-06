import argparse
import shutil
import os
import subprocess
from SDAEChangeDetector import *


def rmDir(dirPath):
    """
    删除文件夹
    :param dirPath: 要删除的文件夹路径
    :return:
    """
    ls = os.listdir(dirPath)
    for i in ls:
        tempPath = os.path.join(dirPath, i)
        if os.path.isdir(tempPath):
            rmDir(tempPath)
        else:
            os.remove(tempPath)
    shutil.rmtree(dirPath)


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--img1Path', type=str, default=None)
parser.add_argument('--img2Path', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--predictMode', type=str, default="build")
args = parser.parse_args()
argsList = [args.img1Path, args.img2Path, args.outPath]


def checkStatus(status):
    if not status:
        sta = 1
        cmd = 'C:\\Python27\\ArcGIS10.2\\python  ' \
              'D:\\new_code\\yptest\\updateChange.py ' \
              '--out_path {} --status {} '.format(argsList[2], sta)
        subprocess.call(cmd, shell=True)


if None in argsList:
    print("路径参数不能为空！")
    checkStatus(False)
else:
    print("正在创建临时工作环境")
    workspace = "E:/ypTest/changeDetect/"
    if not os.path.exists(workspace):
        os.makedirs(workspace)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽 tensorflow 的 debug 信息
    sdaeChangeDetector = SDAEChangeDetector(argsList[0], argsList[1], argsList[2], workspace)
    status = sdaeChangeDetector.splitDualPhaseImage()
    checkStatus(status)
    status = sdaeChangeDetector.batchPredict(batchSize=1000000, predictMode=args.predictMode, imgType=["RGB", "BGR"])
    checkStatus(status)
    if status:
        sta = 0
        cmd = 'C:\\Python27\\ArcGIS10.2\\python  ' \
              'D:\\new_code\\yptest\\updateChange.py ' \
              '--out_path {} --status {} '.format(argsList[2], sta)
        subprocess.call(cmd, shell=True)
    rmDir(workspace)

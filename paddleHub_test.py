#paddlehub


# #下载文件
# import wget
# #网络地址
# ulr='https://paddlehub.bj.bcebos.com/resources/test_image.jpg'
# #获取文件名
# filename=wget.filename_from_url(ulr)
# print(filename)
# #下载文件
# #out可以指定路径，默认为当前目录
# filename=wget.download(ulr)

#加载预训练模型，模型即软件
#hub install/uninstall,完成模型的安装、升级、卸载

import paddlehub as hub
#人像抠图
#指定模型名称,模型名称均通过hub.Module API来指定
# model=hub.Module(name='deeplabv3p_xception65_humanseg')
# #指定待预测的图片路径，输出结果的路径，执行并输出预测结果
# #model.segmentation用于执行图像分割类的预测任务，查阅文档
# #output_dir指定保存路径，不知指定的文件
# res=model.segmentation(paths=['./test_image.jpg'],visualization=True,output_dir='./output')

#人脸检测
model=hub.Module(name='ultra_light_fast_generic_face_detector_1mb_640')
res=model.face_detection(paths=['./test_image.jpg'],visualization=True,output_dir='./output')





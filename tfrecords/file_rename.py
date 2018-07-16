#coding:utf8
import os
import random

def rename():
        i=0
        test_percentage = 0
        path="./YouTu"
        addname_train = 'voc_2007_train'
        addname_test = 'voc_2007_test'
        filelist=os.listdir(path)#该文件夹下所有的文件（包括文件夹）
        filelen = filelist.__len__()
        ind_test = random.sample(range(filelen),round(filelen * test_percentage/100))
        testfilelist = [filelist[i] for i in ind_test]
        trainfilelist = [x for x in filelist if x not in testfilelist]
        for files in trainfilelist:#遍历所有文件
            i=i+1
            Olddir=os.path.join(path,files);#原来的文件路径                
            if os.path.isdir(Olddir):#如果是文件夹则跳过
                    continue
            filename=os.path.splitext(files)[0];#文件名
            filetype=os.path.splitext(files)[1];#文件扩展名

            Newdir=os.path.join(path,addname_train+files);#新的文件路径
            os.rename(Olddir,Newdir)#重命名
        
        for files in testfilelist:#遍历所有文件
            i=i+1
            Olddir=os.path.join(path,files);#原来的文件路径                
            if os.path.isdir(Olddir):#如果是文件夹则跳过
                    continue
            filename=os.path.splitext(files)[0];#文件名
            filetype=os.path.splitext(files)[1];#文件扩展名

            Newdir=os.path.join(path,addname_test+files);#新的文件路径
            os.rename(Olddir,Newdir)#重命名
rename()

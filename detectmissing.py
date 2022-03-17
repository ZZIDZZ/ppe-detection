from cProfile import label
import os, shutil, glob, math

print(os.getcwd())
os.chdir("./dataset2")

# train_percentage = 70
# val_ratio = 20

# def move(file, dir):
#     for f in glob.glob(f"./{file}*"):
#         print(f)
#         shutil.copy(f"{f}", f"./{dir}/{f}")

# def moveExt(ext, dir):
#     try:
#         os.mkdir(dir)
#     except:
#         pass
#     for f in glob.glob(f"./*.{ext}"):
#         print(f)
#         shutil.copy(f"{f}", f"./{dir}/{f}")

imageFile = os.listdir('./images/trainval')
labelFile = os.listdir('./labels/trainval')

for i,f in enumerate(labelFile):
    labelFile[i] = f[:-4]    
print(imageFile)
print(labelFile)
for f in imageFile:
    if f[:-4] not in labelFile:
        print(f)
# for handle no duplicate

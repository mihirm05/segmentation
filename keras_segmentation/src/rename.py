import os 

path = 'annotations_prepped_train'
for f in os.listdir(path):
	old_name = f 
	new_name = f.replace('mask', 'Clipped')
	print(new_name)
	os.rename(path+'/'+old_name, path+'/'+new_name)

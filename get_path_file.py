import os
def get_absolute_path(base_path):
    result = []
    for maindir,subdir,file_name_list in os.walk(base_path):
        for filename in file_name_list:
            filename_split=filename.split('/')
            etx=os.path.splitext(filename_split[len(filename_split)-1])[1]
            if not filename.startswith('.DS_Store'):
                if  '.jpg' in etx :
                    apath = os.path.join(maindir, filename)
                    result.append(apath)
    return result



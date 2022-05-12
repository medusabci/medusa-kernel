
def request_files_to_dict(request):
    """ Extracts the files within a flask request corresponding to parameter key and and returns them in a list"""
    files = dict()
    for key in request.files.keys():
        files[key] = list()
        for value in request.files.getlist(key):
            files[key].append(value.read())

    return files

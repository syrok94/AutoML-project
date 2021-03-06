import pickle

class saving_file:

    def __init__(self):
        pass

    def save_file(self,model,file_path):
        try:
            self.model=model
            self.file_path=file_path

            with open(file_path+"/"+str(self.model)+".pickle","wb") as f:
                pickle.dump(self.model,f)


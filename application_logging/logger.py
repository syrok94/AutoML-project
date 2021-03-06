import datetime as datetime



class app_logger:

    def __init__(self):
        pass

    def log(self,file_object,message):
        self.file_object=file_object
        self.now=datetime.datetime.now()
        self.date=self.now.date()
        self.current_time=self.now.strftime("%H:%M:%S")

        self.file_object.write(str(self.date)+"\t"+str(self.current_time)+"\t\t"+message+"\n")

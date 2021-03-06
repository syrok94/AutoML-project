from flask import Flask,render_template,request,Response
from TrainingFile.training import Training
import warnings
from application_logging.logger import app_logger

warnings. filterwarnings(action='ignore', category=DeprecationWarning)


app=Flask(__name__)
path="pima-indians-diabetes.csv"


@app.route('/',methods=['GET'])
def home():
    #a_log=app_logger()

    #file_object="trainingLOG/train_log.txt"
    #a_log.log(file_object,"training started!")

    model_train = Training(path)

    model_train.train_model()



    return "training complete!"




'''@app.route('/train')
def train():
    try:
        model_train=Training()

        model_train.train_model(path)
    except Exception as e:
        raise e
'''






if __name__=="__main__":
    app.run(debug=True)
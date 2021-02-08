DEBUG = True
DEVELOPMENT = True
SECRET_KEY='mYysupersecretkeyy'
SQLALCHEMY_DATABASE_URI='mysql+pymysql://tomatouser:supersecrettomatopass@localhost/tomatoes'
SQLALCHEMY_TRACK_MODIFICATIONS = False
FILE_PATH = "/FILEPATH"

# models
THREE_CLASS_MODEL = "/home/anton/fermata/codes/a_leaf_classifier/trained_models/cityf_41cl_s3_35_acc946.md"


USING_MODEL_NAME = 'densenet169'
#USING_MODEL_NAME = 'vgg19'
#NUM_CLASSES_USED = 9
#NUM_CLASSES_USED = 11
#NUM_CLASSES_USED = 33
NUM_CLASSES_USED = 41
QUERY_AGE = 4#32000

BLOCKTIME = 10
BLOCKREQUESTS = 10

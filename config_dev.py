DEBUG = True
DEVELOPMENT = True
SECRET_KEY='mYysupersecretkeyy'
SQLALCHEMY_DATABASE_URI='mysql+pymysql://tomatouser:supersecrettomatopass@localhost/tomatoes'
SQLALCHEMY_TRACK_MODIFICATIONS = False
FILE_PATH = "/home/andrey/data/TOMATOES"

# models
THREE_CLASS_MODEL = "/home/anton/fermata/codes/a_leaf_classifier/trained_models/salat_health_cityfarm_50_acc1.md"


USING_MODEL_NAME = 'vgg19'
NUM_CLASSES_USED = 3
QUERY_AGE = 4#32000

BLOCKTIME = 10
BLOCKREQUESTS = 10

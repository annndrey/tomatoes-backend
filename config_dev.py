DEBUG = True
DEVELOPMENT = True
SECRET_KEY='mYysupersecretkeyy'
SQLALCHEMY_DATABASE_URI='mysql+pymysql://tomatouser:supersecrettomatopass@localhost/tomatoes'
SQLALCHEMY_TRACK_MODIFICATIONS = False
FILE_PATH = "/home/andrey/data/TOMATOES"

# models
PLANT_OR_NOT_PATH = "path_to_model.md"
POTATO_FIT_OR_NOT_PATH = "path+to-model.md"


USING_MODEL_NAME = 'vgg19'
NUM_CLASSES_USED = 2
QUERY_AGE = 4

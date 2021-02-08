DEBUG = True
DEVELOPMENT = True
SECRET_KEY='mYysupersecretkeyy'
SQLALCHEMY_DATABASE_URI='mysql+pymysql://tomatouser:supersecrettomatopass@localhost/tomatoes'
SQLALCHEMY_TRACK_MODIFICATIONS = False
FILE_PATH = "/path/TOMATOES"

# models
PLANT_OR_NOT_PATH = "path.md"
LEAF_OR_NOT_PATH = "path.md"
TOMAT_OR_NOT_PATH = "path.md"
PLANT_HEALTH_OR_NOT_PATH = "path.md"
TOMAT_HEALTH_OR_NOT_PATH = "path.md"


USING_MODEL_NAME = 'vgg19'
NUM_CLASSES_USED = 2
QUERY_AGE = 432000

BLOCKTIME = 10
BLOCKREQUESTS = 10
